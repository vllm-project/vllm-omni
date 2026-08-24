# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Length buckets and fixed-shape denoise batches for Irodori-TTS."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from .precision import (
    TRAINED_POLICY,
    IrodoriPrecisionPolicy,
    resolve_precision_policy,
)

if TYPE_CHECKING:
    from .sampler import ConditionBundle, ContextKVCache, IrodoriSamplingState


DEFAULT_LATENT_BUCKET_SECONDS = tuple(float(seconds) for seconds in range(2, 31, 2))
DEFAULT_CONTEXT_BUCKET_TOKENS = (8, 16, 32, 64, 128, 256, 512, 1024)
IRODORI_PRECISION_ENV = "VLLM_OMNI_IRODORI_PRECISION"
IRODORI_FUSED_PROJECTIONS_ENV = "VLLM_OMNI_IRODORI_FUSED_PROJECTIONS"
IRODORI_BATCH_PREPARE_ENV = "VLLM_OMNI_IRODORI_BATCH_PREPARE_ENCODE"


def _environment_bool(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of 1/0, true/false, yes/no, or on/off; got {value!r}.")


def _positive_int(value: Any, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
    return int(value)


def _positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive number, got {value!r}.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}.")
    return result


def _sequence(value: Any, *, name: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError(f"{name} must be a non-empty sequence.")
    return list(value)


def _deduplicate_increasing(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    result: list[int] = []
    for value in values:
        if value <= 0:
            raise ValueError(f"{name} values must be positive, got {value}.")
        if result and value == result[-1]:
            continue
        if result and value < result[-1]:
            raise ValueError(f"{name} must be increasing after runtime conversion.")
        result.append(value)
    if not result:
        raise ValueError(f"{name} produced no usable buckets.")
    return tuple(result)


def _round_up(value: int, granularity: int) -> int:
    return math.ceil(value / granularity) * granularity


@dataclass(frozen=True)
class IrodoriLengthState:
    """Logical and physical audio lengths for one request."""

    valid_codec_frames: int
    valid_latent_len: int
    bucket_latent_len: int
    target_samples: int
    is_dynamic_bucket: bool


@dataclass(frozen=True)
class IrodoriExecutionKey:
    """Worker-side key for one physically homogeneous denoise microbatch."""

    # ``None`` selects packed variable-length execution.  A concrete bucket
    # keeps the padded fallback isolated by physical latent shape.
    bucket_latent_len: int | None
    dtype: torch.dtype
    device_type: str
    device_index: int | None
    cfg_guidance_mode: str
    cfg_layout: tuple[str, ...]
    # Refreshing requests run every CFG branch; reusing requests run only the
    # conditional branch, so the two cannot share one physical forward.
    cfg_refresh: bool = True


@dataclass(frozen=True)
class IrodoriBatchingConfig:
    latent_bucket_seconds: tuple[float, ...]
    overflow_bucket_seconds: float
    context_bucket_tokens: tuple[int, ...]
    cfg_refresh_interval: int
    precision_policy: IrodoriPrecisionPolicy
    fuse_linear_projections: bool
    batch_prepare_encode: bool

    @classmethod
    def from_od_config(cls, od_config: Any) -> IrodoriBatchingConfig:
        extras = dict(getattr(od_config, "extras", {}) or {})
        latent_seconds = tuple(
            _positive_float(value, name="irodori_latent_bucket_seconds")
            for value in _sequence(
                extras.get("irodori_latent_bucket_seconds", DEFAULT_LATENT_BUCKET_SECONDS),
                name="irodori_latent_bucket_seconds",
            )
        )
        if any(right <= left for left, right in zip(latent_seconds, latent_seconds[1:])):
            raise ValueError("irodori_latent_bucket_seconds must be strictly increasing.")

        context_tokens = tuple(
            _positive_int(value, name="irodori_context_bucket_tokens")
            for value in _sequence(
                extras.get("irodori_context_bucket_tokens", DEFAULT_CONTEXT_BUCKET_TOKENS),
                name="irodori_context_bucket_tokens",
            )
        )
        if any(right <= left for left, right in zip(context_tokens, context_tokens[1:])):
            raise ValueError("irodori_context_bucket_tokens must be strictly increasing.")

        if "irodori_batch_prepare_encode" in extras:
            batch_prepare_encode = extras["irodori_batch_prepare_encode"]
            if not isinstance(batch_prepare_encode, bool):
                raise ValueError("irodori_batch_prepare_encode must be a boolean.")
        else:
            batch_prepare_encode = _environment_bool(
                IRODORI_BATCH_PREPARE_ENV,
                default=True,
            )

        if "irodori_fuse_linear_projections" in extras:
            fuse_linear_projections = extras["irodori_fuse_linear_projections"]
            if not isinstance(fuse_linear_projections, bool):
                raise ValueError("irodori_fuse_linear_projections must be a boolean.")
        else:
            fuse_linear_projections = _environment_bool(
                IRODORI_FUSED_PROJECTIONS_ENV,
                default=True,
            )

        return cls(
            latent_bucket_seconds=latent_seconds,
            overflow_bucket_seconds=_positive_float(
                extras.get("irodori_overflow_bucket_seconds", 8.0),
                name="irodori_overflow_bucket_seconds",
            ),
            context_bucket_tokens=context_tokens,
            # Stage ``extras`` is the request/config-level knob.  Keep the
            # environment fallback for process-wide deployments and give it
            # lower precedence than an explicit stage setting.
            precision_policy=resolve_precision_policy(
                extras.get(
                    "irodori_precision_profile",
                    os.environ.get(IRODORI_PRECISION_ENV, TRAINED_POLICY.name),
                )
            ),
            fuse_linear_projections=fuse_linear_projections,
            # Encoding a whole admission group in one pass shifts the encoder
            # output by a couple of bf16 ulp, which is invisible for a request
            # that pins ``seconds`` but can move a *predicted* duration across
            # a rounding boundary — at most one codec frame either way. Set
            # False when a deployment needs predicted lengths to be identical
            # regardless of what else is in flight.
            batch_prepare_encode=batch_prepare_encode,
            # 1 recomputes every CFG branch on every step, matching the
            # reference sampler exactly.  Higher values reuse the last
            # correction in between and trade fidelity for speed.
            cfg_refresh_interval=_positive_int(
                extras.get("irodori_cfg_refresh_interval", 1),
                name="irodori_cfg_refresh_interval",
            ),
        )


class IrodoriLatentBucketPolicy:
    """Convert exact sample lengths into runtime-derived DiT token buckets."""

    def __init__(
        self,
        *,
        sample_rate: int,
        hop_length: int,
        latent_patch_size: int,
        bucket_seconds: Sequence[float],
        overflow_bucket_seconds: float,
        max_output_seconds: float = 30.0,
    ) -> None:
        self.sample_rate = _positive_int(sample_rate, name="sample_rate")
        self.hop_length = _positive_int(hop_length, name="hop_length")
        self.latent_patch_size = _positive_int(latent_patch_size, name="latent_patch_size")
        self.max_output_seconds = _positive_float(max_output_seconds, name="max_output_seconds")
        converted = [self.seconds_to_latent_len(value) for value in bucket_seconds]
        self.standard_buckets = _deduplicate_increasing(
            converted,
            name="irodori_latent_bucket_seconds",
        )
        self.overflow_granularity = self.seconds_to_latent_len(overflow_bucket_seconds)

    def seconds_to_latent_len(self, seconds: float) -> int:
        seconds = _positive_float(seconds, name="bucket seconds")
        samples = math.ceil(seconds * self.sample_rate)
        codec_frames = math.ceil(samples / self.hop_length)
        return math.ceil(codec_frames / self.latent_patch_size)

    def lengths_for_samples(self, target_samples: int) -> IrodoriLengthState:
        target_samples = _positive_int(target_samples, name="target_samples")
        maximum_samples = math.ceil(self.max_output_seconds * self.sample_rate)
        if target_samples > maximum_samples:
            raise ValueError(
                f"Irodori output exceeds the {self.max_output_seconds:g}-second limit: {target_samples} samples."
            )
        valid_codec_frames = math.ceil(target_samples / self.hop_length)
        valid_latent_len = math.ceil(valid_codec_frames / self.latent_patch_size)
        bucket_latent_len, dynamic = self.select_bucket(valid_latent_len)
        return IrodoriLengthState(
            valid_codec_frames=valid_codec_frames,
            valid_latent_len=valid_latent_len,
            bucket_latent_len=bucket_latent_len,
            target_samples=target_samples,
            is_dynamic_bucket=dynamic,
        )

    def lengths_for_predicted_frames(self, valid_codec_frames: int) -> IrodoriLengthState:
        valid_codec_frames = _positive_int(valid_codec_frames, name="valid_codec_frames")
        maximum_frames = math.ceil(self.max_output_seconds * self.sample_rate / self.hop_length)
        if valid_codec_frames > maximum_frames:
            raise ValueError(
                f"Irodori predicted output exceeds the {self.max_output_seconds:g}-second limit: "
                f"{valid_codec_frames} codec frames."
            )
        valid_latent_len = math.ceil(valid_codec_frames / self.latent_patch_size)
        bucket_latent_len, dynamic = self.select_bucket(valid_latent_len)
        return IrodoriLengthState(
            valid_codec_frames=valid_codec_frames,
            valid_latent_len=valid_latent_len,
            bucket_latent_len=bucket_latent_len,
            target_samples=valid_codec_frames * self.hop_length,
            is_dynamic_bucket=dynamic,
        )

    def select_bucket(self, valid_latent_len: int) -> tuple[int, bool]:
        valid_latent_len = _positive_int(valid_latent_len, name="valid_latent_len")
        for bucket in self.standard_buckets:
            if valid_latent_len <= bucket:
                return bucket, False
        return _round_up(valid_latent_len, self.overflow_granularity), True


class IrodoriContextBucketPolicy:
    """Select fixed token buckets for text, speaker, and caption contexts."""

    def __init__(self, buckets: Sequence[int]) -> None:
        self.standard_buckets = _deduplicate_increasing(
            [_positive_int(value, name="context bucket") for value in buckets],
            name="irodori_context_bucket_tokens",
        )

    def select_bucket(self, valid_length: int) -> tuple[int, bool]:
        valid_length = max(1, int(valid_length))
        for bucket in self.standard_buckets:
            if valid_length <= bucket:
                return bucket, False
        return _round_up(valid_length, self.standard_buckets[-1]), True


def _pad_sequence(value: torch.Tensor, target: int) -> torch.Tensor:
    if value.shape[1] < target:
        shape = list(value.shape)
        shape[1] = target - value.shape[1]
        value = torch.cat((value, value.new_zeros(shape)), dim=1)
    return value[:, :target]


def _pack_required(values: Sequence[torch.Tensor], target: int) -> torch.Tensor:
    return torch.cat([_pad_sequence(value, target) for value in values], dim=0).contiguous()


def _pack_optional(values: Sequence[torch.Tensor | None], target: int) -> torch.Tensor | None:
    if not any(value is not None for value in values):
        return None
    if not all(value is not None for value in values):
        raise ValueError("Cannot pack mixed optional Irodori condition tensors.")
    return _pack_required([value for value in values if value is not None], target)


def _source_bucket(
    prefix_lengths: Sequence[int | None],
    *,
    policy: IrodoriContextBucketPolicy,
) -> tuple[int, bool]:
    """Pick one context bucket from per-request prefix lengths.

    The lengths come precomputed from ``IrodoriSamplingState`` so that
    selecting a bucket costs no device synchronization on the step path.
    """
    present = [length for length in prefix_lengths if length is not None]
    if not present:
        return 1, False
    return policy.select_bucket(max(present))


def _pack_bundles(
    bundles: Sequence[ConditionBundle],
    context_buckets: tuple[int, int, int],
) -> ConditionBundle:
    text_bucket, speaker_bucket, caption_bucket = context_buckets
    return (
        _pack_required([bundle[0] for bundle in bundles], text_bucket),
        _pack_required([bundle[1] for bundle in bundles], text_bucket),
        _pack_optional([bundle[2] for bundle in bundles], speaker_bucket),
        _pack_optional([bundle[3] for bundle in bundles], speaker_bucket),
        _pack_optional([bundle[4] for bundle in bundles], caption_bucket),
        _pack_optional([bundle[5] for bundle in bundles], caption_bucket),
    )


def _pack_context_kv(
    states: Sequence[IrodoriSamplingState],
    caches: Sequence[ContextKVCache | None],
    context_buckets: tuple[int, int, int],
) -> ContextKVCache | None:
    if not any(cache is not None for cache in caches):
        return None
    if not all(cache is not None for cache in caches):
        raise ValueError("Cannot pack mixed Irodori context K/V cache modes.")
    present = [cache for cache in caches if cache is not None]
    layer_count = len(present[0])
    if any(len(cache) != layer_count for cache in present):
        raise ValueError("Irodori context K/V layer counts differ within a batch.")

    has_speaker = states[0].condition.speaker_state is not None
    has_caption = states[0].condition.caption_state is not None
    source_buckets = [context_buckets[0], context_buckets[0]]
    if has_speaker:
        source_buckets.extend([context_buckets[1], context_buckets[1]])
    if has_caption:
        source_buckets.extend([context_buckets[2], context_buckets[2]])

    result: list[tuple[torch.Tensor, ...]] = []
    for layer_index in range(layer_count):
        width = len(present[0][layer_index])
        if width != len(source_buckets):
            raise ValueError("Irodori context K/V layout does not match enabled condition sources.")
        if any(len(cache[layer_index]) != width for cache in present):
            raise ValueError("Irodori context K/V layouts differ within a batch.")
        result.append(
            tuple(
                _pack_required(
                    [cache[layer_index][value_index] for cache in present],
                    source_buckets[value_index],
                )
                for value_index in range(width)
            )
        )
    return result


@dataclass
class IrodoriDenoiseBatch:
    """Reusable fixed-shape inputs for one Irodori denoise-and-update call."""

    request_ids: tuple[str, ...]
    cfg_active: bool
    cfg_layout: tuple[str, ...]
    latents: torch.Tensor
    latent_mask: torch.Tensor
    timesteps: torch.Tensor
    dt: torch.Tensor
    cfg_scales: torch.Tensor
    bundle: ConditionBundle
    context_kv_cache: ContextKVCache | None
    context_buckets: tuple[int, int, int]
    # On a refresh step the packed step writes the scaled CFG correction here;
    # on a reuse step it reads the correction carried over from the last
    # refresh instead of running the unconditional branches again.
    cfg_refresh: bool = True
    cfg_correction: torch.Tensor | None = None

    @classmethod
    def make(
        cls,
        request_ids: Sequence[str],
        states: Sequence[IrodoriSamplingState],
        *,
        context_policy: IrodoriContextBucketPolicy,
        cached_batch: IrodoriDenoiseBatch | None = None,
        cfg_refresh: bool = True,
    ) -> IrodoriDenoiseBatch:
        if not states or len(request_ids) != len(states):
            raise ValueError("Irodori packed batch requires matching non-empty request/state lists.")
        cfg_active = states[0].cfg_active[states[0].step_index]
        cfg_layout = states[0].independent_names if cfg_active else ("cond",)
        if any(state.cfg_active[state.step_index] != cfg_active for state in states[1:]):
            raise ValueError("Irodori packed batch has mixed CFG activity.")
        if cfg_active and any(state.independent_names != cfg_layout for state in states[1:]):
            raise ValueError("Irodori packed batch has mixed CFG layouts.")

        bundles = [state.independent_bundle if cfg_active else state.cond_bundle for state in states]
        caches = [state.context_kv_cfg if cfg_active else state.context_kv_cond for state in states]
        prefix_lengths = [state.bundle_prefix_lengths(cfg_active) for state in states]
        text_bucket = _source_bucket(
            [lengths[0] for lengths in prefix_lengths],
            policy=context_policy,
        )
        speaker_bucket = _source_bucket(
            [lengths[1] for lengths in prefix_lengths],
            policy=context_policy,
        )
        caption_bucket = _source_bucket(
            [lengths[2] for lengths in prefix_lengths],
            policy=context_policy,
        )
        context_buckets = (text_bucket[0], speaker_bucket[0], caption_bucket[0])
        latents = torch.cat([state.latents for state in states], dim=0)
        masks = [
            state.latent_mask
            if state.latent_mask is not None
            else torch.ones(state.latents.shape[:2], dtype=torch.bool, device=state.latents.device)
            for state in states
        ]
        latent_mask = torch.cat(masks, dim=0)
        timesteps = torch.stack([state.current_timestep for state in states]).to(
            device=latents.device,
            dtype=latents.dtype,
        )
        next_timesteps = torch.stack([state.t_schedule[state.step_index + 1] for state in states]).to(
            device=latents.device, dtype=latents.dtype
        )
        dt = next_timesteps - timesteps
        scale_names = cfg_layout[1:] if cfg_active else ()
        cfg_scales = latents.new_tensor([[state.cfg_scales[name] for name in scale_names] for state in states]).reshape(
            len(states), len(scale_names)
        )

        reuses_correction = bool(cfg_active) and not cfg_refresh
        if reuses_correction:
            missing = [
                request_id
                for request_id, state in zip(request_ids, states, strict=True)
                if state.cfg_correction is None
            ]
            if missing:
                raise ValueError(f"Irodori CFG reuse step is missing a cached correction for: {missing}.")
            cfg_correction = torch.cat(
                [state.cfg_correction for state in states],  # type: ignore[misc]
                dim=0,
            )
        elif cfg_active:
            # Refresh step: the packed step writes the fresh correction here.
            cfg_correction = torch.zeros_like(latents)
        else:
            cfg_correction = None

        can_reuse = (
            cached_batch is not None
            and cached_batch.request_ids == tuple(request_ids)
            and cached_batch.cfg_active == cfg_active
            and cached_batch.cfg_layout == cfg_layout
            and cached_batch.context_buckets == context_buckets
            and cached_batch.latents.shape == latents.shape
            and cached_batch.cfg_refresh == cfg_refresh
            and (cached_batch.cfg_correction is None) == (cfg_correction is None)
        )
        if can_reuse:
            assert cached_batch is not None
            cached_batch.latents.copy_(latents)
            cached_batch.latent_mask.copy_(latent_mask)
            cached_batch.timesteps.copy_(timesteps)
            cached_batch.dt.copy_(dt)
            cached_batch.cfg_scales.copy_(cfg_scales)
            if cached_batch.cfg_correction is not None:
                assert cfg_correction is not None
                cached_batch.cfg_correction.copy_(cfg_correction)
            return cached_batch

        return cls(
            request_ids=tuple(request_ids),
            cfg_active=cfg_active,
            cfg_layout=cfg_layout,
            latents=latents.contiguous(),
            latent_mask=latent_mask.contiguous(),
            timesteps=timesteps.contiguous(),
            dt=dt.contiguous(),
            cfg_scales=cfg_scales.contiguous(),
            bundle=_pack_bundles(bundles, context_buckets),
            context_kv_cache=_pack_context_kv(states, caches, context_buckets),
            context_buckets=context_buckets,
            cfg_refresh=bool(cfg_refresh),
            cfg_correction=None if cfg_correction is None else cfg_correction.contiguous(),
        )
