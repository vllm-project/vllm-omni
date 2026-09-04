# SPDX-License-Identifier: Apache-2.0
"""Causal autoregressive pipeline for Cosmos-Dreams checkpoints."""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.cosmos3.action import load_action_tensor, pad_action_to_dim
from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
    Cosmos3OmniDiffusersPipeline,
    get_cosmos3_ir_op_priority_func,
    get_cosmos3_post_process_func,
    get_cosmos3_pre_process_func,
)
from vllm_omni.diffusion.models.cosmos_dreams.config import CosmosDreamsManifest, deploy_option
from vllm_omni.diffusion.models.cosmos_dreams.geometry import (
    CosmosDreamsGeometry,
    CosmosDreamsResolutionPolicy,
    resolve_cosmos_dreams_geometry,
)
from vllm_omni.diffusion.models.cosmos_dreams.normalizer import ActionAffineNormalizer
from vllm_omni.diffusion.models.cosmos_dreams.state_cosmos_dreams import (
    CosmosDreamsSessionFingerprint,
    CosmosDreamsSessionState,
    append_dense_kv_history,
)
from vllm_omni.diffusion.models.cosmos_dreams.streaming_vae import decode_wan_causal_chunk
from vllm_omni.diffusion.models.cosmos_dreams.tick_adapter import parse_cosmos_dreams_tick
from vllm_omni.diffusion.models.cosmos_dreams.transformer_cosmos_dreams import (
    CosmosDreamsTransformer,
    CosmosDreamsTransformerOutput,
)
from vllm_omni.diffusion.models.cosmos_dreams.utils import (
    iter_ar_chunk_ranges,
    iter_clean_commit_frames,
    prompt_token_hash,
)
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.experimental.ar_diffusion.capability import (
    ARDiffusionCrossAttentionKVSpec,
    ARDiffusionKVBranchSpec,
    ARDiffusionKVCacheSpec,
    ARDiffusionRequestKVSpec,
    ARDiffusionRequestRejectedError,
)
from vllm_omni.experimental.ar_diffusion.tick_protocol import (
    ARDiffusionChunkMetadata,
    ARDiffusionTickRequest,
)


def _first_not_none(*values: Any) -> Any:
    """Return the first explicitly supplied value, preserving falsey inputs."""

    return next((value for value in values if value is not None), None)


def _admission_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ARDiffusionRequestRejectedError(f"Cosmos-Dreams {name} must be an integer, got {value!r}.") from exc


def _admission_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ARDiffusionRequestRejectedError(f"Cosmos-Dreams {name} must be numeric, got {value!r}.") from exc


@dataclass(frozen=True)
class _RequestControls:
    """One request's control fields, normalized across both tick transports.

    The typed ``ar_diffusion_tick`` envelope and the flat ``extra_args``
    mapping carry the same information, so normalizing once keeps admission
    free of per-field protocol branching. ``None`` means "not supplied": the
    defaults that depend on session state stay at their use site.
    """

    session_id: str
    reset: bool
    close_session: bool
    tick: bool
    measure_tick_latency: bool
    domain_id: Any
    domain_name: Any
    frame_idx: Any
    num_latent_frames: Any
    action: Any


def _resolution_policy(
    od_config: OmniDiffusionConfig,
    manifest: CosmosDreamsManifest,
) -> CosmosDreamsResolutionPolicy:
    return CosmosDreamsResolutionPolicy(
        default_resolution=deploy_option(od_config, "default_resolution", (720, 1280)),
        max_pixels=deploy_option(od_config, "max_pixels", 921_600),
        vae_spatial_compression_factor=manifest.vae_spatial_compression_factor,
        latent_patch_size=manifest.latent_patch_size,
    )


def _request_media(prompt: Any) -> Any:
    if not isinstance(prompt, Mapping):
        return None
    media = prompt.get("multi_modal_data")
    if not isinstance(media, Mapping):
        return None
    return _first_not_none(media.get("image"), media.get("video"))


def get_cosmos_dreams_pre_process_func(od_config: OmniDiffusionConfig):
    """Resolve serializable geometry, delegate media work, and validate its result."""

    manifest = CosmosDreamsManifest.from_od_config(od_config)
    policy = _resolution_policy(od_config, manifest)
    cosmos3_pre_process = get_cosmos3_pre_process_func(od_config)

    def pre_process_func(request):
        sp = request.sampling_params
        prompt = request.prompt
        media = _request_media(prompt)
        geometry = resolve_cosmos_dreams_geometry(sp, media, policy)
        sp.height, sp.width = geometry.height, geometry.width

        result = cosmos3_pre_process(request)
        # Re-resolving explicit final dimensions catches any downstream
        # alignment, area, aspect, or model-grid violation.
        final_sp = result.sampling_params
        geometry = resolve_cosmos_dreams_geometry(final_sp, None, policy)
        final_sp.height, final_sp.width = geometry.height, geometry.width
        return result

    return pre_process_func


# The registry resolves process funcs by name against this model's own
# module, so reusing the Cosmos3 implementations means re-exporting them
# under a Cosmos-Dreams name rather than registering them directly.
def get_cosmos_dreams_post_process_func(od_config: OmniDiffusionConfig):
    return get_cosmos3_post_process_func(od_config)


def get_cosmos_dreams_ir_op_priority_func(od_config: OmniDiffusionConfig):
    return get_cosmos3_ir_op_priority_func(od_config)


class CosmosDreamsPipeline(Cosmos3OmniDiffusersPipeline):
    """Cosmos3-Interactive inference with dense-or-paged persistent GEN K/V.

    The default diffusion engine exercises the dense numerical-oracle path,
    which Cosmos-Dreams-Transfer also runs on. When the AR-Diffusion runner
    binds a state, the exact same attention uses paged storage instead.
    """

    # The engine's generic warmup request is 512x512 with a one-step sampler,
    # while Cosmos-Dreams has request-resolved geometry and a four-step sampler.
    # Skip that incompatible request; AR-Diffusion owns any model-valid rollout
    # warmup when CUDA graphs are enabled.
    dummy_run_num_frames: ClassVar[int] = 0
    _transformer_cls_override: ClassVar[type[CosmosDreamsTransformer]] = CosmosDreamsTransformer
    _MAIN_BRANCH = "main"
    _SESSION_CAPACITY = 1
    _ar_diffusion_kv_state = None
    _bound_session_id: str | None = None

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        self.manifest = CosmosDreamsManifest.from_od_config(od_config)
        self.resolution_policy = _resolution_policy(od_config, self.manifest)
        self.manifest.require_exported_artifact()
        if not isinstance(self.transformer, CosmosDreamsTransformer):
            raise TypeError(
                f"Cosmos-Dreams pipeline resolved the wrong transformer type: {type(self.transformer).__name__}."
            )
        if not self.is_distilled_model:
            raise ValueError("Cosmos-Dreams requires a distilled fixed-step checkpoint.")
        scheduler_t_list = tuple(float(value) for value in self._scheduler_init_t_list)
        if len(scheduler_t_list) != 4:
            raise ValueError(
                f"Cosmos-Dreams requires exactly four distilled denoise steps, got {len(scheduler_t_list)}."
            )
        if len(scheduler_t_list) != len(self.manifest.t_list) or any(
            not math.isclose(scheduler_value, manifest_value, rel_tol=0.0, abs_tol=1e-8)
            for scheduler_value, manifest_value in zip(scheduler_t_list, self.manifest.t_list, strict=True)
        ):
            raise ValueError(
                "Cosmos-Dreams scheduler and transformer manifests define different fixed-step schedules: "
                f"scheduler={scheduler_t_list}, transformer={self.manifest.t_list}."
            )
        scheduler_train_timesteps = int(self.scheduler.config.num_train_timesteps)
        if scheduler_train_timesteps != self.manifest.num_train_timesteps:
            raise ValueError(
                "Cosmos-Dreams scheduler and transformer manifests define different training timestep counts: "
                f"scheduler={scheduler_train_timesteps}, transformer={self.manifest.num_train_timesteps}."
            )
        self._distilled_num_steps = len(scheduler_t_list)
        if od_config.parallel_config.sequence_parallel_size > 1:
            raise ValueError(
                "Cosmos-Dreams supports tensor parallelism but not sequence parallelism; "
                f"got sequence_parallel_size={od_config.parallel_config.sequence_parallel_size}."
            )
        action_schema = self.manifest.action_schema
        if action_schema is None:
            raise ValueError("Cosmos-Dreams action pipeline requires validated schema-v1 action conditioning.")
        action_schema.validate_temporal_compression_factor(self.manifest.temporal_compression_factor)
        self.action_normalizers = {
            embodiment: ActionAffineNormalizer.from_contract(contract)
            for embodiment, contract in action_schema.normalizers.items()
        }
        configured_domain_id = deploy_option(od_config, "default_domain_id")
        artifact_domain_id = action_schema.embodiment_to_domain[action_schema.default_embodiment]
        self.default_domain_id = int(artifact_domain_id if configured_domain_id is None else configured_domain_id)
        self.default_embodiment = action_schema.resolve_embodiment(
            action_schema.default_embodiment,
            self.default_domain_id,
        )
        self.default_fps = float(deploy_option(od_config, "default_fps", 15.0))
        if not math.isfinite(self.default_fps) or self.default_fps <= 0:
            raise ValueError(f"Cosmos-Dreams default_fps must be positive, got {self.default_fps}.")
        self.checkpoint_id = (
            self.manifest.checkpoint_id if self.manifest.checkpoint_id != "unknown" else str(od_config.model)
        )
        self._states: OrderedDict[str, CosmosDreamsSessionState] = OrderedDict()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Reject artifact tensors that do not map into this exact model."""

        allowed = set(self.state_dict())
        tp_aware = {name for name, parameter in self.named_parameters() if hasattr(parameter, "weight_loader")}
        unexpected: list[str] = []

        def is_export_only_tensor(name: str) -> bool:
            key = name.removeprefix("transformer.").removeprefix("model.")
            return key.startswith(("lm_head.", "action_pos_embed."))

        def checked_weights():
            for name, tensor in weights:
                remapped = self._remap_ckpt_key(name)
                if (
                    not is_export_only_tensor(name)
                    and name not in allowed
                    and name not in tp_aware
                    and (remapped is None or (remapped not in allowed and remapped not in tp_aware))
                ):
                    unexpected.append(name)
                yield name, tensor

        loaded = super().load_weights(checked_weights())
        if unexpected:
            preview = ", ".join(sorted(unexpected)[:12])
            suffix = "" if len(unexpected) <= 12 else f" (and {len(unexpected) - 12} more)"
            raise ValueError(f"Cosmos-Dreams checkpoint contains unexpected transformer tensors: {preview}{suffix}")
        return loaded

    # -- AR-Diffusion pipeline capability ---------------------------------

    def _kv_spec_for_geometry(self, geometry: CosmosDreamsGeometry) -> ARDiffusionKVCacheSpec:
        return ARDiffusionKVCacheSpec(
            num_layers=self.transformer.num_hidden_layers,
            num_kv_heads=self.transformer.num_kv_heads_local,
            head_size=self.transformer.head_dim,
            tokens_per_frame=geometry.tokens_per_frame(self.manifest.conditioning_tokens_per_frame),
            frames_per_block=1,
            window_frames=self.manifest.window_frames,
            sink_frames=self.manifest.sink_frames,
            kv_branches=(ARDiffusionKVBranchSpec(self._MAIN_BRANCH, 0),),
            session_capacity=self._SESSION_CAPACITY,
            cross_attention=(ARDiffusionCrossAttentionKVSpec("text", self.manifest.text_cache_max_len),),
            max_scratch_frames_per_branch=self.manifest.chunk_size,
            max_scratch_tokens_per_branch=self.manifest.text_cache_max_len,
        )

    def ar_diffusion_kv_cache_spec(self) -> ARDiffusionKVCacheSpec:
        """Return the default-resolution specification for static consumers."""

        geometry = self.resolution_policy.resolve(*self.resolution_policy.default_resolution)
        return self._kv_spec_for_geometry(geometry)

    def _request_kv_spec(self, geometry: CosmosDreamsGeometry) -> ARDiffusionRequestKVSpec:
        return ARDiffusionRequestKVSpec(
            kv_spec=self._kv_spec_for_geometry(geometry),
            geometry_key=geometry.session_key,
        )

    def ar_diffusion_default_request_spec(self) -> ARDiffusionRequestKVSpec:
        geometry = self.resolution_policy.resolve(*self.resolution_policy.default_resolution)
        return self._request_kv_spec(geometry)

    def ar_diffusion_request_spec(self, request: Any) -> ARDiffusionRequestKVSpec:
        geometry = resolve_cosmos_dreams_geometry(request.sampling_params, None, self.resolution_policy)
        return self._request_kv_spec(geometry)

    def ar_diffusion_worst_case_request_specs(self) -> Iterable[ARDiffusionRequestKVSpec]:
        """Yield the largest admitted request after enumerating the policy space."""

        worst = max(
            self.resolution_policy.iter_valid_geometries(),
            key=lambda geometry: (geometry.vision_tokens_per_frame, geometry.height * geometry.width),
        )
        yield self._request_kv_spec(worst)

    def validate_ar_diffusion_effective_spec(self, spec: ARDiffusionKVCacheSpec) -> None:
        """Validate runner overrides against immutable model structure at load time."""

        expected = self.ar_diffusion_kv_cache_spec()
        fields = (
            "num_layers",
            "num_kv_heads",
            "head_size",
            "frames_per_block",
            "window_frames",
            "sink_frames",
            "reset_at_boundary",
            "kv_branches",
            "session_capacity",
            "cross_attention",
            "max_model_len",
            "max_scratch_frames_per_branch",
            "max_scratch_tokens_per_branch",
            "model_owned_state_bytes_per_session",
        )
        mismatches = {
            name: (getattr(expected, name), getattr(spec, name))
            for name in fields
            if getattr(expected, name) != getattr(spec, name)
        }
        if mismatches:
            detail = ", ".join(
                f"{name}=expected {expected_value!r}, got {actual_value!r}"
                for name, (expected_value, actual_value) in mismatches.items()
            )
            raise ValueError(f"Cosmos-Dreams AR-Diffusion structural specification is invalid ({detail}).")

    def _validate_bound_kv_geometry(
        self,
        state: Any,
        geometry: CosmosDreamsGeometry | None = None,
    ) -> None:
        """Treat every bound-pool mismatch as an internal invariant failure.

        ``window_frames`` and ``sink_frames`` are checkpoint-manifest semantics
        for Cosmos-Dreams, not performance-only engine knobs, but the generic AR
        runner can apply deployment overrides. Startup validates those values;
        this bound-time gate defensively checks the cache that was actually built.
        """

        cache = state.kv_cache
        actual = {
            "num_layers": int(cache.num_layers),
            "num_kv_heads": int(cache.num_kv_heads),
            "head_size": int(cache.head_size),
            "tokens_per_frame": int(cache.block_size),
            "frames_per_block": int(cache.frames_per_block),
            "max_scratch_frames_per_branch": int(cache.max_scratch_frames_per_branch),
            "max_scratch_tokens_per_branch": int(cache.max_scratch_tokens_per_branch),
            "window_frames": int(cache.spec.window_chunks),
            "sink_frames": int(cache.spec.sink_chunks),
            "reset_at_boundary": bool(cache.spec.reset_at_boundary),
            "text_cache_max_len": int(cache.cross_attention_lengths.get("text", -1)),
            "max_model_len": int(cache.max_model_len),
            "kv_branches": tuple(cache.kv_branches),
            "model_owned_state_bytes_per_session": int(cache.model_owned_state_bytes_per_session),
        }
        expected_spec = self._kv_spec_for_geometry(
            geometry or self.resolution_policy.resolve(*self.resolution_policy.default_resolution)
        )
        expected = {
            "num_layers": int(self.transformer.num_hidden_layers),
            "num_kv_heads": int(self.transformer.num_kv_heads_local),
            "head_size": int(self.transformer.head_dim),
            "tokens_per_frame": int(expected_spec.tokens_per_frame),
            "frames_per_block": 1,
            "max_scratch_frames_per_branch": int(self.manifest.chunk_size),
            "max_scratch_tokens_per_branch": int(self.manifest.text_cache_max_len),
            "window_frames": int(self.manifest.window_frames),
            "sink_frames": int(self.manifest.sink_frames),
            "reset_at_boundary": False,
            "text_cache_max_len": int(self.manifest.text_cache_max_len),
            "max_model_len": int(expected_spec.max_model_len),
            "kv_branches": expected_spec.kv_branches,
            "model_owned_state_bytes_per_session": int(expected_spec.model_owned_state_bytes_per_session),
        }
        if geometry is None:
            # The request is resolved independently in ``forward``. At bind
            # time only the geometry-dependent block size is intentionally
            # deferred.
            expected.pop("tokens_per_frame")
            actual.pop("tokens_per_frame")
        mismatches = {name: (expected[name], actual[name]) for name in expected if expected[name] != actual[name]}
        if mismatches:
            details = ", ".join(
                f"{name}=expected {expected_value}, got {actual_value}"
                for name, (expected_value, actual_value) in mismatches.items()
            )
            raise RuntimeError(
                f"Cosmos-Dreams bound AR-Diffusion KV cache violates the resolved model specification ({details})."
            )

    @contextmanager
    def bind_ar_diffusion_state(self, session_id, state):
        if self._ar_diffusion_kv_state is not None:
            raise RuntimeError("Cosmos-Dreams AR-Diffusion state is already bound.")
        if state.session_id != session_id:
            raise ValueError(f"Cosmos-Dreams bound session mismatch: {state.session_id!r} != {session_id!r}.")
        self._validate_bound_kv_geometry(state)
        self._ar_diffusion_kv_state = state
        self._bound_session_id = str(session_id)
        try:
            yield
        finally:
            self._ar_diffusion_kv_state = None
            self._bound_session_id = None

    def reset_ar_diffusion_session(self, session_id: str) -> None:
        self._drop_session(session_id)

    def close_ar_diffusion_session(self, session_id: str) -> None:
        self._drop_session(session_id)

    def _drop_session(self, session_id: str) -> None:
        state = self._states.pop(str(session_id or "default"), None)
        if state is not None:
            if state.vae_decoder_initialized:
                self.vae.clear_cache()
            state.reset()

    def _sync_for_tick_timing(self, enabled: bool) -> None:
        if not enabled or self.device.type != "cuda":
            return
        torch.accelerator.synchronize(self.device)

    @contextmanager
    def _timed_tick_stage(
        self,
        durations: dict[str, float],
        name: str,
        *,
        enabled: bool,
    ):
        if not enabled:
            yield
            return
        self._sync_for_tick_timing(True)
        started = time.perf_counter()
        try:
            yield
        finally:
            self._sync_for_tick_timing(True)
            durations[name] = durations.get(name, 0.0) + (time.perf_counter() - started)

    @staticmethod
    def _validate_session_mode(
        *,
        tick: bool,
        state_was_new: bool,
        session_id: str,
        reset: bool,
        close_session: bool,
    ) -> None:
        """Prevent accidental history reuse by ordinary offline requests."""

        if not tick and state_was_new and not reset and not close_session:
            raise ARDiffusionRequestRejectedError(
                f"Cosmos-Dreams full rollout session {session_id!r} requires reset=True at start "
                "or close_session=True at end."
            )

    def _request_controls(self, sp, extra: Mapping, typed_tick) -> _RequestControls:
        """Read one request's controls from whichever transport supplied them."""

        if typed_tick is not None:
            typed = parse_cosmos_dreams_tick(typed_tick)
            return _RequestControls(
                session_id=str(typed_tick.session_id),
                reset=typed_tick.reset,
                close_session=typed_tick.close_session,
                tick=True,
                measure_tick_latency=typed.measure_tick_latency,
                domain_id=typed.domain_id,
                domain_name=typed.domain_name,
                frame_idx=typed.frame_idx,
                num_latent_frames=typed.num_latent_frames,
                action=typed.action,
            )
        tick = bool(extra.get("chunk_only", False))
        return _RequestControls(
            session_id=str(extra.get("session_id") or self._bound_session_id or "default"),
            reset=bool(extra.get("reset", False)),
            close_session=bool(extra.get("close_session", False)),
            tick=tick,
            measure_tick_latency=tick and bool(extra.get("measure_tick_latency", False)),
            domain_id=self._get_sp_param(sp, "domain_id", None),
            domain_name=self._get_sp_param(sp, "domain_name", None),
            frame_idx=extra.get("frame_idx"),
            num_latent_frames=extra.get("num_latent_frames"),
            action=self._get_sp_param(sp, "action", None),
        )

    # -- Session and conditioning -----------------------------------------

    def _get_or_create_state(self, session_id: str) -> CosmosDreamsSessionState:
        state = self._states.get(session_id)
        if state is None:
            while len(self._states) >= self._SESSION_CAPACITY:
                _, evicted = self._states.popitem(last=False)
                evicted.reset()
            state = CosmosDreamsSessionState(session_id=session_id)
            self._states[session_id] = state
        self._states.move_to_end(session_id)
        return state

    def _ensure_text_kv(
        self,
        state: CosmosDreamsSessionState,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        cached = state.text_kv_by_branch.get(self._MAIN_BRANCH)
        if cached is not None:
            return cached

        paged_state = self._ar_diffusion_kv_state
        if paged_state is not None and paged_state.is_cross_attention_populated(self._MAIN_BRANCH, "text"):
            pooled = paged_state.get_cross_attention_kv(self._MAIN_BRANCH, "text")
            cached = [(entry["k"], entry["v"]) for entry in pooled]
        else:
            raw_kv, real_len = self.transformer.encode_und_kv(text_ids, text_mask)
            if real_len > self.manifest.text_cache_max_len:
                raise ValueError(
                    f"Cosmos-Dreams prompt exceeds text_cache_max_len: {real_len} > {self.manifest.text_cache_max_len}."
                )
            if paged_state is None:
                cached = raw_kv
            else:
                padded = self.transformer.pad_text_kv(
                    raw_kv,
                    max_len=self.manifest.text_cache_max_len,
                )
                paged_state.populate_cross_attention(self._MAIN_BRANCH, "text", padded)
                pooled = paged_state.get_cross_attention_kv(self._MAIN_BRANCH, "text")
                cached = [(entry["k"], entry["v"]) for entry in pooled]
        state.text_kv_by_branch[self._MAIN_BRANCH] = cached
        return cached

    def _fingerprint(
        self,
        text_ids: torch.Tensor,
        *,
        real_text_kv_len: int,
        geometry: CosmosDreamsGeometry,
        fps: float,
        domain_id: int,
        embodiment: str,
    ) -> CosmosDreamsSessionFingerprint:
        return CosmosDreamsSessionFingerprint(
            prompt_hash=prompt_token_hash(text_ids),
            real_text_kv_lengths=((self._MAIN_BRANCH, real_text_kv_len),),
            height=geometry.height,
            width=geometry.width,
            fps=fps,
            domain_id=domain_id,
            embodiment=embodiment,
            action_contract_sha256=self.manifest.action_contract_sha256,
            checkpoint_id=self.checkpoint_id,
            manifest_id=self.manifest.digest,
            sampler_id=self.manifest.sampler_id,
        )

    # -- Dense/paged transformer bridge -----------------------------------

    def _append_dense_kv(
        self,
        state: CosmosDreamsSessionState,
        current_kv: list[tuple[torch.Tensor, torch.Tensor]],
        geometry: CosmosDreamsGeometry,
    ) -> None:
        history = state.dense_kv_by_branch.get(self._MAIN_BRANCH)
        state.dense_kv_by_branch[self._MAIN_BRANCH] = append_dense_kv_history(
            history,
            current_kv,
            tokens_per_frame=geometry.tokens_per_frame(self.manifest.conditioning_tokens_per_frame),
            sink_frames=self.manifest.sink_frames,
            window_frames=self.manifest.window_frames,
        )

    def _transformer_forward(
        self,
        state: CosmosDreamsSessionState,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        *,
        geometry: CosmosDreamsGeometry,
        text_kv: list[tuple[torch.Tensor, torch.Tensor]],
        real_text_kv_len: int,
        frame_start: int,
        fps: float,
        action_latents: torch.Tensor,
        action_domain_ids: torch.Tensor,
        condition_vision: bool,
        null_action_frame_indexes: tuple[int, ...],
        commit_current: bool,
    ) -> CosmosDreamsTransformerOutput:
        paged_state = self._ar_diffusion_kv_state
        tokens_per_frame = geometry.tokens_per_frame(self.manifest.conditioning_tokens_per_frame)
        seq_len = hidden_states.shape[2] * tokens_per_frame
        paged_kv = None
        dense_history = None
        if paged_state is not None:
            paged_kv = paged_state.get_kv_caches(
                self._MAIN_BRANCH,
                seq_len=seq_len,
                commit_current=commit_current,
                extra_visible_tokens=seq_len,
            )
        else:
            dense_history = state.dense_kv_by_branch.get(self._MAIN_BRANCH)

        output = self.transformer(
            hidden_states,
            timestep,
            geometry=geometry,
            text_kv=text_kv,
            real_text_kv_len=real_text_kv_len,
            frame_start=frame_start,
            fps=fps,
            action_latents=action_latents,
            action_domain_ids=action_domain_ids,
            paged_kv=paged_kv,
            dense_history=dense_history,
            condition_vision=condition_vision,
            null_action_frame_indexes=null_action_frame_indexes,
        )
        if paged_state is not None:
            paged_state.commit_paged_context(self._MAIN_BRANCH)
        elif commit_current:
            self._append_dense_kv(state, output.current_kv, geometry)
        return output

    # -- Action and latent preparation ------------------------------------

    def _prepare_raw_action(
        self,
        *,
        embodiment: str,
        action_value: Any,
    ) -> torch.Tensor | None:
        if action_value is None:
            return None
        action = load_action_tensor(action_value)
        expected_raw_action_dim = self.manifest.raw_action_dim_for(embodiment)
        if action.shape[-1] != expected_raw_action_dim:
            raise ValueError(
                f"Cosmos-Dreams embodiment {embodiment!r} requires raw action dimension "
                f"{expected_raw_action_dim}, got {action.shape[-1]}."
            )
        action = self.action_normalizers[embodiment].normalize(action)
        action = pad_action_to_dim(action, self.manifest.max_action_dim)
        return action.to(device=self.device, dtype=self.dtype)

    def _resolve_action_layout(
        self,
        raw_action: torch.Tensor | None,
        *,
        start_frame: int,
        target_frame: int,
    ) -> str | None:
        """Decide once per request how raw action rows are indexed.

        ``global``: row block ``[(f-1)*A, f*A)`` conditions latent frame ``f``
        ``local``:
        rows cover exactly this request's non-prefix frames in order (the
        chunk-per-request tick layout). Resolving once keeps the
        interpretation stable across every chunk of the request and turns
        insufficient coverage into an admission rejection instead of a
        mid-rollout failure after frames were already committed.
        """
        if raw_action is None:
            return None
        action_count = self.manifest.action_tokens_per_frame
        global_rows = max((target_frame - 1) * action_count, 0)
        local_rows = sum(action_count for frame in range(start_frame, target_frame) if frame > 0)
        rows = raw_action.shape[0]
        if rows >= global_rows:
            return "global"
        if rows == local_rows:
            return "local"
        raise ARDiffusionRequestRejectedError(
            "Cosmos-Dreams action length cannot cover the requested latent frames: "
            f"rows={rows}, frame_range=[{start_frame}, {target_frame}), "
            f"expected local rows={local_rows} or at least global rows={global_rows}."
        )

    def _actions_for_frames(
        self,
        raw_action: torch.Tensor | None,
        *,
        layout: str | None,
        request_start_frame: int,
        frame_start: int,
        frame_end: int,
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        action_count = self.manifest.action_tokens_per_frame
        frame_count = frame_end - frame_start
        if raw_action is None or layout is None:
            zeros = torch.zeros(
                1,
                frame_count * action_count,
                self.manifest.max_action_dim,
                device=self.device,
                dtype=self.dtype,
            )
            return zeros, tuple(range(frame_count))

        # First raw row conditions the first non-prefix frame of the request.
        local_base_frame = max(request_start_frame, 1)
        rows: list[torch.Tensor] = []
        null_indexes: list[int] = []
        for local_idx, frame_idx in enumerate(range(frame_start, frame_end)):
            if frame_idx == 0:
                rows.append(raw_action.new_zeros(action_count, self.manifest.max_action_dim))
                null_indexes.append(local_idx)
                continue
            if layout == "global":
                start = (frame_idx - 1) * action_count
            else:
                start = (frame_idx - local_base_frame) * action_count
            rows.append(raw_action[start : start + action_count])
        return torch.cat(rows, dim=0).unsqueeze(0), tuple(null_indexes)

    def _initial_condition_latent(
        self,
        prompt_data: Any,
        sp: Any,
        geometry: CosmosDreamsGeometry,
    ) -> torch.Tensor | None:
        explicit = self._get_sp_param(sp, "initial_latent", None)
        if explicit is not None:
            latent = explicit if isinstance(explicit, torch.Tensor) else torch.as_tensor(explicit)
            if latent.ndim == 4:
                latent = latent.unsqueeze(2)
            expected = (
                1,
                self.transformer.latent_channel_size,
                1,
                geometry.latent_height,
                geometry.latent_width,
            )
            if tuple(latent.shape) != expected:
                raise ValueError(f"Cosmos-Dreams initial_latent must have shape {expected}, got {tuple(latent.shape)}.")
            return latent.to(device=self.device, dtype=self.dtype)

        if isinstance(prompt_data, str):
            return None
        additional = prompt_data.get("additional_information", {}) or {}
        image = additional.get("preprocessed_image")
        video = additional.get("preprocessed_video")
        if image is None and isinstance(video, torch.Tensor):
            if video.ndim != 5:
                raise ValueError(
                    f"Cosmos-Dreams preprocessed video must have shape [1,3,T,H,W], got {tuple(video.shape)}."
                )
            image = video[:, :, 0]
        if image is None:
            return None
        if not isinstance(image, torch.Tensor):
            raise TypeError("Cosmos-Dreams preprocessed image must be a torch.Tensor.")
        latent = self._encode_conditioning_image_latent(image)
        expected = (
            1,
            self.transformer.latent_channel_size,
            1,
            geometry.latent_height,
            geometry.latent_width,
        )
        if tuple(latent.shape) != expected:
            raise ValueError(
                "Cosmos-Dreams encoded conditioning media does not match resolved geometry: "
                f"expected {expected}, got {tuple(latent.shape)}."
            )
        return latent

    def _commit_clean_frame(
        self,
        state: CosmosDreamsSessionState,
        latent: torch.Tensor,
        *,
        geometry: CosmosDreamsGeometry,
        frame_idx: int,
        text_kv: list[tuple[torch.Tensor, torch.Tensor]],
        real_text_kv_len: int,
        fps: float,
        action: torch.Tensor,
        domain_ids: torch.Tensor,
        null_action: bool,
    ) -> None:
        self._transformer_forward(
            state,
            latent.to(self.dtype),
            torch.zeros(1, device=self.device, dtype=torch.float32),
            geometry=geometry,
            text_kv=text_kv,
            real_text_kv_len=real_text_kv_len,
            frame_start=frame_idx,
            fps=fps,
            action_latents=action,
            action_domain_ids=domain_ids,
            condition_vision=True,
            null_action_frame_indexes=(0,) if null_action else (),
            commit_current=True,
        )

    def _denormalize_vae_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.to(device=self.device, dtype=self.vae.dtype)
        if hasattr(self.vae.config, "latents_mean") and hasattr(self.vae.config, "latents_std"):
            latents_mean, latents_std = self._get_latents_mean_std(latents.device, latents.dtype)
            return (latents * latents_std) + latents_mean
        return latents / float(getattr(self.vae.config, "scaling_factor", 1.0))

    def _decode_live_latents(
        self,
        state: CosmosDreamsSessionState,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """Decode only the new tick block with session-owned Wan features."""

        result = decode_wan_causal_chunk(
            self.vae,
            self._denormalize_vae_latents(latents),
            feature_cache=state.vae_decoder_feat_cache,
            initialized=state.vae_decoder_initialized,
        )
        state.record_incremental_decode(
            input_frames=int(latents.shape[2]),
            feature_cache=result.feature_cache,
        )
        return result.video

    # -- Generation --------------------------------------------------------

    def _denoise_chunk(
        self,
        velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        initial_noise: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Run one state-aware chunk with the inherited distilled scheduler."""
        self._set_timesteps(
            self._distilled_num_steps,
            device=initial_noise.device,
            shift=1.0,
        )
        latents = initial_noise.float()
        for timestep in self.scheduler.timesteps:
            model_timestep = timestep.expand(latents.shape[0])
            velocity = velocity_fn(latents, model_timestep)
            latents = self.scheduler.step(
                velocity,
                timestep,
                latents,
                generator=generator,
                return_dict=False,
            )[0]
        return latents

    @torch.no_grad()
    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        try:
            return self._forward_impl(req)
        except ARDiffusionRequestRejectedError:
            # Admission rejection: guaranteed to be raised before any session
            # or KV side effect, so the session (and its paid-for history)
            # survives for a corrected retry or an explicit reset.
            raise
        except Exception:
            extra = req.sampling_params.extra_args or {}
            session_id = str(extra.get("session_id") or self._bound_session_id or "default")
            self._drop_session(session_id)
            raise

    def _forward_impl(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        # ---- Admission (pure) ------------------------------------------------
        # Everything in this section only reads request and session state and
        # raises ARDiffusionRequestRejectedError on invalid input. No session may be
        # created, initialized, evicted, or written before it completes: the
        # rejection contract promises the client an unchanged session.
        if len(req.prompts) != 1:
            raise ARDiffusionRequestRejectedError("CosmosDreamsPipeline supports exactly one prompt per request.")
        prompt_data = req.prompts[0]
        if isinstance(prompt_data, str):
            prompt = prompt_data
        elif isinstance(prompt_data, Mapping):
            prompt = str(prompt_data.get("prompt", ""))
        else:
            raise ARDiffusionRequestRejectedError(
                "Cosmos-Dreams prompt input must be a string or a mapping with a 'prompt' field."
            )
        sp = req.sampling_params
        extra = {} if sp.extra_args is None else sp.extra_args
        if not isinstance(extra, Mapping):
            raise ARDiffusionRequestRejectedError(
                f"Cosmos-Dreams extra_args must be a mapping, got {type(extra).__name__}."
            )
        try:
            typed_tick = ARDiffusionTickRequest.from_extra_args(extra)
            controls = self._request_controls(sp, extra, typed_tick)
        except ValueError as exc:
            raise ARDiffusionRequestRejectedError(str(exc)) from exc
        if typed_tick is not None:
            if self._ar_diffusion_kv_state is None:
                raise ARDiffusionRequestRejectedError(
                    "Cosmos-Dreams typed ticks require ARDiffusionEngine session binding."
                )
            if typed_tick.prompt is not None and typed_tick.prompt != prompt:
                raise ARDiffusionRequestRejectedError(
                    "Cosmos-Dreams ar_diffusion_tick.prompt must match the standard request prompt."
                )
        session_id = controls.session_id
        if self._bound_session_id is not None and session_id != self._bound_session_id:
            raise ARDiffusionRequestRejectedError(
                f"Cosmos-Dreams request session {session_id!r} does not match bound session {self._bound_session_id!r}."
            )

        reset = controls.reset
        close_session = controls.close_session
        tick = controls.tick
        measure_tick_latency = controls.measure_tick_latency
        tick_output_type = "latent" if sp.output_type == "latent" else "video"
        existing_state = self._states.get(session_id)
        if reset or existing_state is None or existing_state.fingerprint is None:
            existing_state = None
        state_was_new = existing_state is None
        self._validate_session_mode(
            tick=tick,
            state_was_new=state_was_new,
            session_id=session_id,
            reset=reset,
            close_session=close_session,
        )

        try:
            geometry = resolve_cosmos_dreams_geometry(sp, None, self.resolution_policy)
        except (TypeError, ValueError) as exc:
            raise ARDiffusionRequestRejectedError(str(exc)) from exc
        if self._ar_diffusion_kv_state is not None:
            # The runner resolved the same serialized H/W to select this pool.
            # A mismatch here is an internal invariant failure, not a client
            # admission error, and therefore follows failed-forward cleanup.
            self._validate_bound_kv_geometry(self._ar_diffusion_kv_state, geometry)
        fps = _admission_float(
            _first_not_none(
                self._get_sp_param(sp, "resolved_frame_rate", None),
                self._get_sp_param(sp, "frame_rate", None),
                self._get_sp_param(sp, "fps", None),
                self.default_fps,
            ),
            "FPS",
        )
        if not math.isfinite(fps) or fps <= 0:
            raise ARDiffusionRequestRejectedError(f"Cosmos-Dreams FPS must be positive, got {fps}.")
        domain_name = controls.domain_name
        domain_value = controls.domain_id
        if domain_value is None and domain_name is None:
            domain_value = self.default_domain_id
        if domain_value is not None:
            domain_id = _admission_int(domain_value, "domain_id")
            if domain_id < 0:
                raise ARDiffusionRequestRejectedError(f"Cosmos-Dreams domain_id must be non-negative, got {domain_id}.")
        else:
            try:
                domain_id = self.manifest.resolve_domain_name(str(domain_name))
            except ValueError as exc:
                raise ARDiffusionRequestRejectedError(str(exc)) from exc
        if domain_id >= self.manifest.num_embodiment_domains:
            raise ARDiffusionRequestRejectedError(
                "Cosmos-Dreams domain_id is outside the exported embodiment table: "
                f"{domain_id} not in [0, {self.manifest.num_embodiment_domains})."
            )
        try:
            embodiment = self.manifest.resolve_embodiment(domain_name, domain_id)
        except ValueError as exc:
            raise ARDiffusionRequestRejectedError(str(exc)) from exc

        guidance_scale = _admission_float(
            _first_not_none(self._get_sp_param(sp, "guidance_scale", None), 1.0),
            "guidance_scale",
        )
        if guidance_scale != 1.0:
            raise ARDiffusionRequestRejectedError(
                f"Cosmos-Dreams distilled inference requires guidance_scale=1.0, got {guidance_scale}."
            )
        if sp.num_inference_steps not in (None, self._distilled_num_steps):
            raise ARDiffusionRequestRejectedError(
                "Cosmos-Dreams distilled inference uses the checkpoint-defined four-step schedule; "
                f"got num_inference_steps={sp.num_inference_steps}."
            )

        text_ids, text_mask = self._tokenize_prompt(
            prompt,
            max_sequence_length=1 << 30,
            use_system_prompt=False,
        )
        real_text_kv_len = int(text_mask[0].sum().item())
        if real_text_kv_len > self.manifest.text_cache_max_len:
            raise ARDiffusionRequestRejectedError(
                "Cosmos-Dreams prompt exceeds text_cache_max_len: "
                f"{real_text_kv_len} > {self.manifest.text_cache_max_len}."
            )
        fingerprint = self._fingerprint(
            text_ids,
            real_text_kv_len=real_text_kv_len,
            geometry=geometry,
            fps=fps,
            domain_id=domain_id,
            embodiment=embodiment,
        )
        start_frame = 0 if state_was_new else existing_state.next_frame_idx
        requested_frame_idx = _admission_int(_first_not_none(controls.frame_idx, start_frame), "frame_idx")
        if state_was_new:
            if requested_frame_idx != 0:
                raise ARDiffusionRequestRejectedError(
                    f"Cosmos-Dreams new sessions must start at latent frame 0; got {requested_frame_idx}."
                )
        else:
            try:
                existing_state.validate_request(fingerprint, frame_idx=requested_frame_idx)
            except (ValueError, RuntimeError) as exc:
                raise ARDiffusionRequestRejectedError(str(exc)) from exc
            if tick and existing_state.tick_output_type not in (None, tick_output_type):
                raise ARDiffusionRequestRejectedError(
                    "Cosmos-Dreams tick output_type cannot change within a session; session reset required."
                )

        try:
            raw_action = self._prepare_raw_action(embodiment=embodiment, action_value=controls.action)
        except (OSError, TypeError, ValueError) as exc:
            raise ARDiffusionRequestRejectedError(str(exc)) from exc
        try:
            initial_latent = self._initial_condition_latent(prompt_data, sp, geometry)
        except (TypeError, ValueError) as exc:
            raise ARDiffusionRequestRejectedError(str(exc)) from exc
        if start_frame > 0 and initial_latent is not None:
            raise ARDiffusionRequestRejectedError(
                "Cosmos-Dreams initial media may only be supplied at frame 0; session reset required."
            )

        if tick:
            tick_frames = _admission_int(
                _first_not_none(controls.num_latent_frames, self.manifest.chunk_size),
                "num_latent_frames",
            )
            if tick_frames <= 0:
                raise ARDiffusionRequestRejectedError(
                    f"Cosmos-Dreams num_latent_frames must be positive, got {tick_frames}."
                )
            target_frame = start_frame + tick_frames
            # Frame zero is the singleton causal prefix. A normal first tick
            # therefore advances through [0, 1) and then one [1, 5) chunk,
            # regardless of whether frame zero is supplied or generated.
            if start_frame == 0:
                target_frame += 1
            if not close_session and (target_frame - 1) % self.manifest.chunk_size != 0:
                raise ARDiffusionRequestRejectedError(
                    "Cosmos-Dreams non-terminal ticks must end on a canonical [1,4,4,...] "
                    f"chunk boundary, got target latent frame {target_frame}."
                )
        else:
            requested_pixel_frames = _admission_int(_first_not_none(sp.num_frames, 1), "num_frames")
            if requested_pixel_frames <= 0:
                raise ARDiffusionRequestRejectedError(
                    f"Cosmos-Dreams num_frames must be positive, got {requested_pixel_frames}."
                )
            target_frame = (requested_pixel_frames - 1) // self.manifest.temporal_compression_factor + 1
            if target_frame < start_frame:
                raise ARDiffusionRequestRejectedError(
                    "Cosmos-Dreams full rollout target precedes existing session state; session reset required."
                )
        action_layout = self._resolve_action_layout(
            raw_action,
            start_frame=start_frame,
            target_frame=target_frame,
        )

        # ---- Side effects begin ---------------------------------------------
        tick_durations: dict[str, float] = {}
        self._sync_for_tick_timing(measure_tick_latency)
        tick_total_started = time.perf_counter() if measure_tick_latency else 0.0
        if reset:
            self._drop_session(session_id)
        state = self._get_or_create_state(session_id)
        if state.fingerprint is None:
            state.initialize(fingerprint)
        if tick and state.tick_output_type is None:
            state.tick_output_type = tick_output_type
        domain_ids = torch.tensor([domain_id], device=self.device, dtype=torch.long)
        text_kv = self._ensure_text_kv(state, text_ids, text_mask)

        terminal_request = close_session or not tick
        seed = self._resolve_seed(sp, sp.generator if isinstance(sp.generator, torch.Generator) else None)
        request_latent_chunks: list[torch.Tensor] = []

        if initial_latent is not None and state.next_frame_idx == 0:
            initial_action, initial_null = self._actions_for_frames(
                raw_action,
                layout=action_layout,
                request_start_frame=start_frame,
                frame_start=0,
                frame_end=1,
            )
            if target_frame > 1 or not terminal_request:
                with self._timed_tick_stage(
                    tick_durations,
                    "clean_cache_commit_s",
                    enabled=measure_tick_latency,
                ):
                    self._commit_clean_frame(
                        state,
                        initial_latent,
                        geometry=geometry,
                        frame_idx=0,
                        text_kv=text_kv,
                        real_text_kv_len=real_text_kv_len,
                        fps=fps,
                        action=initial_action,
                        domain_ids=domain_ids,
                        null_action=bool(initial_null),
                    )
            state.append_chunk(initial_latent, frame_start=0, retain_latent=not tick)
            request_latent_chunks.append(initial_latent)

        generation_start = state.next_frame_idx
        for chunk_start, chunk_end in iter_ar_chunk_ranges(
            generation_start,
            target_frame,
            self.manifest.chunk_size,
        ):
            chunk_frames = chunk_end - chunk_start
            action_chunk, null_action_indexes = self._actions_for_frames(
                raw_action,
                layout=action_layout,
                request_start_frame=start_frame,
                frame_start=chunk_start,
                frame_end=chunk_end,
            )
            noise_generator = torch.Generator(device=self.device).manual_seed(seed + chunk_start)
            initial_noise = torch.randn(
                1,
                self.transformer.latent_channel_size,
                chunk_frames,
                geometry.latent_height,
                geometry.latent_width,
                generator=noise_generator,
                device=self.device,
                # The reference draws checkpoint-dtype noise, then promotes it
                # to fp32 before the scheduler loop.
                dtype=self.dtype,
            )

            def velocity_fn(x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
                output = self._transformer_forward(
                    state,
                    x.to(self.dtype),
                    timestep,
                    geometry=geometry,
                    text_kv=text_kv,
                    real_text_kv_len=real_text_kv_len,
                    frame_start=chunk_start,
                    fps=fps,
                    action_latents=action_chunk,
                    action_domain_ids=domain_ids,
                    condition_vision=False,
                    null_action_frame_indexes=null_action_indexes,
                    commit_current=False,
                )
                return output.video.float()

            with self._timed_tick_stage(
                tick_durations,
                "denoise_s",
                enabled=measure_tick_latency,
            ):
                clean_chunk = self._denoise_chunk(
                    velocity_fn,
                    initial_noise,
                    generator=noise_generator,
                ).to(self.dtype)

            action_count = self.manifest.action_tokens_per_frame
            with self._timed_tick_stage(
                tick_durations,
                "clean_cache_commit_s",
                enabled=measure_tick_latency,
            ):
                for local_idx, frame_idx in iter_clean_commit_frames(
                    chunk_start,
                    chunk_end,
                    target_frame=target_frame,
                    terminal_request=terminal_request,
                ):
                    action_start = local_idx * action_count
                    action_frame = action_chunk[:, action_start : action_start + action_count]
                    self._commit_clean_frame(
                        state,
                        clean_chunk[:, :, local_idx : local_idx + 1],
                        geometry=geometry,
                        frame_idx=frame_idx,
                        text_kv=text_kv,
                        real_text_kv_len=real_text_kv_len,
                        fps=fps,
                        action=action_frame,
                        domain_ids=domain_ids,
                        null_action=local_idx in null_action_indexes,
                    )
            state.append_chunk(clean_chunk, frame_start=chunk_start, retain_latent=not tick)
            request_latent_chunks.append(clean_chunk)

        if not request_latent_chunks:
            raise RuntimeError("Cosmos-Dreams request produced no new latent frames.")
        request_latents = torch.cat(request_latent_chunks, dim=2)
        accumulated = state.accumulated_latents
        if not tick and accumulated is None:
            raise RuntimeError("Cosmos-Dreams full rollout produced no accumulated latent frames.")
        if sp.output_type == "latent":
            output_value = request_latents if tick else accumulated
        else:
            with self._timed_tick_stage(
                tick_durations,
                "vae_decode_s",
                enabled=measure_tick_latency,
            ):
                if tick:
                    output_value = self._decode_live_latents(state, request_latents).clamp(-1, 1)
                else:
                    output_value = self._decode_latents(accumulated).clamp(-1, 1)
            if not tick:
                output_value = output_value[:, :, : int(sp.num_frames or 1)]

        if not tick:
            state.terminal = True
        if measure_tick_latency:
            self._sync_for_tick_timing(True)
            tick_durations["tick_model_total_s"] = time.perf_counter() - tick_total_started
        output: dict[str, Any] = {"video": output_value}
        if typed_tick is not None:
            output = {
                "payload": output,
                "metadata": {
                    "ar_diffusion": ARDiffusionChunkMetadata.from_tick(typed_tick).to_dict(),
                },
            }
        result = DiffusionOutput(output=output, stage_durations=tick_durations)
        if close_session and self._ar_diffusion_kv_state is None:
            self._drop_session(session_id)
        return result
