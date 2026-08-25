# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict batched codec-to-waveform stage for MiniCPM-o 4.5."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.benchmarks.ultra_timeline import (
    emit_ultra_timeline_event,
    ultra_timeline_enabled,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

from .batched_token2wav import (
    BatchedToken2Wav,
    BatchedToken2WavState,
    state_shape_signature,
)

logger = init_logger(__name__)

_ALLOWED_TOKEN2WAV_N_TIMESTEPS = frozenset({6, 8, 10})
_TRUE_CONFIG_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_CONFIG_VALUES = frozenset({"0", "false", "no", "off"})
_CFM_STEPS_ENV = "VLLM_OMNI_MINICPMO45_CFM_STEPS"
_FLOW_FP16_ENV = "VLLM_OMNI_MINICPMO45_FLOW_FP16"


def _batch_error(reason: str, **details: Any) -> RuntimeError:
    payload = {"reason": reason, **details}
    return RuntimeError(f"MiniCPMO45Code2WavBatchError {json.dumps(payload, sort_keys=True)}")


def _scalar(value: Any, default: Any = None) -> Any:
    if isinstance(value, torch.Tensor):
        return value.reshape(-1)[0].item() if value.numel() else default
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _scalar(value[0], default) if value else default
    return default if value is None else value


def _parse_token2wav_n_timesteps(value: Any) -> int:
    """Accept only the bounded CFM quality grid; reject implicit coercion."""
    if type(value) is not int or value not in _ALLOWED_TOKEN2WAV_N_TIMESTEPS:
        allowed = ", ".join(str(step) for step in sorted(_ALLOWED_TOKEN2WAV_N_TIMESTEPS))
        raise ValueError(f"MiniCPM-o token2wav_n_timesteps must be an integer in {{{allowed}}}; got {value!r}")
    return value


def _parse_token2wav_float16(value: Any) -> bool:
    """Parse the precision toggle without Python truthiness surprises."""
    if isinstance(value, bool):
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_CONFIG_VALUES:
            return True
        if normalized in _FALSE_CONFIG_VALUES:
            return False
    raise ValueError(
        "MiniCPM-o token2wav_float16 must be a boolean, 0/1, or one of "
        f"true/false, yes/no, on/off; got {value!r}"
    )


def _resolve_token2wav_n_timesteps(extra: Mapping[str, Any]) -> tuple[int, str]:
    if "token2wav_n_timesteps" in extra:
        return _parse_token2wav_n_timesteps(extra["token2wav_n_timesteps"]), "config"
    raw = os.environ.get(_CFM_STEPS_ENV)
    if raw is None:
        return 10, "default"
    normalized = raw.strip()
    if normalized not in {"6", "8", "10"}:
        raise ValueError(f"MiniCPM-o {_CFM_STEPS_ENV} must be one of 6, 8, 10; got {raw!r}")
    return _parse_token2wav_n_timesteps(int(normalized)), "environment"


def _resolve_token2wav_float16(extra: Mapping[str, Any]) -> tuple[bool, str]:
    if "token2wav_float16" in extra:
        return _parse_token2wav_float16(extra["token2wav_float16"]), "config"
    raw = os.environ.get(_FLOW_FP16_ENV)
    if raw is None:
        return False, "default"
    try:
        return _parse_token2wav_float16(raw), "environment"
    except ValueError as exc:
        raise ValueError(f"MiniCPM-o {_FLOW_FP16_ENV} is invalid: {raw!r}") from exc


def _codec_tensor(value: Any, fallback: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.reshape(-1).to(device=fallback.device, dtype=torch.long)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return torch.as_tensor(value, device=fallback.device, dtype=torch.long).reshape(-1)
    return fallback.reshape(-1).to(dtype=torch.long)


# Keys the runner stamps on every step regardless of stage input (see
# OmniGPUModelRunner._preprocess and the NPU _gather_runtime_additional_information
# override). A step carrying only these has no producer payload at all.
_RUNNER_STAMPED_KEYS = frozenset({"request_id", "req_id", "generated_len", "meta"})


def _carries_stage_payload(info: Mapping[str, Any], meta: Mapping[str, Any]) -> bool:
    """Whether this step carries anything the Talker stage actually sent.

    Any real async-chunk payload brings producer metadata along, whether the
    transport delivers it nested under ``meta`` or as flattened ``meta.*`` keys.
    """
    if any(key not in _RUNNER_STAMPED_KEYS for key in info):
        return True
    return meta is not info and any(key not in _RUNNER_STAMPED_KEYS for key in meta)


@dataclass(frozen=True)
class _RequestState:
    cache_epoch: int
    chunk_seq: int
    prompt_cache_id: str
    prompt_wav: str
    token2wav: BatchedToken2WavState


@dataclass
class _RuntimePrompt:
    cache_id: str
    path: str
    owners: set[str]


@dataclass(frozen=True)
class _WorkItem:
    output_index: int
    state_id: str
    request_id: str
    cache_epoch: int
    chunk_seq: int
    prompt_cache_id: str
    prompt_wav: str
    last_chunk: bool
    tokens: torch.Tensor
    previous: _RequestState | None
    runtime_prompt_key: str | None
    duplex_epoch: int
    duplex_turn_id: int
    segment_text_utf8: torch.Tensor
    tts_is_last_chunk: bool
    segment_end: bool
    turn_end: bool
    init_only: bool
    has_payload: bool = True


class MiniCPMO45Code2Wav(nn.Module):
    """LLM_GENERATION model that admits only true exact-shape GPU batches."""

    input_modalities = "audio"
    have_multimodal_outputs = True
    enable_update_additional_information = True
    requires_raw_input_tokens = True
    requires_request_ids = True
    has_preprocess = False
    has_postprocess = False

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        del prefix
        self.vllm_config = vllm_config
        self.model_path = str(vllm_config.model_config.model)
        self.backend: BatchedToken2Wav | None = None
        self._states: dict[str, _RequestState] = {}
        self._ultra_timeline_enabled = ultra_timeline_enabled()
        self._timeline_pcm_seen: set[tuple[str, int]] = set()
        self._runtime_prompts: dict[str, _RuntimePrompt] = {}
        self._request_prompt_keys: dict[str, str] = {}
        self._runtime_prompt_dir = tempfile.TemporaryDirectory(
            prefix="minicpmo45-runtime-prompts-",
        )
        extra = self._extra_config()
        self._token2wav_n_timesteps, self._token2wav_n_timesteps_source = _resolve_token2wav_n_timesteps(extra)
        self._token2wav_float16, self._token2wav_float16_source = _resolve_token2wav_float16(extra)
        self._min_batch_size = int(extra.get("code2wav_min_batch_size", 1))
        if self._min_batch_size < 1:
            raise ValueError("MiniCPM-o Code2Wav code2wav_min_batch_size must be >= 1")
        self._default_prompt_id = str(extra.get("prompt_cache_id", "HT_ref_audio"))
        self._prompt_wav_explicit = "prompt_wav" in extra
        self._default_prompt_wav = str(
            extra.get(
                "prompt_wav",
                Path(self.model_path) / "assets" / "HT_ref_audio.wav",
            )
        )

    def _resolve_model_root(self) -> Path:
        model_root = Path(self.model_path)
        if model_root.is_dir():
            return model_root

        from vllm_omni.model_executor.model_loader.weight_utils import (
            download_weights_from_hf_specific,
        )

        model_config = self.vllm_config.model_config
        load_config = getattr(self.vllm_config, "load_config", None)
        model_root = Path(
            download_weights_from_hf_specific(
                self.model_path,
                getattr(load_config, "download_dir", None),
                allow_patterns=[
                    "assets/HT_ref_audio.wav",
                    "assets/token2wav/*",
                ],
                revision=getattr(model_config, "revision", None),
                require_all=True,
            )
        )
        self.model_path = str(model_root)
        if not self._prompt_wav_explicit:
            self._default_prompt_wav = str(model_root / "assets" / "HT_ref_audio.wav")
        return model_root

    def _extra_config(self) -> dict[str, Any]:
        model_config = getattr(self.vllm_config, "model_config", None)
        connector = getattr(model_config, "stage_connector_config", None)
        if isinstance(connector, Mapping):
            extra = connector.get("extra", connector)
        else:
            extra = getattr(connector, "extra", None)
        return dict(extra) if isinstance(extra, Mapping) else {}

    def _backend_timeline_context(self, request_ids: list[str]):
        if self.backend is None or not self._ultra_timeline_enabled:
            return nullcontext()
        factory = getattr(self.backend, "timeline_context", None)
        return factory(request_ids) if callable(factory) else nullcontext()

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return torch.zeros((input_ids.numel(), 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: Any, sampling_metadata: Any = None) -> None:
        return None

    def _materialize_runtime_prompt(
        self,
        ref_audio: Any,
        sample_rate: Any,
    ) -> tuple[str, _RuntimePrompt]:
        sample_rate_hz = int(_scalar(sample_rate, 0))
        waveform = torch.as_tensor(ref_audio, dtype=torch.float32).reshape(-1).cpu().contiguous()
        if sample_rate_hz <= 0:
            raise _batch_error("invalid_ref_audio_sample_rate", sample_rate=sample_rate_hz)
        if waveform.numel() == 0:
            raise _batch_error("empty_ref_audio")
        if not bool(torch.isfinite(waveform).all().item()):
            raise _batch_error("non_finite_ref_audio")

        digest = sha256()
        digest.update(waveform.numpy().tobytes())
        digest.update(str(sample_rate_hz).encode())
        cache_key = digest.hexdigest()
        cache_id = f"runtime-ref-{cache_key[:24]}-{sample_rate_hz}"
        path = str(Path(self._runtime_prompt_dir.name) / f"minicpmo45_ref_{cache_key[:24]}_{sample_rate_hz}.wav")
        entry = self._runtime_prompts.get(cache_key)
        if entry is None:
            entry = _RuntimePrompt(cache_id=cache_id, path=path, owners=set())
            self._runtime_prompts[cache_key] = entry
        prompt_path = Path(entry.path)
        if not prompt_path.is_file():
            with tempfile.NamedTemporaryFile(
                dir=prompt_path.parent,
                prefix=f".{prompt_path.stem}-",
                suffix=".wav",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
            try:
                sf.write(
                    temporary_path,
                    waveform.numpy(),
                    sample_rate_hz,
                    format="WAV",
                )
                os.replace(temporary_path, prompt_path)
            finally:
                temporary_path.unlink(missing_ok=True)
        return cache_key, entry

    def _resolve_prompt(
        self,
        state_id: str,
        info: Mapping[str, Any],
        meta: Mapping[str, Any],
        previous: _RequestState | None,
    ) -> tuple[str, str, str | None]:
        codes = info.get("codes")
        ref_audio = codes.get("ref") if isinstance(codes, Mapping) else None
        if ref_audio is not None:
            cache_key, entry = self._materialize_runtime_prompt(
                ref_audio,
                meta.get("ref_audio_sr"),
            )
            return entry.cache_id, entry.path, cache_key

        if previous is not None:
            return previous.prompt_cache_id, previous.prompt_wav, self._request_prompt_keys.get(state_id)

        cache_key = self._request_prompt_keys.get(state_id)
        entry = self._runtime_prompts.get(cache_key) if cache_key is not None else None
        if entry is not None:
            return entry.cache_id, entry.path, cache_key

        return (
            str(_scalar(meta.get("prompt_cache_id"), self._default_prompt_id)),
            str(_scalar(meta.get("prompt_wav"), self._default_prompt_wav)),
            None,
        )

    def _release_request_prompt(self, state_id: str) -> None:
        cache_key = self._request_prompt_keys.pop(state_id, None)
        entry = self._runtime_prompts.get(cache_key) if cache_key is not None else None
        if entry is None:
            return
        entry.owners.discard(state_id)
        if entry.owners:
            return
        if self.backend is not None:
            self.backend.evict_prompt(entry.cache_id, entry.path)
        Path(entry.path).unlink(missing_ok=True)
        self._runtime_prompts.pop(cache_key, None)

    def _commit_runtime_prompt_owners(self, items: list[_WorkItem]) -> None:
        for item in items:
            cache_key = item.runtime_prompt_key
            if cache_key is None:
                continue
            previous_key = self._request_prompt_keys.get(item.state_id)
            if previous_key != cache_key:
                self._release_request_prompt(item.state_id)
            entry = self._runtime_prompts.get(cache_key)
            if entry is not None:
                entry.owners.add(item.state_id)
                self._request_prompt_keys[item.state_id] = cache_key

    def _prune_unowned_runtime_prompts(self) -> None:
        for cache_key, entry in list(self._runtime_prompts.items()):
            if entry.owners:
                continue
            if self.backend is not None:
                self.backend.evict_prompt(entry.cache_id, entry.path)
            Path(entry.path).unlink(missing_ok=True)
            self._runtime_prompts.pop(cache_key, None)

    @staticmethod
    def _split_segments(input_ids: torch.Tensor, counts: Any) -> list[torch.Tensor]:
        flat = input_ids.reshape(-1)
        if counts is None:
            return [flat]
        if not isinstance(counts, Sequence) or isinstance(counts, (str, bytes, bytearray)):
            raise _batch_error("invalid_seq_token_counts", value_type=type(counts).__name__)
        normalized = [int(value) for value in counts]
        if any(value < 0 for value in normalized):
            raise _batch_error("negative_seq_token_count", counts=normalized)
        if sum(normalized) != int(flat.numel()):
            raise _batch_error(
                "seq_token_count_mismatch",
                counts=normalized,
                total=int(flat.numel()),
            )
        if len(normalized) == 1:
            return [flat]
        return list(torch.split(flat, normalized))

    @staticmethod
    def _batch_codec_tokens(items: list[_WorkItem]) -> torch.Tensor:
        if len(items) == 1:
            # The decoder reads codec tokens without modifying them. Keep the
            # singleton payload as a view instead of launching torch.stack.
            return items[0].tokens.unsqueeze(0)
        return torch.stack([item.tokens for item in items], dim=0)

    def _parse_item(
        self,
        index: int,
        state_id: str,
        segment: torch.Tensor,
        info: Mapping[str, Any],
    ) -> _WorkItem:
        meta = info.get("meta")
        if not isinstance(meta, Mapping):
            meta = info
        request_id = str(_scalar(meta.get("request_id"), _scalar(info.get("request_id"), "")))
        if not _carries_stage_payload(info, meta):
            # The producer attached nothing to this step, only the bookkeeping
            # the runner stamps on every request. The stage was scheduled on
            # the placeholder prompt that async-chunk pre-warm submits before
            # the first codec window arrives: those tokens are reserved slots,
            # not codec data, and one bogus frame is shorter than the vocoder's
            # lookahead window. Such a step carries no producer metadata at
            # all, so it cannot be held to the payload contract below either.
            return _WorkItem(
                output_index=index,
                state_id=state_id,
                request_id=request_id or state_id,
                cache_epoch=0,
                chunk_seq=0,
                prompt_cache_id=self._default_prompt_id,
                prompt_wav=self._default_prompt_wav,
                last_chunk=False,
                tokens=segment.new_empty(0, dtype=torch.long),
                previous=None,
                runtime_prompt_key=None,
                duplex_epoch=-1,
                duplex_turn_id=-1,
                segment_text_utf8=torch.empty(0, dtype=torch.uint8),
                tts_is_last_chunk=False,
                segment_end=False,
                turn_end=False,
                init_only=False,
                has_payload=False,
            )
        if not request_id:
            raise _batch_error("missing_request_id", output_index=index)
        cache_epoch = int(_scalar(meta.get("cache_epoch"), 0))
        chunk_seq = int(_scalar(meta.get("chunk_seq"), 0))
        if cache_epoch < 0 or chunk_seq < 0:
            raise _batch_error(
                "negative_stream_position",
                request_id=request_id,
                cache_epoch=cache_epoch,
                chunk_seq=chunk_seq,
            )
        last_chunk = bool(_scalar(meta.get("last_chunk"), False))
        tts_is_last_chunk = bool(_scalar(meta.get("tts_is_last_chunk"), False))
        init_only = bool(_scalar(meta.get("init_only"), False))
        codes = info.get("codes")
        audio = codes.get("audio") if isinstance(codes, Mapping) else None
        tokens = _codec_tensor(audio, segment)
        if int(_scalar(meta.get("code_flat_numel"), tokens.numel())) == 0:
            # The generation scheduler reserves one placeholder token for an
            # empty terminal or segment-boundary chunk. The producer's
            # explicit length is the authority, so do not decode that
            # placeholder as codec data.
            tokens = segment.new_empty(0, dtype=torch.long)
        if init_only and (last_chunk or tokens.numel() != 0):
            raise _batch_error(
                "invalid_init_only_payload",
                request_id=request_id,
                last_chunk=last_chunk,
                code_numel=int(tokens.numel()),
            )
        previous = self._states.get(state_id)
        if previous is None:
            if chunk_seq != 0:
                raise _batch_error(
                    "missing_state_for_chunk",
                    request_id=request_id,
                    cache_epoch=cache_epoch,
                    chunk_seq=chunk_seq,
                )
        elif cache_epoch < previous.cache_epoch:
            raise _batch_error(
                "stale_cache_epoch",
                request_id=request_id,
                expected=previous.cache_epoch,
                actual=cache_epoch,
            )
        elif cache_epoch > previous.cache_epoch:
            if chunk_seq != 0:
                raise _batch_error(
                    "new_epoch_requires_first_chunk",
                    request_id=request_id,
                    cache_epoch=cache_epoch,
                    chunk_seq=chunk_seq,
                )
            previous = None
        elif chunk_seq != previous.chunk_seq + 1:
            raise _batch_error(
                "stale_or_reordered_chunk",
                request_id=request_id,
                expected=previous.chunk_seq + 1,
                actual=chunk_seq,
            )
        if init_only and previous is not None:
            raise _batch_error(
                "duplicate_init_only",
                request_id=request_id,
                cache_epoch=cache_epoch,
                chunk_seq=chunk_seq,
            )
        prompt_cache_id, prompt_wav, runtime_prompt_key = self._resolve_prompt(
            state_id,
            info,
            meta,
            previous,
        )
        if previous is not None and prompt_cache_id != previous.prompt_cache_id:
            raise _batch_error(
                "prompt_changed_midstream",
                request_id=request_id,
                expected=previous.prompt_cache_id,
                actual=prompt_cache_id,
            )
        if previous is not None and prompt_wav != previous.prompt_wav:
            raise _batch_error(
                "prompt_changed_midstream",
                request_id=request_id,
                expected=previous.prompt_wav,
                actual=prompt_wav,
            )
        segment_text_utf8 = meta.get("llm_output_text_utf8")
        if not isinstance(segment_text_utf8, torch.Tensor):
            segment_text_utf8 = torch.empty(0, dtype=torch.uint8)
        return _WorkItem(
            output_index=index,
            state_id=state_id,
            request_id=request_id,
            cache_epoch=cache_epoch,
            chunk_seq=chunk_seq,
            prompt_cache_id=prompt_cache_id,
            prompt_wav=prompt_wav,
            last_chunk=last_chunk,
            tokens=tokens,
            previous=previous,
            runtime_prompt_key=runtime_prompt_key,
            duplex_epoch=int(_scalar(meta.get("duplex_epoch"), -1)),
            duplex_turn_id=int(_scalar(meta.get("duplex_turn_id"), -1)),
            segment_text_utf8=segment_text_utf8,
            tts_is_last_chunk=tts_is_last_chunk,
            segment_end=bool(_scalar(meta.get("segment_end"), False)),
            turn_end=bool(_scalar(meta.get("turn_end"), False)),
            init_only=init_only,
        )

    @staticmethod
    def _bucket_key(item: _WorkItem) -> tuple[Any, ...]:
        cache_signature: Any
        if item.previous is None:
            cache_signature = ("uninitialized",)
        else:
            cache_signature = state_shape_signature(item.previous.token2wav)
        return (
            item.prompt_cache_id,
            item.prompt_wav,
            int(item.tokens.numel()),
            cache_signature,
            item.last_chunk,
            item.tts_is_last_chunk,
            item.cache_epoch,
        )

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        del positions, intermediate_tensors, inputs_embeds
        ids = input_ids if isinstance(input_ids, torch.Tensor) else torch.empty(0, dtype=torch.long)
        segments = self._split_segments(ids, kwargs.get("seq_token_counts"))
        empty = torch.empty(0, dtype=torch.float32, device=ids.device)
        sample_rate = torch.tensor(24000, dtype=torch.int32)
        if not runtime_additional_information:
            count = len(segments)
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": [empty for _ in range(count)],
                    "sr": [sample_rate for _ in range(count)],
                },
            )
        if len(runtime_additional_information) != len(segments):
            raise _batch_error(
                "runtime_info_count_mismatch",
                segments=len(segments),
                runtime_infos=len(runtime_additional_information),
            )
        if self.backend is None:
            # load_format=dummy (CI core_model runs) skips model.load_weights()
            # entirely, but Token2wav's assets live beside the checkpoint rather
            # than in its weight iterator, so they still have to be loaded for
            # this stage to produce anything. Build them on first use, outside
            # inference mode so the parameters are ordinary tensors.
            logger.warning_once(
                "MiniCPM-o Code2Wav backend was not built during weight loading "
                "(load_format=%s); loading Token2wav assets now.",
                getattr(getattr(self.vllm_config, "load_config", None), "load_format", "unknown"),
            )
            with torch.inference_mode(False), torch.no_grad():
                self._build_backend()

        state_ids = kwargs.get("request_ids")
        if state_ids is None:
            state_ids = []
            for index, info in enumerate(runtime_additional_information):
                if not isinstance(info, Mapping):
                    state_ids.append(str(index))
                    continue
                meta = info.get("meta")
                source = meta if isinstance(meta, Mapping) else info
                state_ids.append(str(_scalar(source.get("request_id"), index)))
        if len(state_ids) != len(segments):
            raise _batch_error(
                "request_id_count_mismatch",
                segments=len(segments),
                request_ids=len(state_ids),
            )
        items: list[_WorkItem] = []
        try:
            for index, (state_id, segment, info) in enumerate(
                zip(state_ids, segments, runtime_additional_information, strict=True)
            ):
                if not isinstance(info, Mapping):
                    raise _batch_error(
                        "invalid_runtime_info",
                        output_index=index,
                        value_type=type(info).__name__,
                    )
                items.append(self._parse_item(index, str(state_id), segment, info))
        except Exception:
            self._prune_unowned_runtime_prompts()
            raise
        state_ids = [item.state_id for item in items]
        if len(state_ids) != len(set(state_ids)):
            self._prune_unowned_runtime_prompts()
            raise _batch_error("duplicate_request_in_forward", request_ids=state_ids)
        outputs = [empty for _ in segments]
        init_items = [item for item in items if item.init_only]
        sentinels = [item for item in items if not item.init_only and item.last_chunk and item.tokens.numel() == 0]
        segment_markers = [
            item
            for item in items
            if not item.init_only and not item.last_chunk and item.tts_is_last_chunk and item.tokens.numel() == 0
        ]
        compute_items = [item for item in items if item.tokens.numel() > 0]
        invalid_empty = [
            item.request_id
            for item in items
            if item.has_payload
            and not item.init_only
            and not item.last_chunk
            and not item.tts_is_last_chunk
            and item.tokens.numel() == 0
        ]
        if invalid_empty:
            self._prune_unowned_runtime_prompts()
            raise _batch_error("empty_nonfinal_chunk", request_ids=invalid_empty)

        buckets: dict[tuple[Any, ...], list[_WorkItem]] = {}
        for item in compute_items:
            buckets.setdefault(self._bucket_key(item), []).append(item)
        undersized = [
            {
                "size": len(bucket),
                "request_ids": [item.request_id for item in bucket],
                "codec_len": int(bucket[0].tokens.numel()),
            }
            for bucket in buckets.values()
            if len(bucket) < self._min_batch_size
        ]
        if undersized:
            self._prune_unowned_runtime_prompts()
            raise _batch_error(
                "exact_shape_bucket_below_minimum",
                minimum=self._min_batch_size,
                buckets=undersized,
            )

        pending: dict[str, _RequestState | None] = {item.state_id: None for item in sentinels}
        pending.update(
            {
                item.state_id: _RequestState(
                    cache_epoch=item.cache_epoch,
                    chunk_seq=item.chunk_seq,
                    prompt_cache_id=item.prompt_cache_id,
                    prompt_wav=item.prompt_wav,
                    token2wav=item.previous.token2wav,
                )
                for item in segment_markers
                if item.previous is not None
            }
        )
        initial_marker_buckets: dict[tuple[str, str], list[_WorkItem]] = {}
        for item in [*init_items, *segment_markers]:
            if item.previous is None:
                initial_marker_buckets.setdefault(
                    (item.prompt_cache_id, item.prompt_wav),
                    [],
                ).append(item)
        for bucket in initial_marker_buckets.values():
            try:
                with self._backend_timeline_context([item.request_id for item in bucket]):
                    features = self.backend.prepare_prompt(
                        bucket[0].prompt_cache_id,
                        bucket[0].prompt_wav,
                    )
                    states = self.backend.setup_batch(features, len(bucket))
            except Exception as exc:
                self._prune_unowned_runtime_prompts()
                if isinstance(exc, RuntimeError) and str(exc).startswith("MiniCPMO45Code2WavBatchError "):
                    raise
                raise _batch_error(
                    "backend_unsupported_or_failed",
                    request_ids=[item.request_id for item in bucket],
                    error_type=type(exc).__name__,
                    error=str(exc),
                ) from exc
            if len(states) != len(bucket):
                self._prune_unowned_runtime_prompts()
                raise _batch_error(
                    "backend_result_size_mismatch",
                    expected=len(bucket),
                    states=len(states),
                )
            for item, state in zip(bucket, states, strict=True):
                pending[item.state_id] = _RequestState(
                    cache_epoch=item.cache_epoch,
                    # init_only is outside the codec stream. Keep the next real
                    # chunk at sequence zero and do not advance any codec state.
                    chunk_seq=-1 if item.init_only else item.chunk_seq,
                    prompt_cache_id=item.prompt_cache_id,
                    prompt_wav=item.prompt_wav,
                    token2wav=state,
                )
        for bucket in buckets.values():
            batch_size = len(bucket)
            try:
                with self._backend_timeline_context([item.request_id for item in bucket]):
                    features = self.backend.prepare_prompt(
                        bucket[0].prompt_cache_id,
                        bucket[0].prompt_wav,
                    )
                    if bucket[0].previous is None:
                        states = self.backend.setup_batch(features, batch_size)
                    else:
                        states = [item.previous.token2wav for item in bucket if item.previous is not None]
                    tokens = self._batch_codec_tokens(bucket)
                    audios, next_states = self.backend.decode_batch(
                        tokens,
                        features,
                        states,
                        last_chunk=bucket[0].last_chunk,
                    )
            except Exception as exc:
                self._prune_unowned_runtime_prompts()
                if isinstance(exc, RuntimeError) and str(exc).startswith("MiniCPMO45Code2WavBatchError "):
                    raise
                raise _batch_error(
                    "backend_unsupported_or_failed",
                    request_ids=[item.request_id for item in bucket],
                    error_type=type(exc).__name__,
                    error=str(exc),
                ) from exc
            if len(audios) != batch_size or len(next_states) != batch_size:
                self._prune_unowned_runtime_prompts()
                raise _batch_error(
                    "backend_result_size_mismatch",
                    expected=batch_size,
                    audios=len(audios),
                    states=len(next_states),
                )
            for item, audio, next_state in zip(bucket, audios, next_states, strict=True):
                outputs[item.output_index] = audio.reshape(-1).to(dtype=torch.float32)
                if self._ultra_timeline_enabled:
                    timeline_key = (item.state_id, item.cache_epoch)
                    first_pcm = timeline_key not in self._timeline_pcm_seen
                    emit_ultra_timeline_event(
                        "first_pcm_ready" if first_pcm else "pcm_ready",
                        request_id=item.request_id,
                        turn_id=item.duplex_turn_id if item.duplex_turn_id >= 0 else None,
                        stage=2,
                        chunk_id=item.chunk_seq,
                        stream="audio",
                        shape=tuple(audio.shape),
                        num_bytes=audio.numel() * audio.element_size(),
                        details={
                            "cache_epoch": item.cache_epoch,
                            "last_chunk": item.last_chunk,
                            "dtype": str(audio.dtype),
                        },
                    )
                    self._timeline_pcm_seen.add(timeline_key)
                    if item.last_chunk:
                        emit_ultra_timeline_event(
                            "last_audio_ready",
                            request_id=item.request_id,
                            turn_id=item.duplex_turn_id if item.duplex_turn_id >= 0 else None,
                            stage=2,
                            chunk_id=item.chunk_seq,
                            stream="audio",
                            shape=tuple(audio.shape),
                            num_bytes=audio.numel() * audio.element_size(),
                            details={"cache_epoch": item.cache_epoch, "dtype": str(audio.dtype)},
                        )
                        self._timeline_pcm_seen.discard(timeline_key)
                pending[item.state_id] = (
                    None
                    if item.last_chunk
                    else _RequestState(
                        cache_epoch=item.cache_epoch,
                        chunk_seq=item.chunk_seq,
                        prompt_cache_id=item.prompt_cache_id,
                        prompt_wav=item.prompt_wav,
                        token2wav=next_state,
                    )
                )

        self._commit_runtime_prompt_owners(items)
        for request_id, state in pending.items():
            if state is None:
                self._states.pop(request_id, None)
            else:
                self._states[request_id] = state
        sample_rate_tensor = torch.as_tensor(sample_rate, dtype=torch.int32)
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "model_outputs": outputs,
                "sr": [sample_rate_tensor.clone() for _ in outputs],
                # Generation runner wire payloads are flat and tensor-only.
                # Dotted metadata keys are unflattened again by the output
                # processor before the full-duplex data plane consumes them.
                "meta.duplex_epoch": [torch.tensor(item.duplex_epoch, dtype=torch.int32) for item in items],
                "meta.duplex_turn_id": [torch.tensor(item.duplex_turn_id, dtype=torch.int32) for item in items],
                "meta.llm_output_text_utf8": [item.segment_text_utf8 for item in items],
                "meta.tts_is_last_chunk": [torch.tensor(item.tts_is_last_chunk, dtype=torch.bool) for item in items],
                "meta.segment_end": [torch.tensor(item.segment_end, dtype=torch.bool) for item in items],
                "meta.turn_end": [torch.tensor(item.turn_end, dtype=torch.bool) for item in items],
                "meta.init_only": [torch.tensor(item.init_only, dtype=torch.bool) for item in items],
            },
        )

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        for request_id in finished_req_ids:
            state_id = str(request_id)
            self._states.pop(state_id, None)
            if self._ultra_timeline_enabled:
                self._timeline_pcm_seen = {key for key in self._timeline_pcm_seen if key[0] != state_id}
            self._release_request_prompt(state_id)

    def make_omni_output(self, model_outputs: Any, **_: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        if isinstance(model_outputs, tuple) and len(model_outputs) == len(OmniOutput._fields):
            return OmniOutput(*model_outputs)
        raise TypeError(f"MiniCPMO45Code2Wav expected OmniOutput, got {type(model_outputs).__name__}")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        for _ in weights:
            pass
        self._build_backend()
        # Token2wav loads flow.pt and hift.pt inside its constructor instead of
        # from the parent MiniCPM checkpoint iterator. Report those registered
        # parameters as initialized so vLLM's strict loader audit does not
        # misclassify the independently loaded Stage-2 weights as missing.
        return {name for name, _ in self.named_parameters()}

    def _build_backend(self) -> None:
        """Load the Token2wav assets that back this stage."""
        if self.backend is not None:
            return

        from vllm_omni.platforms import current_omni_platform

        if current_omni_platform.is_npu():
            # NPU/Ascend: the external `stepaudio2` package hard-codes `.cuda()`,
            # so use the in-tree NPU-aware adapter instead. It delegates to
            # StepAudio2Token2WavCore, which auto-applies the Ascend fixes
            # (HiFT linear downsample, DiT mask expand, MATH SDPA) on NPU.
            from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_token2wav import (
                MiniCPMO45Token2wav as Token2wav,
            )
        else:
            from stepaudio2.token2wav import Token2wav

        model_root = self._resolve_model_root()
        prompt_path = Path(self._default_prompt_wav)
        if not prompt_path.is_file():
            raise FileNotFoundError(f"MiniCPM-o Code2Wav prompt audio not found: {prompt_path}")
        token2wav_path = model_root / "assets" / "token2wav"
        if not token2wav_path.is_dir():
            raise FileNotFoundError(f"MiniCPM-o Code2Wav assets not found: {token2wav_path}")
        use_float16 = self._token2wav_float16
        use_npu_flow_autocast = use_float16 and current_omni_platform.is_npu()
        # NPU Flow weights stay FP32 so an unavailable autocast API can fall
        # back safely. CUDA retains the existing half-weight behavior.
        use_half_weights = use_float16 and not use_npu_flow_autocast
        if self._token2wav_n_timesteps != 10 or use_float16:
            logger.warning(
                "MiniCPM-o numerical lab enabled: cfm_steps=%d (%s), flow_fp16=%s (%s); "
                "full Seed-TTS ASV/WER gates are required",
                self._token2wav_n_timesteps,
                self._token2wav_n_timesteps_source,
                use_float16,
                self._token2wav_float16_source,
            )
        previous_dtype = torch.get_default_dtype()
        try:
            # vLLM constructs bf16 models under a bf16 default-dtype context.
            # Token2wav contains fp32-only S3Tokenizer/HiFT modules, so build
            # its independent assets in their native precision.
            torch.set_default_dtype(torch.float32)
            token2wav = Token2wav(
                str(token2wav_path),
                float16=use_half_weights,
                n_timesteps=self._token2wav_n_timesteps,
            )
        finally:
            torch.set_default_dtype(previous_dtype)
        self.backend = BatchedToken2Wav(
            token2wav,
            npu_flow_float16=use_npu_flow_autocast,
        )
