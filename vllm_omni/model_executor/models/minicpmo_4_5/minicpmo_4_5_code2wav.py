# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Strict batched codec-to-waveform stage for MiniCPM-o 4.5."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .batched_token2wav import (
    BatchedToken2Wav,
    BatchedToken2WavState,
    state_shape_signature,
)

logger = init_logger(__name__)


def _resolve_model_dir(model_ref: str, revision: str | None = None) -> str:
    """Resolve ``model_ref`` to a local directory containing the repo assets.

    ``model_config.model`` is a filesystem path in local deployments but a
    Hugging Face repo id in hub/CI deployments; the prompt-audio and
    token2wav asset lookups need a real directory either way.
    """
    if Path(model_ref).is_dir():
        return model_ref
    from huggingface_hub import snapshot_download

    return snapshot_download(model_ref, revision=revision, allow_patterns=["assets/*"])


def _batch_error(reason: str, **details: Any) -> RuntimeError:
    payload = {"reason": reason, **details}
    return RuntimeError(f"MiniCPMO45Code2WavBatchError {json.dumps(payload, sort_keys=True)}")


def _scalar(value: Any, default: Any = None) -> Any:
    if isinstance(value, torch.Tensor):
        return value.reshape(-1)[0].item() if value.numel() else default
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _scalar(value[0], default) if value else default
    return default if value is None else value


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
_PRODUCER_META_KEYS = frozenset(
    {
        "cache_epoch",
        "chunk_seq",
        "code_flat_numel",
        "last_chunk",
        "prompt_cache_id",
        "prompt_wav",
        "ref_audio_sr",
    }
)


def _carries_stage_payload(info: Mapping[str, Any], meta: Mapping[str, Any]) -> bool:
    """Whether this step carries anything the Talker stage actually sent.

    Any real async-chunk payload brings producer metadata along, whether the
    transport delivers it nested under ``meta`` or as flattened ``meta.*`` keys.
    """
    codes = info.get("codes")
    if isinstance(codes, Mapping) and any(value is not None for value in codes.values()):
        return True
    if any(key not in _RUNNER_STAMPED_KEYS for key in info):
        return True
    return any(meta.get(key) is not None for key in _PRODUCER_META_KEYS)


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
    has_payload: bool = True


class MiniCPMO45Code2Wav(nn.Module):
    """LLM_GENERATION model with request-owned state and compatible batching."""

    input_modalities = "audio"
    have_multimodal_outputs = True
    enable_update_additional_information = True
    replace_runtime_additional_information = True
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
        self._model_revision = getattr(vllm_config.model_config, "revision", None)
        self.backend: BatchedToken2Wav | None = None
        self._states: dict[str, _RequestState] = {}
        self._runtime_prompts: dict[str, _RuntimePrompt] = {}
        self._request_prompt_keys: dict[str, str] = {}
        self._runtime_prompt_dir = tempfile.TemporaryDirectory(
            prefix="minicpmo45-runtime-prompts-",
        )
        extra = self._extra_config()
        self._connector_config = {
            "codec_chunk_frames": int(extra.get("codec_chunk_frames", 25)),
            "codec_left_context_frames": int(extra.get("codec_left_context_frames", 3)),
        }
        if self._connector_config["codec_chunk_frames"] <= 0 or self._connector_config["codec_left_context_frames"] < 0:
            raise ValueError(f"Invalid MiniCPM-o connector chunk configuration: {self._connector_config}")
        raw_capture_batch_sizes = extra.get("hift_graph_capture_batch_sizes")
        capture_batch_sizes = [1] if raw_capture_batch_sizes is None else raw_capture_batch_sizes
        self._hift_graph_config = {
            "enabled": bool(extra.get("enable_hift_graph", False)),
            "capture_batch_sizes": capture_batch_sizes,
        }
        self._cfm_graph_config = {
            "enabled": bool(extra.get("enable_cfm_graph", False)),
            "max_graphs": int(extra.get("cfm_max_graphs", 32)),
        }
        self._min_batch_size = int(extra.get("code2wav_min_batch_size", 1))
        if self._min_batch_size < 1:
            raise ValueError("MiniCPM-o Code2Wav code2wav_min_batch_size must be >= 1")
        self._initial_batch_size = int(extra.get("code2wav_initial_batch_size", 0))
        if self._initial_batch_size < 0:
            raise ValueError("MiniCPM-o Code2Wav code2wav_initial_batch_size must be >= 0")
        if self._initial_batch_size and self._initial_batch_size < self._min_batch_size:
            raise ValueError("MiniCPM-o Code2Wav code2wav_initial_batch_size must be 0 or >= code2wav_min_batch_size")
        self._default_prompt_id = str(extra.get("prompt_cache_id", "HT_ref_audio"))
        self._prompt_wav_override = extra.get("prompt_wav")

    @property
    def _default_prompt_wav(self) -> str:
        if self._prompt_wav_override is not None:
            return str(self._prompt_wav_override)
        return str(Path(self.model_path) / "assets" / "HT_ref_audio.wav")

    def _extra_config(self) -> dict[str, Any]:
        model_config = getattr(self.vllm_config, "model_config", None)
        connector = getattr(model_config, "stage_connector_config", None)
        if isinstance(connector, Mapping):
            extra = connector.get("extra", connector)
        else:
            extra = getattr(connector, "extra", None)
        return dict(extra) if isinstance(extra, Mapping) else {}

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
        return list(torch.split(flat, normalized))

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
        codes = info.get("codes")
        audio = codes.get("audio") if isinstance(codes, Mapping) else None
        tokens = _codec_tensor(audio, segment)
        if int(_scalar(meta.get("code_flat_numel"), tokens.numel())) == 0:
            # The generation scheduler reserves one placeholder token for an
            # empty terminal or segment-boundary chunk. The producer's
            # explicit length is the authority, so do not decode that
            # placeholder as codec data.
            tokens = segment.new_empty(0, dtype=torch.long)
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
            cache_signature,
            item.cache_epoch,
        )

    def _iter_limited_batches(
        self,
        buckets: Iterable[list[_WorkItem]],
        *,
        maximum: int,
    ) -> Iterable[list[_WorkItem]]:
        for bucket in buckets:
            if not bucket:
                continue
            if not maximum:
                yield bucket
                continue
            batch_count = (len(bucket) + maximum - 1) // maximum
            if len(bucket) < batch_count * self._min_batch_size:
                raise _batch_error(
                    "initial_batch_partition_below_minimum",
                    size=len(bucket),
                    minimum=self._min_batch_size,
                    maximum=maximum,
                )
            batch_size, larger_batches = divmod(len(bucket), batch_count)
            start = 0
            for batch_index in range(batch_count):
                stop = start + batch_size + (batch_index < larger_batches)
                yield bucket[start:stop]
                start = stop

    def _iter_decode_batches(
        self,
        buckets: Iterable[list[_WorkItem]],
    ) -> Iterable[list[_WorkItem]]:
        for bucket in buckets:
            if not self._initial_batch_size:
                yield bucket
                continue
            initial = [item for item in bucket if item.chunk_seq <= 1]
            steady = [item for item in bucket if item.chunk_seq > 1]

            undersized = [
                {"wave": wave, "size": len(items)}
                for wave, items in (("initial", initial), ("steady", steady))
                if items and len(items) < self._min_batch_size
            ]
            if undersized:
                raise _batch_error(
                    "decode_wave_below_minimum",
                    minimum=self._min_batch_size,
                    waves=undersized,
                )
            yield from self._iter_limited_batches([initial], maximum=self._initial_batch_size)
            if steady:
                yield steady

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
        sentinels = [item for item in items if item.last_chunk and item.tokens.numel() == 0]
        segment_markers = [
            item for item in items if not item.last_chunk and item.tts_is_last_chunk and item.tokens.numel() == 0
        ]
        compute_items = [item for item in items if item.tokens.numel() > 0]
        invalid_empty = [
            item.request_id
            for item in items
            if item.has_payload and not item.last_chunk and not item.tts_is_last_chunk and item.tokens.numel() == 0
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
        for item in segment_markers:
            if item.previous is None:
                initial_marker_buckets.setdefault(
                    (item.prompt_cache_id, item.prompt_wav),
                    [],
                ).append(item)
        for bucket in self._iter_limited_batches(
            initial_marker_buckets.values(),
            maximum=self._initial_batch_size,
        ):
            try:
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
                    chunk_seq=item.chunk_seq,
                    prompt_cache_id=item.prompt_cache_id,
                    prompt_wav=item.prompt_wav,
                    token2wav=state,
                )
        for bucket in self._iter_decode_batches(buckets.values()):
            batch_size = len(bucket)
            try:
                features = self.backend.prepare_prompt(
                    bucket[0].prompt_cache_id,
                    bucket[0].prompt_wav,
                )
                if bucket[0].previous is None:
                    states = self.backend.setup_batch(features, batch_size)
                else:
                    states = [item.previous.token2wav for item in bucket if item.previous is not None]
                token_lengths = {int(item.tokens.numel()) for item in bucket}
                last_chunk_values = {item.last_chunk for item in bucket}
                if len(token_lengths) > 1 or len(last_chunk_values) > 1:
                    audios, next_states = self.backend.decode_ragged_batch(
                        [item.tokens for item in bucket],
                        features,
                        states,
                        last_chunks=[item.last_chunk for item in bucket],
                    )
                else:
                    tokens = torch.stack([item.tokens for item in bucket], dim=0)
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
            },
        )

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        for request_id in finished_req_ids:
            state_id = str(request_id)
            self._states.pop(state_id, None)
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

        # In-tree adapter over StepAudio2Token2WavCore on every platform. It
        # matches the external `stepaudio2-minicpmo` Token2wav bit-for-bit on
        # CUDA (same cosyvoice2 flow/DiT modules and weights) while dropping
        # that package's hard-coded `.cuda()` calls, and auto-applies the
        # Ascend fixes (HiFT linear downsample, DiT mask expand, MATH SDPA)
        # on NPU.
        from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_token2wav import (
            MiniCPMO45Token2wav as Token2wav,
        )
        from vllm_omni.platforms import current_omni_platform

        extra = self._extra_config()
        # Hub repo ids only need to become local directories once the vocoder
        # assets are actually read; unit tests construct this model with fake
        # paths and must not trigger a hub download (#5442).
        self.model_path = _resolve_model_dir(self.model_path, self._model_revision)
        prompt_path = Path(self._default_prompt_wav)
        if not prompt_path.is_file():
            raise FileNotFoundError(f"MiniCPM-o Code2Wav prompt audio not found: {prompt_path}")
        token2wav_path = Path(self.model_path) / "assets" / "token2wav"
        if not token2wav_path.is_dir():
            raise FileNotFoundError(f"MiniCPM-o Code2Wav assets not found: {token2wav_path}")
        use_float16 = bool(extra.get("token2wav_float16", False))
        previous_dtype = torch.get_default_dtype()
        try:
            # vLLM constructs bf16 models under a bf16 default-dtype context.
            # Token2wav contains fp32-only S3Tokenizer/HiFT modules, so build
            # its independent assets in their native precision.
            torch.set_default_dtype(torch.float32)
            token2wav = Token2wav(
                str(token2wav_path),
                float16=use_float16,
                n_timesteps=int(extra.get("token2wav_n_timesteps", 10)),
            )
        finally:
            torch.set_default_dtype(previous_dtype)

        trt_stepper = None
        use_trt = bool(extra.get("token2wav_trt", False)) or os.environ.get("MINICPMO_TOKEN2WAV_TRT", "") == "1"
        # TensorRT is CUDA-only; other platforms ignore the toggle.
        if use_trt and current_omni_platform.is_cuda():
            from vllm_omni.model_executor.models.step_audio2.step_audio2_dit_trt import build_dit_trt_stepper

            dtype_name = str(
                extra.get("token2wav_trt_dtype", os.environ.get("MINICPMO_TOKEN2WAV_TRT_DTYPE", "fp16"))
            ).lower()
            trt_dtype = torch.float32 if dtype_name in ("fp32", "float32") else torch.float16
            max_batch = int(extra.get("token2wav_trt_max_batch", 16))
            device = next(token2wav.flow.parameters()).device
            trt_stepper = build_dit_trt_stepper(
                token2wav.flow.decoder.estimator,
                device=device,
                dtype=trt_dtype,
                max_batch=max_batch,
            )
            logger.info("MiniCPM-o Code2Wav: DiT estimator running on TensorRT (%s)", dtype_name)
            token2wav.enable_trt_spk_embedding()
            logger.info("MiniCPM-o Code2Wav: campplus speaker embedding running on TensorRT")
        self.backend = BatchedToken2Wav(
            token2wav,
            trt_stepper=trt_stepper,
            connector_config=self._connector_config,
            hift_graph_config=self._hift_graph_config,
            cfm_graph_config=self._cfm_graph_config,
            bfloat16_attention_cache=bool(extra.get("code2wav_bfloat16_attention_cache", False)),
        )
