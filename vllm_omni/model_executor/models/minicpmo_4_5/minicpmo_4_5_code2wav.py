# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict batched codec-to-waveform stage for MiniCPM-o 4.5."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import torch.nn as nn
from vllm.config import VllmConfig

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .batched_token2wav import (
    BatchedToken2Wav,
    BatchedToken2WavState,
    state_shape_signature,
)


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


@dataclass(frozen=True)
class _RequestState:
    cache_epoch: int
    chunk_seq: int
    prompt_cache_id: str
    prompt_wav: str
    token2wav: BatchedToken2WavState


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
    segment_text: str
    duplex_turn_id: int | None
    duplex_epoch: int | None
    segment_end: bool
    turn_end: bool


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
        self._owned_prompt_wavs: dict[str, tuple[str, str]] = {}
        extra = self._extra_config()
        self._min_batch_size = int(extra.get("code2wav_min_batch_size", 1))
        if self._min_batch_size < 1:
            raise ValueError("MiniCPM-o Code2Wav code2wav_min_batch_size must be >= 1")
        self._default_prompt_id = str(extra.get("prompt_cache_id", "HT_ref_audio"))
        self._default_prompt_wav = str(
            extra.get(
                "prompt_wav",
                Path(self.model_path) / "assets" / "HT_ref_audio.wav",
            )
        )

    def _extra_config(self) -> dict[str, Any]:
        model_config = getattr(self.vllm_config, "model_config", None)
        connector = getattr(model_config, "stage_connector_config", None)
        if isinstance(connector, Mapping):
            extra = connector.get("extra", connector)
        else:
            extra = getattr(connector, "extra", None)
        return dict(extra) if isinstance(extra, Mapping) else {}

    def _release_prompt(self, state_id: str) -> None:
        owned = self._owned_prompt_wavs.pop(state_id, None)
        if owned is None:
            return
        prompt_cache_id, prompt_wav = owned
        if self.backend is not None:
            self.backend.evict_prompt(prompt_cache_id, prompt_wav)
        try:
            os.unlink(prompt_wav)
        except FileNotFoundError:
            pass

    def _write_reference_prompt(
        self,
        state_id: str,
        reference: object,
        sample_rate: object,
    ) -> tuple[str, str]:
        waveform = torch.as_tensor(reference, dtype=torch.float32).reshape(-1).cpu()
        if waveform.numel() == 0:
            return self._default_prompt_id, self._default_prompt_wav
        sample_rate_hz = int(_scalar(sample_rate, 24000))
        if sample_rate_hz <= 0:
            raise _batch_error(
                "invalid_ref_audio_sample_rate",
                request_id=state_id,
                sample_rate=sample_rate_hz,
            )
        digest = hashlib.sha256()
        digest.update(str(sample_rate_hz).encode("ascii"))
        digest.update(waveform.numpy().tobytes())
        prompt_cache_id = f"{state_id}:{digest.hexdigest()}"
        current = self._owned_prompt_wavs.get(state_id)
        if current is not None and current[0] == prompt_cache_id and os.path.exists(current[1]):
            return current

        self._release_prompt(state_id)
        handle = tempfile.NamedTemporaryFile(prefix="minicpmo45-ref-", suffix=".wav", delete=False)
        prompt_wav = handle.name
        handle.close()
        try:
            sf.write(prompt_wav, waveform.numpy(), sample_rate_hz, format="WAV")
        except Exception:
            try:
                os.unlink(prompt_wav)
            except FileNotFoundError:
                pass
            raise
        self._owned_prompt_wavs[state_id] = (prompt_cache_id, prompt_wav)
        return prompt_cache_id, prompt_wav

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return torch.zeros((input_ids.numel(), 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: Any, sampling_metadata: Any = None) -> None:
        return None

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
        codes = info.get("codes")
        audio = codes.get("audio") if isinstance(codes, Mapping) else None
        reference = codes.get("ref") if isinstance(codes, Mapping) else None
        tokens = _codec_tensor(audio, segment)
        if last_chunk and int(_scalar(meta.get("code_flat_numel"), tokens.numel())) == 0:
            # The generation scheduler reserves one placeholder token for an
            # empty terminal chunk. The producer's explicit length is the
            # authority, so do not decode that placeholder as codec data.
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
            self._release_prompt(state_id)
            previous = None
        elif chunk_seq != previous.chunk_seq + 1:
            raise _batch_error(
                "stale_or_reordered_chunk",
                request_id=request_id,
                expected=previous.chunk_seq + 1,
                actual=chunk_seq,
            )

        explicit_prompt_id = _scalar(meta.get("prompt_cache_id"))
        explicit_prompt_wav = _scalar(meta.get("prompt_wav"))
        if previous is not None and explicit_prompt_id is None and explicit_prompt_wav is None:
            prompt_cache_id = previous.prompt_cache_id
            prompt_wav = previous.prompt_wav
        elif reference is not None:
            prompt_cache_id, prompt_wav = self._write_reference_prompt(
                state_id,
                reference,
                meta.get("ref_audio_sr"),
            )
        else:
            prompt_cache_id = str(explicit_prompt_id or self._default_prompt_id)
            prompt_wav = str(explicit_prompt_wav or self._default_prompt_wav)

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
        duplex_turn_id = _scalar(meta.get("duplex_turn_id"))
        duplex_epoch = _scalar(meta.get("duplex_epoch"))
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
            segment_text=str(_scalar(meta.get("native_duplex_segment_text"), "")),
            duplex_turn_id=int(duplex_turn_id) if duplex_turn_id is not None else None,
            duplex_epoch=int(duplex_epoch) if duplex_epoch is not None else None,
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
            int(item.tokens.numel()),
            cache_signature,
            item.last_chunk,
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
            raise _batch_error("backend_not_loaded")

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
        state_ids = [item.state_id for item in items]
        if len(state_ids) != len(set(state_ids)):
            raise _batch_error("duplicate_request_in_forward", request_ids=state_ids)
        outputs = [empty for _ in segments]
        sentinels = [item for item in items if item.last_chunk and item.tokens.numel() == 0]
        compute_items = [item for item in items if item.tokens.numel() > 0]
        invalid_empty = [item.request_id for item in items if not item.last_chunk and item.tokens.numel() == 0]
        if invalid_empty:
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
            raise _batch_error(
                "exact_shape_bucket_below_minimum",
                minimum=self._min_batch_size,
                buckets=undersized,
            )

        pending: dict[str, _RequestState | None] = {item.state_id: None for item in sentinels}
        for bucket in buckets.values():
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
                tokens = torch.stack([item.tokens for item in bucket], dim=0)
                audios, next_states = self.backend.decode_batch(
                    tokens,
                    features,
                    states,
                    last_chunk=bucket[0].last_chunk,
                )
            except Exception as exc:
                if isinstance(exc, RuntimeError) and str(exc).startswith("MiniCPMO45Code2WavBatchError "):
                    raise
                raise _batch_error(
                    "backend_unsupported_or_failed",
                    request_ids=[item.request_id for item in bucket],
                    error_type=type(exc).__name__,
                    error=str(exc),
                ) from exc
            if len(audios) != batch_size or len(next_states) != batch_size:
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

        for request_id, state in pending.items():
            if state is None:
                self._states.pop(request_id, None)
                self._release_prompt(request_id)
            else:
                self._states[request_id] = state
        sample_rate_tensor = torch.as_tensor(sample_rate, dtype=torch.int32)
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "model_outputs": outputs,
                "sr": [sample_rate_tensor.clone() for _ in outputs],
                "meta.llm_output_text_utf8": [
                    torch.tensor(list(item.segment_text.encode("utf-8")), dtype=torch.uint8) for item in items
                ],
                "meta.tts_is_last_chunk": [torch.tensor(item.last_chunk, dtype=torch.bool) for item in items],
                "meta.segment_end": [torch.tensor(item.segment_end, dtype=torch.bool) for item in items],
                "meta.turn_end": [torch.tensor(item.turn_end, dtype=torch.bool) for item in items],
                "meta.duplex_turn_id": [
                    torch.tensor(item.duplex_turn_id, dtype=torch.int64)
                    if item.duplex_turn_id is not None
                    else torch.empty(0, dtype=torch.int64)
                    for item in items
                ],
                "meta.duplex_epoch": [
                    torch.tensor(item.duplex_epoch, dtype=torch.int64)
                    if item.duplex_epoch is not None
                    else torch.empty(0, dtype=torch.int64)
                    for item in items
                ],
            },
        )

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        for request_id in finished_req_ids:
            state_id = str(request_id)
            self._states.pop(state_id, None)
            self._release_prompt(state_id)

    def make_omni_output(self, model_outputs: Any, **_: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        if isinstance(model_outputs, tuple) and len(model_outputs) == len(OmniOutput._fields):
            return OmniOutput(*model_outputs)
        raise TypeError(f"MiniCPMO45Code2Wav expected OmniOutput, got {type(model_outputs).__name__}")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        for _ in weights:
            pass
        if self.backend is not None:
            return {name for name, _ in self.named_parameters()}
        from stepaudio2.token2wav import Token2wav

        extra = self._extra_config()
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
        self.backend = BatchedToken2Wav(token2wav)
        # Token2wav loads flow.pt and hift.pt inside its constructor instead of
        # from the parent MiniCPM checkpoint iterator. Report those registered
        # parameters as initialized so vLLM's strict loader audit does not
        # misclassify the independently loaded Stage-2 weights as missing.
        return {name for name, _ in self.named_parameters()}
