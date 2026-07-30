# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field

import numpy as np
from vllm.logger import init_logger

from vllm_omni.experimental.fullduplex.engine.contracts import (
    duplex_resource_request_belongs_to_session,
)

logger = init_logger(__name__)

EncodeAudio = Callable[[object, int, str, float | None], str | None]


@dataclass(frozen=True, slots=True)
class DataPlaneContext:
    epoch: int = 0
    turn_id: int = 0
    active_response_turn_id: int | None = None
    active_response_id: str | None = None
    auto_responds: bool = False
    response_format: str = "wav"
    speed: float | None = None
    modalities: tuple[str, ...] = ()


@dataclass(slots=True)
class TurnState:
    sent_segment_text: str = ""
    has_text: bool = False
    tts_eos_done: bool = False
    turn_eos_done: bool = False


@dataclass(slots=True)
class RequestState:
    audio_offset: int = 0
    uses_segment_text_metadata: bool = False
    pending_audio_without_text: list[dict[str, object]] = field(default_factory=list)
    terminal: bool = False
    turns: dict[int | None, TurnState] = field(default_factory=dict)

    def turn(self, turn_id: int | None) -> TurnState:
        return self.turns.setdefault(turn_id, TurnState())


class BaseDataPlaneSession(ABC):
    """Request/turn lifecycle and audio projection base class."""

    def __init__(self, encode_audio: EncodeAudio) -> None:
        self._encode_audio = encode_audio
        self._requests: dict[str, RequestState] = {}

    def begin_request(self, request_id: str) -> None:
        state = self._requests.setdefault(request_id, RequestState())
        state.terminal = False
        request_turn = state.turns.get(None)
        if request_turn is not None:
            request_turn.tts_eos_done = False
            request_turn.turn_eos_done = False

    def is_terminal(self, request_id: str | None) -> bool:
        if request_id is None:
            return False
        state = self._requests.get(request_id)
        return state is not None and state.terminal

    def mark_terminal(self, request_id: str) -> None:
        self._requests.setdefault(request_id, RequestState()).terminal = True

    def close_stream(self, request_id: str) -> None:
        state = self._requests.get(request_id)
        if state is None:
            return
        state.audio_offset = 0
        state.turns.pop(None, None)

    def close_request(self, request_id: str) -> None:
        self._requests.pop(request_id, None)

    def close_session(self, session_id: str, *, active_request_id: str | None = None) -> None:
        if active_request_id is not None:
            self.close_request(active_request_id)
        for request_id in list(self._requests):
            if self.request_belongs_to_session(request_id, session_id):
                self.close_request(request_id)

    def has_request(self, request_id: str) -> bool:
        return request_id in self._requests

    def has_pending_audio(self, request_id: str) -> bool:
        state = self._requests.get(request_id)
        return bool(state and state.pending_audio_without_text)

    def audio_offset(self, request_id: str | None) -> int | None:
        if request_id is None:
            return None
        state = self._requests.get(request_id)
        return state.audio_offset if state is not None and state.audio_offset > 0 else None

    def project(
        self,
        result: object,
        *,
        context: DataPlaneContext | None = None,
    ) -> Iterator[dict[str, object]]:
        if not isinstance(result, dict):
            return
        outputs = result.get("data_plane_outputs")
        if not isinstance(outputs, list):
            return
        for output in outputs:
            yield from self.project_output(output, context=context)

    @abstractmethod
    def project_output(
        self,
        output: object,
        *,
        context: DataPlaneContext | None = None,
    ) -> Iterator[dict[str, object]]: ...

    def segment_text_delta(self, request_id: str | None, text: object, *, turn_id: int | None = None) -> str:
        if not isinstance(text, str) or not text:
            return ""
        if request_id is None:
            return text
        turn_state = self._requests.setdefault(request_id, RequestState()).turn(turn_id)
        sent_text = turn_state.sent_segment_text
        if not sent_text:
            delta_text = text
        elif text == sent_text:
            delta_text = ""
        elif text.startswith(sent_text):
            delta_text = text[len(sent_text) :]
        else:
            delta_text = text
        turn_state.sent_segment_text = text
        return delta_text

    def slice_cumulative_audio(self, request_id: str | None, audio_data: object) -> object:
        if request_id is None:
            return audio_data
        num_samples = audio_num_samples(audio_data)
        if num_samples is None or num_samples <= 0:
            return audio_data
        state = self._requests.setdefault(request_id, RequestState())
        prev_samples = state.audio_offset
        if prev_samples <= 0:
            state.audio_offset = num_samples
            return audio_data
        if num_samples == prev_samples:
            return None
        if num_samples < prev_samples:
            state.audio_offset = num_samples
            return audio_data
        state.audio_offset = num_samples
        try:
            import torch

            if isinstance(audio_data, torch.Tensor):
                return audio_data.reshape(-1)[prev_samples:].contiguous()
            return np.asarray(audio_data, dtype=np.float32).reshape(-1)[prev_samples:]
        except Exception:
            logger.exception("Failed to slice cumulative duplex audio output")
            return audio_data

    def encode_audio_chunks_with_duration(
        self,
        mm_output: dict[str, object],
        *,
        request_id: str | None,
        response_format: str,
        speed: float | None,
    ) -> Iterator[tuple[str, int]]:
        sample_rate = sample_rate_hz(mm_output)
        audio_data = next((mm_output[key] for key in ("audio", "model_outputs", "latent") if key in mm_output), None)
        if isinstance(audio_data, list):
            for value in audio_data:
                encoded = self._encode_audio(value, sample_rate, response_format, speed)
                if encoded:
                    duration_ms = int((audio_num_samples(value) or 0) * 1000 / max(1, sample_rate))
                    yield encoded, duration_ms
            return
        sliced = self.slice_cumulative_audio(request_id, audio_data)
        encoded = self._encode_audio(sliced, sample_rate, response_format, speed)
        if encoded:
            duration_ms = int((audio_num_samples(sliced) or 0) * 1000 / max(1, sample_rate))
            yield encoded, duration_ms

    @staticmethod
    def request_belongs_to_session(request_id: str, session_id: str) -> bool:
        return (
            duplex_resource_request_belongs_to_session(request_id, session_id)
            or request_id.startswith(f"duplex-{session_id}-")
            or request_id.startswith(f"chatcmpl-duplex-{session_id}-")
        )


# ---------------------------------------------------------------------------
# Helper functions (model-agnostic)
# ---------------------------------------------------------------------------


def coerce_int(value: object) -> int | None:
    if hasattr(value, "detach"):
        try:
            value = value.detach().cpu().reshape(-1)
            if value.numel() == 0:
                return None
            value = value[0].item()
        except Exception:
            return None
    elif hasattr(value, "reshape") and hasattr(value, "size"):
        try:
            value = value.reshape(-1)
            if int(value.size) == 0:
                return None
            item = value[0]
            value = item.item() if hasattr(item, "item") else item
        except Exception:
            return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def coerce_int_list(value: object) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        try:
            value = value.detach().cpu().reshape(-1).tolist()
        except Exception:
            return []
    if not isinstance(value, (list, tuple)):
        return []
    return [item for raw in value if (item := coerce_int(raw)) is not None]


def payload_turn_id(payload: object) -> int | None:
    if not isinstance(payload, Mapping):
        return None
    return coerce_int(payload.get("duplex_turn_id"))


def output_turn_id_from_metadata(mm_output: dict[str, object]) -> int | None:
    return _first_metadata_int(mm_output, "duplex_turn_id", "turn_id")


def output_epoch_from_metadata(mm_output: dict[str, object]) -> int | None:
    return _first_metadata_int(mm_output, "duplex_epoch", "epoch")


def _first_metadata_int(mm_output: dict[str, object], *names: str) -> int | None:
    candidates: list[object] = []
    meta = mm_output.get("meta")
    for name in names:
        candidates.extend((mm_output.get(name), mm_output.get(f"meta.{name}")))
        if isinstance(meta, dict):
            candidates.append(meta.get(name))
    for value in candidates:
        result = coerce_int(value)
        if result is not None:
            return result
    return None


def runtime_result(**values: object) -> dict[str, object]:
    return {
        "supported": True,
        **values,
        "uses_model_runner_scheduler": True,
        "runner_kv_backed": True,
        "runtime_impl": "scheduler_data_plane",
        "owned_runtime": False,
    }


def output_stage_metrics(output: object) -> dict[str, dict[str, object]] | None:
    metrics = getattr(output, "metrics", None)
    if not isinstance(metrics, Mapping):
        return None
    stage_metrics = metrics.get("stage_metrics")
    if not isinstance(stage_metrics, Mapping):
        return None
    snapshot = {
        str(stage_id): dict(values) for stage_id, values in stage_metrics.items() if isinstance(values, Mapping)
    }
    return snapshot or None


def special_token_ids(mm_output: dict[str, object]) -> dict[str, int]:
    sources: list[object] = []
    raw_special = mm_output.get("special_token_ids")
    if isinstance(raw_special, dict):
        sources.append(raw_special)
    raw_meta = mm_output.get("meta")
    if isinstance(raw_meta, dict):
        sources.append(raw_meta)
    sources.append(
        {
            key.removeprefix("meta."): value
            for key, value in mm_output.items()
            if isinstance(key, str) and key.startswith("meta.")
        }
    )
    out: dict[str, int] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key, value in source.items():
            if not isinstance(key, str):
                continue
            token_id = coerce_int(value)
            if token_id is not None and token_id >= 0:
                out[key] = token_id
    return out


def completion_token_ids(completion: object) -> list[int]:
    if completion is None:
        return []
    for candidate in (
        getattr(completion, "token_ids", None),
        getattr(completion, "cumulative_token_ids", None),
    ):
        token_ids = coerce_int_list(candidate)
        if token_ids:
            return token_ids
    return []


def first_completion(output: object) -> object | None:
    outputs = getattr(output, "outputs", None)
    return outputs[0] if isinstance(outputs, list) and outputs else None


def multimodal_output(output: object, completion: object | None) -> dict[str, object]:
    mm = getattr(output, "multimodal_output", None)
    if not isinstance(mm, Mapping):
        mm = getattr(completion, "multimodal_output", {}) if completion is not None else {}
    if not mm:
        inner = getattr(output, "request_output", None)
        if inner is not None and inner is not output:
            inner_mm = getattr(inner, "multimodal_output", None)
            if isinstance(inner_mm, Mapping) and inner_mm:
                mm = inner_mm
    return dict(mm) if isinstance(mm, Mapping) else {}


def audio_num_samples(audio_data: object) -> int | None:
    try:
        import torch

        if isinstance(audio_data, torch.Tensor):
            return int(audio_data.numel())
        return int(np.asarray(audio_data, dtype=np.float32).size)
    except Exception:
        return None


def bool_metadata(
    mm_output: dict[str, object],
    names: tuple[str, ...],
    *,
    default: bool,
) -> bool:
    def coerce(value: object) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        try:
            import torch

            if isinstance(value, torch.Tensor):
                return bool(value.reshape(-1)[-1].item()) if value.numel() else None
        except Exception:
            pass
        if isinstance(value, np.ndarray):
            return bool(value.reshape(-1)[-1].item()) if value.size else None
        if isinstance(value, np.generic):
            return bool(value.item())
        if isinstance(value, (list, tuple)):
            return coerce(value[-1]) if value else None
        if isinstance(value, (int, float)):
            return bool(value)
        return None

    meta = mm_output.get("meta")
    for name in names:
        for key in (name, f"meta.{name}"):
            result = coerce(mm_output.get(key))
            if result is not None:
                return result
        if isinstance(meta, dict):
            result = coerce(meta.get(name))
            if result is not None:
                return result
    return default


def sample_rate_hz(mm_output: dict[str, object]) -> int:
    sr_raw = mm_output.get("sr")
    if sr_raw is None:
        sr_raw = mm_output.get("sample_rate_hz", mm_output.get("sample_rate"))
    meta = mm_output.get("meta")
    if sr_raw is None and isinstance(meta, dict):
        sr_raw = meta.get("sr") or meta.get("sample_rate_hz") or meta.get("sample_rate")
    if sr_raw is None:
        sr_raw = mm_output.get("meta.sr") or mm_output.get("meta.sample_rate_hz") or mm_output.get("meta.sample_rate")
    if hasattr(sr_raw, "item"):
        try:
            return int(sr_raw.item())
        except Exception:
            return 24000
    return int(sr_raw) if isinstance(sr_raw, (int, float)) else 24000


def audio_text_marks(mm_output: dict[str, object]) -> list[dict[str, object]]:
    names = ("audio_text_marks", "text_audio_marks", "audio_text_alignment", "alignment_marks")
    candidates = [mm_output.get(name) for name in names]
    meta = mm_output.get("meta")
    if isinstance(meta, dict):
        candidates.extend(meta.get(name) for name in names)
    candidates.extend(
        value
        for key, value in mm_output.items()
        if isinstance(key, str) and key.startswith("meta.") and key.rsplit(".", 1)[-1] in names
    )
    for raw in candidates:
        if not isinstance(raw, list):
            continue
        marks: list[dict[str, object]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            text_chars = item.get("text_chars")
            audio_end_ms = item.get("audio_end_ms", item.get("audio_ms"))
            if isinstance(text_chars, (int, float)) and isinstance(audio_end_ms, (int, float)):
                marks.append({"text_chars": max(0, int(text_chars)), "audio_end_ms": max(0, int(audio_end_ms))})
        if marks:
            return marks
    return []


def llm_output_text(mm_output: dict[str, object]) -> str:
    candidates: list[object] = [
        mm_output.get("llm_output_text"),
        mm_output.get("text"),
        mm_output.get("llm_output_text_utf8"),
        mm_output.get("meta.llm_output_text_utf8"),
    ]
    meta = mm_output.get("meta")
    if isinstance(meta, dict):
        candidates.extend((meta.get("llm_output_text"), meta.get("text"), meta.get("llm_output_text_utf8")))
    candidates.extend(mm_output.get(key) for key in ("meta.llm_output_text", "meta.text"))
    for value in candidates:
        if isinstance(value, str) and value:
            return value
        decoded = decode_text_tensor(value)
        if decoded:
            return decoded
        if isinstance(value, list):
            text_chunks = [item for item in value if isinstance(item, str)]
            if text_chunks:
                return "".join(text_chunks)
    return ""


def decode_text_tensor(value: object) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, np.ndarray):
            raw = value.astype(np.uint8, copy=False).reshape(-1).tobytes()
        elif hasattr(value, "detach"):
            raw = value.detach().cpu().numpy().astype(np.uint8, copy=False).reshape(-1).tobytes()
        else:
            return ""
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def fallback_audio_text_marks(
    audio_chunks: list[tuple[str, int]],
    delta_text: str,
) -> list[list[dict[str, int]] | None]:
    if not delta_text:
        return []
    total_duration_ms = sum(max(0, int(duration_ms)) for _, duration_ms in audio_chunks)
    cumulative_duration_ms = 0
    marks: list[list[dict[str, int]] | None] = []
    for _, duration_ms in audio_chunks:
        cumulative_duration_ms += max(0, int(duration_ms))
        if total_duration_ms <= 0:
            marks.append(None)
            continue
        text_chars = int(len(delta_text) * min(1.0, cumulative_duration_ms / float(total_duration_ms)))
        marks.append([{"text_chars": max(0, text_chars), "audio_end_ms": max(0, cumulative_duration_ms)}])
    return marks
