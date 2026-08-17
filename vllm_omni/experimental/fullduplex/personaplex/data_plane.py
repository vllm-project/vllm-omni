# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass

import numpy as np

from vllm_omni.experimental.fullduplex.engine.contracts import (
    duplex_resource_request_belongs_to_session,
)

EncodeAudio = Callable[[object, int, str, float | None], str | None]


@dataclass(frozen=True, slots=True)
class PersonaPlexDataPlaneContext:
    epoch: int = 0
    turn_id: int = 0
    active_response_turn_id: int | None = None
    active_response_id: str | None = None
    auto_responds: bool = True
    response_format: str = "wav"
    speed: float | None = None
    modalities: tuple[str, ...] = ("audio", "text")


@dataclass(slots=True)
class _RequestCursor:
    audio_samples: int = 0
    text: str = ""
    terminal: bool = False


class PersonaPlexDataPlaneSession:
    """Project cumulative staged PersonaPlex output into Realtime deltas."""

    def __init__(self, encode_audio: EncodeAudio) -> None:
        self._encode_audio = encode_audio
        self._requests: dict[str, _RequestCursor] = {}

    def begin_request(self, request_id: str) -> None:
        self._requests.setdefault(request_id, _RequestCursor()).terminal = False

    def is_terminal(self, request_id: str | None) -> bool:
        if request_id is None:
            return False
        state = self._requests.get(request_id)
        return state is not None and state.terminal

    def mark_terminal(self, request_id: str) -> None:
        self._requests.setdefault(request_id, _RequestCursor()).terminal = True

    def close_stream(self, request_id: str) -> None:
        self._requests.pop(request_id, None)

    def close_session(self, session_id: str, *, active_request_id: str | None = None) -> None:
        if active_request_id is not None:
            self._requests.pop(active_request_id, None)
        for request_id in list(self._requests):
            if duplex_resource_request_belongs_to_session(request_id, session_id):
                self._requests.pop(request_id, None)

    def project(
        self,
        result: object,
        *,
        context: PersonaPlexDataPlaneContext | None = None,
    ) -> Iterator[dict[str, object]]:
        if not isinstance(result, dict):
            return
        outputs = result.get("data_plane_outputs")
        if not isinstance(outputs, list):
            return
        context = context or PersonaPlexDataPlaneContext()
        for output in outputs:
            projected = self._project_output(output, context=context)
            if projected is not None:
                yield projected

    def _project_output(
        self,
        output: object,
        *,
        context: PersonaPlexDataPlaneContext,
    ) -> dict[str, object] | None:
        output, completion = _unwrap_output(output)
        request_id = getattr(output, "request_id", None)
        if not isinstance(request_id, str) or not request_id:
            request_id = None
        state = self._requests.setdefault(request_id, _RequestCursor()) if request_id is not None else _RequestCursor()
        multimodal = _multimodal_output(output, completion)
        audio = _audio_value(multimodal)
        audio_delta = _slice_audio(audio, state.audio_samples)
        audio_samples = _num_samples(audio)
        if audio_samples is not None:
            state.audio_samples = audio_samples

        text = _text_value(multimodal, completion)
        text_delta = _text_delta(text, state.text)
        if text:
            state.text = text

        sample_rate_hz = _sample_rate(multimodal)
        encoded = self._encode_audio(audio_delta, sample_rate_hz, context.response_format, context.speed)
        if not encoded and not text_delta:
            return None
        delta_samples = _num_samples(audio_delta) or 0
        return {
            "supported": True,
            "stage_role": "tts",
            "is_listen": False,
            "data_plane_request_id": request_id,
            "text": text_delta,
            "audio_data": encoded or "",
            "audio_format": context.response_format,
            "sample_rate_hz": sample_rate_hz,
            "audio_duration_ms": round(delta_samples * 1000 / max(1, sample_rate_hz)),
            "end_of_turn": False,
            "uses_model_runner_scheduler": True,
            "runner_kv_backed": True,
            "runtime_impl": "scheduler_data_plane",
            "owned_runtime": False,
        }


def _unwrap_output(output: object) -> tuple[object, object | None]:
    inner = getattr(output, "request_output", None)
    if inner is not None and inner is not output:
        output = inner
    outputs = getattr(output, "outputs", None)
    completion = outputs[0] if isinstance(outputs, list) and outputs else None
    return output, completion


def _multimodal_output(output: object, completion: object | None) -> dict[str, object]:
    for candidate in (
        getattr(output, "multimodal_output", None),
        getattr(completion, "multimodal_output", None) if completion is not None else None,
    ):
        if isinstance(candidate, Mapping) and candidate:
            return dict(candidate)
    return {}


def _audio_value(multimodal: dict[str, object]) -> object | None:
    value = next(
        (multimodal[key] for key in ("audio", "model_outputs", "latent") if key in multimodal),
        None,
    )
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _text_value(multimodal: dict[str, object], completion: object | None) -> str:
    for candidate in (
        multimodal.get("text"),
        multimodal.get("llm_output_text"),
        getattr(completion, "text", None) if completion is not None else None,
    ):
        if isinstance(candidate, str) and candidate:
            return candidate
    return ""


def _text_delta(text: str, previous: str) -> str:
    if not text:
        return ""
    if text == previous:
        return ""
    if previous and text.startswith(previous):
        return text[len(previous) :]
    return text


def _slice_audio(audio: object | None, offset: int) -> object | None:
    samples = _num_samples(audio)
    if samples is None or samples <= 0:
        return None
    if offset <= 0 or samples < offset:
        return audio
    if samples == offset:
        return None
    try:
        import torch

        if isinstance(audio, torch.Tensor):
            return audio.reshape(-1)[offset:].contiguous()
    except Exception:
        pass
    return np.asarray(audio, dtype=np.float32).reshape(-1)[offset:]


def _num_samples(audio: object | None) -> int | None:
    if audio is None:
        return None
    try:
        import torch

        if isinstance(audio, torch.Tensor):
            return int(audio.numel())
    except Exception:
        pass
    try:
        return int(np.asarray(audio, dtype=np.float32).size)
    except Exception:
        return None


def _sample_rate(multimodal: dict[str, object]) -> int:
    value = multimodal.get("sr", multimodal.get("sample_rate_hz", 24000))
    if isinstance(value, list) and value:
        value = value[0]
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            value = 24000
    return int(value) if isinstance(value, int | float) else 24000


__all__ = [
    "PersonaPlexDataPlaneContext",
    "PersonaPlexDataPlaneSession",
]
