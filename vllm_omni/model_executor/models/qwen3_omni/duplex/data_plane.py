# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Project Qwen3-Omni stage outputs into duplex internal events.

Stage 0 (Thinker) surfaces text via ``observe_stage_output``. Stage 2
(Code2Wav) surfaces PCM. Stage 1 (Talker) is forwarded to the next stage.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

import numpy as np

from vllm_omni.engine.duplex.contracts import duplex_resource_request_belongs_to_session
from vllm_omni.engine.duplex.plugin import DuplexDataPlane, EncodeAudio
from vllm_omni.outputs.duplex import get_duplex_output_decision

_CODE2WAV_STAGE_ID = 2
_DEFAULT_SAMPLE_RATE_HZ = 24000


@dataclass(frozen=True, slots=True)
class Qwen3OmniDataPlaneContext:
    epoch: int = 0
    turn_id: int = 0
    auto_responds: bool = False
    response_format: str = "wav"
    speed: float | None = None
    modalities: tuple[str, ...] = ("text", "audio")


@dataclass(slots=True)
class _RequestState:
    text_sent: str = ""
    audio_offset: int = 0
    chunks_drained: int = 0
    terminal: bool = False
    stage_seen: set[int] = field(default_factory=set)


def _unwrap(output: object) -> tuple[object, object | None, int | None]:
    stage_id = getattr(output, "stage_id", None)
    inner = getattr(output, "request_output", None)
    if inner is not None and inner is not output:
        output = inner
    outputs = getattr(output, "outputs", None)
    completion = outputs[0] if isinstance(outputs, list) and outputs else None
    if stage_id is None:
        stage_id = getattr(output, "stage_id", None)
    return output, completion, int(stage_id) if isinstance(stage_id, int) else None


def _text_from(output: object, completion: object | None) -> str:
    for candidate in (completion, output):
        if candidate is None:
            continue
        for attr in ("cumulative_text", "text"):
            value = getattr(candidate, attr, None)
            if isinstance(value, str) and value:
                return value
    return ""


def _multimodal(output: object, completion: object | None) -> dict[str, object]:
    decision = get_duplex_output_decision(output)
    metadata = getattr(decision, "metadata", None)
    if isinstance(metadata, Mapping) and metadata:
        return dict(metadata)
    for candidate in (
        getattr(output, "multimodal_output", None),
        getattr(completion, "multimodal_output", None) if completion is not None else None,
    ):
        if isinstance(candidate, Mapping) and candidate:
            return dict(candidate)
    return {}


def _audio_payload(metadata: Mapping[str, object]) -> object | None:
    """Code2Wav PCM under the ``audio`` key."""
    return metadata.get("audio")


def _context_response_format(context: object | None) -> str:
    value = getattr(context, "response_format", None) if context is not None else None
    return value if isinstance(value, str) and value else "wav"


def _context_speed(context: object | None) -> float | None:
    value = getattr(context, "speed", None) if context is not None else None
    return float(value) if isinstance(value, int | float) else None


def _slice_cumulative_audio(audio: object, offset: int) -> object | None:
    samples = _audio_num_samples(audio)
    if samples <= 0 or samples <= offset:
        return None
    if offset <= 0:
        return audio
    try:
        import torch

        if isinstance(audio, torch.Tensor):
            return audio.reshape(-1)[offset:].contiguous()
    except Exception:
        pass
    try:
        return np.asarray(audio, dtype=np.float32).reshape(-1)[offset:]
    except (TypeError, ValueError):
        return None


def _iter_new_audio(audio: object, state: _RequestState) -> Iterator[object]:
    """Yield newly arrived samples or chunks for this request."""
    if isinstance(audio, list):
        new_chunks = audio[state.chunks_drained :]
        state.chunks_drained = len(audio)
        for chunk in new_chunks:
            if chunk is not None:
                yield chunk
        return
    sliced = _slice_cumulative_audio(audio, state.audio_offset)
    total = _audio_num_samples(audio)
    if total > state.audio_offset:
        state.audio_offset = total
    if sliced is not None:
        yield sliced


def _sample_rate(metadata: Mapping[str, object]) -> int:
    value = metadata.get("sr", metadata.get("sample_rate_hz", _DEFAULT_SAMPLE_RATE_HZ))
    if isinstance(value, list) and value:
        value = value[0]
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    return int(value) if isinstance(value, int | float) else _DEFAULT_SAMPLE_RATE_HZ


def _audio_num_samples(audio: object) -> int:
    numel = getattr(audio, "numel", None)
    if callable(numel):
        try:
            return int(numel())
        except (TypeError, ValueError):
            return 0
    try:
        return int(np.asarray(audio, dtype=np.float32).size)
    except (TypeError, ValueError):
        return 0


class Qwen3OmniDataPlaneSession(DuplexDataPlane):
    """Map Thinker text + Code2Wav audio onto duplex events."""

    def __init__(self, encode_audio: EncodeAudio) -> None:
        self._encode_audio = encode_audio
        self._requests: dict[str, _RequestState] = {}

    def begin_request(self, request_id: str) -> None:
        state = self._requests.setdefault(request_id, _RequestState())
        state.terminal = False

    def is_terminal(self, request_id: str | None) -> bool:
        if request_id is None:
            return False
        state = self._requests.get(request_id)
        return state is not None and state.terminal

    def mark_terminal(self, request_id: str) -> None:
        self._requests.setdefault(request_id, _RequestState()).terminal = True

    def close_stream(self, request_id: str) -> None:
        state = self._requests.get(request_id)
        if state is not None:
            state.audio_offset = 0
            state.chunks_drained = 0

    def close_session(self, session_id: str, *, active_request_id: str | None = None) -> None:
        if active_request_id is not None:
            self._requests.pop(active_request_id, None)
        for request_id in list(self._requests):
            if duplex_resource_request_belongs_to_session(request_id, session_id):
                self._requests.pop(request_id, None)

    def project(self, result: object, *, context: object | None = None) -> Iterator[dict[str, object]]:
        if not isinstance(result, dict):
            return
        outputs = result.get("data_plane_outputs")
        if not isinstance(outputs, list):
            return
        for output in outputs:
            yield from self.project_output(output, context=context)

    def project_output(self, result: object, *, context: object | None = None) -> Iterator[dict[str, object]]:
        turn_id = getattr(context, "turn_id", None) if context is not None else None
        request_id = getattr(result, "request_id", None)
        if not isinstance(request_id, str) or not request_id:
            return
        outer_finished = bool(getattr(result, "finished", False))
        output, completion, stage_id = _unwrap(result)
        state = self._requests.setdefault(request_id, _RequestState())
        if stage_id is not None:
            state.stage_seen.add(stage_id)

        text = _text_from(output, completion)
        if text and text != state.text_sent:
            delta = text[len(state.text_sent) :] if text.startswith(state.text_sent) else text
            state.text_sent = text
            if delta:
                event = {
                    "stage_role": "thinker",
                    "is_listen": False,
                    "data_plane_request_id": request_id,
                    "text": delta,
                    "end_of_turn": False,
                }
                if isinstance(turn_id, int):
                    event["model_turn_id"] = turn_id
                yield event

        finished = bool(outer_finished or getattr(output, "finished", False))
        is_final_audio_stage = stage_id is None or stage_id >= _CODE2WAV_STAGE_ID
        mm = _multimodal(output, completion)
        audio = _audio_payload(mm)
        if audio is not None and is_final_audio_stage:
            sample_rate = _sample_rate(mm)
            response_format = _context_response_format(context)
            speed = _context_speed(context)
            new_pieces = list(_iter_new_audio(audio, state))
            for index, piece in enumerate(new_pieces):
                encoded = self._encode_audio(piece, sample_rate, response_format, speed)
                if not encoded:
                    continue
                delta_samples = _audio_num_samples(piece)
                end_of_turn = bool(finished and is_final_audio_stage and index == len(new_pieces) - 1)
                event = {
                    "stage_role": "tts",
                    "is_listen": False,
                    "data_plane_request_id": request_id,
                    "audio": encoded,
                    "audio_format": response_format,
                    "sample_rate_hz": sample_rate,
                    "audio_duration_ms": int(delta_samples * 1000 / max(1, sample_rate)),
                    "end_of_turn": end_of_turn,
                }
                if isinstance(turn_id, int):
                    event["model_turn_id"] = turn_id
                yield event
                if end_of_turn:
                    state.terminal = True
                    return

        if finished and is_final_audio_stage and not state.terminal:
            event = {
                "stage_role": "tts",
                "is_listen": False,
                "data_plane_request_id": request_id,
                "text": "",
                "end_of_turn": True,
            }
            if isinstance(turn_id, int):
                event["model_turn_id"] = turn_id
            yield event
            state.terminal = True


__all__ = ["Qwen3OmniDataPlaneContext", "Qwen3OmniDataPlaneSession"]
