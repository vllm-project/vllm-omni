# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Project AURA stage outputs into duplex internal events."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

import numpy as np
import torch

from vllm_omni.engine.duplex.contracts import (
    duplex_resource_request_belongs_to_session,
    duplex_turn_id_from_request_id,
)
from vllm_omni.engine.duplex.plugin import DuplexDataPlane, EncodeAudio
from vllm_omni.model_executor.stage_input_processors.aura_omni import (
    SILENT_TEXT,
    is_effectively_silent,
    is_silent_text_prefix,
)
from vllm_omni.outputs.duplex import get_duplex_output_decision


@dataclass(frozen=True, slots=True)
class AuraDataPlaneContext:
    epoch: int = 0
    turn_id: int = 0
    auto_responds: bool = True
    response_format: str = "wav"
    speed: float | None = None
    modalities: tuple[str, ...] = ("text", "audio")


@dataclass(slots=True)
class _RequestState:
    text_sent: str = ""
    text_emitted: str = ""
    audio_offset: int = 0
    silent: bool = False
    terminal: bool = False
    context_committed: bool = False
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


def _strip_think(text: str) -> str:
    """Drop Qwen3-VL ``<think>...</think>`` wrappers from assistant text."""
    from vllm_omni.model_executor.stage_input_processors.aura_omni import _strip_assistant_text

    return _strip_assistant_text(text)


def _text_from(output: object, completion: object | None) -> str:
    for candidate in (completion, output):
        if candidate is None:
            continue
        for attr in ("cumulative_text", "text"):
            value = getattr(candidate, attr, None)
            if isinstance(value, str) and value:
                return _strip_think(value)
    return ""


def _multimodal(output: object, completion: object | None) -> dict[str, object]:
    """Prefer decision metadata, then completion multimodal payload.

    Empty mappings must not short-circuit: OmniRequestOutput.multimodal_output
    returns ``{}`` when completion payloads are falsy, which would otherwise
    hide a real MultimodalPayload on the completion (MiniCPM data-plane does
    the same ``if not mm`` fallthrough).
    """
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


def _audio_value(metadata: Mapping[str, object]) -> object | None:
    """Extract PCM/latent audio from a multimodal mapping.

    Code2Wav wire payloads use producer key ``model_outputs``; after
    ``MultimodalPayload.from_raw(..., modality_key)`` that becomes ``audio``
    when modality is audio, or another modality key (e.g. ``text``/``hidden``)
    when the stage output_modality was wrongly tagged. Accept common aliases and
    finally the first non-sr tensor so duplex still surfaces Stage3 PCM.
    """
    value = next(
        (metadata[key] for key in ("audio", "model_outputs", "latent", "hidden", "text") if key in metadata),
        None,
    )
    if value is None:
        primary = getattr(metadata, "primary_tensor", None)
        if primary is not None:
            value = primary
    if value is None:
        for key, candidate in metadata.items():
            if key in {"sr", "sample_rate", "sample_rate_hz", "audio_sample_rate"}:
                continue
            if isinstance(candidate, torch.Tensor) or (
                isinstance(candidate, list) and candidate and isinstance(candidate[0], torch.Tensor)
            ):
                value = candidate
                break
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _sample_rate(metadata: Mapping[str, object]) -> int:
    value = metadata.get("sr", metadata.get("sample_rate_hz", 24000))
    if isinstance(value, list) and value:
        value = value[0]
    if hasattr(value, "item"):
        value = value.item()
    return int(value) if isinstance(value, int | float) else 24000


def _pcm_sample_count(audio: object) -> int:
    if isinstance(audio, torch.Tensor):
        return int(audio.detach().reshape(-1).numel())
    try:
        return int(np.asarray(audio, dtype=np.float32).reshape(-1).size)
    except (TypeError, ValueError):
        return 0


def _flat_audio(audio: object) -> torch.Tensor | np.ndarray | None:
    if isinstance(audio, torch.Tensor):
        return audio.detach().reshape(-1)
    try:
        return np.asarray(audio, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None


def _new_audio_samples(audio: object, already_sent: int) -> tuple[torch.Tensor | np.ndarray, int] | None:
    """Return only samples not yet sent.

    Code2Wav hands back a growing waveform. A longer buffer is a cumulative
    snapshot: send the tail. An equal buffer is a duplicate. A shorter buffer
    is a real delta (or a restart) and is sent whole.
    """
    flat = _flat_audio(audio)
    if flat is None:
        return None
    count = int(flat.numel()) if isinstance(flat, torch.Tensor) else int(flat.size)
    if count > already_sent:
        return flat[already_sent:], count
    if count == already_sent:
        return None
    return flat, already_sent + count


def _requested_audio_format(context: object | None) -> tuple[str, float | None]:
    """Honor the session response format. Missing context stays PCM16 (Realtime default)."""
    fmt = getattr(context, "response_format", None)
    speed = getattr(context, "speed", None)
    if not isinstance(fmt, str) or not fmt:
        fmt = "pcm16"
    if not isinstance(speed, int | float):
        speed = None
    return fmt, float(speed) if speed is not None else None


class AuraDataPlaneSession(DuplexDataPlane):
    """Map Stage1 text + Stage3 audio (or silent) onto duplex events."""

    def __init__(self, encode_audio: EncodeAudio) -> None:
        self._encode_audio = encode_audio
        self._requests: dict[str, _RequestState] = {}
        self._closed: set[str] = set()

    def begin_request(self, request_id: str) -> None:
        self._closed.discard(request_id)
        state = self._requests.setdefault(request_id, _RequestState())
        state.terminal = False

    def is_terminal(self, request_id: str | None) -> bool:
        if request_id is None:
            return False
        if request_id in self._closed:
            return True
        state = self._requests.get(request_id)
        return state is not None and state.terminal

    def mark_terminal(self, request_id: str) -> None:
        # The shared session already calls this when a turn finishes. Drop the
        # projector row here so completed turns do not accumulate; the id stays
        # closed so a late chunk cannot open a second turn.
        self._requests.pop(request_id, None)
        self._closed.add(request_id)

    def close_stream(self, request_id: str) -> None:
        state = self._requests.get(request_id)
        if state is not None:
            state.audio_offset = 0

    def close_session(self, session_id: str, *, active_request_id: str | None = None) -> None:
        if active_request_id is not None:
            self._requests.pop(active_request_id, None)
        for request_id in list(self._requests):
            if duplex_resource_request_belongs_to_session(request_id, session_id):
                self._requests.pop(request_id, None)
        self._closed = {
            request_id
            for request_id in self._closed
            if not duplex_resource_request_belongs_to_session(request_id, session_id)
        }
        from vllm_omni.model_executor.models.aura_omni.duplex.history import drop_session_history

        drop_session_history(session_id)

    def project(self, result: object, *, context: object | None = None) -> Iterator[dict[str, object]]:
        if not isinstance(result, dict):
            return
        outputs = result.get("data_plane_outputs")
        if not isinstance(outputs, list):
            return
        for output in outputs:
            yield from self.project_output(output, context=context)

    def project_output(self, result: object, *, context: object | None = None) -> Iterator[dict[str, object]]:
        response_format, speed = _requested_audio_format(context)
        request_id = getattr(result, "request_id", None)
        if not isinstance(request_id, str) or not request_id:
            return
        if request_id in self._closed:
            return
        model_turn_id = duplex_turn_id_from_request_id(request_id)
        outer_finished = bool(getattr(result, "finished", False))
        output, completion, stage_id = _unwrap(result)
        state = self._requests.setdefault(request_id, _RequestState())
        if stage_id is not None:
            state.stage_seen.add(stage_id)

        def _event(**fields: object) -> dict[str, object]:
            payload = dict(fields)
            if model_turn_id is not None:
                payload["model_turn_id"] = model_turn_id
            return payload

        decision = get_duplex_output_decision(result)
        metadata = dict(getattr(decision, "metadata", {}) or {})
        if metadata.get("model_listen") or metadata.get("duplex_native_decision") == "listen":
            state.silent = True
            # DIRECT_RESPONSE skips aura2tts. The session runner commits
            # model context from model_context_text; this projector does not.
            yield _event(
                stage_role="thinker",
                is_listen=True,
                data_plane_request_id=request_id,
                text="",
                end_of_turn=True,
                silent=True,
                model_context_text=SILENT_TEXT,
                # Prewarm already reserved Stage2/3 on Stage0 submit. Silent
                # short-circuit never feeds codec chunks; abort frees those seats.
                abort_data_plane_request=True,
            )
            state.terminal = True
            return

        text = _text_from(output, completion)
        if text and text != state.text_sent:
            state.text_sent = text
            if is_effectively_silent(text):
                state.silent = True
            # Hold deltas that are still a <|silent|> prefix until short-circuit.
            if not state.silent and not is_silent_text_prefix(text):
                emit = (
                    text
                    if not state.text_emitted
                    else (text[len(state.text_emitted) :] if text.startswith(state.text_emitted) else text)
                )
                if emit:
                    state.text_emitted = text
                    yield _event(
                        stage_role="thinker",
                        is_listen=False,
                        data_plane_request_id=request_id,
                        text=emit,
                        end_of_turn=False,
                    )

        # RequestOutput.finished / envelope finished is the stream EOS.
        # CompletionOutput.finished is true on every Code2Wav chunk and must
        # not close the duplex turn (runner would then drop via is_terminal).
        finished = bool(outer_finished or getattr(output, "finished", False))
        if (
            finished
            and stage_id == 1
            and not state.silent
            and not state.context_committed
            and state.text_sent
            and not is_effectively_silent(state.text_sent)
        ):
            state.context_committed = True
            yield _event(
                stage_role="thinker",
                is_listen=False,
                data_plane_request_id=request_id,
                text="",
                end_of_turn=False,
                model_context_text=state.text_sent,
            )
        is_final_audio_stage = stage_id is None or stage_id >= 3
        mm = _multimodal(output, completion)
        audio = _audio_value(mm)
        if audio is not None and not state.silent:
            sample_rate = _sample_rate(mm)
            delta = _new_audio_samples(audio, state.audio_offset)
            encoded = None
            n = 0
            if delta is not None:
                samples, new_offset = delta
                encoded = self._encode_audio(samples, sample_rate, response_format, speed)
                if encoded:
                    n = _pcm_sample_count(samples)
                    state.audio_offset = new_offset
            if encoded:
                duration_ms = round(n * 1000 / max(1, sample_rate))
                end_of_turn = bool(finished and is_final_audio_stage)
                yield _event(
                    stage_role="tts",
                    is_listen=False,
                    data_plane_request_id=request_id,
                    audio=encoded,
                    audio_format=response_format,
                    sample_rate_hz=sample_rate,
                    audio_duration_ms=duration_ms,
                    end_of_turn=end_of_turn,
                )
                if end_of_turn:
                    state.terminal = True
                    return

        if finished and is_final_audio_stage and not state.terminal:
            yield _event(
                stage_role="tts",
                is_listen=False,
                data_plane_request_id=request_id,
                text="",
                end_of_turn=True,
            )
            state.terminal = True
        elif finished and state.silent and not state.terminal:
            yield _event(
                stage_role="thinker",
                is_listen=True,
                data_plane_request_id=request_id,
                text="",
                end_of_turn=True,
                silent=True,
                model_context_text=SILENT_TEXT,
                abort_data_plane_request=True,
            )
            state.terminal = True


__all__ = ["AuraDataPlaneContext", "AuraDataPlaneSession"]
