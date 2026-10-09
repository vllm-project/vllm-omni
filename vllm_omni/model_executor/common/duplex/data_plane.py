# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Data plane for models whose final stage streams cumulative audio and text.

The stage runner re-sends the whole waveform (and transcript) it has produced
so far for one request; the session needs only what is new. This data plane
keeps one cursor per request, slices the delta, encodes it for the wire and
reports it as one internal ``response.output_audio.delta``-shaped result.
Model-specific constants (stage role, runtime metadata, default rate) are
class attributes for a subclass to set.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from vllm_omni.engine.duplex.contracts import duplex_resource_request_belongs_to_session
from vllm_omni.engine.duplex.plugin import DuplexDataPlane, DuplexDataPlaneContext
from vllm_omni.model_executor.common.request_outputs import (
    audio_sample_count,
    audio_value,
    multimodal_output,
    sample_rate_hz,
    slice_audio_delta,
    text_delta,
    text_value,
    unwrap_request_output,
)

EncodeAudio = Callable[[object, int, str, float | None], str | None]


@dataclass(slots=True)
class _RequestCursor:
    audio_samples: int = 0
    text: str = ""
    terminal: bool = False


class CumulativeAudioTextDataPlane(DuplexDataPlane):
    """Project cumulative staged audio/text output into per-request deltas."""

    stage_role: str = "tts"
    runtime_impl: str = "scheduler_data_plane"
    uses_model_runner_scheduler: bool = True
    runner_kv_backed: bool = True
    owned_runtime: bool = False
    default_sample_rate_hz: int = 24000

    def __init__(self, encode_audio: EncodeAudio) -> None:
        self._encode_audio = encode_audio
        self._requests: dict[str, _RequestCursor] = {}

    # ---- request / session bookkeeping ----

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

    # ---- projection ----

    def project(self, result: object, *, context: object | None = None) -> Iterator[dict[str, object]]:
        if not isinstance(result, dict):
            return
        outputs = result.get("data_plane_outputs")
        if not isinstance(outputs, list):
            return
        if not isinstance(context, DuplexDataPlaneContext):
            context = DuplexDataPlaneContext(response_format="wav", modalities=("audio", "text"))
        for output in outputs:
            projected = self._project_output(output, context=context)
            if projected is not None:
                yield projected

    def _project_output(self, output: object, *, context: DuplexDataPlaneContext) -> dict[str, object] | None:
        output, completion = unwrap_request_output(output)
        request_id = getattr(output, "request_id", None)
        if not isinstance(request_id, str) or not request_id:
            request_id = None
        state = self._requests.setdefault(request_id, _RequestCursor()) if request_id is not None else _RequestCursor()
        multimodal = multimodal_output(output, completion)
        audio = audio_value(multimodal)
        audio_delta = slice_audio_delta(audio, state.audio_samples)
        total_samples = audio_sample_count(audio)
        if total_samples is not None:
            state.audio_samples = total_samples

        text = text_value(multimodal, completion)
        delta_text = text_delta(text, state.text)
        if text:
            state.text = text

        rate = sample_rate_hz(multimodal, default=self.default_sample_rate_hz)
        encoded = self._encode_audio(audio_delta, rate, context.response_format, context.speed)
        if not encoded and not delta_text:
            return None
        delta_samples = audio_sample_count(audio_delta) or 0
        return {
            "supported": True,
            "stage_role": self.stage_role,
            "is_listen": False,
            "data_plane_request_id": request_id,
            "text": delta_text,
            "audio_data": encoded or "",
            "audio_format": context.response_format,
            "sample_rate_hz": rate,
            "audio_duration_ms": round(delta_samples * 1000 / max(1, rate)),
            "end_of_turn": False,
            "uses_model_runner_scheduler": self.uses_model_runner_scheduler,
            "runner_kv_backed": self.runner_kv_backed,
            "runtime_impl": self.runtime_impl,
            "owned_runtime": self.owned_runtime,
        }


__all__ = ["CumulativeAudioTextDataPlane"]
