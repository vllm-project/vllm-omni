# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Completed JoyAI actions and Code2Wav audio on the unified duplex event stream."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field

import numpy as np

from vllm_omni.engine.duplex.contracts import duplex_resource_request_belongs_to_session
from vllm_omni.engine.duplex.plugin import DefaultDuplexModelSessionState, DuplexDataPlane, DuplexDataPlaneContext
from vllm_omni.experimental.fullduplex.joyvl.decision.output_parser import Action, ParsedAction, parse_action
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.input import JoyAIFrameBuffer


@dataclass
class JoyAISessionState(DefaultDuplexModelSessionState):
    audio_buffer: JoyAIFrameBuffer = field(default_factory=JoyAIFrameBuffer)
    frames: list[str] = field(default_factory=list)
    conversation_images: tuple[str, ...] = ()
    last_response_text: str = ""


@dataclass
class Observation:
    state: JoyAISessionState | None = None
    frames: list[str] = field(default_factory=list)
    conversation_images: tuple[str, ...] = ()
    modalities: tuple[str, ...] = ("text", "audio")
    previous_response: str = ""
    text_sent: bool = False
    response_text: str = ""

    def parse(self, output: object) -> ParsedAction:
        completion = next(iter(output.outputs or ()), None)
        raw = getattr(completion, "cumulative_text", None) or getattr(completion, "text", "")
        return parse_action(raw)

    def is_repeat(self, action: ParsedAction) -> bool:
        return action.action is Action.RESPONSE and action.text == self.previous_response


class JoyAIDataPlane(DuplexDataPlane):
    def __init__(self, encode_audio: Callable[..., str]) -> None:
        self.encode_audio = encode_audio
        self.requests: dict[str, Observation] = {}

    def begin_request(self, request_id: str) -> None:
        self.requests.setdefault(request_id, Observation())

    def is_terminal(self, request_id: str | None) -> bool:
        return request_id is not None and request_id not in self.requests

    def mark_terminal(self, request_id: str) -> None:
        self.requests.pop(request_id, None)

    def close_stream(self, request_id: str) -> None:
        self.requests.pop(request_id, None)

    def close_session(self, session_id: str, *, active_request_id: str | None = None) -> None:
        for request_id in list(self.requests):
            if duplex_resource_request_belongs_to_session(request_id, session_id):
                self.close_stream(request_id)

    def project(self, result: object, *, context: DuplexDataPlaneContext | None = None) -> Iterator[dict[str, object]]:
        if not isinstance(result, dict):
            return
        context = context or DuplexDataPlaneContext()
        for output in result.get("data_plane_outputs", ()):
            request_id = output.request_id
            observation = self.requests.get(request_id)
            if observation is None:
                continue
            stage_id = output.stage_id
            event = {
                "supported": True,
                "data_plane_request_id": request_id,
                "model_turn_id": context.active_response_turn_id
                if context.active_response_turn_id is not None
                else context.turn_id,
                "text": "",
                "audio_data": "",
                "audio_format": context.response_format,
                "end_of_turn": False,
            }
            if stage_id == 0:
                if observation.text_sent:
                    continue
                action = observation.parse(output)
                observation.text_sent = True
                if action.action is Action.DELEGATE:
                    event.update(
                        error_code="unsupported_tool_call",
                        error="JoyAI delegation requires the tool adapter",
                        end_of_turn=True,
                    )
                elif action.action is Action.SILENCE or not action.text or observation.is_repeat(action):
                    # Match JoyVLPolicy: genuine silence resets deduplication;
                    # suppressing a repeat keeps the last spoken response.
                    if observation.state is not None and not observation.is_repeat(action):
                        observation.state.last_response_text = ""
                    # A completed observation, not a resumable LISTEN unit:
                    # the ordinary terminal path advances the turn identity.
                    event["end_of_turn"] = True
                else:
                    observation.response_text = action.text
                    event["text"] = action.text
                    event["text_requires_complete_audio"] = "audio" in context.modalities
                    if "audio" not in observation.modalities:
                        event["end_of_turn"] = True
                        if observation.state is not None:
                            observation.state.last_response_text = action.text
                yield event
            elif stage_id == 2:
                completion = next(iter(output.outputs or ()), None)
                mm = output.multimodal_output or getattr(completion, "multimodal_output", None) or {}
                audio = mm.get("audio") if isinstance(mm, Mapping) else None
                if audio is None and not output.finished:
                    continue
                rate = mm.get("sr", mm.get("sample_rate", 24000))
                if isinstance(rate, (list, tuple)):
                    rate = rate[0]
                rate = int(rate)
                if rate <= 0:
                    raise ValueError("Invalid JoyAI output audio sample rate")
                chunks = [] if audio is None else audio if isinstance(audio, (list, tuple)) else [audio]
                arrays = [
                    np.asarray(
                        chunk.detach().float().cpu().numpy() if hasattr(chunk, "detach") else chunk, dtype=np.float32
                    ).reshape(-1)
                    for chunk in chunks
                ]
                samples = np.concatenate(arrays) if arrays else np.empty(0, dtype=np.float32)
                if output.finished and not samples.size:
                    event.update(
                        error_code="missing_audio", error="JoyAI Code2Wav completed without audio", end_of_turn=True
                    )
                    yield event
                    continue
                if output.finished and samples.size and observation.state is not None:
                    observation.state.last_response_text = observation.response_text
                event.update(
                    audio_data=self.encode_audio(samples, rate, context.response_format, context.speed)
                    if samples.size and "audio" in context.modalities
                    else "",
                    sample_rate_hz=rate,
                    audio_duration_ms=samples.size * 1000 / rate if "audio" in context.modalities else 0,
                    audio_complete=bool(output.finished),
                    text_requires_complete_audio="audio" in context.modalities,
                    end_of_turn=bool(output.finished),
                )
                yield event
