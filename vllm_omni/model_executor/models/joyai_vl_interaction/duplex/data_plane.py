# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Completed JoyAI actions and Code2Wav audio on the unified duplex event stream."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace

import numpy as np

from vllm_omni.engine.duplex.contracts import duplex_resource_request_belongs_to_session
from vllm_omni.engine.duplex.plugin import DefaultDuplexModelSessionState, DuplexDataPlane, DuplexDataPlaneContext
from vllm_omni.experimental.fullduplex.joyvl.decision.output_parser import Action, ParsedAction, parse_action
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.delegation import (
    DELEGATION_TOOL_NAME,
    delegation_arguments,
)
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.input import JoyAIFrameBuffer
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.memory import (
    RECENT_MESSAGES,
    SUMMARY_CHUNK_MESSAGES,
    EncodedVisualImage,
    JoyAISummaryMemory,
    JoyAIVisualMemory,
    VisualMemorySource,
)


@dataclass(frozen=True)
class ResponseReference:
    item_id: str
    time_s: float
    source_frame_id: int


@dataclass(frozen=True)
class QueryReference:
    # None identifies initial_user_text, which has no conversation item.
    item_id: str | None
    text_digest: str
    time_s: float
    first_frame_id: int
    responses: tuple[ResponseReference, ...] = ()


@dataclass(frozen=True)
class WorkingObservation:
    """Accepted visual input order; frame pixels and played answers have other owners."""

    request_id: str
    frame_ids: tuple[int, ...]
    query: str
    query_index: int = -1
    silent: bool = False
    query_item_id: str | None = None
    history_anchor_item_ids: tuple[str, ...] = ()
    response_item_id: str | None = None


@dataclass
class JoyAISessionState(DefaultDuplexModelSessionState):
    audio_buffer: JoyAIFrameBuffer = field(default_factory=JoyAIFrameBuffer)
    frames: list[str] = field(default_factory=list)
    conversation_images: tuple[str, ...] = ()
    last_response_text: str = ""
    memory: JoyAISummaryMemory = field(default_factory=JoyAISummaryMemory)
    visual_memory: JoyAIVisualMemory = field(default_factory=JoyAIVisualMemory)
    visual_source: VisualMemorySource | None = None
    observations: list[WorkingObservation] = field(default_factory=list)
    queries: list[QueryReference] = field(default_factory=list)
    closed_frame_id: int = -1


def advance_chunk(*, current: object, spoken: bool, has_camera: bool) -> tuple[str, bool]:
    """Return the ``(chunk id, spoken)`` pair the next observation belongs to.

    JoyAI answers the goal of the current chunk and keeps that answer as the
    goal until the chunk closes. The chunk closes when a reply of it stored
    text and the next observation arrives with a new camera frame: that is the
    same point at which the retained-answer deduplication lets a reply be
    repeated.

    The pair lives in the session runtime config, not on a plugin-level slot:
    one plugin instance serves every session of a deployment, so a shared slot
    would let one session advance another session's chunk.
    """
    chunk = current if isinstance(current, str) and current else "chunk-0"
    if not spoken or not has_camera:
        return chunk, spoken
    index = chunk.removeprefix("chunk-")
    next_index = int(index) + 1 if index.isdigit() else 1
    return f"chunk-{next_index}", False


@dataclass
class Observation:
    state: JoyAISessionState | None = None
    frames: list[str] = field(default_factory=list)
    conversation_images: tuple[str, ...] = ()
    modalities: tuple[str, ...] = ("text", "audio")
    previous_response: str = ""
    new_frames: list[tuple[EncodedVisualImage, int]] = field(default_factory=list)
    frame_time_s: float = 0.0
    query: str = ""
    query_index: int = -1
    query_item_id: str | None = None
    history_anchor_item_ids: tuple[str, ...] = ()
    visual_generation: int = 0
    delegation_tools: bool = False
    chunk_id: str | None = None
    text_sent: bool = False
    call_sent: bool = False
    response_text: str = ""
    query_reference: QueryReference | None = None
    response_frame_id: int = -1
    response_time_s: float = 0.0

    def parse(self, output: object) -> ParsedAction:
        completion = next(iter(output.outputs or ()), None)
        raw = getattr(completion, "cumulative_text", None) or getattr(completion, "text", "")
        return parse_action(raw)

    def is_repeat(self, action: ParsedAction) -> bool:
        return action.action is Action.RESPONSE and action.text == self.previous_response


class JoyAIDataPlane(DuplexDataPlane):
    joyai_stage_id = 0
    audio_stage_id = 2

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
            if stage_id == self.joyai_stage_id:
                action = observation.parse(output)
                if observation.delegation_tools and action.action is Action.DELEGATE:
                    # The call is one result of its own: the recognised call
                    # short-circuits the rest of this output, so a spoken note
                    # is projected separately and may still enter TTS.
                    yield from self._project_delegation(observation, action, event, context)
                    continue
                if observation.text_sent:
                    continue
                observation.text_sent = True
                if action.action is Action.DELEGATE:
                    event.update(
                        error_code="unsupported_tool_call",
                        error="JoyAI delegation requires the session to declare the delegate tool",
                        end_of_turn=True,
                    )
                elif action.action is Action.SILENCE or not action.text or observation.is_repeat(action):
                    # Match JoyVLPolicy: genuine silence resets deduplication;
                    # suppressing a repeat keeps the last spoken response.
                    if observation.state is not None and not observation.is_repeat(action):
                        observation.state.last_response_text = ""
                    # Keep the model's silent decision next to its accepted
                    # visual input. Spoken Q&A continues to come exclusively
                    # from the Session's playback-committed conversation.
                    if (
                        observation.state is not None
                        and observation.visual_generation == observation.state.visual_memory.generation
                    ):
                        observation.state.observations = [
                            replace(item, silent=True) if item.request_id == request_id else item
                            for item in observation.state.observations
                        ]
                    # A completed observation, not a resumable LISTEN unit:
                    # the ordinary terminal path advances the turn identity.
                    event["end_of_turn"] = True
                else:
                    observation.response_text = action.text
                    if (
                        observation.state is not None
                        and observation.visual_generation == observation.state.visual_memory.generation
                        and context.active_response_id is not None
                    ):
                        # Resolve only the Session's committed portion when a
                        # later prompt is built; generated speech may be unheard.
                        observation.state.observations = [
                            replace(item, response_item_id=f"item_{context.active_response_id}")
                            if item.request_id == request_id
                            else item
                            for item in observation.state.observations
                        ]
                        if observation.query_reference is not None:
                            response = ResponseReference(
                                f"item_{context.active_response_id}",
                                observation.response_time_s,
                                observation.response_frame_id,
                            )
                            observation.state.queries = [
                                replace(
                                    query,
                                    responses=(*query.responses, response)[
                                        -(RECENT_MESSAGES + SUMMARY_CHUNK_MESSAGES) :
                                    ],
                                )
                                if (query.item_id, query.text_digest)
                                == (observation.query_reference.item_id, observation.query_reference.text_digest)
                                else query
                                for query in observation.state.queries
                            ]
                    event["text"] = action.text
                    event["model_context_text"] = action.text
                    event["text_requires_complete_audio"] = "audio" in context.modalities
                    if "audio" not in observation.modalities:
                        event["end_of_turn"] = True
                        if observation.state is not None:
                            observation.state.last_response_text = action.text
                yield event
            elif stage_id == self.audio_stage_id:
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

    def _project_delegation(
        self,
        observation: Observation,
        action: ParsedAction,
        event: dict[str, object],
        context: DuplexDataPlaneContext,
    ) -> Iterator[dict[str, object]]:
        """Project one delegated action as a call plus, at most, its spoken note.

        A recognised call short-circuits the rest of the output, so the call
        result carries the whole output's ``end_of_turn``: nothing else of this
        output will carry it. The note is the ordinary spoken reply that
        follows, so it carries the text of its own response whenever the audio
        stage is going to render it. A delegation without a note still owes the
        observation a terminal of its own: the call is published, but the shared
        call handler returns before the response lifecycle runs, so an empty
        terminal releases the request instead of leaving it open.
        """
        if observation.call_sent or observation.text_sent:
            return
        audio = "audio" in context.modalities
        call_event = dict(event)
        note_event = dict(event)
        note_event.update(
            text=action.text,
            model_context_text=action.text,
            # The call already ended the turn; only a text-only observation
            # has no audio stage left to produce the terminal.
            end_of_turn=not audio,
            # Without this the session would commit the note as spoken history
            # when its audio never completed, e.g. a cancel between projection
            # and the first Code2Wav output.
            text_requires_complete_audio=audio,
        )
        arguments = delegation_arguments(action)
        observation.call_sent = True
        call_event["end_of_turn"] = True
        if arguments is None:
            # Nothing to hand over: the note is an ordinary observation and no
            # call is exposed.
            observation.text_sent = True
            if not action.text:
                note_event.update(
                    text="",
                    model_context_text="",
                    end_of_turn=True,
                    text_requires_complete_audio=False,
                )
            yield note_event
            return
        observation.text_sent = True
        observation.response_text = action.text
        call_event["joyai_function_call"] = {
            "call_id": f"call_{uuid.uuid4().hex}",
            "name": DELEGATION_TOOL_NAME,
            "arguments": arguments,
        }
        yield call_event
        if action.text:
            yield note_event
            return
        # A note-free call still ends its observation. The call result above is
        # published, but nothing else of this output will produce a terminal.
        terminal_event = dict(event)
        terminal_event.update(
            text="",
            model_context_text="",
            end_of_turn=True,
            text_requires_complete_audio=False,
        )
        yield terminal_event
