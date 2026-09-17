# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Session-internal duplex events -> typed :class:`DuplexEvent` objects (stateful projection).

This is the former ``entrypoints/duplex/realtime_output.py`` projector plus the
projection-relevant fields of ``realtime_state.py``, now owned by the session
runner through :class:`RealtimeProjectionState`. The projection consumes the
state (response / item ids, content-part bookkeeping) when it *constructs*
events; rendering an event to wire JSON (``event.to_realtime()``) is pure and
lives on the event classes in ``vllm_omni.engine.duplex.events``.

Besides the output projection (:func:`project_internal_event`) the state also
carries the input-side bookkeeping the old input translator kept (input buffer
flags, conversation items, response-id fallbacks); the ``resolve_*`` /
``note_*`` helpers give the runner the same behaviour for the corresponding
commands.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from vllm_omni.engine.duplex.audio import convert_output_audio
from vllm_omni.engine.duplex.commands import (
    AppendAudio,
    CancelResponse,
    ClearOutputAudio,
    Commit,
    CreateItem,
    DeleteItem,
    TruncateItem,
)
from vllm_omni.engine.duplex.events import (
    AudioDelta,
    AudioDone,
    ContentPartAdded,
    ContentPartDone,
    DuplexEvent,
    DuplexRawEvent,
    FunctionCallArgumentsDelta,
    FunctionCallArgumentsDone,
    InputCleared,
    InputCommitted,
    InputTranscriptionCompleted,
    ItemAdded,
    ItemCreated,
    ItemDeleted,
    ItemDone,
    ItemRetrieved,
    ItemTruncated,
    Listen,
    OutputAudioCleared,
    OutputItemAdded,
    OutputItemDone,
    RateLimitsUpdated,
    ResponseCreated,
    ResponseDone,
    SessionClosed,
    SessionCreated,
    SessionReplaced,
    SessionResumed,
    SessionResyncRequired,
    SessionUpdated,
    Speak,
    SpeechStarted,
    SpeechStopped,
    TextDelta,
    TextDone,
    TranscriptDelta,
    TranscriptDone,
    error_event,
)
from vllm_omni.engine.duplex.realtime_commands import (
    RealtimeInputDefaults,
    apply_realtime_session_defaults,
    build_append_audio,
    copy_realtime_input_hints,
    input_looks_like_speech,
    input_transcript_from_item,
    parse_realtime_audio_format,
    realtime_audio_format_object,
    realtime_output_format,
    truncate_realtime_item_content,
    validate_realtime_item_truncate,
)

if TYPE_CHECKING:
    # Runtime import pulls in the VAD backend; the projection only needs the type.
    from vllm_omni.engine.duplex.turn_detection import TurnDetectionResult


def _str_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _int_or(value: object, default: int = 0) -> int:
    return int(value) if isinstance(value, int | float) else default


@dataclass(slots=True)
class _ResponseProjection:
    item_id: str
    transcript_parts: list[str] = field(default_factory=list)
    text_parts: list[str] = field(default_factory=list)
    audio_duration_ms: int | None = None
    audio_text_marks: list[dict[str, int]] = field(default_factory=list)
    audio_delta_emitted: bool = False
    audio_done_emitted: bool = False
    audio_part_added: bool = False
    audio_part_done: bool = False
    text_part_added: bool = False
    text_part_done: bool = False
    output_text_done: bool = False
    output_item_done: bool = False
    conversation_item_done: bool = False
    speak_emitted: bool = False
    done_emitted: bool = False

    @property
    def transcript(self) -> str:
        return "".join(self.transcript_parts)

    @property
    def text(self) -> str:
        return "".join(self.text_parts)


@dataclass(slots=True)
class RealtimeProjectionState:
    """Per-session Realtime projection state (lives on the engine-side runner)."""

    session_id: str
    model: str | None = None
    default_payload: Mapping[str, object] | None = None
    #: Wire defaults (input/output formats, rates, silence rms) declared by the session object.
    defaults: RealtimeInputDefaults = field(default_factory=RealtimeInputDefaults)
    #: The first ``session.created`` also emits ``session.updated`` (client sent session.update to open).
    initial_session_update: bool = False
    # ---- response / item projection ----
    response_states: dict[str | int, _ResponseProjection] = field(default_factory=dict)
    item_truncation_cursors: dict[str, tuple[int, int]] = field(default_factory=dict)
    active_response_id: str | None = None
    last_response_id: str | None = None
    conversation_items: dict[str, dict[str, object]] = field(default_factory=dict)
    last_conversation_item_id: str | None = None
    pending_commit_item_ids: list[str] = field(default_factory=list)
    # ---- input buffer projection (was on the input translator) ----
    input_speech_started: bool = False
    active_input_item_id: str | None = None
    input_audio_buffer_has_audio: bool = False
    input_audio_buffer_had_non_speech: bool = False
    input_audio_buffer_transcript_parts: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.default_payload is not None:
            self.apply_session_defaults(self.default_payload)

    # ---- session defaults ----

    def apply_session_defaults(self, session_payload: Mapping[str, object]) -> None:
        self.defaults = apply_realtime_session_defaults(self.defaults, session_payload)

    @property
    def input_audio_format(self) -> str:
        return self.defaults.input_audio_format

    @property
    def input_sample_rate_hz(self) -> int:
        return self.defaults.input_sample_rate_hz

    @property
    def output_audio_format(self) -> str:
        return self.defaults.output_audio_format

    @property
    def output_sample_rate_hz(self) -> int | None:
        return self.defaults.output_sample_rate_hz

    @property
    def overlap_silence_rms(self) -> float:
        return self.defaults.overlap_silence_rms


# ---- response projection helpers ----


def _response_state(
    state: RealtimeProjectionState,
    response_id: object,
    *,
    event: Mapping[str, object] | None = None,
    create: bool = True,
) -> _ResponseProjection | None:
    if not (isinstance(response_id, str) and response_id):
        # An event without a response id has no projection to attach to; a
        # key derived from the transient event object would alias later
        # anonymous events (and leak an entry per terminal).
        return None
    key = response_id
    item_id = f"item_{response_id}"
    projection = state.response_states.get(key)
    if projection is None and create:
        projection = _ResponseProjection(item_id=item_id)
        state.response_states[key] = projection
    return projection


def response_is_done(state: RealtimeProjectionState, response_id: object) -> bool:
    projection = _response_state(state, response_id, create=False)
    return projection is not None and projection.done_emitted


def _response_item_id(state: RealtimeProjectionState, response_id: object) -> str:
    projection = _response_state(state, response_id)
    if projection is not None:
        return projection.item_id
    return f"item_{uuid4().hex}"


def _response_content_part(*, transcript: str = "") -> dict[str, object]:
    return {"type": "audio", "transcript": transcript}


def _response_text_content_part(*, text: str = "") -> dict[str, object]:
    return {"type": "text", "text": text}


def _response_item_content_part(
    *,
    transcript: str = "",
    audio_duration_ms: int | None = None,
    audio_text_marks: list[dict[str, int]] | None = None,
) -> dict[str, object]:
    part: dict[str, object] = {"type": "output_audio", "transcript": transcript}
    if audio_duration_ms is not None:
        part["audio_duration_ms"] = int(audio_duration_ms)
    if audio_text_marks:
        part["audio_text_marks"] = [dict(mark) for mark in audio_text_marks]
    return part


def _response_item_text_content_part(*, text: str = "") -> dict[str, object]:
    return {"type": "output_text", "text": text}


def _previous_item_id(state: RealtimeProjectionState, item_id: str) -> str | None:
    previous: str | None = None
    for known_id in state.conversation_items:
        if known_id == item_id:
            return previous
        previous = known_id
    if state.last_conversation_item_id == item_id:
        return previous
    return state.last_conversation_item_id


def _conversation_item_added_events(state: RealtimeProjectionState, item: dict[str, object]) -> list[DuplexEvent]:
    item_id = item.get("id")
    explicit_previous_item_id = item.pop("_previous_item_id", None)
    previous_item_id = (
        explicit_previous_item_id if isinstance(explicit_previous_item_id, str) else state.last_conversation_item_id
    )
    if isinstance(item_id, str) and item_id:
        state.last_conversation_item_id = item_id
    return [
        ItemAdded(previous_item_id=previous_item_id, item=item),
        ItemCreated(previous_item_id=previous_item_id, item=item),
    ]


def _conversation_item_done_event(state: RealtimeProjectionState, item: dict[str, object]) -> ItemDone:
    item_id = item.get("id")
    previous_item_id = _previous_item_id(state, item_id) if isinstance(item_id, str) else None
    if isinstance(item_id, str) and item_id:
        state.conversation_items[item_id] = item
        state.last_conversation_item_id = item_id
    return ItemDone(previous_item_id=previous_item_id, item=item)


def _remove_conversation_item(state: RealtimeProjectionState, item_id: str) -> bool:
    removed = state.conversation_items.pop(item_id, None) is not None
    if state.last_conversation_item_id == item_id:
        remaining_ids = list(state.conversation_items)
        state.last_conversation_item_id = remaining_ids[-1] if remaining_ids else None
    state.item_truncation_cursors.pop(item_id, None)
    return removed


def _apply_pending_item_truncation(state: RealtimeProjectionState, item: dict[str, object]) -> None:
    item_id = item.get("id")
    if not isinstance(item_id, str) or not item_id:
        return
    cursor = state.item_truncation_cursors.get(item_id)
    if cursor is None:
        return
    content_index, audio_end_ms = cursor
    truncate_realtime_item_content(item, content_index=content_index, audio_end_ms=audio_end_ms)


def _response_done_output_item(
    state: RealtimeProjectionState, response_id: object, *, status: str
) -> dict[str, object]:
    item_id = _response_item_id(state, response_id)
    projection = _response_state(state, response_id)
    transcript = projection.transcript if projection is not None else ""
    text = projection.text if projection is not None else ""
    audio_duration_ms = projection.audio_duration_ms if projection is not None else None
    audio_text_marks = projection.audio_text_marks if projection is not None else None
    content: list[dict[str, object]] = []
    if (projection is not None and projection.audio_part_added) or transcript or audio_duration_ms is not None:
        content.append(
            _response_item_content_part(
                transcript=transcript,
                audio_duration_ms=audio_duration_ms,
                audio_text_marks=audio_text_marks,
            )
        )
    if text:
        content.append(_response_item_text_content_part(text=text))
    item = {
        "id": item_id,
        "object": "realtime.item",
        "type": "message",
        "role": "assistant",
        "status": status,
        "content": content,
    }
    _apply_pending_item_truncation(state, item)
    return item


def _refresh_in_progress_response_item(state: RealtimeProjectionState, response_id: object) -> None:
    if not isinstance(response_id, str) or not response_id:
        return
    item_id = _response_item_id(state, response_id)
    item = state.conversation_items.get(item_id)
    if not isinstance(item, dict):
        return
    content = item.get("content")
    if not isinstance(content, list):
        content = []
        item["content"] = content
    projection = _response_state(state, response_id)
    if projection is None:
        return
    transcript = projection.transcript
    audio_duration_ms = projection.audio_duration_ms
    audio_text_marks = projection.audio_text_marks
    has_audio = projection.audio_part_added or bool(transcript) or audio_duration_ms is not None
    if has_audio:
        audio_part = _response_item_content_part(
            transcript=transcript,
            audio_duration_ms=audio_duration_ms,
            audio_text_marks=audio_text_marks,
        )
        if content and isinstance(content[0], dict) and content[0].get("type") in {"audio", "output_audio"}:
            content[0] = audio_part
        else:
            content.insert(0, audio_part)
    text = projection.text
    if text:
        text_index = (
            1 if content and isinstance(content[0], dict) and content[0].get("type") in {"audio", "output_audio"} else 0
        )
        text_part = _response_item_text_content_part(text=text)
        if (
            len(content) > text_index
            and isinstance(content[text_index], dict)
            and content[text_index].get("type") in {"text", "output_text"}
        ):
            content[text_index] = text_part
        else:
            content.insert(text_index, text_part)
    _apply_pending_item_truncation(state, item)


def _append_response_transcript(state: RealtimeProjectionState, response_id: object, text: str) -> None:
    projection = _response_state(state, response_id)
    if projection is None or not text:
        return
    projection.transcript_parts.append(text)


def _ensure_response_text_part_added(state: RealtimeProjectionState, response_id: object) -> list[DuplexEvent]:
    projection = _response_state(state, response_id)
    if projection is None or projection.text_part_added:
        return []
    projection.text_part_added = True
    return [
        ContentPartAdded(
            response_id=_str_or_none(response_id),
            item_id=_response_item_id(state, response_id),
            content_index=1 if projection.audio_part_added else 0,
            part=_response_text_content_part(),
        )
    ]


def _ensure_response_audio_part_added(state: RealtimeProjectionState, response_id: object) -> list[DuplexEvent]:
    projection = _response_state(state, response_id)
    if projection is None or projection.audio_part_added:
        return []
    projection.audio_part_added = True
    return [
        ContentPartAdded(
            response_id=_str_or_none(response_id),
            item_id=_response_item_id(state, response_id),
            part=_response_content_part(),
        )
    ]


def _remember_response_audio_metadata(
    state: RealtimeProjectionState, response_id: object, event: Mapping[str, object]
) -> None:
    projection = _response_state(state, response_id)
    if projection is None:
        return
    duration = event.get("audio_duration_ms")
    playback = event.get("playback")
    if not isinstance(duration, int | float) and isinstance(playback, dict):
        duration = playback.get("sent_ms") or playback.get("generated_ms")
    if isinstance(duration, int | float):
        projection.audio_duration_ms = max(projection.audio_duration_ms or 0, int(duration))
    marks = event.get("audio_text_marks")
    if not isinstance(marks, list):
        return
    clean_marks: list[dict[str, int]] = []
    for mark in marks:
        if not isinstance(mark, dict):
            continue
        text_chars = mark.get("text_chars")
        audio_end_ms = mark.get("audio_end_ms", mark.get("audio_ms"))
        if not isinstance(text_chars, int | float) or not isinstance(audio_end_ms, int | float):
            continue
        clean_marks.append({"text_chars": max(0, int(text_chars)), "audio_end_ms": max(0, int(audio_end_ms))})
    if clean_marks:
        merged = list(projection.audio_text_marks)
        merged.extend(clean_marks)
        deduped: dict[tuple[int, int], dict[str, int]] = {}
        for mark in merged:
            deduped[(int(mark["audio_end_ms"]), int(mark["text_chars"]))] = mark
        projection.audio_text_marks = sorted(
            deduped.values(), key=lambda mark: (mark["audio_end_ms"], mark["text_chars"])
        )


def _response_created_event(event: Mapping[str, object]) -> ResponseCreated:
    response_id = event.get("response_id")
    metadata = event.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    metadata = {**metadata, "duplex_event": dict(event)}
    return ResponseCreated(
        response_id=_str_or_none(response_id),
        response={
            "id": response_id,
            "object": "realtime.response",
            "status": "in_progress",
            "status_details": None,
            "output": [],
            "modalities": event.get("modalities") or ["audio", "text"],
            "metadata": metadata,
        },
    )


def _response_speak_metadata(event: Mapping[str, object]) -> dict[str, object]:
    return {key: event[key] for key in ("session_id", "epoch", "model_speak") if key in event}


def _realtime_audio_delta_events(
    state: RealtimeProjectionState,
    event: Mapping[str, object],
    response_id: object,
    audio: str,
) -> list[DuplexEvent]:
    item_id = _response_item_id(state, response_id)
    fmt, format_rate = parse_realtime_audio_format(event.get("format", "wav"))
    source_fmt = realtime_output_format(fmt)
    source_sample_rate_hz = event.get("sample_rate_hz") or format_rate
    target_sample_rate_hz = (
        state.output_sample_rate_hz if state.output_audio_format in {"g711_ulaw", "g711_alaw"} else None
    )
    audio, fmt, converted_sample_rate_hz = convert_output_audio(
        audio,
        source_fmt=source_fmt,
        target_fmt=state.output_audio_format,
        source_sample_rate_hz=(int(source_sample_rate_hz) if isinstance(source_sample_rate_hz, int | float) else None),
        target_sample_rate_hz=target_sample_rate_hz,
    )
    sample_rate_hz = (
        converted_sample_rate_hz
        if fmt in {"g711_ulaw", "g711_alaw"}
        else event.get("sample_rate_hz") or format_rate or state.output_sample_rate_hz
    )
    metadata: dict[str, object] = {}
    for key in ("session_id", "epoch", "model_speak", "end_of_turn", "playback", "vllm_omni"):
        if key in event:
            metadata[key] = event[key]
    duration_ms = event.get("audio_duration_ms")
    if isinstance(duration_ms, int | float):
        metadata["audio_duration_ms"] = int(duration_ms)
    marks = event.get("audio_text_marks")
    if isinstance(marks, list):
        metadata["audio_text_marks"] = marks
    projection = _response_state(state, response_id)
    if projection is not None:
        projection.audio_delta_emitted = True
        _remember_response_audio_metadata(state, response_id, event)
    events: list[DuplexEvent] = []
    if projection is not None and not projection.speak_emitted and metadata.get("model_speak") is True:
        projection.speak_emitted = True
        events.append(
            Speak(response_id=_str_or_none(response_id), item_id=item_id, metadata=_response_speak_metadata(event))
        )
    events.append(
        AudioDelta(
            response_id=_str_or_none(response_id),
            item_id=item_id,
            delta=str(audio),
            format=str(fmt),
            sample_rate_hz=int(sample_rate_hz) if isinstance(sample_rate_hz, int | float) else None,
            metadata=metadata or None,
        )
    )
    return events


def _realtime_audio_done_events(
    state: RealtimeProjectionState, event: Mapping[str, object], response_id: object
) -> list[DuplexEvent]:
    item_id = _response_item_id(state, response_id)
    projection = _response_state(state, response_id, event=event)
    transcript = projection.transcript if projection is not None else ""
    events: list[DuplexEvent] = []
    if (
        isinstance(response_id, str)
        and projection is not None
        and not projection.audio_delta_emitted
        and not transcript
    ):
        return events
    if projection is None or not projection.audio_done_emitted:
        if projection is not None:
            projection.audio_done_emitted = True
        events.append(AudioDone(response_id=_str_or_none(response_id), item_id=item_id))
        if transcript:
            events.append(TranscriptDone(response_id=_str_or_none(response_id), item_id=item_id, transcript=transcript))
    return events


def _realtime_response_done_event(
    state: RealtimeProjectionState,
    event: Mapping[str, object],
    *,
    projection: _ResponseProjection | None = None,
    status: str = "completed",
    status_details: dict[str, object] | None = None,
) -> ResponseDone | None:
    response_id = event.get("response_id")
    if projection is None:
        projection = _response_state(state, response_id, event=event)
    if projection is not None:
        if projection.done_emitted:
            return None
        projection.done_emitted = True
    return ResponseDone(
        response_id=_str_or_none(response_id),
        response={
            "id": response_id,
            "object": "realtime.response",
            "status": status,
            "status_details": status_details,
            "output": [_response_done_output_item(state, response_id, status=status)],
            "metadata": dict(event),
        },
    )


def _realtime_response_terminal_events(
    state: RealtimeProjectionState,
    event: Mapping[str, object],
    response_id: object,
    *,
    status: str = "completed",
    status_details: dict[str, object] | None = None,
) -> list[DuplexEvent]:
    item_id = _response_item_id(state, response_id)
    projection = _response_state(state, response_id, event=event)
    transcript = projection.transcript if projection is not None else ""
    rid = _str_or_none(response_id)
    events: list[DuplexEvent] = []
    if projection is not None and projection.audio_part_added and not projection.audio_part_done:
        projection.audio_part_done = True
        events.append(
            ContentPartDone(response_id=rid, item_id=item_id, part=_response_content_part(transcript=transcript))
        )
    if projection is not None and projection.text_parts and not projection.output_text_done:
        projection.output_text_done = True
        events.append(
            TextDone(
                response_id=rid,
                item_id=item_id,
                content_index=1 if projection.audio_part_added else 0,
                text=projection.text,
            )
        )
    if projection is not None and projection.text_parts and not projection.text_part_done:
        projection.text_part_done = True
        events.append(
            ContentPartDone(
                response_id=rid,
                item_id=item_id,
                content_index=1 if projection.audio_part_added else 0,
                part=_response_text_content_part(text=projection.text),
            )
        )
    if projection is None or not projection.output_item_done:
        if projection is not None:
            projection.output_item_done = True
        item = _response_done_output_item(state, response_id, status=status)
        events.append(OutputItemDone(response_id=rid, item=item))
        if projection is None or not projection.conversation_item_done:
            if projection is not None:
                projection.conversation_item_done = True
            events.append(_conversation_item_done_event(state, item))
    done_event = _realtime_response_done_event(
        state,
        {**event, "response_id": response_id},
        projection=projection,
        status=status,
        status_details=status_details,
    )
    if done_event is not None:
        events.append(done_event)
        # vLLM-Omni does not expose a Realtime-specific quota budget; emit the
        # terminal event with an empty list so clients that sequence on it work.
        events.append(RateLimitsUpdated())
        if isinstance(response_id, str) and response_id == state.active_response_id:
            state.active_response_id = None
    return events


def _user_item_content_from_duplex_message(message: object) -> list[dict[str, object]]:
    if not isinstance(message, dict):
        return [{"type": "input_audio", "transcript": ""}]
    message_transcript = message.get("transcript")
    content = message.get("content")
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    if not isinstance(content, list):
        return [
            {"type": "input_audio", "transcript": message_transcript if isinstance(message_transcript, str) else ""}
        ]
    parts: list[dict[str, object]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type in {"text", "input_text"} and isinstance(part.get("text"), str):
            parts.append({"type": "input_text", "text": part["text"]})
            continue
        if part_type == "audio_url":
            transcript = (
                part.get("transcript")
                if isinstance(part.get("transcript"), str)
                else message_transcript
                if isinstance(message_transcript, str)
                else ""
            )
            parts.append({"type": "input_audio", "transcript": transcript})
            continue
        if part_type in {"audio", "input_audio"}:
            transcript = part.get("transcript") if isinstance(part.get("transcript"), str) else ""
            parts.append({"type": "input_audio", "transcript": transcript})
    return parts or [{"type": "input_audio", "transcript": ""}]


def _input_audio_transcription_completed_event(
    item_id: str, item: Mapping[str, object]
) -> InputTranscriptionCompleted | None:
    content = item.get("content")
    if not isinstance(content, list):
        return None
    for index, part in enumerate(content):
        if not isinstance(part, dict) or part.get("type") != "input_audio":
            continue
        transcript = part.get("transcript")
        if isinstance(transcript, str) and transcript:
            return InputTranscriptionCompleted(item_id=item_id, content_index=index, transcript=transcript)
    return None


def _pop_pending_commit_item_id(state: RealtimeProjectionState) -> str:
    if state.pending_commit_item_ids:
        return state.pending_commit_item_ids.pop(0)
    return f"item_{uuid4().hex}"


def realtime_session_payload(state: RealtimeProjectionState, session: object) -> dict[str, object]:
    """Fill the Realtime ``session`` object defaults for session.created/updated."""
    payload = dict(session) if isinstance(session, Mapping) else {}
    payload.setdefault("object", "realtime.session")
    payload.setdefault("type", "realtime")
    payload.setdefault("id", payload.get("id") or state.session_id)
    payload.setdefault("model", payload.get("model") or state.model)
    payload.setdefault("input_audio_format", state.input_audio_format)
    payload.setdefault("output_audio_format", state.output_audio_format)
    payload.setdefault("modalities", payload.get("modalities") or ["text", "audio"])
    payload.setdefault("output_modalities", payload.get("output_modalities") or payload.get("modalities"))
    payload.setdefault(
        "audio",
        {
            "input": {
                "format": realtime_audio_format_object(
                    state.input_audio_format, sample_rate_hz=state.input_sample_rate_hz
                ),
                "sample_rate_hz": state.input_sample_rate_hz,
            },
            "output": {
                "format": realtime_audio_format_object(
                    state.output_audio_format, sample_rate_hz=state.output_sample_rate_hz
                ),
            },
        },
    )
    payload.setdefault("turn_detection", payload.get("turn_detection"))
    payload.setdefault("input_audio_transcription", payload.get("input_audio_transcription"))
    payload.setdefault("tracing", payload.get("tracing"))
    return payload


# ---- main projection ----


def project_internal_event(state: RealtimeProjectionState, event: Mapping[str, object]) -> list[DuplexEvent]:
    """Project one session-internal event onto 0..n typed public events."""
    return _project(state, dict(event))


def _project(state: RealtimeProjectionState, event: dict[str, object]) -> list[DuplexEvent]:
    event_type = event.get("type")
    if event_type == "session.created":
        session = realtime_session_payload(state, event.get("session"))
        events: list[DuplexEvent] = [
            SessionCreated(
                session=session,
                attachment_generation=(
                    cast("int", event["attachment_generation"])
                    if isinstance(event.get("attachment_generation"), int)
                    else None
                ),
                resume_token=_str_or_none(event.get("resume_token")),
            )
        ]
        if state.initial_session_update:
            events.append(SessionUpdated(session=session))
            state.initial_session_update = False
        return events
    if event_type == "session.updated":
        return [SessionUpdated(session=realtime_session_payload(state, event.get("session")))]
    if event_type == "session.resumed":
        return [
            SessionResumed(
                session=realtime_session_payload(state, event.get("session")),
                attachment_generation=(
                    cast("int", event["attachment_generation"])
                    if isinstance(event.get("attachment_generation"), int)
                    else None
                ),
                resume_token=_str_or_none(event.get("resume_token")),
            )
        ]
    if event_type == "session.replaced":
        return [SessionReplaced(attachment_generation=_int_or(event.get("attachment_generation")))]
    if event_type == "session.resync_required":
        return [SessionResyncRequired(reason=str(event.get("reason") or "journal_gap"))]
    if event_type == "session.closed":
        return [SessionClosed(reason=str(event.get("reason") or "closed"), details=event)]
    if event_type == "response.created":
        response_id = event.get("response_id")
        if isinstance(response_id, str) and response_id:
            state.active_response_id = response_id
            state.last_response_id = response_id
        item_id = _response_item_id(state, response_id)
        modalities = event.get("modalities")
        has_audio_modality = not isinstance(modalities, list) or "audio" in modalities
        item = {
            "id": item_id,
            "object": "realtime.item",
            "type": "message",
            "role": "assistant",
            "status": "in_progress",
            "content": [],
        }
        state.conversation_items[item_id] = item
        events = [
            _response_created_event(event),
            *_conversation_item_added_events(state, item),
            OutputItemAdded(response_id=_str_or_none(response_id), item=item),
        ]
        if has_audio_modality:
            events.extend(_ensure_response_audio_part_added(state, response_id))
        return events
    if event_type == "response.listen":
        return [Listen(response_id=_str_or_none(event.get("response_id")), details=event)]
    if event_type == "response.speak":
        response_id = event.get("response_id")
        projection = _response_state(state, response_id)
        if projection is not None:
            if projection.speak_emitted:
                return []
            projection.speak_emitted = True
        return [
            Speak(
                response_id=_str_or_none(response_id),
                item_id=_response_item_id(state, response_id),
                metadata=_response_speak_metadata(event),
            )
        ]
    if event_type == "response.output_audio.delta":
        response_id = event.get("response_id")
        audio = event.get("audio", "")
        events = []
        if isinstance(audio, str) and audio:
            events.extend(_ensure_response_audio_part_added(state, response_id))
            events.extend(_realtime_audio_delta_events(state, event, response_id, audio))
            _refresh_in_progress_response_item(state, response_id)
        text = event.get("text")
        has_text = isinstance(text, str) and bool(text)
        if has_text:
            _append_response_transcript(state, response_id, cast("str", text))
            _refresh_in_progress_response_item(state, response_id)
        # Keep the audio.delta + transcript.delta pair invariant even for
        # text-less units so clients that treat the pair as unit-complete work.
        if has_text or (isinstance(audio, str) and bool(audio)):
            events.append(
                TranscriptDelta(
                    response_id=_str_or_none(response_id),
                    item_id=_response_item_id(state, response_id),
                    delta=cast("str", text) if has_text else "",
                )
            )
        if event.get("end_of_turn") is True:
            events.extend(_realtime_audio_done_events(state, event, response_id))
            events.extend(
                _realtime_response_terminal_events(
                    state,
                    event,
                    response_id,
                    status="completed",
                    status_details={"type": "completed", "reason": event.get("finish_reason") or "stop"},
                )
            )
        return events
    if event_type == "response.text.delta":
        response_id = event.get("response_id")
        text = event.get("delta", "")
        projection = _response_state(state, response_id)
        if projection is not None and isinstance(text, str) and text:
            projection.text_parts.append(text)
            _refresh_in_progress_response_item(state, response_id)
        events = _ensure_response_text_part_added(state, response_id)
        events.append(
            TextDelta(
                response_id=_str_or_none(response_id),
                item_id=_response_item_id(state, response_id),
                content_index=1 if projection is not None and projection.audio_part_added else 0,
                delta=str(text) if isinstance(text, str) else "",
            )
        )
        return events
    if event_type == "response.done":
        response_id = event.get("response_id")
        status = cast("str", event.get("status")) if isinstance(event.get("status"), str) else "completed"
        status_details = (
            cast("dict[str, object]", event.get("status_details"))
            if isinstance(event.get("status_details"), dict)
            else None
        )
        return [
            *_realtime_audio_done_events(state, event, response_id),
            *_realtime_response_terminal_events(
                state, event, response_id, status=status, status_details=status_details
            ),
        ]
    if event_type == "input.committed":
        event_item_id = event.get("realtime_item_id")
        item_id = (
            event_item_id if isinstance(event_item_id, str) and event_item_id else _pop_pending_commit_item_id(state)
        )
        item = state.conversation_items.get(item_id)
        events = []
        if item is None:
            message = event.get("message")
            no_response = event.get("no_response") is True
            is_speech = event.get("is_speech")
            item = {
                "id": item_id,
                "object": "realtime.item",
                "type": "message",
                "role": "user",
                "status": "completed",
                "content": (
                    [{"type": "input_audio", "transcript": "", "is_speech": False}]
                    if no_response and is_speech is False
                    else _user_item_content_from_duplex_message(message)
                ),
            }
            state.conversation_items[item_id] = item
            events.extend(_conversation_item_added_events(state, item))
        item["status"] = "completed"
        events.append(
            InputCommitted(previous_item_id=_previous_item_id(state, item_id), item_id=item_id, details=event)
        )
        transcription_event = _input_audio_transcription_completed_event(item_id, item)
        if transcription_event is not None:
            events.append(transcription_event)
        events.append(_conversation_item_done_event(state, item))
        return events
    if event_type == "input.cancelled":
        return [InputCleared()]
    if event_type == "audio.cancelled":
        response_id = event.get("response_id")
        events = []
        if event.get("reason") == "output_audio_buffer_clear":
            if not isinstance(response_id, str) or not response_id:
                response_id = state.active_response_id or state.last_response_id
            events.append(OutputAudioCleared(response_id=_str_or_none(response_id)))
            if not isinstance(response_id, str) or not response_id:
                return events
        elif not isinstance(response_id, str) or not response_id:
            response_id = state.active_response_id
        if not isinstance(response_id, str) or not response_id:
            return events
        if response_is_done(state, response_id):
            return events
        committed_ms = event.get("committed_ms")
        if isinstance(committed_ms, int | float):
            item_id = _response_item_id(state, response_id)
            committed_audio_ms = max(0, int(committed_ms))
            state.item_truncation_cursors[item_id] = (0, committed_audio_ms)
            item = state.conversation_items.get(item_id)
            if item is not None:
                truncate_realtime_item_content(item, content_index=0, audio_end_ms=committed_audio_ms)
        events.extend(_realtime_audio_done_events(state, event, response_id))
        events.extend(
            _realtime_response_terminal_events(
                state,
                event,
                response_id,
                status="cancelled",
                status_details={"type": "cancelled", "reason": event.get("reason") or "client_cancelled"},
            )
        )
        if isinstance(response_id, str) and response_id == state.active_response_id:
            state.active_response_id = None
        return events
    if event_type == "conversation.item.created":
        item = event.get("item")
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            item_id = str(item["id"])
            already_known = item_id in state.conversation_items
            state.conversation_items[item_id] = item
            if already_known:
                if item.get("status") == "completed":
                    return [_conversation_item_done_event(state, item)]
                return []
            events = _conversation_item_added_events(state, item)
            if item.get("status") == "completed":
                events.append(_conversation_item_done_event(state, item))
            return events
        return [ItemCreated(item=item if isinstance(item, Mapping) else {})]
    if event_type == "conversation.item.deleted":
        item_id = event.get("item_id")
        if isinstance(item_id, str):
            _remove_conversation_item(state, item_id)
        return [ItemDeleted(item_id=_str_or_none(item_id), details=event)]
    if event_type == "conversation.item.truncated":
        item_id = event.get("item_id")
        audio_end_ms = event.get("audio_end_ms")
        content_index = event.get("content_index", 0)
        if isinstance(item_id, str):
            item = state.conversation_items.get(item_id)
            if item is not None:
                truncate_realtime_item_content(
                    item,
                    content_index=_int_or(content_index),
                    audio_end_ms=_int_or(audio_end_ms),
                )
        return [
            ItemTruncated(
                item_id=_str_or_none(item_id),
                content_index=_int_or(content_index),
                audio_end_ms=_int_or(audio_end_ms),
                details=event,
            )
        ]
    if event_type == "conversation.item.retrieved":
        item = event.get("item")
        return [ItemRetrieved(item=item if isinstance(item, Mapping) else {})]
    if event_type == "response.output_item.done":
        response_id = event.get("response_id")
        return _realtime_response_terminal_events(
            state,
            event,
            response_id,
            status="completed",
            status_details={"type": "completed", "reason": event.get("finish_reason") or "stop"},
        )
    if event_type == "function_call.done":
        return _function_call_done_events(state, event)
    if event_type == "input_audio_buffer.speech_started":
        return [
            SpeechStarted(
                audio_start_ms=_int_or(event.get("audio_start_ms")), item_id=_str_or_none(event.get("item_id"))
            )
        ]
    if event_type == "input_audio_buffer.speech_stopped":
        return [
            SpeechStopped(audio_end_ms=_int_or(event.get("audio_end_ms")), item_id=_str_or_none(event.get("item_id")))
        ]
    return [DuplexRawEvent(internal_type=str(event_type), details=event)]


def _function_call_done_events(state: RealtimeProjectionState, event: dict[str, object]) -> list[DuplexEvent]:
    call_id = event.get("call_id")
    name = event.get("name")
    arguments = event.get("arguments", "")
    if not isinstance(call_id, str) or not call_id or not isinstance(name, str) or not name:
        return [DuplexRawEvent(internal_type="function_call.done", details=event)]
    if not isinstance(arguments, str):
        arguments = str(arguments)
    response_id = f"resp_{uuid4().hex}"
    item_id = f"item_{uuid4().hex}"
    item = {
        "id": item_id,
        "object": "realtime.item",
        "type": "function_call",
        "status": "completed",
        "name": name,
        "call_id": call_id,
        "arguments": arguments,
    }
    state.conversation_items[item_id] = item
    response = {
        "id": response_id,
        "object": "realtime.response",
        "status": "completed",
        "status_details": {"type": "completed", "reason": "completed"},
        "output": [dict(item)],
        "metadata": {},
    }
    return [
        ResponseCreated(response_id=response_id, response={**response, "status": "in_progress", "output": []}),
        *_conversation_item_added_events(state, item),
        OutputItemAdded(response_id=response_id, item=dict(item)),
        FunctionCallArgumentsDelta(response_id=response_id, item_id=item_id, call_id=call_id, delta=arguments),
        FunctionCallArgumentsDone(response_id=response_id, item_id=item_id, call_id=call_id, arguments=arguments),
        OutputItemDone(response_id=response_id, item=dict(item)),
        _conversation_item_done_event(state, item),
        ResponseDone(response_id=response_id, response=response),
    ]


# ---- input-side helpers for the runner (were on the input translator) ----


def register_user_item(
    state: RealtimeProjectionState,
    item: Mapping[str, object],
    *,
    previous_item_id: str | None = None,
) -> list[DuplexEvent]:
    """Record a client-created user message item and return its ack events.

    Replaces the old translator's ``_send_realtime_input_ack``: call it when
    handling a ``CreateItem`` whose item role is ``user`` (before the internal
    ``conversation.item.created`` is emitted, which then projects to ``done``).
    """
    normalized = dict(item)
    if isinstance(previous_item_id, str):
        normalized["_previous_item_id"] = previous_item_id
    if normalized.get("role") != "user":
        return []
    state.conversation_items[str(normalized["id"])] = normalized
    return _conversation_item_added_events(state, normalized)


def retrieve_item_events(state: RealtimeProjectionState, payload: Mapping[str, object]) -> list[DuplexEvent]:
    """Answer ``conversation.item.retrieve`` from the projected conversation items."""
    event_id = payload.get("event_id")
    item_id = payload.get("item_id")
    if not isinstance(item_id, str) or not item_id:
        return [error_event("missing_item_id", "conversation.item.retrieve requires item_id", event_id=event_id)]
    item = state.conversation_items.get(item_id)
    if item is None:
        return [error_event("item_not_found", f"Conversation item not found: {item_id}", event_id=event_id)]
    return [ItemRetrieved(item=item)]


def emit_input_speech_started(state: RealtimeProjectionState, audio_start_ms: object = 0) -> list[DuplexEvent]:
    if state.input_speech_started:
        return []
    state.input_speech_started = True
    if state.active_input_item_id is None:
        state.active_input_item_id = f"item_{uuid4().hex}"
    return [SpeechStarted(audio_start_ms=_int_or(audio_start_ms), item_id=state.active_input_item_id)]


def emit_input_speech_stopped(
    state: RealtimeProjectionState, *, item_id: str, audio_end_ms: object = 0
) -> list[DuplexEvent]:
    if not state.input_speech_started:
        return []
    state.input_speech_started = False
    return [SpeechStopped(audio_end_ms=_int_or(audio_end_ms), item_id=item_id)]


def _remember_input_transcript_hint(state: RealtimeProjectionState, event: Mapping[str, object]) -> None:
    transcript = event.get("transcript")
    if not isinstance(transcript, str):
        transcript = event.get("text") if isinstance(event.get("text"), str) else None
    if not isinstance(transcript, str):
        hints = event.get("hints")
        if isinstance(hints, dict):
            transcript = hints.get("transcript")
            if not isinstance(transcript, str):
                transcript = hints.get("text") if isinstance(hints.get("text"), str) else None
    if isinstance(transcript, str) and transcript:
        parts = state.input_audio_buffer_transcript_parts
        if parts and parts[-1] == transcript:
            return
        parts.append(transcript)


def _consume_input_transcript_hint(state: RealtimeProjectionState) -> str:
    transcript = "".join(state.input_audio_buffer_transcript_parts).strip()
    state.input_audio_buffer_transcript_parts.clear()
    return transcript


def note_input_append(
    state: RealtimeProjectionState,
    payload: dict[str, object],
    *,
    vad_result: TurnDetectionResult | None = None,
) -> list[DuplexEvent]:
    """Update the input-buffer projection for one appended chunk; returns typed events.

    ``payload`` is the internal ``input_audio_buffer.append`` dictionary
    (``AppendAudio.payload()``), optionally after ``apply_turn_detection_result``;
    ``vad_result`` is the ``TurnDetectionResult`` when server VAD is active. The
    returned events are ``input_audio_buffer.speech_started`` /
    ``speech_stopped`` exactly as the old translator produced them.
    """
    audio = payload.get("audio")
    looks_like_speech = bool(vad_result.is_speech) if vad_result is not None else payload.get("is_speech") is not False
    has_audio = isinstance(audio, str) and bool(audio)
    state.input_audio_buffer_has_audio = state.input_audio_buffer_has_audio or (looks_like_speech and has_audio)
    state.input_audio_buffer_had_non_speech = state.input_audio_buffer_had_non_speech or (
        not looks_like_speech and has_audio
    )
    events: list[DuplexEvent] = []
    stop_ms: object = payload.get("audio_end_ms", payload.get("audio_ms", 0))
    if vad_result is not None:
        if vad_result.speech_stopped and vad_result.audio_end_ms is not None:
            stop_ms = vad_result.audio_end_ms
        if vad_result.speech_stopped and vad_result.speech_active:
            events.extend(
                emit_input_speech_stopped(
                    state,
                    item_id=state.active_input_item_id or f"item_{uuid4().hex}",
                    audio_end_ms=stop_ms,
                )
            )
        if vad_result.speech_started:
            start_ms = (
                vad_result.audio_start_ms if vad_result.audio_start_ms is not None else payload.get("audio_start_ms", 0)
            )
            events.extend(emit_input_speech_started(state, start_ms))
        if vad_result.speech_stopped and not vad_result.speech_active:
            events.extend(
                emit_input_speech_stopped(
                    state,
                    item_id=state.active_input_item_id or f"item_{uuid4().hex}",
                    audio_end_ms=stop_ms,
                )
            )
    elif looks_like_speech:
        events.extend(emit_input_speech_started(state, payload.get("audio_start_ms", 0)))
    if looks_like_speech:
        _remember_input_transcript_hint(state, payload)
    return events


def discard_pending_input_audio(state: RealtimeProjectionState, audio_end_ms: int | None = None) -> list[DuplexEvent]:
    """Drop Realtime input-buffer state that was consumed as overlap (returns typed events)."""
    events: list[DuplexEvent] = []
    if state.input_speech_started and state.active_input_item_id is not None:
        events.append(SpeechStopped(audio_end_ms=max(0, int(audio_end_ms or 0)), item_id=state.active_input_item_id))
    state.input_speech_started = False
    state.active_input_item_id = None
    state.input_audio_buffer_has_audio = False
    state.input_audio_buffer_had_non_speech = False
    state.input_audio_buffer_transcript_parts.clear()
    return events


def clear_input_buffer(state: RealtimeProjectionState) -> None:
    """Input-buffer projection reset for ``input_audio_buffer.clear``."""
    state.input_speech_started = False
    state.active_input_item_id = None
    state.input_audio_buffer_has_audio = False
    state.input_audio_buffer_had_non_speech = False
    state.input_audio_buffer_transcript_parts.clear()


@dataclass(frozen=True, slots=True)
class ResolvedCommit:
    """What a ``Commit`` means given the projected input buffer."""

    #: Internal ``input_audio_buffer.commit`` payload to run, or None when rejected.
    payload: dict[str, object] | None
    #: Events to emit before running (speech_stopped) or instead (error).
    events: list[DuplexEvent]
    #: True when the VAD detector should be reset.
    reset_vad: bool = False


def resolve_commit(state: RealtimeProjectionState, command: Commit) -> ResolvedCommit:
    """Apply the old translator's commit rules (empty buffer, non-speech commit, item id)."""
    if command.realtime_item_id is None and not state.input_audio_buffer_has_audio:
        if state.input_audio_buffer_had_non_speech:
            state.input_audio_buffer_had_non_speech = False
            state.active_input_item_id = None
            payload: dict[str, object] = {
                "type": "input_audio_buffer.commit",
                "final": command.final,
                "response_create": False,
                "is_speech": False,
            }
            if command.event_id is not None:
                payload["realtime_event_id"] = command.event_id
            return ResolvedCommit(payload=payload, events=[], reset_vad=True)
        return ResolvedCommit(
            payload=None,
            events=[
                error_event(
                    "input_audio_buffer_empty",
                    "input_audio_buffer.commit requires a non-empty input audio buffer",
                    event_id=command.event_id,
                )
            ],
        )
    item_id = command.realtime_item_id or state.active_input_item_id or f"item_{uuid4().hex}"
    state.pending_commit_item_ids.append(item_id)
    events = emit_input_speech_stopped(state, item_id=item_id, audio_end_ms=0)
    transcript = _consume_input_transcript_hint(state)
    state.active_input_item_id = None
    state.input_audio_buffer_has_audio = False
    state.input_audio_buffer_had_non_speech = False
    payload = {
        "type": "input_audio_buffer.commit",
        "final": command.final,
        "realtime_item_id": item_id,
        "response_create": bool(command.create_response) if command.create_response is not None else False,
    }
    if command.is_speech is not None:
        payload["is_speech"] = command.is_speech
    if transcript:
        payload["transcript"] = transcript
    if command.event_id is not None:
        payload["realtime_event_id"] = command.event_id
    return ResolvedCommit(payload=payload, events=events, reset_vad=True)


@dataclass(frozen=True, slots=True)
class ResolvedControl:
    """Result of resolving a cancel/clear/delete/truncate against the projection."""

    #: Internal payload(s) to run in order (empty when nothing to do).
    payloads: list[dict[str, object]]
    #: Events to emit instead (already-done response ack, errors).
    events: list[DuplexEvent]


def resolve_cancel_response(state: RealtimeProjectionState, command: CancelResponse) -> ResolvedControl:
    payload: dict[str, object] = {"type": "response.cancel", "reason": "response.cancel"}
    response_id = command.response_id or state.active_response_id or state.last_response_id
    if response_is_done(state, response_id):
        return ResolvedControl(
            payloads=[],
            events=[
                error_event(
                    "response_not_active", f"Response is already complete: {response_id}", event_id=command.event_id
                )
            ],
        )
    if isinstance(response_id, str) and response_id:
        payload["response_id"] = response_id
    if command.event_id is not None:
        payload["realtime_event_id"] = command.event_id
    return ResolvedControl(payloads=[payload], events=[])


def resolve_clear_output_audio(state: RealtimeProjectionState, command: ClearOutputAudio) -> ResolvedControl:
    payload: dict[str, object] = {"type": "output_audio_buffer.clear", "reason": "output_audio_buffer.clear"}
    response_id = command.response_id or state.active_response_id or state.last_response_id
    if response_is_done(state, response_id):
        return ResolvedControl(payloads=[], events=[OutputAudioCleared(response_id=_str_or_none(response_id))])
    if isinstance(response_id, str) and response_id:
        payload["response_id"] = response_id
    if command.event_id is not None:
        payload["realtime_event_id"] = command.event_id
    return ResolvedControl(payloads=[payload], events=[])


def resolve_delete_item(state: RealtimeProjectionState, command: DeleteItem) -> ResolvedControl:
    if command.item_id not in state.conversation_items:
        return ResolvedControl(
            payloads=[],
            events=[
                error_event(
                    "item_not_found", f"Conversation item not found: {command.item_id}", event_id=command.event_id
                )
            ],
        )
    _remove_conversation_item(state, command.item_id)
    return ResolvedControl(payloads=[command.payload()], events=[])


def resolve_truncate_item(state: RealtimeProjectionState, command: TruncateItem) -> ResolvedControl:
    item = state.conversation_items.get(command.item_id)
    if not isinstance(item, dict):
        return ResolvedControl(
            payloads=[],
            events=[
                error_event(
                    "item_not_found", f"Conversation item not found: {command.item_id}", event_id=command.event_id
                )
            ],
        )
    truncate_error = validate_realtime_item_truncate(
        item, content_index=command.content_index, audio_end_ms=command.audio_end_ms
    )
    if truncate_error is not None:
        return ResolvedControl(
            payloads=[], events=[error_event("bad_event", truncate_error, event_id=command.event_id)]
        )
    state.item_truncation_cursors[command.item_id] = (command.content_index, command.audio_end_ms)
    truncate_realtime_item_content(item, content_index=command.content_index, audio_end_ms=command.audio_end_ms)
    ack_payload: dict[str, object] = {
        "type": "playback.ack",
        "item_id": command.item_id,
        "committed_ms": int(command.audio_end_ms),
        "played_ms": int(command.audio_end_ms),
        "truncate": True,
    }
    return ResolvedControl(payloads=[command.payload(), ack_payload], events=[])


def _duplicate_function_call_output(state: RealtimeProjectionState, call_id: object) -> tuple[bool, bool]:
    matching_call = any(
        known.get("type") == "function_call" and known.get("call_id") == call_id
        for known in state.conversation_items.values()
    )
    duplicate_output = any(
        known.get("type") == "function_call_output" and known.get("call_id") == call_id
        for known in state.conversation_items.values()
    )
    return matching_call, duplicate_output


def resolve_create_item(state: RealtimeProjectionState, command: CreateItem) -> ResolvedControl:
    """Expand ``conversation.item.create`` the way the old translator did.

    Assistant/system/function items and text-only user items become one
    ``turn.signal conversation.item.create`` payload. User items carrying audio
    become one ``input_audio_buffer.append`` per audio part followed by an
    ``input_audio_buffer.commit`` (no response). The user-item ack events are
    returned in ``events`` and must be emitted *before* running the payloads.
    """
    item = dict(command.item)
    if isinstance(command.previous_item_id, str):
        item["_previous_item_id"] = command.previous_item_id
    item_id = str(item["id"])
    item_type = item.get("type")
    role = item.get("role")
    if item_type == "function_call_output":
        call_id = item.get("call_id")
        output = item.get("output")
        matching_call, duplicate_output = _duplicate_function_call_output(state, call_id)
        if not isinstance(call_id, str) or not call_id or not matching_call:
            return ResolvedControl(
                payloads=[],
                events=[
                    error_event(
                        "invalid_function_call_output",
                        "function_call_output requires the call_id of a completed function call",
                        event_id=command.event_id,
                        param="item.call_id",
                    )
                ],
            )
        if not isinstance(output, str) or duplicate_output:
            return ResolvedControl(
                payloads=[],
                events=[
                    error_event(
                        "invalid_function_call_output",
                        (
                            "function_call_output requires a string output"
                            if not isinstance(output, str)
                            else f"function_call_output already exists for call_id {call_id}"
                        ),
                        event_id=command.event_id,
                        param="item.output",
                    )
                ],
            )
    ack_events = register_user_item(state, item) if item_type == "message" and role == "user" else []
    signal_payload = {
        "type": "turn.signal",
        "event": "conversation.item.create",
        "payload": {"item": item},
    }
    if command.event_id is not None:
        signal_payload["realtime_event_id"] = command.event_id
    if item_type != "message" or role in {"assistant", "system"}:
        return ResolvedControl(payloads=[signal_payload], events=ack_events)
    content = item.get("content")
    if not isinstance(content, list):
        return ResolvedControl(payloads=[], events=ack_events)
    text_chunks: list[str] = []
    audio_payloads: list[dict[str, object]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") in {"input_text", "text"} and isinstance(part.get("text"), str):
            text_chunks.append(str(part["text"]))
        if part.get("type") in {"input_audio", "audio"}:
            audio = part.get("audio") or part.get("data")
            if not isinstance(audio, str) or not audio:
                continue
            try:
                append = build_append_audio(
                    {
                        "audio": audio,
                        "format": part.get("format"),
                        "sample_rate_hz": part.get("sample_rate_hz") or part.get("sample_rate"),
                        "event_id": command.event_id,
                    },
                    defaults=state.defaults,
                    hints_source={**item, **part},
                )
            except Exception as exc:  # DuplexCommandError from validation/conversion
                code = getattr(exc, "code", "bad_event")
                if code == "unsupported_audio_format":
                    continue
                return ResolvedControl(
                    payloads=[],
                    events=[error_event(code, str(exc), event_id=command.event_id, param="sample_rate_hz")],
                )
            speech_hints = {**item, **part}
            if not input_looks_like_speech(
                speech_hints, audio=append.audio, fmt=append.format, overlap_silence_rms=state.overlap_silence_rms
            ):
                continue
            state.input_speech_started = True
            payload = append.payload()
            copy_realtime_input_hints(part, payload)
            copy_realtime_input_hints(item, payload)
            audio_payloads.append(payload)
    if audio_payloads:
        state.input_audio_buffer_has_audio = True
        transcript = input_transcript_from_item(item)
        commit_payload: dict[str, object] = {
            "type": "input_audio_buffer.commit",
            "final": True,
            "realtime_item_id": item_id,
            "response_create": False,
        }
        if transcript:
            commit_payload["transcript"] = transcript
        return ResolvedControl(payloads=[*audio_payloads, commit_payload], events=ack_events)
    if not text_chunks:
        return ResolvedControl(payloads=[], events=ack_events)
    text_item = dict(item)
    text_item["status"] = "completed"
    signal_payload["payload"] = {"item": text_item}
    return ResolvedControl(payloads=[signal_payload], events=ack_events)


def append_audio_payload(command: AppendAudio) -> dict[str, object]:
    """Internal ``input_audio_buffer.append`` payload for a command (convenience for the runner)."""
    return command.payload()


__all__ = [
    "RealtimeProjectionState",
    "ResolvedCommit",
    "ResolvedControl",
    "append_audio_payload",
    "clear_input_buffer",
    "discard_pending_input_audio",
    "emit_input_speech_started",
    "emit_input_speech_stopped",
    "note_input_append",
    "project_internal_event",
    "realtime_session_payload",
    "register_user_item",
    "resolve_cancel_response",
    "resolve_clear_output_audio",
    "resolve_commit",
    "resolve_create_item",
    "resolve_delete_item",
    "resolve_truncate_item",
    "response_is_done",
    "retrieve_item_events",
]
