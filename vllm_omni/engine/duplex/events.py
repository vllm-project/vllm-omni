# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Typed events emitted by a duplex session.

Every public event is a frozen dataclass with explicit fields. The Realtime
wire JSON is *derived* from those fields by :meth:`DuplexEvent.to_realtime`,
which is pure: it never consults session state. The stateful part (response /
item ids, content-part bookkeeping) lives in
``vllm_omni.engine.duplex.realtime_events`` on the session runner and is
consumed when the events are *constructed*.

``session_id`` / ``epoch`` are bound by
``DuplexSessionManager.emit`` when an event leaves the runner; producers
(projector, serving handshake helpers) build events with the defaults.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import ClassVar, cast
from uuid import uuid4


def new_event_id() -> str:
    return f"event_{uuid4().hex}"


#: OpenAI Realtime ``error.type`` for each internal error code.
REALTIME_ERROR_TYPES_BY_CODE: dict[str, str] = {
    "bad_event": "invalid_request_error",
    "bad_audio": "invalid_request_error",
    "config_timeout": "invalid_request_error",
    "invalid_json": "invalid_request_error",
    "event_too_large": "invalid_request_error",
    "unknown_event": "invalid_request_error",
    "internal_error": "server_error",
    "runtime_append_failed": "server_error",
    "runtime_append_task_failed": "server_error",
    "runtime_signal_failed": "server_error",
    "runtime_abort_failed": "server_error",
    "runtime_data_plane_stream_failed": "server_error",
    "runtime_data_plane_text_without_audio": "server_error",
    "resource_exhausted": "rate_limit_error",
    "session_exists": "invalid_request_error",
    "session_closed": "invalid_request_error",
    "unknown_session": "invalid_request_error",
    "invalid_duplex_runtime_config": "invalid_request_error",
    "instructions_update_unsupported": "invalid_request_error",
    "persona_update_unsupported": "invalid_request_error",
    "voice_update_unsupported": "invalid_request_error",
    "unsupported_nemotron_duplex_mode": "invalid_request_error",
    "unsupported_native_response_options": "invalid_request_error",
    "runtime_touch_failed": "server_error",
    "engine_error": "server_error",
    "input_backpressure": "rate_limit_error",
    "response_already_active": "invalid_request_error",
    "response_not_active": "invalid_request_error",
    "response_create_without_input": "invalid_request_error",
    "input_audio_buffer_empty": "invalid_request_error",
    "missing_item_id": "invalid_request_error",
    "item_not_found": "invalid_request_error",
    "playback_item_mismatch": "invalid_request_error",
    "playback_item_not_found": "invalid_request_error",
    "playback_ack_too_late": "invalid_request_error",
    "unsupported_audio_format": "invalid_request_error",
    "unsupported_turn_detection": "invalid_request_error",
    "unsupported_ref_audio_path": "invalid_request_error",
    "ref_audio_required": "invalid_request_error",
    "model_update_unsupported": "invalid_request_error",
    "voice_update_after_audio_unsupported": "invalid_request_error",
    "ref_audio_update_unsupported": "invalid_request_error",
    "native_text_append_unsupported": "invalid_request_error",
    "invalid_video_frames": "invalid_request_error",
    "invalid_function_call_output": "invalid_request_error",
    "server_vad_unavailable": "server_error",
}

_IDENTITY_FIELDS = frozenset({"session_id", "epoch", "event_id"})


def _wire_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(k): _wire_value(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [_wire_value(v) for v in value]
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class DuplexEvent:
    """Base of every public session event."""

    #: Realtime ``type`` this class renders to.
    wire_type: ClassVar[str] = ""
    #: Fields that are omitted from the wire object when ``None``.
    optional_wire_fields: ClassVar[frozenset[str]] = frozenset()

    session_id: str = ""
    epoch: int | None = None
    #: Server event id (OpenAI ``event_id``); generated at construction.
    event_id: str = field(default_factory=new_event_id)

    # ---- identity ----

    @property
    def type(self) -> str:
        return self.wire_type

    @property
    def is_terminal(self) -> bool:
        return False

    # ---- generic accessors (None when the class has no such field) ----

    @property
    def response_id(self) -> str | None:
        return None

    @property
    def item_id(self) -> str | None:
        return None

    @property
    def text(self) -> str | None:
        return None

    @property
    def audio(self) -> bytes | None:
        return None

    # ---- wire rendering (pure) ----

    def _wire_fields(self) -> dict[str, object]:
        data: dict[str, object] = {}
        for f in fields(self):
            if f.name in _IDENTITY_FIELDS:
                continue
            value = getattr(self, f.name)
            if value is None and f.name in self.optional_wire_fields:
                continue
            data[f.name] = _wire_value(value)
        return data

    def to_realtime(self) -> dict[str, object]:
        """The Realtime wire JSON object for this event (derived from the fields)."""
        return {"type": self.wire_type, "event_id": self.event_id, **self._wire_fields()}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(type={self.wire_type!r}, session_id={self.session_id!r}, epoch={self.epoch!r})"


# ---- session lifecycle ----


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionCreated(DuplexEvent):
    wire_type = "session.created"
    optional_wire_fields = frozenset({"attachment_generation", "resume_token"})

    session: Mapping[str, object] = field(default_factory=dict)
    #: Transport resume credentials (set by the websocket handler when resume is supported).
    attachment_generation: int | None = None
    resume_token: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionUpdated(DuplexEvent):
    wire_type = "session.updated"

    session: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionResumed(SessionCreated):
    wire_type = "session.resumed"

    def _wire_fields(self) -> dict[str, object]:
        return {"session_id": self.session_id, **DuplexEvent._wire_fields(self)}


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionReplaced(DuplexEvent):
    """A newer connection took the session over (sent to the replaced socket)."""

    wire_type = "session.replaced"

    attachment_generation: int = 0

    def _wire_fields(self) -> dict[str, object]:
        return {"session_id": self.session_id, **DuplexEvent._wire_fields(self)}


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionResyncRequired(DuplexEvent):
    """The replay journal cannot bridge the gap; the client must start over."""

    wire_type = "session.resync_required"

    reason: str = "journal_gap"

    def _wire_fields(self) -> dict[str, object]:
        return {"session_id": self.session_id, **DuplexEvent._wire_fields(self)}


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionClosed(DuplexEvent):
    wire_type = "session.closed"

    reason: str = "closed"
    #: The session-internal close event (kept for clients that inspect it).
    details: Mapping[str, object] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return True

    def _wire_fields(self) -> dict[str, object]:
        return {"session_id": self.session_id, "reason": self.reason, "event": _wire_value(self.details)}


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionExpired(SessionClosed):
    wire_type = "session.expired"

    def _wire_fields(self) -> dict[str, object]:
        return {"session_id": self.session_id, "reason": self.reason}


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionHeartbeatAck(DuplexEvent):
    wire_type = "session.heartbeat_ack"

    def _wire_fields(self) -> dict[str, object]:
        return {"session_id": self.session_id}


@dataclass(frozen=True, slots=True, kw_only=True)
class TurnEvent(DuplexEvent):
    """Turn-state transition (``turn.event``)."""

    wire_type = "turn.event"

    event: str = ""
    turn_state: str = ""

    def _wire_fields(self) -> dict[str, object]:
        return {"session_id": self.session_id, "event": self.event, "turn_state": self.turn_state, "epoch": self.epoch}


# ---- response lifecycle ----


@dataclass(frozen=True, slots=True, kw_only=True)
class _ResponseEvent(DuplexEvent):
    """Events addressed to one response (and usually one output item / content part)."""

    response_id: str | None = None  # type: ignore[assignment]
    item_id: str | None = None  # type: ignore[assignment]
    output_index: int = 0
    content_index: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class ResponseCreated(DuplexEvent):
    wire_type = "response.created"

    response_id: str | None = None  # type: ignore[assignment]
    response: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class Listen(DuplexEvent):
    """The model decided to keep listening (no spoken response for this turn)."""

    wire_type = "response.listen"
    optional_wire_fields = frozenset({"response_id"})

    response_id: str | None = None  # type: ignore[assignment]
    details: Mapping[str, object] = field(default_factory=dict)

    def _wire_fields(self) -> dict[str, object]:
        response: dict[str, object] = {
            "object": "realtime.response",
            "status": "listening",
            "metadata": _wire_value(self.details),
        }
        data: dict[str, object] = {"session_id": self.session_id, "epoch": self.epoch, "response": response}
        if self.response_id:
            response["id"] = self.response_id
            data["response_id"] = self.response_id
        return data


@dataclass(frozen=True, slots=True, kw_only=True)
class Speak(_ResponseEvent):
    wire_type = "response.speak"

    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class OverlapDecision(DuplexEvent):
    wire_type = "overlap.decision"

    policy: str | None = None
    action: str | None = None
    reason: str | None = None
    details: Mapping[str, object] = field(default_factory=dict)

    def _wire_fields(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "epoch": self.epoch,
            "policy": self.policy,
            "action": self.action,
            "reason": self.reason,
            "metadata": _wire_value(self.details),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class OutputItemAdded(DuplexEvent):
    wire_type = "response.output_item.added"

    response_id: str | None = None  # type: ignore[assignment]
    output_index: int = 0
    item: Mapping[str, object] = field(default_factory=dict)

    @property
    def item_id(self) -> str | None:
        value = self.item.get("id")
        return value if isinstance(value, str) else None


@dataclass(frozen=True, slots=True, kw_only=True)
class OutputItemDone(OutputItemAdded):
    wire_type = "response.output_item.done"


@dataclass(frozen=True, slots=True, kw_only=True)
class ContentPartAdded(_ResponseEvent):
    wire_type = "response.content_part.added"

    part: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class ContentPartDone(ContentPartAdded):
    wire_type = "response.content_part.done"


@dataclass(frozen=True, slots=True, kw_only=True)
class AudioDelta(_ResponseEvent):
    wire_type = "response.output_audio.delta"
    optional_wire_fields = frozenset({"sample_rate_hz", "metadata"})

    #: Base64 audio in ``format``.
    delta: str = ""
    format: str = "pcm16"
    sample_rate_hz: int | None = None
    metadata: Mapping[str, object] | None = None

    @property
    def audio(self) -> bytes | None:
        try:
            return base64.b64decode(self.delta)
        except (ValueError, TypeError):
            return None


@dataclass(frozen=True, slots=True, kw_only=True)
class AudioDone(_ResponseEvent):
    wire_type = "response.output_audio.done"


@dataclass(frozen=True, slots=True, kw_only=True)
class TranscriptDelta(_ResponseEvent):
    wire_type = "response.output_audio_transcript.delta"

    delta: str = ""

    @property
    def text(self) -> str | None:
        return self.delta


@dataclass(frozen=True, slots=True, kw_only=True)
class TranscriptDone(_ResponseEvent):
    wire_type = "response.output_audio_transcript.done"

    transcript: str = ""

    @property
    def text(self) -> str | None:
        return self.transcript


@dataclass(frozen=True, slots=True, kw_only=True)
class TextDelta(_ResponseEvent):
    wire_type = "response.output_text.delta"

    delta: str = ""

    @property
    def text(self) -> str | None:
        return self.delta


@dataclass(frozen=True, slots=True, kw_only=True)
class TextDone(_ResponseEvent):
    wire_type = "response.output_text.done"

    text: str = ""  # type: ignore[assignment]


@dataclass(frozen=True, slots=True, kw_only=True)
class FunctionCallArgumentsDelta(DuplexEvent):
    wire_type = "response.function_call_arguments.delta"

    response_id: str | None = None  # type: ignore[assignment]
    item_id: str | None = None  # type: ignore[assignment]
    output_index: int = 0
    call_id: str = ""
    delta: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class FunctionCallArgumentsDone(DuplexEvent):
    wire_type = "response.function_call_arguments.done"

    response_id: str | None = None  # type: ignore[assignment]
    item_id: str | None = None  # type: ignore[assignment]
    output_index: int = 0
    call_id: str = ""
    arguments: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class ResponseDone(DuplexEvent):
    wire_type = "response.done"

    response_id: str | None = None  # type: ignore[assignment]
    response: Mapping[str, object] = field(default_factory=dict)

    @property
    def status(self) -> str | None:
        value = self.response.get("status")
        return value if isinstance(value, str) else None


@dataclass(frozen=True, slots=True, kw_only=True)
class RateLimitsUpdated(DuplexEvent):
    wire_type = "rate_limits.updated"

    rate_limits: tuple[Mapping[str, object], ...] = ()


# ---- input buffer / conversation ----


@dataclass(frozen=True, slots=True, kw_only=True)
class InputCommitted(DuplexEvent):
    wire_type = "input_audio_buffer.committed"

    previous_item_id: str | None = None
    item_id: str | None = None  # type: ignore[assignment]
    #: The session-internal ``input.committed`` event.
    details: Mapping[str, object] = field(default_factory=dict)

    def _wire_fields(self) -> dict[str, object]:
        return {
            "previous_item_id": self.previous_item_id,
            "item_id": self.item_id,
            "event": _wire_value(self.details),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class InputCleared(DuplexEvent):
    wire_type = "input_audio_buffer.cleared"


@dataclass(frozen=True, slots=True, kw_only=True)
class SpeechStarted(DuplexEvent):
    wire_type = "input_audio_buffer.speech_started"

    audio_start_ms: int = 0
    item_id: str | None = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True, kw_only=True)
class SpeechStopped(DuplexEvent):
    wire_type = "input_audio_buffer.speech_stopped"

    audio_end_ms: int = 0
    item_id: str | None = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True, kw_only=True)
class _ItemEvent(DuplexEvent):
    previous_item_id: str | None = None
    item: Mapping[str, object] = field(default_factory=dict)

    @property
    def item_id(self) -> str | None:
        value = self.item.get("id")
        return value if isinstance(value, str) else None


@dataclass(frozen=True, slots=True, kw_only=True)
class ItemAdded(_ItemEvent):
    wire_type = "conversation.item.added"


@dataclass(frozen=True, slots=True, kw_only=True)
class ItemCreated(_ItemEvent):
    wire_type = "conversation.item.created"


@dataclass(frozen=True, slots=True, kw_only=True)
class ItemDone(_ItemEvent):
    wire_type = "conversation.item.done"


@dataclass(frozen=True, slots=True, kw_only=True)
class ItemRetrieved(DuplexEvent):
    wire_type = "conversation.item.retrieved"

    item: Mapping[str, object] = field(default_factory=dict)

    @property
    def item_id(self) -> str | None:
        value = self.item.get("id")
        return value if isinstance(value, str) else None


@dataclass(frozen=True, slots=True, kw_only=True)
class ItemDeleted(DuplexEvent):
    wire_type = "conversation.item.deleted"

    item_id: str | None = None  # type: ignore[assignment]
    details: Mapping[str, object] = field(default_factory=dict)

    def _wire_fields(self) -> dict[str, object]:
        return {"item_id": self.item_id, "event": _wire_value(self.details)}


@dataclass(frozen=True, slots=True, kw_only=True)
class ItemTruncated(DuplexEvent):
    wire_type = "conversation.item.truncated"

    item_id: str | None = None  # type: ignore[assignment]
    content_index: int = 0
    audio_end_ms: int = 0
    details: Mapping[str, object] = field(default_factory=dict)

    def _wire_fields(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "content_index": self.content_index,
            "audio_end_ms": self.audio_end_ms,
            "event": _wire_value(self.details),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class InputTranscriptionCompleted(DuplexEvent):
    wire_type = "conversation.item.input_audio_transcription.completed"

    item_id: str | None = None  # type: ignore[assignment]
    content_index: int = 0
    transcript: str = ""

    @property
    def text(self) -> str | None:
        return self.transcript


# ---- playback / control ----


@dataclass(frozen=True, slots=True, kw_only=True)
class OutputAudioCleared(DuplexEvent):
    wire_type = "output_audio_buffer.cleared"

    response_id: str | None = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True, kw_only=True)
class PlaybackAcknowledged(DuplexEvent):
    wire_type = "playback.acknowledged"

    #: The session-internal ``playback.acknowledged`` event (cursor, committed_ms ...).
    details: Mapping[str, object] = field(default_factory=dict)

    def _wire_fields(self) -> dict[str, object]:
        return {"event": _wire_value(self.details)}


@dataclass(frozen=True, slots=True, kw_only=True)
class ErrorEvent(DuplexEvent):
    wire_type = "error"

    code: str = "internal_error"
    message: str = ""
    #: Client ``event_id`` this error answers (OpenAI ``error.event_id``).
    related_event_id: str | None = None
    param: str | None = None
    #: Extra keys merged into the wire ``error`` object (``retryable`` ...).
    extra: Mapping[str, object] = field(default_factory=dict)

    @property
    def error_type(self) -> str:
        return REALTIME_ERROR_TYPES_BY_CODE.get(self.code, "invalid_request_error")

    @property
    def error(self) -> dict[str, object]:
        error: dict[str, object] = {"type": self.error_type, "code": self.code, "message": self.message}
        if self.related_event_id:
            error["event_id"] = self.related_event_id
        if self.param:
            error["param"] = self.param
        error.update(cast("Mapping[str, object]", _wire_value(self.extra)))
        return error

    @property
    def text(self) -> str | None:
        return self.message

    def _wire_fields(self) -> dict[str, object]:
        return {"error": self.error}


@dataclass(frozen=True, slots=True, kw_only=True)
class DuplexRawEvent(DuplexEvent):
    """A session-internal event with no dedicated Realtime type (``duplex.<internal type>``)."""

    wire_type = "duplex.raw"

    internal_type: str = ""
    details: Mapping[str, object] = field(default_factory=dict)

    @property
    def type(self) -> str:
        return f"duplex.{self.internal_type}"

    def to_realtime(self) -> dict[str, object]:
        return {"type": self.type, "event_id": self.event_id, "event": _wire_value(self.details)}


def error_event(
    code: str,
    message: str,
    *,
    event_id: object | None = None,
    param: object | None = None,
    extra: Mapping[str, object] | None = None,
) -> ErrorEvent:
    """Build the error event for an internal code; ``event_id`` is the *client* event id."""
    return ErrorEvent(
        code=code,
        message=message,
        related_event_id=event_id if isinstance(event_id, str) and event_id else None,
        param=param if isinstance(param, str) and param else None,
        extra=dict(extra or {}),
    )


#: Internal events that terminate a response/session and must never be dropped as stale.
DOMAIN_TERMINAL_EVENTS = frozenset(
    {
        "response.done",
        "response.listen",
        "audio.cancelled",
        "input.cancelled",
        "session.closed",
    }
)

#: Internal streaming model-output events subject to epoch/close stale filtering.
MODEL_OUTPUT_EVENTS = frozenset(
    {
        "response.created",
        "response.listen",
        "response.speak",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_audio.delta",
        "response.output_audio.delta",
        "response.output_audio.done",
        "response.output_audio.done",
        "response.output_text.delta",
        "response.output_text.done",
        "response.text.delta",
        "response.text.done",
        "response.message",
        "response.output_item.done",
        "response.content_part.done",
        "response.done",
    }
)


__all__ = [
    "DOMAIN_TERMINAL_EVENTS",
    "MODEL_OUTPUT_EVENTS",
    "REALTIME_ERROR_TYPES_BY_CODE",
    "AudioDelta",
    "AudioDone",
    "ContentPartAdded",
    "ContentPartDone",
    "DuplexEvent",
    "DuplexRawEvent",
    "ErrorEvent",
    "FunctionCallArgumentsDelta",
    "FunctionCallArgumentsDone",
    "InputCleared",
    "InputCommitted",
    "InputTranscriptionCompleted",
    "ItemAdded",
    "ItemCreated",
    "ItemDeleted",
    "ItemDone",
    "ItemRetrieved",
    "ItemTruncated",
    "Listen",
    "OutputAudioCleared",
    "OutputItemAdded",
    "OutputItemDone",
    "OverlapDecision",
    "PlaybackAcknowledged",
    "RateLimitsUpdated",
    "ResponseCreated",
    "ResponseDone",
    "SessionClosed",
    "SessionCreated",
    "SessionExpired",
    "SessionHeartbeatAck",
    "SessionReplaced",
    "SessionResumed",
    "SessionResyncRequired",
    "SessionUpdated",
    "Speak",
    "SpeechStarted",
    "SpeechStopped",
    "TextDelta",
    "TextDone",
    "TranscriptDelta",
    "TranscriptDone",
    "TurnEvent",
    "error_event",
    "new_event_id",
]
