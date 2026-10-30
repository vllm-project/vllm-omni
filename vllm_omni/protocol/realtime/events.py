# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""OpenAI Realtime server events: typed classes and their wire rendering.

Every event is a frozen dataclass with explicit fields, and the Realtime wire
JSON is *derived* from those fields by :meth:`RealtimeEvent.to_realtime`. That
rendering is pure --- it never consults session state --- which is why it can
live here, away from whoever owns the session. The stateful part (response and
item ids, content-part bookkeeping) is consumed when the events are
*constructed*, by the consumer.

This module holds the 30 events OpenAI's Realtime API defines. vLLM-Omni's own
additions to the wire (listen/speak turn-taking, playback acknowledgement,
session resume) are a separate vocabulary in
``vllm_omni.protocol.duplex.events``, so a consumer that only speaks GA does
not inherit them.

``session_id`` / ``epoch`` are bound by the consumer when an event leaves it;
producers build events with the defaults.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import ClassVar
from uuid import uuid4


def new_event_id() -> str:
    return f"event_{uuid4().hex}"


_IDENTITY_FIELDS = frozenset({"session_id", "epoch", "event_id"})


def wire_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(k): wire_value(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [wire_value(v) for v in value]
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class RealtimeEvent:
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
            data[f.name] = wire_value(value)
        return data

    def to_realtime(self) -> dict[str, object]:
        """The Realtime wire JSON object for this event (derived from the fields)."""
        return {"type": self.wire_type, "event_id": self.event_id, **self._wire_fields()}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(type={self.wire_type!r}, session_id={self.session_id!r}, epoch={self.epoch!r})"


# ---- shared bases ----


@dataclass(frozen=True, slots=True, kw_only=True)
class ResponseScopedEvent(RealtimeEvent):
    """Events addressed to one response (and usually one output item / content part)."""

    response_id: str | None = None  # type: ignore[assignment]
    item_id: str | None = None  # type: ignore[assignment]
    output_index: int = 0
    content_index: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class _ItemEvent(RealtimeEvent):
    previous_item_id: str | None = None
    item: Mapping[str, object] = field(default_factory=dict)

    @property
    def item_id(self) -> str | None:
        value = self.item.get("id")
        return value if isinstance(value, str) else None


# ---- GA server events ----


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionCreated(RealtimeEvent):
    wire_type = "session.created"

    session: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionUpdated(RealtimeEvent):
    wire_type = "session.updated"

    session: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class ResponseCreated(RealtimeEvent):
    wire_type = "response.created"

    response: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class OutputItemAdded(RealtimeEvent):
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
class ContentPartAdded(ResponseScopedEvent):
    wire_type = "response.content_part.added"

    part: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class ContentPartDone(ContentPartAdded):
    wire_type = "response.content_part.done"


@dataclass(frozen=True, slots=True, kw_only=True)
class AudioDelta(ResponseScopedEvent):
    wire_type = "response.output_audio.delta"

    #: Base64 audio in ``format``.
    delta: str = ""

    @property
    def audio(self) -> bytes | None:
        try:
            return base64.b64decode(self.delta)
        except (ValueError, TypeError):
            return None


@dataclass(frozen=True, slots=True, kw_only=True)
class AudioDone(ResponseScopedEvent):
    wire_type = "response.output_audio.done"


@dataclass(frozen=True, slots=True, kw_only=True)
class TranscriptDelta(ResponseScopedEvent):
    wire_type = "response.output_audio_transcript.delta"

    delta: str = ""

    @property
    def text(self) -> str | None:
        return self.delta


@dataclass(frozen=True, slots=True, kw_only=True)
class TranscriptDone(ResponseScopedEvent):
    wire_type = "response.output_audio_transcript.done"

    transcript: str = ""

    @property
    def text(self) -> str | None:
        return self.transcript


@dataclass(frozen=True, slots=True, kw_only=True)
class TextDelta(ResponseScopedEvent):
    wire_type = "response.output_text.delta"

    delta: str = ""

    @property
    def text(self) -> str | None:
        return self.delta


@dataclass(frozen=True, slots=True, kw_only=True)
class TextDone(ResponseScopedEvent):
    wire_type = "response.output_text.done"

    text: str = ""  # type: ignore[assignment]


@dataclass(frozen=True, slots=True, kw_only=True)
class FunctionCallArgumentsDelta(RealtimeEvent):
    wire_type = "response.function_call_arguments.delta"

    response_id: str | None = None  # type: ignore[assignment]
    item_id: str | None = None  # type: ignore[assignment]
    output_index: int = 0
    call_id: str = ""
    delta: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class FunctionCallArgumentsDone(RealtimeEvent):
    wire_type = "response.function_call_arguments.done"

    response_id: str | None = None  # type: ignore[assignment]
    item_id: str | None = None  # type: ignore[assignment]
    output_index: int = 0
    call_id: str = ""
    arguments: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class ResponseDone(RealtimeEvent):
    wire_type = "response.done"

    response: Mapping[str, object] = field(default_factory=dict)

    @property
    def status(self) -> str | None:
        value = self.response.get("status")
        return value if isinstance(value, str) else None


@dataclass(frozen=True, slots=True, kw_only=True)
class RateLimitsUpdated(RealtimeEvent):
    wire_type = "rate_limits.updated"

    rate_limits: tuple[Mapping[str, object], ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class InputCommitted(RealtimeEvent):
    wire_type = "input_audio_buffer.committed"

    previous_item_id: str | None = None
    item_id: str | None = None  # type: ignore[assignment]

    def _wire_fields(self) -> dict[str, object]:
        return {"previous_item_id": self.previous_item_id, "item_id": self.item_id}


@dataclass(frozen=True, slots=True, kw_only=True)
class InputCleared(RealtimeEvent):
    wire_type = "input_audio_buffer.cleared"


@dataclass(frozen=True, slots=True, kw_only=True)
class SpeechStarted(RealtimeEvent):
    wire_type = "input_audio_buffer.speech_started"

    audio_start_ms: int = 0
    item_id: str | None = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True, kw_only=True)
class SpeechStopped(RealtimeEvent):
    wire_type = "input_audio_buffer.speech_stopped"

    audio_end_ms: int = 0
    item_id: str | None = None  # type: ignore[assignment]


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
class ItemRetrieved(RealtimeEvent):
    wire_type = "conversation.item.retrieved"

    item: Mapping[str, object] = field(default_factory=dict)

    @property
    def item_id(self) -> str | None:
        value = self.item.get("id")
        return value if isinstance(value, str) else None


@dataclass(frozen=True, slots=True, kw_only=True)
class ItemDeleted(RealtimeEvent):
    wire_type = "conversation.item.deleted"

    item_id: str | None = None  # type: ignore[assignment]

    def _wire_fields(self) -> dict[str, object]:
        return {"item_id": self.item_id}


@dataclass(frozen=True, slots=True, kw_only=True)
class ItemTruncated(RealtimeEvent):
    wire_type = "conversation.item.truncated"

    item_id: str | None = None  # type: ignore[assignment]
    content_index: int = 0
    audio_end_ms: int = 0

    def _wire_fields(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "content_index": self.content_index,
            "audio_end_ms": self.audio_end_ms,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class InputTranscriptionCompleted(RealtimeEvent):
    wire_type = "conversation.item.input_audio_transcription.completed"

    item_id: str | None = None  # type: ignore[assignment]
    content_index: int = 0
    transcript: str = ""

    @property
    def text(self) -> str | None:
        return self.transcript


@dataclass(frozen=True, slots=True, kw_only=True)
class OutputAudioCleared(RealtimeEvent):
    wire_type = "output_audio_buffer.cleared"

    response_id: str | None = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True, kw_only=True)
class ErrorEvent(RealtimeEvent):
    wire_type = "error"

    code: str = "internal_error"
    message: str = ""
    #: Client ``event_id`` this error answers (OpenAI ``error.event_id``).
    related_event_id: str | None = None
    param: str | None = None

    @property
    def error_type(self) -> str:
        """OpenAI's error class for this error.

        OpenAI standardises only the three classes, not the codes, so the base
        reports the default. A consumer with its own code vocabulary overrides
        this --- see ``vllm_omni.protocol.duplex.events.ErrorEvent``.
        """
        return "invalid_request_error"

    @property
    def error(self) -> dict[str, object]:
        error: dict[str, object] = {"type": self.error_type, "code": self.code, "message": self.message}
        if self.related_event_id:
            error["event_id"] = self.related_event_id
        if self.param:
            error["param"] = self.param
        return error

    @property
    def text(self) -> str | None:
        return self.message

    def _wire_fields(self) -> dict[str, object]:
        return {"error": self.error}


__all__ = [
    "AudioDelta",
    "AudioDone",
    "ContentPartAdded",
    "ContentPartDone",
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
    "OutputAudioCleared",
    "OutputItemAdded",
    "OutputItemDone",
    "RateLimitsUpdated",
    "RealtimeEvent",
    "ResponseCreated",
    "ResponseDone",
    "ResponseScopedEvent",
    "SessionCreated",
    "SessionUpdated",
    "SpeechStarted",
    "SpeechStopped",
    "TextDelta",
    "TextDone",
    "TranscriptDelta",
    "TranscriptDone",
    "new_event_id",
    "wire_value",
]
