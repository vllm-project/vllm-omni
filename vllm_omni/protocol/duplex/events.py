# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""vLLM-Omni's half of the Realtime server-event vocabulary.

``docs/serving/realtime_duplex_api.md`` sorts every message into three tiers.
Tier 1 --- identical to OpenAI --- is ``vllm_omni.protocol.realtime.events``.
The other two are here:

**Tier 2, OpenAI names carrying vLLM-Omni extensions.** Same wire type, same
OpenAI fields, plus additive keys a stock client ignores. Each one subclasses
its Tier 1 twin and declares only what it adds, so the OpenAI surface stays
honest: a consumer that wants pure GA takes ``protocol/realtime`` and gets
exactly OpenAI's fields.

**Tier 3, vLLM-Omni only.** Events OpenAI has no equivalent for, because they
only make sense when the model and the user can talk at the same time: the
model decides when to listen and when to speak, the client reports how much
audio it played, overlapping speech needs a recorded decision, and a dropped
socket can re-attach to a live session.

Same-named classes are deliberate --- ``duplex.events.SessionCreated`` *is*
``realtime.events.SessionCreated`` with our extensions. Import whichever
surface you mean.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

from vllm_omni.protocol.duplex.errors import REALTIME_ERROR_TYPES_BY_CODE
from vllm_omni.protocol.realtime import events as realtime_events
from vllm_omni.protocol.realtime.events import (
    AudioDone,
    ContentPartAdded,
    ContentPartDone,
    FunctionCallArgumentsDelta,
    FunctionCallArgumentsDone,
    InputCleared,
    InputTranscriptionCompleted,
    ItemAdded,
    ItemCreated,
    ItemDone,
    ItemRetrieved,
    OutputAudioCleared,
    OutputItemAdded,
    OutputItemDone,
    RateLimitsUpdated,
    RealtimeEvent,
    ResponseScopedEvent,
    SessionUpdated,
    SpeechStarted,
    SpeechStopped,
    TextDelta,
    TextDone,
    TranscriptDone,
    new_event_id,
    wire_value,
)

#: The duplex event base class *is* the Realtime one: unlike a command, an
#: event has no engine-internal half to carry --- what the session emits is
#: exactly what the socket sends. Named here rather than in the engine so the
#: duplex vocabulary is described in one place, and kept as an alias rather
#: than an empty subclass so ``isinstance`` and the class identity tests in
#: ``tests/protocol/`` stay meaningful.
DuplexEvent = RealtimeEvent

# ---- Tier 2: OpenAI names carrying vLLM-Omni extensions ----


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionCreated(realtime_events.SessionCreated):
    """Tier 2: OpenAI's ``SessionCreated`` plus the resume credentials a reconnecting client needs."""

    #: Transport resume credentials (set by the websocket handler when resume is supported).
    optional_wire_fields = frozenset({"attachment_generation", "resume_token"})
    attachment_generation: int | None = None
    resume_token: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ResponseCreated(realtime_events.ResponseCreated):
    """Tier 2: OpenAI's ``ResponseCreated`` plus a top-level ``response_id``.

    OpenAI carries the id nested as ``response.id``; the duplex lane also
    reports it at the top level, where every other response-scoped event
    already has one.
    """

    response_id: str | None = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True, kw_only=True)
class ResponseDone(realtime_events.ResponseDone):
    """Tier 2: OpenAI's ``ResponseDone`` plus a top-level ``response_id``.

    OpenAI carries the id nested as ``response.id``; the duplex lane also
    reports it at the top level, where every other response-scoped event
    already has one.
    """

    response_id: str | None = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True, kw_only=True)
class AudioDelta(realtime_events.AudioDelta):
    """Tier 2: OpenAI's ``AudioDelta`` plus the chunk's own format/rate and the duplex playback metadata."""

    optional_wire_fields = frozenset({"sample_rate_hz", "metadata"})
    format: str = "pcm16"
    sample_rate_hz: int | None = None
    metadata: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class TranscriptDelta(realtime_events.TranscriptDelta):
    """Tier 2: OpenAI's ``TranscriptDelta`` plus duplex request-start metrics on text-only units."""

    optional_wire_fields = frozenset({"metadata"})
    metadata: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class InputCommitted(realtime_events.InputCommitted):
    """Tier 2: OpenAI's ``InputCommitted`` plus the native ``input.committed`` event echoed under ``event``."""

    #: The session-internal ``input.committed`` event.
    details: Mapping[str, object] = field(default_factory=dict)

    def _wire_fields(self) -> dict[str, object]:
        # Explicit parent call, not ``super()``: ``@dataclass(slots=True)``
        # rebuilds the class, so the zero-argument form's ``__class__`` cell
        # points at the pre-slots class and is no longer in the MRO.
        base = realtime_events.__dict__[type(self).__name__]._wire_fields(self)
        return {**base, "event": wire_value(self.details)}


@dataclass(frozen=True, slots=True, kw_only=True)
class ItemDeleted(realtime_events.ItemDeleted):
    """Tier 2: OpenAI's ``ItemDeleted`` plus the originating event echoed under ``event``."""

    details: Mapping[str, object] = field(default_factory=dict)

    def _wire_fields(self) -> dict[str, object]:
        # Explicit parent call, not ``super()``: ``@dataclass(slots=True)``
        # rebuilds the class, so the zero-argument form's ``__class__`` cell
        # points at the pre-slots class and is no longer in the MRO.
        base = realtime_events.__dict__[type(self).__name__]._wire_fields(self)
        return {**base, "event": wire_value(self.details)}


@dataclass(frozen=True, slots=True, kw_only=True)
class ItemTruncated(realtime_events.ItemTruncated):
    """Tier 2: OpenAI's ``ItemTruncated`` plus the originating event echoed under ``event``."""

    details: Mapping[str, object] = field(default_factory=dict)

    def _wire_fields(self) -> dict[str, object]:
        # Explicit parent call, not ``super()``: ``@dataclass(slots=True)``
        # rebuilds the class, so the zero-argument form's ``__class__`` cell
        # points at the pre-slots class and is no longer in the MRO.
        base = realtime_events.__dict__[type(self).__name__]._wire_fields(self)
        return {**base, "event": wire_value(self.details)}


@dataclass(frozen=True, slots=True, kw_only=True)
class ErrorEvent(realtime_events.ErrorEvent):
    """Tier 2: OpenAI's ``ErrorEvent`` plus vLLM-Omni's error codes and any extra keys merged into the envelope."""

    #: Extra keys merged into the wire ``error`` object (``retryable`` ...).
    extra: Mapping[str, object] = field(default_factory=dict)

    @property
    def error_type(self) -> str:
        return REALTIME_ERROR_TYPES_BY_CODE.get(self.code, "invalid_request_error")

    @property
    def error(self) -> dict[str, object]:
        # Explicit parent call, not ``super()``: ``@dataclass(slots=True)``
        # rebuilds the class, so the zero-argument form's ``__class__`` cell
        # points at the pre-slots class and is no longer in the MRO. Reached
        # through ``__dict__`` because attribute access on the class gives the
        # getter function, not the ``property`` that owns it.
        error = dict(realtime_events.ErrorEvent.__dict__["error"].fget(self))
        error.update(cast("Mapping[str, object]", wire_value(self.extra)))
        return error


# ---- Tier 3: vLLM-Omni only ----


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionResumed(SessionCreated):
    wire_type = "session.resumed"

    def _wire_fields(self) -> dict[str, object]:
        return {"session_id": self.session_id, **RealtimeEvent._wire_fields(self)}


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionReplaced(RealtimeEvent):
    """A newer connection took the session over (sent to the replaced socket)."""

    wire_type = "session.replaced"

    attachment_generation: int = 0

    def _wire_fields(self) -> dict[str, object]:
        return {"session_id": self.session_id, **RealtimeEvent._wire_fields(self)}


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionResyncRequired(RealtimeEvent):
    """The replay journal cannot bridge the gap; the client must start over."""

    wire_type = "session.resync_required"

    reason: str = "journal_gap"

    def _wire_fields(self) -> dict[str, object]:
        return {"session_id": self.session_id, **RealtimeEvent._wire_fields(self)}


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionClosed(RealtimeEvent):
    wire_type = "session.closed"

    reason: str = "closed"
    #: The session-internal close event (kept for clients that inspect it).
    details: Mapping[str, object] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return True

    def _wire_fields(self) -> dict[str, object]:
        return {"session_id": self.session_id, "reason": self.reason, "event": wire_value(self.details)}


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionExpired(SessionClosed):
    wire_type = "session.expired"

    def _wire_fields(self) -> dict[str, object]:
        return {"session_id": self.session_id, "reason": self.reason}


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionHeartbeatAck(RealtimeEvent):
    wire_type = "session.heartbeat_ack"

    def _wire_fields(self) -> dict[str, object]:
        return {"session_id": self.session_id}


@dataclass(frozen=True, slots=True, kw_only=True)
class TurnEvent(RealtimeEvent):
    """Turn-state transition (``turn.event``)."""

    wire_type = "turn.event"

    event: str = ""
    turn_state: str = ""

    def _wire_fields(self) -> dict[str, object]:
        return {"session_id": self.session_id, "event": self.event, "turn_state": self.turn_state, "epoch": self.epoch}


@dataclass(frozen=True, slots=True, kw_only=True)
class Listen(RealtimeEvent):
    """The model decided to keep listening (no spoken response for this turn)."""

    wire_type = "response.listen"
    optional_wire_fields = frozenset({"response_id"})

    response_id: str | None = None  # type: ignore[assignment]
    details: Mapping[str, object] = field(default_factory=dict)

    def _wire_fields(self) -> dict[str, object]:
        response: dict[str, object] = {
            "object": "realtime.response",
            "status": "listening",
            "metadata": wire_value(self.details),
        }
        data: dict[str, object] = {"session_id": self.session_id, "epoch": self.epoch, "response": response}
        if self.response_id:
            response["id"] = self.response_id
            data["response_id"] = self.response_id
        return data


@dataclass(frozen=True, slots=True, kw_only=True)
class Speak(ResponseScopedEvent):
    wire_type = "response.speak"

    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class OverlapDecision(RealtimeEvent):
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
            "metadata": wire_value(self.details),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class PlaybackAcknowledged(RealtimeEvent):
    wire_type = "playback.acknowledged"

    #: The session-internal ``playback.acknowledged`` event (cursor, committed_ms ...).
    details: Mapping[str, object] = field(default_factory=dict)

    def _wire_fields(self) -> dict[str, object]:
        return {"event": wire_value(self.details)}


@dataclass(frozen=True, slots=True, kw_only=True)
class DuplexRawEvent(RealtimeEvent):
    """A session-internal event with no dedicated Realtime type (``duplex.<internal type>``)."""

    wire_type = "duplex.raw"

    internal_type: str = ""
    details: Mapping[str, object] = field(default_factory=dict)

    @property
    def type(self) -> str:
        return f"duplex.{self.internal_type}"

    def to_wire(self) -> dict[str, object]:
        return {"type": self.type, "event_id": self.event_id, "event": wire_value(self.details)}


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


#: The complete event vocabulary a duplex client may receive: 21 Tier 1
#: classes re-exported unchanged, the 9 Tier 2 ones defined above, and 12
#: Tier 3 ones. A duplex consumer imports from here and never reaches past
#: this module into ``vllm_omni.protocol.realtime``.
__all__ = [
    # Tier 1 --- re-exported from ``vllm_omni.protocol.realtime.events``.
    "AudioDone",
    "ContentPartAdded",
    "ContentPartDone",
    "FunctionCallArgumentsDelta",
    "FunctionCallArgumentsDone",
    "InputCleared",
    "InputTranscriptionCompleted",
    "ItemAdded",
    "ItemCreated",
    "ItemDone",
    "ItemRetrieved",
    "OutputAudioCleared",
    "OutputItemAdded",
    "OutputItemDone",
    "RateLimitsUpdated",
    "SessionUpdated",
    "SpeechStarted",
    "SpeechStopped",
    "TextDelta",
    "TextDone",
    "TranscriptDone",
    # Tier 2 --- an OpenAI event plus duplex-only fields.
    "AudioDelta",
    "ErrorEvent",
    "InputCommitted",
    "ItemDeleted",
    "ItemTruncated",
    "ResponseCreated",
    "ResponseDone",
    "SessionCreated",
    "TranscriptDelta",
    # Tier 3 --- vLLM-Omni only.
    "DuplexRawEvent",
    "Listen",
    "OverlapDecision",
    "PlaybackAcknowledged",
    "SessionClosed",
    "SessionExpired",
    "SessionHeartbeatAck",
    "SessionReplaced",
    "SessionResumed",
    "SessionResyncRequired",
    "Speak",
    "TurnEvent",
    # Bases and helpers shared with the Tier 1 surface.
    "DuplexEvent",
    "REALTIME_ERROR_TYPES_BY_CODE",
    "RealtimeEvent",
    "ResponseScopedEvent",
    "error_event",
    "new_event_id",
    "wire_value",
]
