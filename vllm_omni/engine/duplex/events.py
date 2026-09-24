# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The event vocabulary one duplex session speaks, assembled from two layers.

The classes themselves no longer live here. ``vllm_omni.protocol.duplex.events``
carries the whole vocabulary --- 21 Tier 1 classes re-exported from
``vllm_omni.protocol.realtime.events``, 9 Tier 2 ones and 12 Tier 3 ones --- so
a non-duplex Realtime consumer can take the Tier 1 half and its wire rendering
without the duplex session control plane (RFC #6592 P0a). The engine imports
the duplex module and never the Tier 1 one directly.

What is genuinely engine-side stays: :data:`DOMAIN_TERMINAL_EVENTS` and
:data:`MODEL_OUTPUT_EVENTS` are the session runner's epoch-filter policy, not
wire contract --- they name *internal* event types, and which of them may never
be dropped as stale is a runtime decision.

``DuplexEvent`` is re-exported as the name the engine and the clients already
use. It is an alias of
:class:`~vllm_omni.protocol.realtime.events.RealtimeEvent`, not a subclass,
because an event has no engine-internal half --- what the session emits is
exactly what the socket sends. A *command* does have one (``payload()`` renders
the runner's mailbox), which is why ``DuplexCommand`` is a real class.
"""

from __future__ import annotations

from vllm_omni.protocol.duplex.errors import REALTIME_ERROR_TYPES_BY_CODE
from vllm_omni.protocol.duplex.events import (
    AudioDelta,
    AudioDone,
    ContentPartAdded,
    ContentPartDone,
    DuplexEvent,
    DuplexRawEvent,
    ErrorEvent,
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
    OverlapDecision,
    PlaybackAcknowledged,
    RateLimitsUpdated,
    ResponseCreated,
    ResponseDone,
    SessionClosed,
    SessionCreated,
    SessionExpired,
    SessionHeartbeatAck,
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
    TurnEvent,
    error_event,
    new_event_id,
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

MODEL_OUTPUT_EVENTS = frozenset(
    {
        "response.created",
        "response.listen",
        "response.speak",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_audio.delta",
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
    "AudioDelta",
    "AudioDone",
    "ContentPartAdded",
    "ContentPartDone",
    "DOMAIN_TERMINAL_EVENTS",
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
    "MODEL_OUTPUT_EVENTS",
    "OutputAudioCleared",
    "OutputItemAdded",
    "OutputItemDone",
    "OverlapDecision",
    "PlaybackAcknowledged",
    "REALTIME_ERROR_TYPES_BY_CODE",
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
