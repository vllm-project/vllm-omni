# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""OpenAI Realtime client events as typed commands.

A Realtime server's first job is turning a client JSON event into something
typed. These are the 10 client events OpenAI's Realtime API defines, as frozen
dataclasses carrying the fields that survive decoding.

vLLM-Omni's own client events (barge-in, playback acknowledgement, text append,
heartbeat, explicit close, generic turn signals) are a separate vocabulary in
``vllm_omni.protocol.duplex.commands``.

Decoding a payload *into* these stays with the consumer, because which command
a given event becomes can depend on what that consumer supports --- see
``vllm_omni.engine.duplex.realtime_commands.translate_realtime_command``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar


@dataclass(frozen=True, slots=True, kw_only=True)
class RealtimeCommand:
    """One decoded OpenAI Realtime client event.

    ``wire_type`` is the event type the client sent; the typed fields are what
    survived decoding, so a consumer reads them instead of re-parsing JSON.

    There is deliberately no rendering method here. A server receives commands,
    it does not emit them, and how a runtime represents one internally is its
    own business --- the duplex engine renders its mailbox dictionary in
    ``vllm_omni.engine.duplex.commands``.
    """

    #: OpenAI Realtime client event this command decodes from.
    wire_type: ClassVar[str] = ""
    #: Client correlation id (OpenAI ``event_id``), echoed on error events.
    event_id: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class UpdateSession(RealtimeCommand):
    """``session.update``: a Realtime ``session`` object patch."""

    wire_type: ClassVar[str] = "session.update"
    patch: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class AppendAudio(RealtimeCommand):
    wire_type: ClassVar[str] = "input_audio_buffer.append"
    #: Raw audio bytes in ``format`` at ``sample_rate_hz`` (base64 only on the wire).
    #: Empty means video-only / frames-only append (validated against duplex capabilities).
    audio: bytes = b""


@dataclass(frozen=True, slots=True, kw_only=True)
class Commit(RealtimeCommand):
    wire_type: ClassVar[str] = "input_audio_buffer.commit"
    final: bool = True


@dataclass(frozen=True, slots=True, kw_only=True)
class ClearInput(RealtimeCommand):
    wire_type: ClassVar[str] = "input_audio_buffer.clear"


@dataclass(frozen=True, slots=True, kw_only=True)
class ClearOutputAudio(RealtimeCommand):
    wire_type: ClassVar[str] = "output_audio_buffer.clear"
    #: Explicit response to clear; ``None`` targets the active/last response.
    response_id: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class CreateResponse(RealtimeCommand):
    wire_type: ClassVar[str] = "response.create"
    #: Raw Realtime ``response`` object (instructions, voice, modalities, ...).
    options: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class CancelResponse(RealtimeCommand):
    wire_type: ClassVar[str] = "response.cancel"
    response_id: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class CreateItem(RealtimeCommand):
    """``conversation.item.create`` (history injection or function-call output)."""

    wire_type: ClassVar[str] = "conversation.item.create"
    item: Mapping[str, object]
    previous_item_id: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class DeleteItem(RealtimeCommand):
    wire_type: ClassVar[str] = "conversation.item.delete"
    item_id: str


@dataclass(frozen=True, slots=True, kw_only=True)
class TruncateItem(RealtimeCommand):
    wire_type: ClassVar[str] = "conversation.item.truncate"
    item_id: str
    audio_end_ms: int
    content_index: int = 0


__all__ = [
    "AppendAudio",
    "CancelResponse",
    "ClearInput",
    "ClearOutputAudio",
    "Commit",
    "CreateItem",
    "CreateResponse",
    "DeleteItem",
    "RealtimeCommand",
    "TruncateItem",
    "UpdateSession",
]
