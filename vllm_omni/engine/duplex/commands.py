# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Typed commands accepted by a duplex session.

``DuplexSessionHandle.submit()`` takes one of these; the websocket handler and
``InlineDuplexClient`` build them with :func:`command_from_realtime` from the
OpenAI Realtime client event vocabulary. Every command can also render the
session-internal mailbox payload (``payload()``), which is the dictionary
vocabulary the session runner bodies were written against.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from vllm_omni.engine.duplex.realtime_commands import RealtimeInputDefaults


class DuplexCommandError(ValueError):
    """A client payload could not be turned into a command."""

    def __init__(self, message: str, *, code: str = "bad_event", event_id: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.event_id = event_id


@dataclass(frozen=True, slots=True, kw_only=True)
class DuplexCommand:
    #: Mailbox event type this command renders to (see ``payload()``).
    type: ClassVar[str] = ""
    #: Client correlation id (OpenAI ``event_id``), echoed on error events.
    event_id: str | None = None

    def payload(self) -> dict[str, object]:
        """Render the session-internal mailbox dictionary."""
        data: dict[str, object] = {"type": self.type}
        for f in fields(self):
            if f.name == "event_id":
                if self.event_id is not None:
                    data["realtime_event_id"] = self.event_id
                continue
            value = getattr(self, f.name)
            if value is None:
                continue
            if isinstance(value, tuple):
                value = list(value)
            elif isinstance(value, Mapping):
                value = dict(value)
            data[f.name] = value
        return data


@dataclass(frozen=True, slots=True, kw_only=True)
class AppendAudio(DuplexCommand):
    type: ClassVar[str] = "input_audio_buffer.append"
    #: Raw audio bytes in ``format`` at ``sample_rate_hz`` (base64 only on the wire).
    audio: bytes
    format: str = "pcm16"
    sample_rate_hz: int | None = None
    is_speech: bool | None = None
    video_frames: tuple[str, ...] = ()
    duration_ms: int | None = None
    audio_end_ms: int | None = None
    #: Model-neutral hints carried through from the wire (rms, vad, transcript hints ...).
    hints: Mapping[str, object] = field(default_factory=dict)

    def payload(self) -> dict[str, object]:
        data = DuplexCommand.payload(self)
        data["audio"] = base64.b64encode(self.audio).decode("ascii")
        hints = data.pop("hints", None)
        if isinstance(hints, Mapping):
            data.update(hints)
        if not data.get("video_frames"):
            data.pop("video_frames", None)
        return data


@dataclass(frozen=True, slots=True, kw_only=True)
class AppendText(DuplexCommand):
    type: ClassVar[str] = "input.text.append"
    text: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Commit(DuplexCommand):
    type: ClassVar[str] = "input_audio_buffer.commit"
    final: bool = True
    #: ``None`` means "no explicit request": the runner decides on commit (auto-response
    #: sessions answer on their own); ``True`` / ``False`` force it.
    create_response: bool | None = None
    is_speech: bool | None = None
    #: Realtime conversation item created for this commit (wire correlation only).
    realtime_item_id: str | None = None

    def payload(self) -> dict[str, object]:
        data = DuplexCommand.payload(self)
        if self.create_response is not None:
            data["response_create"] = data.pop("create_response")
        return data


@dataclass(frozen=True, slots=True, kw_only=True)
class CreateResponse(DuplexCommand):
    type: ClassVar[str] = "response.create"
    #: Raw Realtime ``response`` object (instructions, voice, modalities, ...).
    options: Mapping[str, object] = field(default_factory=dict)

    def payload(self) -> dict[str, object]:
        data = DuplexCommand.payload(self)
        options = data.pop("options", None)
        if isinstance(options, Mapping):
            data["response"] = dict(options)
        return data


@dataclass(frozen=True, slots=True, kw_only=True)
class ClearInput(DuplexCommand):
    type: ClassVar[str] = "input_audio_buffer.clear"


@dataclass(frozen=True, slots=True, kw_only=True)
class CancelInput(DuplexCommand):
    type: ClassVar[str] = "input.cancel"


@dataclass(frozen=True, slots=True, kw_only=True)
class CancelResponse(DuplexCommand):
    type: ClassVar[str] = "response.cancel"
    response_id: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class BargeIn(DuplexCommand):
    type: ClassVar[str] = "barge_in"


@dataclass(frozen=True, slots=True, kw_only=True)
class ClearOutputAudio(DuplexCommand):
    type: ClassVar[str] = "output_audio_buffer.clear"
    #: Explicit response to clear; ``None`` targets the active/last response.
    response_id: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class SignalTurn(DuplexCommand):
    """Generic ``turn.signal`` (local turn transitions such as ``user_started``)."""

    type: ClassVar[str] = "turn.signal"
    event: str
    signal_payload: Mapping[str, object] = field(default_factory=dict)

    def payload(self) -> dict[str, object]:
        data = DuplexCommand.payload(self)
        signal_payload = data.pop("signal_payload", None)
        if isinstance(signal_payload, Mapping) and signal_payload:
            data["payload"] = dict(signal_payload)
        return data


@dataclass(frozen=True, slots=True, kw_only=True)
class UpdateSession(DuplexCommand):
    """``session.update``: a Realtime ``session`` object patch."""

    type: ClassVar[str] = "turn.signal"
    patch: Mapping[str, object] = field(default_factory=dict)

    def payload(self) -> dict[str, object]:
        data = DuplexCommand.payload(self)
        data["event"] = "session.update"
        data["payload"] = dict(data.pop("patch", {}) or {})
        return data


@dataclass(frozen=True, slots=True, kw_only=True)
class AckPlayback(DuplexCommand):
    type: ClassVar[str] = "playback.ack"
    played_ms: int
    committed_ms: int | None = None
    response_id: str | None = None
    item_id: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Heartbeat(DuplexCommand):
    type: ClassVar[str] = "session.heartbeat"


@dataclass(frozen=True, slots=True, kw_only=True)
class CreateItem(DuplexCommand):
    """``conversation.item.create`` (history injection or function-call output)."""

    type: ClassVar[str] = "turn.signal"
    item: Mapping[str, object]
    previous_item_id: str | None = None

    def payload(self) -> dict[str, object]:
        data = DuplexCommand.payload(self)
        data["event"] = "conversation.item.create"
        payload: dict[str, object] = {"item": dict(data.pop("item"))}
        previous = data.pop("previous_item_id", None)
        if previous is not None:
            payload["previous_item_id"] = previous
        data["payload"] = payload
        return data


@dataclass(frozen=True, slots=True, kw_only=True)
class DeleteItem(DuplexCommand):
    type: ClassVar[str] = "turn.signal"
    item_id: str

    def payload(self) -> dict[str, object]:
        data = DuplexCommand.payload(self)
        data["event"] = "conversation.item.delete"
        data["payload"] = {"item_id": data.pop("item_id")}
        return data


@dataclass(frozen=True, slots=True, kw_only=True)
class TruncateItem(DuplexCommand):
    type: ClassVar[str] = "turn.signal"
    item_id: str
    audio_end_ms: int
    content_index: int = 0

    def payload(self) -> dict[str, object]:
        data = DuplexCommand.payload(self)
        data["event"] = "conversation.item.truncate"
        data["payload"] = {
            "item_id": data.pop("item_id"),
            "audio_end_ms": data.pop("audio_end_ms"),
            "content_index": data.pop("content_index", 0),
        }
        return data


@dataclass(frozen=True, slots=True, kw_only=True)
class CloseSession(DuplexCommand):
    """Graceful close requested through the command stream (``session.close``)."""

    type: ClassVar[str] = "session.close"
    reason: str = "client_close"


#: Realtime client event types that map onto commands (wire vocabulary).
REALTIME_COMMAND_TYPES: frozenset[str] = frozenset(
    {
        "input_audio_buffer.append",
        "input_audio_buffer.commit",
        "input_audio_buffer.clear",
        "output_audio_buffer.clear",
        "response.create",
        "response.cancel",
        "conversation.item.create",
        "conversation.item.delete",
        "conversation.item.truncate",
        "session.update",
        "playback.ack",
        "session.heartbeat",
        "session.close",
        "turn.signal",
        "input.text.append",
        "input.cancel",
        "barge_in",
    }
)


def command_from_realtime(
    payload: Mapping[str, object],
    *,
    defaults: RealtimeInputDefaults | None = None,
) -> DuplexCommand:
    """Translate one OpenAI Realtime client event into a command.

    Envelope concerns (``event_id`` acknowledgements, resume bookkeeping) are
    handled by the transport; this function only validates and maps the
    payload shape. ``defaults`` carries the audio format / sample rate / VAD
    defaults the session declared, so every transport decodes an append the
    same way. Raises :class:`DuplexCommandError` for malformed or unsupported
    payloads.
    """
    from vllm_omni.engine.duplex.realtime_commands import translate_realtime_command

    return translate_realtime_command(payload, defaults=defaults)


__all__ = [
    "REALTIME_COMMAND_TYPES",
    "AckPlayback",
    "AppendAudio",
    "AppendText",
    "BargeIn",
    "CancelInput",
    "CancelResponse",
    "ClearInput",
    "ClearOutputAudio",
    "CloseSession",
    "Commit",
    "CreateItem",
    "CreateResponse",
    "DeleteItem",
    "DuplexCommand",
    "DuplexCommandError",
    "Heartbeat",
    "SignalTurn",
    "TruncateItem",
    "UpdateSession",
    "command_from_realtime",
]
