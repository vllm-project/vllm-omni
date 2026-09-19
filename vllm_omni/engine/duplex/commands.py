# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The command vocabulary one duplex session accepts, plus its mailbox rendering.

``DuplexSessionHandle.submit()`` takes one of these; the websocket handler and
``InlineDuplexClient`` build them with :func:`command_from_realtime`.

The *wire* half of each command --- which client event it decodes from and what
fields survive decoding --- now lives in ``vllm_omni.protocol.duplex.commands``,
which carries the whole vocabulary: the eight Tier 1 classes re-exported from
``vllm_omni.protocol.realtime.commands``, the two Tier 2 ones and the seven
Tier 3 ones. The engine imports that module and never the Tier 1 one directly,
so a command that later grows a duplex extension changes one file
(RFC #6592 P0a).

The *engine* half stays here, because it is not wire contract at all:
``payload()`` renders the session-internal mailbox dictionary the runner bodies
were written against, and its ``type`` is the mailbox channel rather than the
client event. Those two genuinely differ --- ``session.update`` and
``conversation.item.create`` / ``.delete`` / ``.truncate`` all travel on the
``turn.signal`` channel --- which is why the halves are separated instead of
the classes being relocated wholesale.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, ClassVar

from vllm_omni.protocol.duplex import RealtimeProtocolError
from vllm_omni.protocol.duplex import commands as _duplex_wire
from vllm_omni.protocol.duplex.commands import RealtimeCommand

if TYPE_CHECKING:
    from vllm_omni.protocol.duplex import RealtimeInputDefaults


class DuplexCommandError(RealtimeProtocolError):
    """A client payload could not be turned into a duplex command.

    The duplex name for a Realtime protocol error: same ``code`` /
    ``event_id`` contract, so the error envelope is rendered identically
    whichever consumer raised it.
    """


@dataclass(frozen=True, slots=True, kw_only=True)
class DuplexCommand(RealtimeCommand):
    """A Realtime command as the duplex engine handles it.

    Adds the mailbox channel (``type``) and its rendering on top of the wire
    command; every concrete class below pairs this with its protocol twin.
    """

    #: Mailbox event type this command renders to (see ``payload()``).
    type: ClassVar[str] = ""

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


# ---- GA commands ----


@dataclass(frozen=True, slots=True, kw_only=True)
class UpdateSession(DuplexCommand, _duplex_wire.UpdateSession):
    type: ClassVar[str] = "turn.signal"

    def payload(self) -> dict[str, object]:
        data = DuplexCommand.payload(self)
        data["event"] = "session.update"
        data["payload"] = dict(data.pop("patch", {}) or {})
        return data


@dataclass(frozen=True, slots=True, kw_only=True)
class AppendAudio(DuplexCommand, _duplex_wire.AppendAudio):
    type: ClassVar[str] = "input_audio_buffer.append"

    def payload(self) -> dict[str, object]:
        data = DuplexCommand.payload(self)
        hints = data.pop("hints", None)
        if isinstance(hints, Mapping):
            # Hints are raw wire values; the typed fields went through
            # ``build_append_audio``'s normalization, so they have to win. An
            # unset typed field is simply absent here (``payload`` skips None),
            # which is what lets a hint still carry it. Merging the other way
            # round let a client's ``"is_speech": 0`` override the computed
            # ``bool | None`` and silently miss the runner's silent-commit path.
            merged: dict[str, object] = dict(hints)
            merged.update(data)
            data = merged
        data["audio"] = base64.b64encode(self.audio).decode("ascii")
        if not data.get("video_frames"):
            data.pop("video_frames", None)
        return data


@dataclass(frozen=True, slots=True, kw_only=True)
class Commit(DuplexCommand, _duplex_wire.Commit):
    type: ClassVar[str] = "input_audio_buffer.commit"

    def payload(self) -> dict[str, object]:
        data = DuplexCommand.payload(self)
        if self.create_response is not None:
            data["response_create"] = data.pop("create_response")
        return data


@dataclass(frozen=True, slots=True, kw_only=True)
class ClearInput(DuplexCommand, _duplex_wire.ClearInput):
    type: ClassVar[str] = "input_audio_buffer.clear"


@dataclass(frozen=True, slots=True, kw_only=True)
class ClearOutputAudio(DuplexCommand, _duplex_wire.ClearOutputAudio):
    type: ClassVar[str] = "output_audio_buffer.clear"


@dataclass(frozen=True, slots=True, kw_only=True)
class CreateResponse(DuplexCommand, _duplex_wire.CreateResponse):
    type: ClassVar[str] = "response.create"

    def payload(self) -> dict[str, object]:
        data = DuplexCommand.payload(self)
        options = data.pop("options", None)
        if isinstance(options, Mapping):
            data["response"] = dict(options)
        return data


@dataclass(frozen=True, slots=True, kw_only=True)
class CancelResponse(DuplexCommand, _duplex_wire.CancelResponse):
    type: ClassVar[str] = "response.cancel"


@dataclass(frozen=True, slots=True, kw_only=True)
class CreateItem(DuplexCommand, _duplex_wire.CreateItem):
    type: ClassVar[str] = "turn.signal"

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
class DeleteItem(DuplexCommand, _duplex_wire.DeleteItem):
    type: ClassVar[str] = "turn.signal"

    def payload(self) -> dict[str, object]:
        data = DuplexCommand.payload(self)
        data["event"] = "conversation.item.delete"
        data["payload"] = {"item_id": data.pop("item_id")}
        return data


@dataclass(frozen=True, slots=True, kw_only=True)
class TruncateItem(DuplexCommand, _duplex_wire.TruncateItem):
    type: ClassVar[str] = "turn.signal"

    def payload(self) -> dict[str, object]:
        data = DuplexCommand.payload(self)
        data["event"] = "conversation.item.truncate"
        data["payload"] = {
            "item_id": data.pop("item_id"),
            "audio_end_ms": data.pop("audio_end_ms"),
            "content_index": data.pop("content_index", 0),
        }
        return data


# ---- duplex extension commands ----


@dataclass(frozen=True, slots=True, kw_only=True)
class AppendText(DuplexCommand, _duplex_wire.AppendText):
    type: ClassVar[str] = "input.text.append"


@dataclass(frozen=True, slots=True, kw_only=True)
class CancelInput(DuplexCommand, _duplex_wire.CancelInput):
    type: ClassVar[str] = "input.cancel"


@dataclass(frozen=True, slots=True, kw_only=True)
class BargeIn(DuplexCommand, _duplex_wire.BargeIn):
    type: ClassVar[str] = "barge_in"


@dataclass(frozen=True, slots=True, kw_only=True)
class SignalTurn(DuplexCommand, _duplex_wire.SignalTurn):
    type: ClassVar[str] = "turn.signal"

    def payload(self) -> dict[str, object]:
        data = DuplexCommand.payload(self)
        signal_payload = data.pop("signal_payload", None)
        if isinstance(signal_payload, Mapping) and signal_payload:
            data["payload"] = dict(signal_payload)
        return data


@dataclass(frozen=True, slots=True, kw_only=True)
class AckPlayback(DuplexCommand, _duplex_wire.AckPlayback):
    type: ClassVar[str] = "playback.ack"


@dataclass(frozen=True, slots=True, kw_only=True)
class Heartbeat(DuplexCommand, _duplex_wire.Heartbeat):
    type: ClassVar[str] = "session.heartbeat"


@dataclass(frozen=True, slots=True, kw_only=True)
class CloseSession(DuplexCommand, _duplex_wire.CloseSession):
    type: ClassVar[str] = "session.close"


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
        "conversation.item.retrieve",
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
    "REALTIME_COMMAND_TYPES",
    "RealtimeCommand",
    "SignalTurn",
    "TruncateItem",
    "UpdateSession",
    "command_from_realtime",
]
