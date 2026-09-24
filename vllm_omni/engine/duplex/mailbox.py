# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The duplex engine's binding of the wire protocol: intake and mailbox rendering.

The protocol package (``vllm_omni.protocol.duplex``) decodes a client event
into a typed :data:`~vllm_omni.protocol.duplex.commands.DuplexCommand` and
knows nothing about how a runtime represents one. This module is the
engine-side half of that story:

* :data:`DUPLEX_REALTIME_CAPABILITIES` --- what the duplex engine can serve,
  for the shared session-object validator. It lives here rather than in the
  protocol package because the turn-detection answer comes from the Silero
  VAD backend.
* :func:`command_from_realtime` --- the protocol decoder bound to those
  capabilities. Every transport into the engine (the websocket handler, the
  inline client) decodes through this one function.
* :func:`mailbox_payload` --- renders a typed command as the session-internal
  *mailbox* dictionary the session runner bodies were written against. The
  mailbox channel is not the client event type for four commands
  (``session.update`` and the three ``conversation.item.*`` commands all travel
  on ``turn.signal``), which is why this rendering is engine business and not
  a method on the protocol classes.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import fields
from typing import TypeVar

import pybase64 as base64

from vllm_omni.protocol.duplex import RealtimeInputDefaults, RealtimeProtocolCapabilities, decode_duplex_command
from vllm_omni.protocol.duplex.commands import (
    AckPlayback,
    AppendAudio,
    AppendText,
    BargeIn,
    CancelInput,
    CancelResponse,
    ClearInput,
    ClearOutputAudio,
    CloseSession,
    Commit,
    CreateItem,
    CreateResponse,
    DeleteItem,
    DuplexCommand,
    Heartbeat,
    SignalTurn,
    TruncateItem,
    UpdateSession,
)

__all__ = [
    "DUPLEX_REALTIME_CAPABILITIES",
    "command_from_realtime",
    "mailbox_channel",
    "mailbox_payload",
]


# ---- capabilities and intake ----


def _validate_duplex_turn_detection(session_payload: Mapping[str, object]) -> str | None:
    """The duplex engine's answer for ``turn_detection`` on a session object.

    Imported lazily: the validator lives with the Silero VAD backend, and a
    connection that never sends a session object should not pay for loading it.
    """
    from vllm_omni.engine.duplex.turn_detection import validate_realtime_turn_detection

    return validate_realtime_turn_detection(session_payload)


#: What the duplex engine can serve, for the shared session-object validator.
DUPLEX_REALTIME_CAPABILITIES = RealtimeProtocolCapabilities(
    validate_turn_detection=_validate_duplex_turn_detection,
)


def command_from_realtime(
    payload: Mapping[str, object],
    *,
    defaults: RealtimeInputDefaults | None = None,
) -> DuplexCommand:
    """Translate one client event into a command the duplex engine accepts.

    :func:`~vllm_omni.protocol.duplex.commands.decode_duplex_command` bound to
    :data:`DUPLEX_REALTIME_CAPABILITIES`, so a ``session.update`` is accepted or
    refused identically whichever door it came in. Raises
    :class:`~vllm_omni.protocol.duplex.RealtimeProtocolError` for malformed or
    unsupported payloads.
    """
    return decode_duplex_command(payload, defaults=defaults, capabilities=DUPLEX_REALTIME_CAPABILITIES)


# ---- mailbox rendering ----

#: Mailbox channel (the internal ``type``) each command renders to.
_MAILBOX_CHANNELS: dict[type[DuplexCommand], str] = {
    UpdateSession: "turn.signal",
    AppendAudio: "input_audio_buffer.append",
    Commit: "input_audio_buffer.commit",
    ClearInput: "input_audio_buffer.clear",
    ClearOutputAudio: "output_audio_buffer.clear",
    CreateResponse: "response.create",
    CancelResponse: "response.cancel",
    CreateItem: "turn.signal",
    DeleteItem: "turn.signal",
    TruncateItem: "turn.signal",
    AppendText: "input.text.append",
    CancelInput: "input.cancel",
    BargeIn: "barge_in",
    SignalTurn: "turn.signal",
    AckPlayback: "playback.ack",
    Heartbeat: "session.heartbeat",
    CloseSession: "session.close",
}


_T = TypeVar("_T")


def _for_class(table: Mapping[type[DuplexCommand], _T], command: DuplexCommand | type[DuplexCommand]) -> _T | None:
    """The table entry for a command's class or its nearest registered base; ``None`` when unregistered."""
    cls = command if isinstance(command, type) else type(command)
    for base in cls.__mro__:
        if base in table:
            return table[base]
    return None


def mailbox_channel(command: DuplexCommand | type[DuplexCommand]) -> str:
    """The session-internal ``type`` a command (or command class) travels on."""
    channel = _for_class(_MAILBOX_CHANNELS, command)
    if channel is None:
        cls = command if isinstance(command, type) else type(command)
        raise TypeError(f"{cls.__name__} is not a duplex command the engine knows how to render")
    return channel


def _base_payload(command: DuplexCommand) -> dict[str, object]:
    data: dict[str, object] = {"type": mailbox_channel(command)}
    for f in fields(command):
        if f.name == "event_id":
            if command.event_id is not None:
                data["realtime_event_id"] = command.event_id
            continue
        value = getattr(command, f.name)
        if value is None:
            continue
        if isinstance(value, tuple):
            value = list(value)
        elif isinstance(value, Mapping):
            value = dict(value)
        data[f.name] = value
    return data


def _render_update_session(command: UpdateSession, data: dict[str, object]) -> dict[str, object]:
    data["event"] = "session.update"
    data["payload"] = dict(data.pop("patch", {}) or {})
    return data


def _render_append_audio(command: AppendAudio, data: dict[str, object]) -> dict[str, object]:
    hints = data.pop("hints", None)
    if isinstance(hints, Mapping):
        # Hints are raw wire values; the typed fields went through
        # ``build_append_audio``'s normalization, so they have to win. An
        # unset typed field is simply absent here (``_base_payload`` skips
        # None), which is what lets a hint still carry it. Merging the other
        # way round let a client's ``"is_speech": 0`` override the computed
        # ``bool | None`` and silently miss the runner's silent-commit path.
        merged: dict[str, object] = dict(hints)
        merged.update(data)
        data = merged
    if command.audio:
        data["audio"] = base64.b64encode(command.audio).decode("ascii")
    else:
        # Empty audio with video frames is a legal frames-only append (#7633).
        data.pop("audio", None)
    if not data.get("video_frames"):
        data.pop("video_frames", None)
    return data


def _render_commit(command: Commit, data: dict[str, object]) -> dict[str, object]:
    if command.create_response is not None:
        data["response_create"] = data.pop("create_response")
    return data


def _render_create_response(command: CreateResponse, data: dict[str, object]) -> dict[str, object]:
    options = data.pop("options", None)
    if isinstance(options, Mapping):
        data["response"] = dict(options)
    return data


def _render_create_item(command: CreateItem, data: dict[str, object]) -> dict[str, object]:
    data["event"] = "conversation.item.create"
    payload: dict[str, object] = {"item": dict(data.pop("item"))}
    previous = data.pop("previous_item_id", None)
    if previous is not None:
        payload["previous_item_id"] = previous
    data["payload"] = payload
    return data


def _render_delete_item(command: DeleteItem, data: dict[str, object]) -> dict[str, object]:
    data["event"] = "conversation.item.delete"
    data["payload"] = {"item_id": data.pop("item_id")}
    return data


def _render_truncate_item(command: TruncateItem, data: dict[str, object]) -> dict[str, object]:
    data["event"] = "conversation.item.truncate"
    data["payload"] = {
        "item_id": data.pop("item_id"),
        "audio_end_ms": data.pop("audio_end_ms"),
        "content_index": data.pop("content_index", 0),
    }
    return data


def _render_signal_turn(command: SignalTurn, data: dict[str, object]) -> dict[str, object]:
    signal_payload = data.pop("signal_payload", None)
    if isinstance(signal_payload, Mapping) and signal_payload:
        data["payload"] = dict(signal_payload)
    return data


#: Commands whose mailbox dictionary is not just their fields.
_MAILBOX_RENDERERS: dict[type[DuplexCommand], Callable[..., dict[str, object]]] = {
    UpdateSession: _render_update_session,
    AppendAudio: _render_append_audio,
    Commit: _render_commit,
    CreateResponse: _render_create_response,
    CreateItem: _render_create_item,
    DeleteItem: _render_delete_item,
    TruncateItem: _render_truncate_item,
    SignalTurn: _render_signal_turn,
}


def mailbox_payload(command: DuplexCommand) -> dict[str, object]:
    """Render a typed command as the session runner's mailbox dictionary.

    ``event_id`` travels as ``realtime_event_id``; ``None`` fields are omitted;
    tuples and mappings become lists and dicts. The commands in
    :data:`_MAILBOX_RENDERERS` then reshape the result to the channel's own
    layout (``turn.signal`` carries an ``event`` and a ``payload``, an append
    re-encodes its audio, ...).
    """
    data = _base_payload(command)
    renderer = _for_class(_MAILBOX_RENDERERS, command)
    return renderer(command, data) if renderer is not None else data
