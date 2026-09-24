# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The client-event vocabulary a full-duplex session accepts, and its decoder.

``docs/serving/realtime_duplex_api.md`` sorts every client event into three
tiers. Tier 1 --- identical to OpenAI --- is
``vllm_omni.protocol.realtime.commands`` and is re-exported here unchanged, so
this module is the *whole* vocabulary and a duplex consumer never imports the
Tier 1 package directly. The other two tiers are declared here:

**Tier 2, OpenAI names carrying vLLM-Omni extensions.** ``AppendAudio`` and
``Commit`` subclass their Tier 1 twin and add only the duplex fields.

**Tier 3, vLLM-Omni only.** Events OpenAI has no equivalent for, because they
only make sense when the model and the user can talk at the same time: the
user cuts in (``barge_in``), the client reports how much audio it played
(``playback.ack``), a turn transition is signalled locally (``turn.signal``),
input is abandoned without committing (``input.cancel``), text is pushed into
a live turn (``input.text.append``), and the session is kept alive or closed
explicitly (``session.heartbeat`` / ``session.close``).

:data:`DuplexCommand` is :class:`~vllm_omni.protocol.realtime.commands.RealtimeCommand`
itself: a duplex command carries nothing a Realtime command does not. How the
duplex engine *represents* one internally (its runner mailbox) is the engine's
own business and lives in ``vllm_omni.engine.duplex.mailbox``.

:func:`decode_duplex_command` is the wire decoder: one client JSON event in,
one typed command out. It is stateless and consumer-neutral; what the consumer
can serve (audio formats, turn detection) is injected through
:class:`~vllm_omni.protocol.realtime.capabilities.RealtimeProtocolCapabilities`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar, cast

from vllm_omni.protocol.realtime import commands as realtime_commands
from vllm_omni.protocol.realtime.audio_input import decode_audio_append
from vllm_omni.protocol.realtime.capabilities import RealtimeProtocolCapabilities, validate_session_payload
from vllm_omni.protocol.realtime.commands import (
    CancelResponse,
    ClearInput,
    ClearOutputAudio,
    CreateItem,
    CreateResponse,
    DeleteItem,
    RealtimeCommand,
    TruncateItem,
    UpdateSession,
)
from vllm_omni.protocol.realtime.errors import RealtimeProtocolError
from vllm_omni.protocol.realtime.formats import (
    validate_conversation_item_audio_formats,
    validate_realtime_response_audio_formats,
)
from vllm_omni.protocol.realtime.items import normalize_conversation_item
from vllm_omni.protocol.realtime.session import RealtimeInputDefaults

#: The duplex command base class *is* the Realtime one. Kept as an alias rather
#: than an empty subclass so ``isinstance`` and the class identity tests in
#: ``tests/protocol/`` stay meaningful.
DuplexCommand = RealtimeCommand

# ---- Tier 2: OpenAI names carrying vLLM-Omni extensions ----


@dataclass(frozen=True, slots=True, kw_only=True)
class AppendAudio(realtime_commands.AppendAudio):
    """Tier 2: OpenAI's ``input_audio_buffer.append`` plus the duplex hints.

    OpenAI's append is just base64 audio. The duplex lane also lets a client
    declare this chunk's own format and rate, whether it believes the chunk is
    speech, camera frames captured alongside it, and its timing --- all
    additive, all ignored by a stock client. Empty ``audio`` with
    ``video_frames`` is legal when session capabilities allow video without
    required audio.
    """

    #: Model-neutral hints carried through from the wire (rms, vad, transcript hints ...).
    format: str = "pcm16"
    sample_rate_hz: int | None = None
    is_speech: bool | None = None
    video_frames: tuple[str, ...] = ()
    duration_ms: int | None = None
    audio_end_ms: int | None = None
    hints: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class Commit(realtime_commands.Commit):
    """Tier 2: OpenAI's ``input_audio_buffer.commit`` plus duplex turn control.

    OpenAI's commit carries nothing and always produces an item. Here a commit
    may be non-final, may decline to start a response, and may declare itself
    silence --- the semantic divergence documented in
    ``docs/serving/realtime_duplex_api.md`` ("Commit != response").
    """

    #: Realtime conversation item created for this commit (wire correlation only).
    #: ``None`` means "no explicit request": the runner decides on commit (auto-response
    #: sessions answer on their own); ``True`` / ``False`` force it.
    create_response: bool | None = None
    is_speech: bool | None = None
    realtime_item_id: str | None = None


# ---- Tier 3: vLLM-Omni only ----


@dataclass(frozen=True, slots=True, kw_only=True)
class AppendText(DuplexCommand):
    wire_type: ClassVar[str] = "input.text.append"
    text: str


@dataclass(frozen=True, slots=True, kw_only=True)
class CancelInput(DuplexCommand):
    wire_type: ClassVar[str] = "input.cancel"


@dataclass(frozen=True, slots=True, kw_only=True)
class BargeIn(DuplexCommand):
    wire_type: ClassVar[str] = "barge_in"


@dataclass(frozen=True, slots=True, kw_only=True)
class SignalTurn(DuplexCommand):
    """Generic ``turn.signal`` (local turn transitions such as ``user_started``)."""

    wire_type: ClassVar[str] = "turn.signal"
    event: str
    signal_payload: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class AckPlayback(DuplexCommand):
    wire_type: ClassVar[str] = "playback.ack"
    played_ms: int
    committed_ms: int | None = None
    response_id: str | None = None
    item_id: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class Heartbeat(DuplexCommand):
    wire_type: ClassVar[str] = "session.heartbeat"


@dataclass(frozen=True, slots=True, kw_only=True)
class CloseSession(DuplexCommand):
    """Graceful close requested through the command stream (``session.close``)."""

    wire_type: ClassVar[str] = "session.close"
    reason: str = "client_close"


# ---- decoding ----


def build_append_audio(
    event: Mapping[str, object],
    *,
    defaults: RealtimeInputDefaults,
    hints_source: Mapping[str, object] | None = None,
) -> AppendAudio:
    """Decode one audio append (shared codec) and pack it as the duplex command.

    Raises :class:`~vllm_omni.protocol.realtime.errors.RealtimeProtocolError`
    for malformed audio, an unsupported format or rate, or invalid video frames.
    """
    decoded = decode_audio_append(event, defaults=defaults, hints_source=hints_source)
    return AppendAudio(
        event_id=decoded.event_id,
        audio=decoded.audio,
        format=decoded.format,
        sample_rate_hz=decoded.sample_rate_hz,
        is_speech=decoded.is_speech,
        video_frames=decoded.video_frames,
        duration_ms=decoded.duration_ms,
        audio_end_ms=decoded.audio_end_ms,
        hints=decoded.hints,
    )


def decode_duplex_command(
    payload: Mapping[str, object],
    *,
    defaults: RealtimeInputDefaults | None = None,
    capabilities: RealtimeProtocolCapabilities | None = None,
) -> DuplexCommand:
    """Map one client event (OpenAI Realtime or duplex extension) onto a typed command.

    Envelope concerns (``event_id`` acknowledgements, resume bookkeeping) are
    handled by the transport; this function only validates and maps the
    payload shape. ``defaults`` carries the audio format / sample rate / VAD
    defaults the session declared, so every transport decodes an append the
    same way. ``capabilities`` is the consumer's answer to what a ``session``
    object may ask for; the permissive default accepts every format the codec
    decodes and does not check ``turn_detection`` at all.

    Everything that needs per-session state (input-buffer emptiness for
    commits, response-id fallbacks for cancels, conversation-item lookups,
    VAD) is the consumer's to resolve; the commands produced here carry the
    raw client intent only.

    Raises :class:`~vllm_omni.protocol.realtime.errors.RealtimeProtocolError`
    for malformed or unsupported payloads. ``session.resume`` and
    ``session.event_ack`` are transport concerns and are rejected with
    ``code="unknown_event"``.
    """
    defaults = defaults or RealtimeInputDefaults()
    capabilities = capabilities or RealtimeProtocolCapabilities()
    event_type = payload.get("type")
    event_id = cast("str", payload.get("event_id")) if isinstance(payload.get("event_id"), str) else None
    if not isinstance(event_type, str):
        raise RealtimeProtocolError("Duplex event missing string type", code="bad_event", event_id=event_id)

    if event_type == "session.update":
        session = payload.get("session")
        session_payload: Mapping[str, object] = session if isinstance(session, dict) else payload
        rejection = validate_session_payload(session_payload, capabilities=capabilities)
        if rejection is not None:
            raise RealtimeProtocolError(rejection.message, code=rejection.code, event_id=event_id)
        return UpdateSession(event_id=event_id, patch=dict(session_payload))

    if event_type == "conversation.item.create":
        item = payload.get("item")
        format_error = validate_conversation_item_audio_formats(item)
        if format_error is not None:
            raise RealtimeProtocolError(format_error, code="unsupported_audio_format", event_id=event_id)
        if not isinstance(item, dict):
            raise RealtimeProtocolError("conversation.item.create requires item", code="bad_event", event_id=event_id)
        previous_item_id = payload.get("previous_item_id")
        return CreateItem(
            event_id=event_id,
            item=normalize_conversation_item(item),
            previous_item_id=previous_item_id if isinstance(previous_item_id, str) else None,
        )

    if event_type == "conversation.item.delete":
        item_id = payload.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise RealtimeProtocolError(
                "conversation.item.delete requires item_id", code="missing_item_id", event_id=event_id
            )
        return DeleteItem(event_id=event_id, item_id=item_id)

    if event_type == "conversation.item.truncate":
        item_id = payload.get("item_id")
        audio_end_ms = payload.get("audio_end_ms")
        content_index = payload.get("content_index", 0)
        if not isinstance(item_id, str) or not item_id:
            raise RealtimeProtocolError(
                "conversation.item.truncate requires item_id", code="missing_item_id", event_id=event_id
            )
        if not isinstance(audio_end_ms, int | float):
            raise RealtimeProtocolError(
                "conversation.item.truncate requires numeric audio_end_ms", code="bad_event", event_id=event_id
            )
        return TruncateItem(
            event_id=event_id,
            item_id=item_id,
            audio_end_ms=int(audio_end_ms),
            content_index=int(content_index) if isinstance(content_index, int | float) else 0,
        )

    if event_type == "input_audio_buffer.append":
        return build_append_audio(dict(payload), defaults=defaults)

    if event_type in {"input_audio_buffer.commit", "input.commit"}:
        final = payload.get("final", True)
        create_response = payload.get("response_create", payload.get("create_response"))
        is_speech = payload.get("is_speech")
        return Commit(
            event_id=event_id,
            final=bool(final) if isinstance(final, bool) else True,
            create_response=(
                bool(create_response)
                if isinstance(create_response, bool)
                else (True if event_type == "input.commit" and create_response is None else None)
            ),
            is_speech=is_speech if isinstance(is_speech, bool) else None,
        )

    if event_type == "input_audio_buffer.clear":
        return ClearInput(event_id=event_id)

    if event_type == "output_audio_buffer.clear":
        response_id = payload.get("response_id")
        return ClearOutputAudio(
            event_id=event_id,
            response_id=response_id if isinstance(response_id, str) and response_id else None,
        )

    if event_type == "response.cancel":
        response_id = payload.get("response_id")
        return CancelResponse(
            event_id=event_id,
            response_id=response_id if isinstance(response_id, str) and response_id else None,
        )

    if event_type == "response.create":
        response_payload = payload.get("response")
        if isinstance(response_payload, dict):
            format_error = validate_realtime_response_audio_formats(response_payload)
            if format_error is not None:
                raise RealtimeProtocolError(format_error, code="unsupported_audio_format", event_id=event_id)
        return CreateResponse(
            event_id=event_id,
            options=dict(response_payload) if isinstance(response_payload, dict) else {},
        )

    if event_type in {"playback.ack", "audio.playback_ack"}:
        played_ms = payload.get("played_ms")
        if not isinstance(played_ms, int | float):
            raise RealtimeProtocolError("playback.ack requires numeric played_ms", code="bad_event", event_id=event_id)
        committed_ms = payload.get("committed_ms")
        response_id = payload.get("response_id")
        item_id = payload.get("item_id")
        return AckPlayback(
            event_id=event_id,
            played_ms=int(played_ms),
            committed_ms=int(committed_ms) if isinstance(committed_ms, int | float) else None,
            response_id=response_id if isinstance(response_id, str) and response_id else None,
            item_id=item_id if isinstance(item_id, str) and item_id else None,
        )

    if event_type == "session.heartbeat":
        return Heartbeat(event_id=event_id)

    if event_type == "conversation.item.retrieve":
        # The projected conversation items live engine-side; the runner answers.
        return SignalTurn(event_id=event_id, event="conversation.item.retrieve", signal_payload=dict(payload))

    if event_type in {"session.close", "close", "close_session"}:
        reason = payload.get("reason")
        return CloseSession(event_id=event_id, reason=reason if isinstance(reason, str) and reason else "client_close")

    if event_type in {"input.text.append", "input_text.append", "push_text"}:
        text = payload.get("text")
        if not isinstance(text, str):
            raise RealtimeProtocolError("input.text.append requires text", code="bad_event", event_id=event_id)
        return AppendText(event_id=event_id, text=text)

    if event_type == "input.cancel":
        return CancelInput(event_id=event_id)

    if event_type == "barge_in":
        return BargeIn(event_id=event_id)

    if event_type in {"turn.signal", "signal_turn"}:
        signal_event = payload.get("event")
        if not isinstance(signal_event, str) or not signal_event:
            raise RealtimeProtocolError("turn.signal requires event", code="bad_event", event_id=event_id)
        signal_payload = payload.get("payload")
        signal_payload = dict(signal_payload) if isinstance(signal_payload, dict) else {}
        if signal_event == "input.cancel":
            return CancelInput(event_id=event_id)
        if signal_event == "barge_in":
            return BargeIn(event_id=event_id)
        if signal_event == "response.cancel":
            response_id = signal_payload.get("response_id", payload.get("response_id"))
            return CancelResponse(
                event_id=event_id,
                response_id=response_id if isinstance(response_id, str) and response_id else None,
            )
        if signal_event == "session.update":
            return UpdateSession(event_id=event_id, patch=signal_payload)
        if signal_event == "conversation.item.create":
            item = signal_payload.get("item")
            if not isinstance(item, dict):
                raise RealtimeProtocolError(
                    "conversation.item.create requires item", code="bad_event", event_id=event_id
                )
            previous_item_id = signal_payload.get("previous_item_id")
            return CreateItem(
                event_id=event_id,
                item=normalize_conversation_item(item),
                previous_item_id=previous_item_id if isinstance(previous_item_id, str) else None,
            )
        if signal_event == "conversation.item.delete":
            item_id = signal_payload.get("item_id")
            if not isinstance(item_id, str) or not item_id:
                raise RealtimeProtocolError(
                    "conversation.item.delete requires item_id", code="missing_item_id", event_id=event_id
                )
            return DeleteItem(event_id=event_id, item_id=item_id)
        if signal_event == "conversation.item.truncate":
            item_id = signal_payload.get("item_id")
            audio_end_ms = signal_payload.get("audio_end_ms")
            content_index = signal_payload.get("content_index", 0)
            if not isinstance(item_id, str) or not item_id or not isinstance(audio_end_ms, int | float):
                raise RealtimeProtocolError(
                    "conversation.item.truncate requires item_id and numeric audio_end_ms",
                    code="bad_event",
                    event_id=event_id,
                )
            return TruncateItem(
                event_id=event_id,
                item_id=item_id,
                audio_end_ms=int(audio_end_ms),
                content_index=int(content_index) if isinstance(content_index, int | float) else 0,
            )
        return SignalTurn(event_id=event_id, event=signal_event, signal_payload=signal_payload)

    raise RealtimeProtocolError(f"Unknown duplex event type: {event_type}", code="unknown_event", event_id=event_id)


#: The complete command vocabulary a duplex client may send: the eight Tier 1
#: classes re-exported unchanged, the two Tier 2 classes defined above, and the
#: seven Tier 3 ones. A duplex consumer imports from here and never reaches
#: past this module into ``vllm_omni.protocol.realtime``, so a command that
#: later grows a duplex extension changes only this file.
__all__ = [
    # Tier 1 --- re-exported from ``vllm_omni.protocol.realtime.commands``.
    "CancelResponse",
    "ClearInput",
    "ClearOutputAudio",
    "CreateItem",
    "CreateResponse",
    "DeleteItem",
    "RealtimeCommand",
    "TruncateItem",
    "UpdateSession",
    # Tier 2 --- an OpenAI command plus duplex-only fields.
    "AppendAudio",
    "Commit",
    # Tier 3 --- vLLM-Omni only.
    "AckPlayback",
    "AppendText",
    "BargeIn",
    "CancelInput",
    "CloseSession",
    "Heartbeat",
    "SignalTurn",
    # Base, vocabulary and decoder.
    "DuplexCommand",
    "RealtimeProtocolError",
    "build_append_audio",
    "decode_duplex_command",
]
