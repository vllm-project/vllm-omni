# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Thin websocket serving for ``/v1/realtime?duplex=1``.

Everything about a duplex *session* lives inside the engine (see
``vllm_omni.entrypoints.duplex_omni``). This handler only does transport:

* accept the socket and run the Realtime handshake (``session.update`` opens a
  session through ``DuplexOmni.open_session``; ``session.resume`` re-attaches
  to an existing one);
* translate each client JSON event into a ``DuplexCommand`` and submit it in
  arrival order;
* pump the session's typed events to the wire, journaling them for resume;
* manage attachments: resume tokens, replay, takeover, detach on disconnect.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, replace

from fastapi import WebSocket, WebSocketDisconnect
from vllm.logger import init_logger

from vllm_omni.engine.duplex.commands import DuplexCommand, DuplexCommandError
from vllm_omni.engine.duplex.events import (
    DuplexEvent,
    SessionClosed,
    SessionCreated,
    SessionReplaced,
    SessionResumed,
    SessionResyncRequired,
)
from vllm_omni.engine.duplex.messages import DuplexSessionError
from vllm_omni.entrypoints.duplex.realtime_input import RealtimeEnvelope, parse_resume_request
from vllm_omni.entrypoints.duplex.session_attachment import (
    DuplexJournalGapError,
    DuplexJournalOverflowError,
    DuplexSessionAttachmentRegistry,
    InvalidResumeTokenError,
    ResumeToken,
)
from vllm_omni.entrypoints.duplex.websocket import (
    MAX_EVENT_BYTES,
    SendJson,
    attachment_callbacks,
    receive_text_with_timeout,
)
from vllm_omni.entrypoints.duplex_omni import DuplexOmni, DuplexSessionHandle

logger = init_logger(__name__)

__all__ = ["OmniDuplexSessionHandler"]

_DEFAULT_CONFIG_TIMEOUT_S = 10.0
_DEFAULT_IDLE_TIMEOUT_S = 300.0
#: session.created is journaled by the registry when it hands out the resume credential.
_UNJOURNALED_EVENTS = (SessionCreated,)


@dataclass
class _Attachment:
    """What one websocket connection knows about the session it is attached to."""

    handle: DuplexSessionHandle
    generation: int


@dataclass(frozen=True)
class _ResumeCredentials:
    """Transport credentials stamped onto ``session.created`` for a resumable session."""

    attachment_generation: int
    resume_token: str


class OmniDuplexSessionHandler:
    """WebSocket transport for engine-resident duplex sessions."""

    def __init__(
        self,
        *,
        duplex_omni: DuplexOmni,
        config_timeout_s: float = _DEFAULT_CONFIG_TIMEOUT_S,
        idle_timeout_s: float = _DEFAULT_IDLE_TIMEOUT_S,
    ) -> None:
        self._omni = duplex_omni
        self._config_timeout_s = config_timeout_s
        self._idle_timeout_s = idle_timeout_s
        runtime_config = duplex_omni.duplex_session_config
        self._attachment_registry = DuplexSessionAttachmentRegistry(
            replay_ttl_s=runtime_config.resume_replay_ttl_s,
            replay_max_bytes_per_session=runtime_config.resume_replay_max_bytes_per_session,
        )
        self._resync_required_sessions: set[str] = set()
        self._pumps: dict[str, asyncio.Task[None]] = {}

    # ------------------------------------------------------------------ #
    # Entry point                                                         #
    # ------------------------------------------------------------------ #

    async def handle_realtime_session(self, websocket: WebSocket) -> None:
        await websocket.accept()
        envelope = RealtimeEnvelope.from_query_params(websocket.query_params)

        async def send_json(payload: Mapping[str, object]) -> None:
            await websocket.send_json(dict(payload))

        attachment: _Attachment | None = None
        pending_command: dict[str, object] | None = None
        try:
            open_payload = envelope.initial_open_payload()
            if open_payload is None:
                first = await self._receive_first_message(websocket, envelope, send_json)
                if first is None:
                    return
                handshake = envelope.first_message(first)
                if handshake.kind == "resume":
                    attachment = await self._resume(websocket, envelope, handshake.resume_payload, send_json)
                    if attachment is None:
                        return
                else:
                    open_payload = handshake.session_payload
                    pending_command = handshake.pending_command_payload
            if attachment is None:
                assert open_payload is not None
                attachment = await self._open(websocket, envelope, open_payload, send_json)
                if attachment is None:
                    return
            if pending_command is not None:
                await self._submit_wire_event(attachment, envelope, pending_command, send_json)
            await self._read_loop(websocket, envelope, attachment, send_json)
        except WebSocketDisconnect:
            if attachment is not None:
                await self._on_disconnect(attachment)
        except Exception as exc:
            logger.exception("Duplex websocket session failed: %s", exc)
            if attachment is not None and await self._attachment_registry.is_current_attachment(
                attachment.handle.session_id, attachment.generation
            ):
                with suppress(Exception):
                    await self._attachment_registry.close(attachment.handle.session_id)
                with suppress(Exception):
                    await attachment.handle.close(reason="transport_error")
            with suppress(Exception):
                await send_json(envelope.error_payload("realtime_input_failed", str(exc)))

    async def _receive_first_message(
        self, websocket: WebSocket, envelope: RealtimeEnvelope, send_json: SendJson
    ) -> dict[str, object] | None:
        raw = await receive_text_with_timeout(websocket, self._config_timeout_s)
        if raw is None:
            await send_json(envelope.error_payload("config_timeout", "Timeout waiting for session.update"))
            return None
        try:
            first = json.loads(raw)
        except json.JSONDecodeError:
            await send_json(envelope.error_payload("invalid_json", "Invalid JSON in session.update"))
            return None
        if not isinstance(first, dict):
            await send_json(envelope.error_payload("bad_event", "Realtime event must be a JSON object"))
            return None
        return first

    # ------------------------------------------------------------------ #
    # Handshake                                                           #
    # ------------------------------------------------------------------ #

    async def _open(
        self,
        websocket: WebSocket,
        envelope: RealtimeEnvelope,
        session_payload: Mapping[str, object],
        send_json: SendJson,
    ) -> _Attachment | None:
        # Any ``session_id`` / ``id`` in the payload is ignored: the server
        # allocates the id and announces it in ``session.created``.
        try:
            handle = await self._omni.open_session(session_payload)
        except DuplexSessionError as exc:
            await send_json(envelope.error_payload(exc.code, str(exc)))
            return None
        except (TypeError, ValueError) as exc:
            await send_json(envelope.error_payload("invalid_session_config", str(exc)))
            return None
        attachment_send, attachment_close = attachment_callbacks(websocket)
        created = await self._attachment_registry.create(
            handle.session_id,
            send=attachment_send,
            close=attachment_close,
        )
        resume_supported = bool(handle.capabilities.supports_session_resume)
        attachment = _Attachment(handle=handle, generation=created.attachment_generation)
        credentials: _ResumeCredentials | None = None
        if resume_supported:
            credentials = _ResumeCredentials(
                attachment_generation=created.attachment_generation,
                resume_token=created.resume_token.plaintext,
            )
        self._start_pump(handle, credentials)
        return attachment

    async def _resume(
        self,
        websocket: WebSocket,
        envelope: RealtimeEnvelope,
        event: Mapping[str, object],
        send_json: SendJson,
    ) -> _Attachment | None:
        request = parse_resume_request(event)
        if request is None:
            await send_json(
                envelope.error_payload(
                    "invalid_session_resume",
                    "session.resume requires session_id, resume_token, and a non-negative event sequence",
                )
            )
            return None
        session_id = request.session_id
        handle = self._omni.get_session(session_id)
        if handle is None or handle.closed:
            await send_json(
                envelope.error_payload("session_resume_expired", f"Unknown or expired duplex session: {session_id}")
            )
            return None
        if not handle.capabilities.supports_session_resume:
            await send_json(
                envelope.error_payload("unsupported_session_resume", f"Session does not support resume: {session_id}")
            )
            return None
        try:
            await self._attachment_registry.authenticate_resume(
                session_id,
                resume_token=request.resume_token,
                last_received_server_event_seq=request.last_received_server_event_seq,
            )
        except InvalidResumeTokenError:
            await send_json(envelope.error_payload("invalid_resume_token", "Invalid duplex session resume token"))
            return None
        except DuplexJournalGapError:
            await send_json(SessionResyncRequired(session_id=session_id, reason="journal_gap").to_realtime())
            return None
        except (KeyError, ValueError) as exc:
            await send_json(envelope.error_payload("session_resume_conflict", str(exc)))
            return None
        try:
            await self._omni.resume_session(
                session_id,
                expected_lease_generation=handle.lease_generation,
            )
        except DuplexSessionError as exc:
            await send_json(envelope.error_payload("runtime_resume_failed", str(exc)))
            return None

        attachment_send, attachment_close = attachment_callbacks(websocket)

        def activation_payload_factory(token: ResumeToken, generation: int) -> dict[str, object]:
            return SessionResumed(
                session_id=session_id,
                session=dict(handle.public_session),
                attachment_generation=generation,
                resume_token=token.plaintext,
            ).to_realtime()

        try:
            resumed = await self._attachment_registry.resume(
                session_id,
                resume_token=request.resume_token,
                last_received_server_event_seq=request.last_received_server_event_seq,
                send=attachment_send,
                close=attachment_close,
                activation_payload_factory=activation_payload_factory,
            )
        except Exception as exc:
            # The engine lease was already resumed above: put the session back
            # into its disconnect grace instead of leaving it attached to nothing.
            with suppress(DuplexSessionError):
                await self._omni.detach_session(session_id)
            await send_json(envelope.error_payload("session_resume_conflict", str(exc)))
            return None
        replaced = resumed.replaced_attachment
        if replaced is not None:
            with suppress(Exception):
                await replaced.send(
                    SessionReplaced(session_id=session_id, attachment_generation=replaced.generation).to_realtime()
                )
            with suppress(Exception):
                await replaced.close("session_replaced")
        self._start_pump(handle, None)
        return _Attachment(handle=handle, generation=resumed.attachment_generation)

    # ------------------------------------------------------------------ #
    # Outbound pump (session-scoped, survives reconnects)                #
    # ------------------------------------------------------------------ #

    def _start_pump(self, handle: DuplexSessionHandle, credentials: _ResumeCredentials | None) -> None:
        existing = self._pumps.get(handle.session_id)
        if existing is not None and not existing.done():
            return
        self._pumps[handle.session_id] = asyncio.create_task(
            self._pump_events(handle, credentials),
            name=f"duplex-session-pump-{handle.session_id}",
        )

    async def _pump_events(self, handle: DuplexSessionHandle, credentials: _ResumeCredentials | None) -> None:
        session_id = handle.session_id
        close_reason = "session_closed"
        try:
            async for event in handle.events():
                if isinstance(event, SessionCreated) and credentials is not None:
                    event = replace(
                        event,
                        attachment_generation=credentials.attachment_generation,
                        resume_token=credentials.resume_token,
                    )
                await self._send_event(session_id, event)
                if isinstance(event, SessionClosed):
                    close_reason = event.reason or event.type
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Duplex event pump failed for %s: %s", session_id, exc)
        finally:
            self._pumps.pop(session_id, None)
            self._resync_required_sessions.discard(session_id)
            attachment = None
            with suppress(Exception):
                attachment = await self._attachment_registry.close(session_id)
            if attachment is not None:
                with suppress(Exception):
                    await attachment.close(close_reason)

    async def _send_event(self, session_id: str, event: DuplexEvent) -> None:
        payload = event.to_realtime()
        journal = not isinstance(event, _UNJOURNALED_EVENTS) and session_id not in self._resync_required_sessions
        try:
            try:
                await self._attachment_registry.send_event(session_id, payload, journal=journal)
            except DuplexJournalOverflowError:
                first_overflow = session_id not in self._resync_required_sessions
                self._resync_required_sessions.add(session_id)
                if first_overflow:
                    resync = SessionResyncRequired(session_id=session_id, reason="journal_overflow")
                    await self._attachment_registry.send_event(session_id, resync.to_realtime(), journal=False)
                await self._attachment_registry.send_event(session_id, payload, journal=False)
        except KeyError:
            # Attachment already closed (takeover or teardown); the journal is gone.
            pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # The socket died mid-send. The session is alive in the engine:
            # treat it as a disconnect of the current attachment (the pump
            # keeps journaling for a resume) instead of tearing the session down.
            logger.info("Duplex transport send failed for %s: %s", session_id, exc)
            await self._detach_current_attachment(session_id)

    async def _detach_current_attachment(self, session_id: str) -> None:
        """Disconnect semantics for the socket currently attached to ``session_id``.

        The pump is session-scoped and outlives any one connection, so it can
        only name the session: the registry detaches whichever socket is
        attached, which is the one whose send just failed.
        """
        handle = self._omni.get_session(session_id)
        if handle is None or handle.closed:
            return
        if handle.capabilities.supports_session_resume:
            if await self._attachment_registry.detach(session_id, attachment_generation=None):
                with suppress(DuplexSessionError):
                    await self._omni.detach_session(session_id)
            return
        with suppress(DuplexSessionError):
            await handle.close(reason="disconnect")

    # ------------------------------------------------------------------ #
    # Inbound                                                             #
    # ------------------------------------------------------------------ #

    async def _read_loop(
        self,
        websocket: WebSocket,
        envelope: RealtimeEnvelope,
        attachment: _Attachment,
        send_json: SendJson,
    ) -> None:
        handle = attachment.handle
        idle_timeout_s = self._idle_timeout_s
        config_timeout = handle.public_session.get("idle_timeout_s") if handle.public_session else None
        if isinstance(config_timeout, int | float) and config_timeout > 0:
            idle_timeout_s = float(config_timeout)
        while not handle.closed:
            raw = await receive_text_with_timeout(websocket, idle_timeout_s)
            if not await self._attachment_registry.is_current_attachment(handle.session_id, attachment.generation):
                # A newer connection took the session over; this socket is done.
                return
            if raw is None:
                # Nothing from this socket for the idle window: same as a
                # disconnect (engine grace for resumable sessions); the engine
                # lease decides whether the session itself expires.
                await self._on_disconnect(attachment)
                return
            if len(raw.encode("utf-8")) > MAX_EVENT_BYTES:
                await send_json(envelope.error_payload("event_too_large", "Duplex event too large"))
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                await send_json(envelope.error_payload("invalid_json", "Invalid JSON event"))
                continue
            if not isinstance(payload, dict) or not isinstance(payload.get("type"), str):
                await send_json(envelope.error_payload("bad_event", "Duplex event must be a JSON object with a type"))
                continue
            if envelope.is_envelope_event(payload):
                await self._handle_envelope_event(attachment, envelope, payload, send_json)
                continue
            await self._submit_wire_event(attachment, envelope, payload, send_json)

    async def _handle_envelope_event(
        self,
        attachment: _Attachment,
        envelope: RealtimeEnvelope,
        payload: dict[str, object],
        send_json: SendJson,
    ) -> None:
        handle = attachment.handle
        event_type = payload.get("type")
        event_id = payload.get("event_id")
        if event_type == "session.event_ack":
            acknowledged = payload.get("server_event_seq")
            if not isinstance(acknowledged, int) or acknowledged < 0:
                message = "session.event_ack requires a non-negative server_event_seq"
                await send_json(envelope.error_payload("invalid_event_ack", message, event_id=event_id))
                return
            try:
                await self._attachment_registry.acknowledge(handle.session_id, acknowledged)
            except ValueError as exc:
                await send_json(envelope.error_payload("invalid_event_ack", str(exc), event_id=event_id))
            return
        message = "session.resume must be the first message of a new connection"
        await send_json(envelope.error_payload("unsupported_session_resume", message, event_id=event_id))

    async def _submit_wire_event(
        self,
        attachment: _Attachment,
        envelope: RealtimeEnvelope,
        payload: dict[str, object],
        send_json: SendJson,
    ) -> None:
        try:
            command = envelope.translate(payload)
        except DuplexCommandError as exc:
            await send_json(envelope.command_error_payload(exc))
            return
        await self._submit_command(attachment, envelope, command, send_json)

    async def _submit_command(
        self,
        attachment: _Attachment,
        envelope: RealtimeEnvelope,
        command: DuplexCommand,
        send_json: SendJson,
    ) -> None:
        try:
            await attachment.handle.submit(command)
        except DuplexSessionError as exc:
            await send_json(envelope.error_payload(exc.code, str(exc), event_id=command.event_id))

    # ------------------------------------------------------------------ #
    # Disconnect                                                          #
    # ------------------------------------------------------------------ #

    async def _on_disconnect(self, attachment: _Attachment) -> None:
        handle = attachment.handle
        if handle.closed:
            return
        if handle.capabilities.supports_session_resume:
            if await self._attachment_registry.detach(handle.session_id, attachment_generation=attachment.generation):
                # Engine-owned disconnect grace: expiry arrives as session.expired.
                with suppress(DuplexSessionError):
                    await self._omni.detach_session(handle.session_id)
            # Otherwise a newer connection already took the session over; the
            # replaced socket's disconnect must not touch it.
            return
        with suppress(DuplexSessionError):
            await handle.close(reason="disconnect")
