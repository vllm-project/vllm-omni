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
from functools import partial

from fastapi import WebSocket, WebSocketDisconnect
from vllm.logger import init_logger

from vllm_omni.engine.duplex.commands import DuplexCommand, DuplexCommandError, UpdateSession
from vllm_omni.engine.duplex.events import (
    DuplexEvent,
    SessionClosed,
    SessionCreated,
    SessionReplaced,
    SessionResumed,
    SessionResyncRequired,
    SessionUpdated,
)
from vllm_omni.engine.duplex.messages import DuplexSessionError
from vllm_omni.entrypoints.duplex.realtime_input import RealtimeEnvelope, ResumeRequest, parse_resume_request
from vllm_omni.entrypoints.duplex.session_attachment import (
    DuplexDetachedAttachment,
    DuplexJournalGapError,
    DuplexJournalOverflowError,
    DuplexSessionAttachmentRegistry,
    DuplexSessionResumeResult,
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
#: How long the endpoint waits for the session pump to deliver its terminal
#: event once the session is closed. Bounded so a wedged pump cannot pin the
#: connection open; the pump itself closes the socket on its way out.
_PUMP_DRAIN_TIMEOUT_S = 5.0
#: session.created is journaled by the registry when it hands out the resume credential.
_UNJOURNALED_EVENTS = (SessionCreated,)


def _log_rollback_failure(done: asyncio.Task[None]) -> None:
    """A shielded resume rollback outlives its caller; its failure must still be seen."""
    if not done.cancelled() and done.exception() is not None:
        logger.error("duplex resume rollback %s failed: %r", done.get_name(), done.exception())


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
        #: Live connection envelopes keyed by session id. ``session.updated``
        #: arrives on the session-scoped pump; the envelope that must apply the
        #: new wire defaults lives on the read loop, so the pump looks it up here.
        #: Defaults stay on that envelope. A reconnect copies them off the
        #: previous entry before replacing it. The pump drops the entry when
        #: the session ends.
        self._live_envelopes: dict[str, RealtimeEnvelope] = {}
        self._pumps: dict[str, asyncio.Task[None]] = {}
        #: One submitted ``session.update`` whose ``session.updated`` or rejection
        #: is not yet visible. The next format-dependent append waits on it.
        self._pending_session_updates: dict[str, asyncio.Future[SessionUpdated | None]] = {}

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
            await self._drain_terminal_pump(attachment)
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
        # The registry keeps the engine lease generation each attachment is
        # serving, so every detach is fenced on the dropped connection's own
        # lease (see ``release_attachment``).
        created = await self._attachment_registry.create(
            handle.session_id,
            send=attachment_send,
            close=attachment_close,
            lease_generation=handle.lease_generation,
        )
        resume_supported = bool(handle.capabilities.supports_session_resume)
        attachment = _Attachment(handle=handle, generation=created.attachment_generation)
        credentials: _ResumeCredentials | None = None
        if resume_supported:
            credentials = _ResumeCredentials(
                attachment_generation=created.attachment_generation,
                resume_token=created.resume_token.plaintext,
            )
        # Bind before the pump starts so a session.updated emitted during open
        # applies onto this connection's envelope.
        self._live_envelopes[handle.session_id] = envelope
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
        # The claim tells the registry a resume is about to activate: a lease
        # generation orphaned meanwhile (another resume abandoned mid-RPC) is
        # parked for this activation instead of being handed to the socket it
        # is about to replace. The claim ends whatever happens below, and an
        # orphan nobody else will serve is then ours to put into grace.
        await self._attachment_registry.begin_resume(session_id)
        try:
            return await self._resume_claimed(websocket, envelope, request, handle, send_json)
        finally:
            await self._end_resume_claim(session_id)

    async def _resume_claimed(
        self,
        websocket: WebSocket,
        envelope: RealtimeEnvelope,
        request: ResumeRequest,
        handle: DuplexSessionHandle,
        send_json: SendJson,
    ) -> _Attachment | None:
        session_id = request.session_id
        try:
            # If this task is cancelled while the RPC is in flight, DuplexOmni
            # observes the outcome and settles a resume that landed through
            # ``_settle_abandoned_resume``: the generation goes to the resume
            # waiting to activate, else to whoever is attached, else the lease
            # goes back into disconnect grace.
            await self._omni.resume_session(
                session_id,
                expected_lease_generation=handle.lease_generation,
                on_abandoned=lambda lease_generation: self._settle_abandoned_resume(session_id, lease_generation),
            )
        except DuplexSessionError as exc:
            await send_json(envelope.error_payload("runtime_resume_failed", str(exc)))
            return None
        # The generation this resume produced, read before anything else can
        # run: the handle is shared by every connection of the session, and a
        # concurrent resume adopts its own, newer generation into it.
        lease_generation = handle.lease_generation

        attachment_send, attachment_close = attachment_callbacks(websocket)

        def activation_payload_factory(token: ResumeToken, generation: int) -> dict[str, object]:
            return SessionResumed(
                session_id=session_id,
                session=dict(handle.public_session),
                attachment_generation=generation,
                resume_token=token.plaintext,
            ).to_realtime()

        # From here on the engine lease is resumed (``detached_at`` cleared), so
        # every exit that does not hand the attachment to the caller has to
        # put the lease back into its disconnect grace, or the session sits
        # attached to nothing until idle expiry. That includes cancellation of
        # the handler task, which is not an ``Exception``.
        resumed: DuplexSessionResumeResult | None = None
        try:
            resumed = await self._attachment_registry.resume(
                session_id,
                resume_token=request.resume_token,
                last_received_server_event_seq=request.last_received_server_event_seq,
                send=attachment_send,
                close=attachment_close,
                activation_payload_factory=activation_payload_factory,
                lease_generation=lease_generation,
            )
            replaced = resumed.replaced_attachment
            if replaced is not None:
                with suppress(Exception):
                    await replaced.send(
                        SessionReplaced(session_id=session_id, attachment_generation=replaced.generation).to_realtime()
                    )
                with suppress(Exception):
                    await replaced.close("session_replaced")
        except asyncio.CancelledError:
            await self._abandon_resume(
                session_id,
                attachment_generation=resumed.attachment_generation if resumed is not None else None,
                lease_generation=lease_generation,
            )
            raise
        except Exception as exc:
            await self._abandon_resume(session_id, attachment_generation=None, lease_generation=lease_generation)
            await send_json(envelope.error_payload("session_resume_conflict", str(exc)))
            return None
        # A reconnect brings a fresh envelope carrying pcm16/16 kHz wire
        # defaults. The negotiated input format is a wire default, not part of
        # the public session object, so copy it from the envelope this session
        # is still holding before replacing the map entry. Otherwise the first
        # append that omits format/rate is decoded as pcm16.
        previous = self._live_envelopes.get(session_id)
        if previous is not None:
            envelope.defaults = previous.defaults
        self._live_envelopes[session_id] = envelope
        self._start_pump(handle, None)
        return _Attachment(handle=handle, generation=resumed.attachment_generation)

    async def _abandon_resume(
        self,
        session_id: str,
        *,
        attachment_generation: int | None,
        lease_generation: int,
    ) -> None:
        """Roll the engine lease of a resume this connection will never serve back into disconnect grace.

        Only what this attempt owns is rolled back. ``attachment_generation``
        is the attachment it activated, or ``None`` when activation did not
        happen (or was rolled back by the registry): then the generation this
        attempt's engine resume produced is settled like an abandoned resume
        (``_settle_abandoned_resume``). The registry knows more than this
        method does: a rolled-back activation may have been handed a newer
        generation meanwhile, which the registry parks as the orphan, and the
        claim ending in ``_resume`` gives it back if nobody else will serve
        it. Every engine detach is fenced on the generation being given up, so
        a newer lease a concurrent resume produced is never touched.

        The rollback runs shielded: a second cancellation landing between the
        registry detach and the engine detach would otherwise strand the
        session exactly the way this method compensates for.
        """
        rollback = asyncio.ensure_future(
            self._roll_back_resume(
                session_id,
                attachment_generation=attachment_generation,
                lease_generation=lease_generation,
            )
        )
        rollback.add_done_callback(_log_rollback_failure)
        await asyncio.shield(rollback)

    async def _roll_back_resume(
        self,
        session_id: str,
        *,
        attachment_generation: int | None,
        lease_generation: int,
    ) -> None:
        if attachment_generation is None:
            await self._settle_abandoned_resume(session_id, lease_generation)
            return
        released = await self._attachment_registry.release_attachment(
            session_id, attachment_generation=attachment_generation
        )
        if released is not None:
            await self._detach_lease(released)

    async def _settle_abandoned_resume(self, session_id: str, lease_generation: int) -> None:
        """Settle the engine lease generation of a resume its connection never got to serve.

        The engine already bumped the lease to ``lease_generation`` and
        cleared its disconnect grace. The registry picks the owner: a resume
        waiting to activate (it will serve the session), else the connection
        attached right now (a takeover that never activated leaves the
        previous socket serving, and its own disconnect must be able to
        detach this lease), else nobody, and then the lease goes back into
        disconnect grace here, fenced on the generation so a later resume is
        never detached.
        """
        orphan = await self._attachment_registry.settle_lease_generation(session_id, lease_generation)
        if orphan is not None:
            with suppress(DuplexSessionError):
                await self._omni.detach_session(session_id, expected_lease_generation=orphan)

    async def _end_resume_claim(self, session_id: str) -> None:
        """End this connection's resume claim; detach an orphaned lease nobody else will serve.

        Runs from a ``finally``, so possibly during cancellation: the detach
        is shielded like the other rollbacks.
        """
        orphan = await self._attachment_registry.end_resume(session_id)
        if orphan is None:
            return

        async def detach_orphan() -> None:
            with suppress(DuplexSessionError):
                await self._omni.detach_session(session_id, expected_lease_generation=orphan)

        rollback = asyncio.ensure_future(detach_orphan())
        rollback.add_done_callback(_log_rollback_failure)
        await asyncio.shield(rollback)

    async def _detach_lease(self, released: DuplexDetachedAttachment) -> None:
        """Engine-owned disconnect grace for the lease a dropped connection was serving."""
        with suppress(DuplexSessionError):
            await self._omni.detach_session(released.session_id, expected_lease_generation=released.lease_generation)

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
                if isinstance(event, SessionUpdated):
                    # Queued updates are also applied before the next translate.
                    # This path still covers a client that waits for the event
                    # itself and sends nothing in between.
                    self._apply_accepted_session_defaults(session_id, event.session)
                await self._send_event(session_id, event, handle=handle)
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
            self._live_envelopes.pop(session_id, None)
            pending = self._pending_session_updates.pop(session_id, None)
            if pending is not None:
                pending.cancel()
            attachment = None
            with suppress(Exception):
                attachment = await self._attachment_registry.close(session_id)
            if attachment is not None:
                with suppress(Exception):
                    await attachment.close(close_reason)

    async def _drain_terminal_pump(self, attachment: _Attachment) -> None:
        """Let the pump deliver ``session.closed`` before the endpoint returns.

        ``DuplexSessionHandle._deliver`` queues ``SessionClosed`` on the outbox
        and marks the handle closed in the same synchronous step, so
        ``handle.closed`` is already true while that event is still sitting in
        the queue. ``_read_loop`` stops on exactly that flag: if the engine
        closes the session between two reads, the loop returns, this endpoint
        returns, and the ASGI server tears the socket down with the terminal
        event unsent -- the client sees an abrupt close instead of
        ``session.closed``. Waiting for the pump keeps the two in order; it is
        the pump that sends the event and then closes with code 1000.

        Only for a closed session. A takeover or a resumable disconnect leaves
        the pump running for the next attachment, and must not be waited on.
        """
        if not attachment.handle.closed:
            return
        pump = self._pumps.get(attachment.handle.session_id)
        if pump is None or pump.done():
            return
        # shield: a drain timeout must not cancel a pump that is still the
        # session's only writer.
        with suppress(Exception):
            await asyncio.wait_for(asyncio.shield(pump), _PUMP_DRAIN_TIMEOUT_S)

    async def _send_event(self, session_id: str, event: DuplexEvent, *, handle: DuplexSessionHandle) -> None:
        payload = event.to_realtime()
        is_handshake = isinstance(event, SessionCreated)
        # ``SessionCreated`` is unjournaled. ``on_accepted`` then runs only after
        # a successful transport send, not when a detached session merely returns.
        # Journal acceptance (replay recorded, socket already gone) is not delivery.
        journal = not isinstance(event, _UNJOURNALED_EVENTS) and session_id not in self._resync_required_sessions
        transport_delivered = False

        def mark_transport_delivered() -> None:
            nonlocal transport_delivered
            transport_delivered = True

        event_guard = partial(handle.output_guard, event)
        try:
            try:
                await self._attachment_registry.send_event(
                    session_id,
                    payload,
                    journal=journal,
                    on_accepted=mark_transport_delivered if is_handshake else None,
                    event_guard=event_guard,
                )
            except DuplexJournalOverflowError:
                first_overflow = session_id not in self._resync_required_sessions
                self._resync_required_sessions.add(session_id)
                if first_overflow:
                    resync = SessionResyncRequired(session_id=session_id, reason="journal_overflow")
                    await self._attachment_registry.send_event(session_id, resync.to_realtime(), journal=False)
                await self._attachment_registry.send_event(session_id, payload, journal=False, event_guard=event_guard)
        except KeyError:
            # Attachment registry already torn down. A handshake that never left
            # the process still has to release the engine session.
            if is_handshake:
                await self._close_undeliverable_session(session_id)
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # The socket died mid-send. Usual case: keep the engine session for
            # resume (detach the current attachment; the pump keeps journaling).
            # Exception: ``session.created`` itself never reached the client, so
            # nobody has the session id / resume token -- detach would burn an
            # admission slot until idle expiry. Close instead (#7636 issue 9).
            # An in-flight send that then raises takes this path; one that
            # completes has already set ``transport_delivered`` and must not.
            logger.info("Duplex transport send failed for %s: %s", session_id, exc)
            if is_handshake:
                await self._close_undeliverable_session(session_id)
                return
            await self._detach_current_attachment(session_id)
            return
        if is_handshake and not transport_delivered:
            # Reader detached before the pump reached ``session.created``.
            # ``send_event`` returns without raising when there is no attachment
            # and the event is not journaled.
            await self._close_undeliverable_session(session_id)

    async def _close_undeliverable_session(self, session_id: str) -> None:
        """Tear down a session whose ``session.created`` never reached the client.

        Covers a raised send and a detach that happens before the send.
        Production admission frees after ``handle.close`` waits for
        ``session.closed`` (manager cleanup). If close raises, the handle is
        already marked closed and we must not detach.
        """
        handle = self._omni.get_session(session_id)
        if handle is None or handle.closed:
            return
        with suppress(DuplexSessionError):
            await handle.close(reason="session_created_undelivered")

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
            # The lease generation comes with the dropped attachment, captured
            # under the registry lock: the shared handle may already carry the
            # generation of a resume that is waiting to activate behind this
            # very send, and that lease is not ours to detach.
            released = await self._attachment_registry.release_attachment(session_id, attachment_generation=None)
            if released is not None:
                await self._detach_lease(released)
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
        # An append that omits its format has to see the outcome of a
        # session.update already submitted on this connection. Acceptance may
        # still be inside the engine, so it is not enough that a
        # session.updated is sitting in the outbox. A later update waits too,
        # so only one update is in flight.
        if payload.get("type") in {"input_audio_buffer.append", "session.update"}:
            await self._wait_for_pending_session_update(attachment.handle)
            if attachment.handle.closed:
                return
        # An accepted session.update is already on the outbox once the engine
        # emits it. Apply that before translate so the next append does not
        # keep the previous format while the pump is still sending an earlier
        # event. A rejected update never queues session.updated.
        self._apply_queued_session_updates(attachment.handle)
        try:
            command = envelope.translate(payload)
        except DuplexCommandError as exc:
            await send_json(envelope.command_error_payload(exc))
            return
        await self._submit_command(attachment, envelope, command, send_json)

    async def _wait_for_pending_session_update(self, handle: DuplexSessionHandle) -> None:
        """Block until the in-flight ``session.update`` is accepted or rejected.

        Acceptance is the enqueued ``session.updated``; rejection is settled
        explicitly by the engine owner. Neither waits for socket delivery.
        """
        session_id = handle.session_id
        pending = self._pending_session_updates.get(session_id)
        if pending is None:
            return
        # Disconnect cancels this reader, not an already submitted operation.
        # The same session retains the result for its next attachment.
        outcome = await asyncio.shield(pending)
        if isinstance(outcome, SessionUpdated):
            self._apply_accepted_session_defaults(session_id, outcome.session)
        if self._pending_session_updates.get(session_id) is pending:
            self._pending_session_updates.pop(session_id, None)

    def _apply_queued_session_updates(self, handle: DuplexSessionHandle) -> None:
        """Apply ``session.updated`` events the pump has not dequeued yet."""
        for event in handle.queued_events():
            if isinstance(event, SessionUpdated):
                self._apply_accepted_session_defaults(handle.session_id, event.session)

    def _apply_accepted_session_defaults(self, session_id: str, session_payload: Mapping[str, object]) -> None:
        """Apply wire defaults only after the engine accepted a session.update."""
        envelope = self._live_envelopes.get(session_id)
        if envelope is None:
            return
        envelope.apply_accepted_session(session_payload)

    async def _submit_command(
        self,
        attachment: _Attachment,
        envelope: RealtimeEnvelope,
        command: DuplexCommand,
        send_json: SendJson,
    ) -> None:
        pending = None
        session_id = attachment.handle.session_id
        if isinstance(command, UpdateSession):
            pending = attachment.handle.observe_session_update()
            self._pending_session_updates[session_id] = pending
        try:
            await attachment.handle.submit(command)
        except DuplexSessionError as exc:
            if pending is not None:
                pending.cancel()
                if self._pending_session_updates.get(session_id) is pending:
                    self._pending_session_updates.pop(session_id, None)
            await send_json(envelope.error_payload(exc.code, str(exc), event_id=command.event_id))

    # ------------------------------------------------------------------ #
    # Disconnect                                                          #
    # ------------------------------------------------------------------ #

    async def _on_disconnect(self, attachment: _Attachment) -> None:
        handle = attachment.handle
        if handle.closed:
            return
        if handle.capabilities.supports_session_resume:
            released = await self._attachment_registry.release_attachment(
                handle.session_id, attachment_generation=attachment.generation
            )
            if released is not None:
                # Engine-owned disconnect grace: expiry arrives as session.expired.
                await self._detach_lease(released)
            # Otherwise a newer connection already took the session over; the
            # replaced socket's disconnect must not touch it.
            return
        with suppress(DuplexSessionError):
            await handle.close(reason="disconnect")
