# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The thin websocket handler: handshake, command translation, event pump, resume/takeover."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

import pytest
from fastapi import WebSocketDisconnect

from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.config import DuplexCapabilities
from vllm_omni.engine.duplex.delivery import DuplexOutputBuffer
from vllm_omni.engine.duplex.events import AudioDelta, DuplexEvent, SessionClosed, SessionCreated
from vllm_omni.engine.duplex.messages import DuplexSessionError
from vllm_omni.entrypoints.duplex.realtime_input import RealtimeEnvelope, parse_resume_request
from vllm_omni.entrypoints.duplex.serving import OmniDuplexSessionHandler
from vllm_omni.entrypoints.duplex.websocket import MAX_EVENT_BYTES

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DISCONNECT = object()


class FakeWebSocket:
    def __init__(self, query: dict[str, str] | None = None) -> None:
        self.query_params = dict(query or {})
        self.sent: list[dict[str, Any]] = []
        self.accepted = False
        self.closed: list[tuple[int, str]] = []
        self._send_failure: str | None = None
        self._inbound: asyncio.Queue[Any] = asyncio.Queue()

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, payload: dict[str, Any]) -> None:
        if self.closed or self._send_failure is not None:
            raise RuntimeError(
                self._send_failure or "Unexpected ASGI message 'websocket.send', after sending 'websocket.close'."
            )
        self.sent.append(json.loads(json.dumps(payload)))

    async def receive_text(self) -> str:
        item = await self._inbound.get()
        if item is _DISCONNECT:
            raise WebSocketDisconnect(code=1000)
        return item

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed.append((code, reason))
        # A closed socket ends the reader exactly like a client disconnect.
        self._inbound.put_nowait(_DISCONNECT)

    # ---- test helpers ----

    def feed(self, payload: dict[str, Any] | str) -> None:
        self._inbound.put_nowait(payload if isinstance(payload, str) else json.dumps(payload))

    def disconnect(self) -> None:
        self._inbound.put_nowait(_DISCONNECT)

    def break_sends(self) -> None:
        """Kill the write half only: the reader stays parked, as it does in practice."""
        self._send_failure = "Unexpected ASGI message 'websocket.send', after sending 'websocket.close'."

    def types(self) -> list[str]:
        return [payload["type"] for payload in self.sent]

    async def wait_for(self, wire_type: str, *, timeout_s: float = 2.0) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while loop.time() < deadline:
            for payload in self.sent:
                if payload["type"] == wire_type:
                    return payload
            await asyncio.sleep(0.005)
        raise AssertionError(f"no {wire_type!r} in {self.types()}")


class FakeHandle:
    def __init__(self, session_id: str, capabilities: DuplexCapabilities, *, idle_timeout_s: float = 300) -> None:
        self.session_id = session_id
        self.capabilities = capabilities
        self.public_session: dict[str, Any] = {"id": session_id, "idle_timeout_s": idle_timeout_s}
        self.lease_generation = 0
        self.closed = False
        self.close_reasons: list[str] = []
        self.commands: list[commands.DuplexCommand] = []
        self._outbox = DuplexOutputBuffer(max_bytes=2 * 1024 * 1024, max_events=512)

    def deliver(self, event: DuplexEvent) -> None:
        self._outbox.put(event)

    def output_guard(self, event: DuplexEvent):
        return self._outbox.guard(event)

    async def submit(self, command: commands.DuplexCommand) -> None:
        if self.closed:
            raise DuplexSessionError("closed", code="session_closed", session_id=self.session_id)
        self.commands.append(command)

    async def events(self):
        while True:
            event = await self._outbox.get()
            if event is None:
                return
            if not self._outbox.is_valid(event):
                continue
            yield event
            if event.is_terminal:
                return

    async def close(self, *, reason: str = "client_close", timeout: float | None = None) -> None:
        self.close_reasons.append(reason)
        if self.closed:
            return
        self.closed = True
        self.deliver(SessionClosed(session_id=self.session_id, reason=reason))


class FakeOmni:
    def __init__(
        self, *, resumable: bool = True, replay_max_bytes: int = 64 * 1024, idle_timeout_s: float = 300
    ) -> None:
        self.duplex_session_config = DuplexSessionRuntimeConfig(
            resume_replay_ttl_s=60.0, resume_replay_max_bytes_per_session=replay_max_bytes
        )
        self.capabilities = DuplexCapabilities(supports_session_resume=resumable)
        self.idle_timeout_s = idle_timeout_s
        self.opened: list[dict[str, Any]] = []
        self.handles: dict[str, FakeHandle] = {}
        self.resumed: list[tuple[str, int]] = []
        self.detached: list[str] = []
        #: Detaches the engine refused because the caller's lease generation was stale.
        self.detach_refused: list[tuple[str, int]] = []
        self.open_error: DuplexSessionError | None = None
        #: When set, ``resume_session`` parks on it once the engine applied the
        #: resume: the lease generation is bumped, the result is still in flight.
        self.resume_gate: asyncio.Event | None = None
        self.resume_started = asyncio.Event()
        #: Settlements of resumes whose caller was cancelled mid-RPC (DuplexOmni's contract).
        self.compensations: list[asyncio.Task[None]] = []

    async def open_session(self, config: Any) -> FakeHandle:
        self.opened.append(dict(config))
        if self.open_error is not None:
            raise self.open_error
        session_id = f"duplex-{len(self.handles) + 1:032x}"
        handle = FakeHandle(session_id, self.capabilities, idle_timeout_s=self.idle_timeout_s)
        self.handles[session_id] = handle
        handle.deliver(SessionCreated(session_id=session_id, session={"id": session_id, "model": config.get("model")}))
        return handle

    def get_session(self, session_id: str) -> FakeHandle | None:
        return self.handles.get(session_id)

    async def resume_session(
        self,
        session_id: str,
        *,
        expected_lease_generation: int,
        on_abandoned: Callable[[int], Awaitable[None]] | None = None,
    ) -> FakeHandle:
        self.resumed.append((session_id, expected_lease_generation))
        handle = self.handles[session_id]
        lease_generation = expected_lease_generation + 1
        handle.lease_generation = lease_generation
        if self.resume_gate is not None:
            self.resume_started.set()
            gate = self.resume_gate
            try:
                await gate.wait()
            except asyncio.CancelledError:
                # DuplexOmni's contract: the engine applied the resume; once
                # its answer arrives the landed generation is settled off the
                # cancelled task, through the caller's callback if it gave one.
                async def settle() -> None:
                    await gate.wait()
                    if on_abandoned is not None:
                        await on_abandoned(lease_generation)
                    else:
                        with suppress(DuplexSessionError):
                            await self.detach_session(session_id, expected_lease_generation=lease_generation)

                self.compensations.append(asyncio.create_task(settle()))
                raise
        return handle

    async def detach_session(self, session_id: str, *, expected_lease_generation: int | None = None) -> None:
        handle = self.handles[session_id]
        if expected_lease_generation is not None and expected_lease_generation != handle.lease_generation:
            # The engine's fence: a detach of a lease the caller no longer holds is refused.
            self.detach_refused.append((session_id, expected_lease_generation))
            raise DuplexSessionError(
                "duplex lease generation mismatch", code="session_resume_conflict", session_id=session_id
            )
        self.detached.append(session_id)


def _handler(omni: FakeOmni, **kwargs: Any) -> OmniDuplexSessionHandler:
    kwargs.setdefault("config_timeout_s", 1.0)
    kwargs.setdefault("idle_timeout_s", 5.0)
    return OmniDuplexSessionHandler(duplex_omni=omni, **kwargs)


def _session_update(**session: Any) -> dict[str, Any]:
    return {"type": "session.update", "session": {"model": "test-model", "modalities": ["audio", "text"], **session}}


async def _open(
    handler: OmniDuplexSessionHandler, omni: FakeOmni, **session: Any
) -> tuple[FakeWebSocket, FakeHandle, asyncio.Task]:
    ws = FakeWebSocket({"duplex": "1", "autostart": "0"})
    task = asyncio.create_task(handler.handle_realtime_session(ws))
    ws.feed(_session_update(**session))
    created = await ws.wait_for("session.created")
    handle = omni.handles[created["session"]["id"]]
    return ws, handle, task


# --------------------------------------------------------------------------- #
# Handshake and commands                                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_session_update_opens_a_session_and_announces_the_server_id() -> None:
    omni = FakeOmni()
    handler = _handler(omni)
    ws, handle, task = await _open(handler, omni, session_id="mine", instructions="hi")

    assert ws.accepted
    created = ws.sent[0]
    assert created["type"] == "session.created"
    assert created["session"]["id"] == handle.session_id != "mine"
    assert created["attachment_generation"] == 1
    assert isinstance(created["resume_token"], str) and created["resume_token"]
    assert "incarnation" not in created
    assert "server_event_seq" not in created
    # The whole session object went to open_session; the id inside it is ignored there.
    assert omni.opened == [
        {"model": "test-model", "modalities": ["audio", "text"], "session_id": "mine", "instructions": "hi"}
    ]

    ws.feed({"type": "input_audio_buffer.append", "audio": base64.b64encode(b"\x00\x01" * 8).decode("ascii")})
    ws.feed({"type": "input_audio_buffer.commit", "event_id": "evt-c"})
    await asyncio.sleep(0.05)
    assert [type(command) for command in handle.commands] == [commands.AppendAudio, commands.Commit]
    assert handle.commands[1].event_id == "evt-c"

    ws.disconnect()
    await asyncio.wait_for(task, timeout=2.0)
    # A resumable session is detached (engine-owned grace), not closed.
    assert omni.detached == [handle.session_id]
    assert handle.close_reasons == []


@pytest.mark.asyncio
async def test_session_events_are_journaled_and_sent_in_order() -> None:
    omni = FakeOmni()
    handler = _handler(omni)
    ws, handle, task = await _open(handler, omni)

    handle.deliver(AudioDelta(session_id=handle.session_id, response_id="r1", delta="aGk="))
    delta = await ws.wait_for("response.output_audio.delta")
    assert delta["delta"] == "aGk=" and delta["server_event_seq"] == 1

    ws.feed({"type": "session.event_ack", "server_event_seq": 1})
    ws.feed({"type": "session.event_ack", "server_event_seq": 9, "event_id": "evt-ack"})
    error = await ws.wait_for("error")
    assert error["error"]["code"] == "invalid_event_ack" and error["error"]["event_id"] == "evt-ack"

    await handle.close(reason="client_close")
    closed = await ws.wait_for("session.closed")
    assert closed["reason"] == "client_close"
    await asyncio.wait_for(task, timeout=2.0)
    assert ws.closed[0][1] == "client_close"


@pytest.mark.asyncio
async def test_envelope_errors_are_answered_locally_without_touching_the_session() -> None:
    omni = FakeOmni()
    handler = _handler(omni)
    ws, handle, task = await _open(handler, omni)

    ws.feed("not json")
    ws.feed("x" * (MAX_EVENT_BYTES + 1))
    ws.feed({"type": "totally.unknown", "event_id": "evt-u"})
    ws.feed({"type": "session.resume", "event_id": "evt-r"})
    ws.feed({"type": "playback.ack", "event_id": "evt-p"})  # played_ms missing
    await asyncio.sleep(0.1)

    codes = [payload["error"]["code"] for payload in ws.sent if payload["type"] == "error"]
    assert codes == ["invalid_json", "event_too_large", "unknown_event", "unsupported_session_resume", "bad_event"]
    errors = [payload for payload in ws.sent if payload["type"] == "error"]
    assert errors[2]["error"]["event_id"] == "evt-u"
    assert errors[4]["error"]["event_id"] == "evt-p"
    assert handle.commands == []
    ws.disconnect()
    await asyncio.wait_for(task, timeout=2.0)


@pytest.mark.asyncio
async def test_open_rejection_is_reported_and_the_socket_ends() -> None:
    omni = FakeOmni()
    omni.open_error = DuplexSessionError("no room", code="resource_exhausted", retryable=True)
    handler = _handler(omni)
    ws = FakeWebSocket({"duplex": "1"})
    task = asyncio.create_task(handler.handle_realtime_session(ws))
    ws.feed(_session_update())
    await asyncio.wait_for(task, timeout=2.0)
    assert ws.types() == ["error"]
    assert ws.sent[0]["error"]["code"] == "resource_exhausted"


@pytest.mark.asyncio
async def test_config_timeout_and_first_message_validation() -> None:
    omni = FakeOmni()
    handler = _handler(omni, config_timeout_s=0.05)
    ws = FakeWebSocket({"duplex": "1"})
    await asyncio.wait_for(handler.handle_realtime_session(ws), timeout=2.0)
    assert ws.sent[0]["error"]["code"] == "config_timeout"

    ws = FakeWebSocket({"duplex": "1"})
    task = asyncio.create_task(handler.handle_realtime_session(ws))
    ws.feed("{not json")
    await asyncio.wait_for(task, timeout=2.0)
    assert ws.sent[0]["error"]["code"] == "invalid_json"
    assert omni.opened == []


@pytest.mark.asyncio
async def test_idle_timeout_detaches_a_resumable_session_like_a_disconnect() -> None:
    omni = FakeOmni(idle_timeout_s=0.05)
    handler = _handler(omni, idle_timeout_s=0.05)
    ws, handle, task = await _open(handler, omni)

    await asyncio.wait_for(task, timeout=2.0)
    # Serving makes no session-lifetime decision: the engine lease decides
    # whether a silent, detached session expires.
    assert handle.close_reasons == []
    assert omni.detached == [handle.session_id]


@pytest.mark.asyncio
async def test_idle_timeout_closes_a_non_resumable_session() -> None:
    omni = FakeOmni(idle_timeout_s=0.05, resumable=False)
    handler = _handler(omni, idle_timeout_s=0.05)
    ws, handle, task = await _open(handler, omni)

    await asyncio.wait_for(task, timeout=2.0)
    assert handle.close_reasons == ["disconnect"]
    assert omni.detached == []


@pytest.mark.asyncio
async def test_transport_send_failure_detaches_the_session_instead_of_closing_it() -> None:
    omni = FakeOmni()
    handler = _handler(omni)
    ws, handle, task = await _open(handler, omni)

    # Only the write half dies, so the pump's send is the one and only report
    # of the broken socket (the reader is still parked on receive).
    ws.break_sends()
    handle.deliver(AudioDelta(session_id=handle.session_id, response_id="r1", delta="aGk="))
    await asyncio.sleep(0.05)

    # The engine session is alive and resumable, so this is a disconnect of
    # the current attachment, not a session close; the pump keeps journaling.
    assert handle.close_reasons == []
    assert omni.detached == [handle.session_id]
    assert not handler._pumps[handle.session_id].done()

    # The reader sees the same broken socket a moment later. The attachment is
    # already gone, so this must not detach a second time: another detach would
    # restart the engine's disconnect grace window.
    ws.disconnect()
    await asyncio.wait_for(task, timeout=2.0)
    assert omni.detached == [handle.session_id]
    assert handle.close_reasons == []

    handle.deliver(SessionClosed(session_id=handle.session_id, reason="client_close"))
    await asyncio.sleep(0.05)
    assert handle.session_id not in handler._pumps


@pytest.mark.asyncio
async def test_non_resumable_session_is_closed_on_disconnect() -> None:
    omni = FakeOmni(resumable=False)
    handler = _handler(omni)
    ws, handle, task = await _open(handler, omni)
    assert "resume_token" not in ws.sent[0]

    ws.disconnect()
    await asyncio.wait_for(task, timeout=2.0)
    assert handle.close_reasons == ["disconnect"]
    assert omni.detached == []


@pytest.mark.asyncio
async def test_a_client_close_still_delivers_session_closed_before_the_socket_goes() -> None:
    """The read loop must not outrun the pump's terminal event.

    ``DuplexSessionHandle._deliver`` queues ``SessionClosed`` and marks the
    handle closed in one synchronous step, so ``handle.closed`` is already true
    while the event is still sitting in the outbox. ``_read_loop`` loops on
    exactly that flag: it returned, the endpoint returned, and the ASGI server
    tore the socket down with ``session.closed`` unsent -- the client saw an
    abrupt close (no close frame) instead of the terminal event it was waiting
    for.

    The teardown is what makes this observable, so it is modelled here: a real
    connection stops accepting writes the moment the endpoint returns, which is
    why a pump that is merely *still scheduled* is already too late.
    """
    omni = FakeOmni()
    handler = _handler(omni)
    ws = FakeWebSocket({"duplex": "1", "autostart": "0"})

    async def serve() -> None:
        try:
            await handler.handle_realtime_session(ws)
        finally:
            # The ASGI server drops the connection when the endpoint returns;
            # anything the pump writes after this never reaches the client.
            ws.break_sends()

    task = asyncio.create_task(serve())
    ws.feed(_session_update())
    created = await ws.wait_for("session.created")
    handle = omni.handles[created["session"]["id"]]
    submit = handle.submit

    async def closing_submit(command: commands.DuplexCommand) -> None:
        await submit(command)
        if isinstance(command, commands.CloseSession):
            # Same order as the real handle: queue the event, then flip the flag.
            handle.deliver(SessionClosed(session_id=handle.session_id, reason="client_close"))
            handle.closed = True

    handle.submit = closing_submit  # type: ignore[method-assign]

    ws.feed({"type": "session.close"})
    await asyncio.wait_for(task, timeout=2.0)

    assert ws.types()[-1] == "session.closed", f"terminal event never reached the wire: {ws.types()}"
    assert ws.closed and ws.closed[0][0] == 1000, "the socket must close with a normal close frame"


# --------------------------------------------------------------------------- #
# Resume and takeover                                                         #
# --------------------------------------------------------------------------- #


async def _resume(
    handler: OmniDuplexSessionHandler,
    session_id: str,
    token: str,
    *,
    last_seq: int = 0,
) -> tuple[FakeWebSocket, asyncio.Task]:
    ws = FakeWebSocket({"duplex": "1", "resume": "1"})
    task = asyncio.create_task(handler.handle_realtime_session(ws))
    ws.feed(
        {
            "type": "session.resume",
            "session_id": session_id,
            "resume_token": token,
            "last_received_server_event_seq": last_seq,
        }
    )
    return ws, task


@pytest.mark.asyncio
async def test_resume_after_disconnect_replays_missed_events_and_rotates_the_token() -> None:
    omni = FakeOmni()
    handler = _handler(omni)
    ws, handle, task = await _open(handler, omni)
    token = ws.sent[0]["resume_token"]
    handle.deliver(AudioDelta(session_id=handle.session_id, response_id="r1", delta="one"))
    await ws.wait_for("response.output_audio.delta")
    ws.disconnect()
    await asyncio.wait_for(task, timeout=2.0)
    assert omni.detached == [handle.session_id]
    # Emitted while detached: journaled for replay only.
    handle.deliver(AudioDelta(session_id=handle.session_id, response_id="r1", delta="two"))
    await asyncio.sleep(0.05)

    ws2, task2 = await _resume(handler, handle.session_id, token, last_seq=1)
    resumed = await ws2.wait_for("session.resumed")
    assert resumed["session_id"] == handle.session_id
    assert resumed["attachment_generation"] == 2
    assert resumed["resume_token"] != token
    assert "incarnation" not in resumed
    replayed = await ws2.wait_for("response.output_audio.delta")
    assert replayed["delta"] == "two" and replayed["server_event_seq"] == 2
    assert omni.resumed == [(handle.session_id, 0)]

    # The old token is revoked by the rotation.
    ws3, task3 = await _resume(handler, handle.session_id, token)
    await asyncio.wait_for(task3, timeout=2.0)
    assert ws3.sent[0]["error"]["code"] == "invalid_resume_token"

    ws2.feed({"type": "session.close"})
    await asyncio.sleep(0.05)
    ws2.disconnect()
    await asyncio.wait_for(task2, timeout=2.0)


@pytest.mark.asyncio
async def test_resume_takes_over_a_live_attachment() -> None:
    omni = FakeOmni()
    handler = _handler(omni)
    ws, handle, task = await _open(handler, omni)
    token = ws.sent[0]["resume_token"]

    ws2, task2 = await _resume(handler, handle.session_id, token)
    await ws2.wait_for("session.resumed")
    replaced = await ws.wait_for("session.replaced")
    assert replaced["attachment_generation"] == 1
    assert ws.closed and ws.closed[0][1] == "session_replaced"
    # The replaced socket's later input is ignored; the new one drives the session.
    ws.feed({"type": "input_audio_buffer.clear"})
    ws2.feed({"type": "input_audio_buffer.clear", "event_id": "evt-new"})
    await asyncio.sleep(0.05)
    assert [command.event_id for command in handle.commands] == ["evt-new"]
    await asyncio.wait_for(task, timeout=2.0)

    ws2.disconnect()
    await asyncio.wait_for(task2, timeout=2.0)
    assert omni.detached == [handle.session_id]


@pytest.mark.asyncio
async def test_resume_validation_errors() -> None:
    omni = FakeOmni()
    handler = _handler(omni)
    ws = FakeWebSocket({"duplex": "1", "resume": "1"})
    task = asyncio.create_task(handler.handle_realtime_session(ws))
    ws.feed({"type": "session.resume", "session_id": "duplex-x"})
    await asyncio.wait_for(task, timeout=2.0)
    assert ws.sent[0]["error"]["code"] == "invalid_session_resume"

    ws, task = await _resume(handler, "duplex-unknown", "token")
    await asyncio.wait_for(task, timeout=2.0)
    assert ws.sent[0]["error"]["code"] == "session_resume_expired"


@pytest.mark.asyncio
async def test_journal_overflow_degrades_to_live_delivery_with_resync_required() -> None:
    omni = FakeOmni(replay_max_bytes=256)
    handler = _handler(omni)
    ws, handle, task = await _open(handler, omni)

    handle.deliver(AudioDelta(session_id=handle.session_id, response_id="r1", delta="x" * 400))
    resync = await ws.wait_for("session.resync_required")
    assert resync["reason"] == "journal_overflow"
    delta = await ws.wait_for("response.output_audio.delta")
    assert "server_event_seq" not in delta
    assert ws.types().index("session.resync_required") < ws.types().index("response.output_audio.delta")

    ws.disconnect()
    await asyncio.wait_for(task, timeout=2.0)


# --------------------------------------------------------------------------- #
# Envelope helpers                                                            #
# --------------------------------------------------------------------------- #


def test_realtime_envelope_query_rules() -> None:
    envelope = RealtimeEnvelope.from_query_params({"model": "m"})
    assert envelope.initial_open_payload() == {"model": "m"}
    assert envelope.initial_open_payload() is None  # autostart happens once

    envelope = RealtimeEnvelope.from_query_params({"model": "m", "autostart": "0"})
    assert envelope.resume_only is True
    assert envelope.initial_open_payload() is None
    assert RealtimeEnvelope.from_query_params({"resume": "1"}).resume_only is True
    assert RealtimeEnvelope.from_query_params({"session_id": "mine"}).default_session_payload() == {"model": None}


def test_realtime_envelope_first_message_classification() -> None:
    envelope = RealtimeEnvelope.from_query_params({"model": "m", "autostart": "0"})
    resume = envelope.first_message({"type": "session.resume", "session_id": "s"})
    assert resume.kind == "resume" and resume.resume_payload["session_id"] == "s"

    envelope = RealtimeEnvelope.from_query_params({"model": "m", "autostart": "0"})
    opened = envelope.first_message({"type": "session.update", "session": {"model": "x", "instructions": "hi"}})
    assert opened.kind == "open" and opened.session_payload == {"model": "x", "instructions": "hi"}
    assert opened.pending_command_payload is None

    envelope = RealtimeEnvelope.from_query_params({"model": "m", "autostart": "0"})
    autostarted = envelope.first_message({"type": "input_audio_buffer.commit"})
    assert autostarted.kind == "open" and autostarted.session_payload == {"model": "m"}
    assert autostarted.pending_command_payload == {"type": "input_audio_buffer.commit"}


def test_parse_resume_request_requires_the_three_fields_only() -> None:
    request = parse_resume_request({"session_id": "s", "resume_token": "t", "last_received_server_event_seq": 3})
    assert request is not None
    assert (request.session_id, request.resume_token, request.last_received_server_event_seq) == ("s", "t", 3)
    assert parse_resume_request({"session_id": "s", "resume_token": "t"}).last_received_server_event_seq == 0
    assert parse_resume_request({"session_id": "s", "resume_token": "t", "incarnation": 1}) is not None
    assert parse_resume_request({"session_id": "s"}) is None
    assert parse_resume_request({"session_id": "s", "resume_token": "t", "last_received_server_event_seq": -1}) is None


@pytest.mark.asyncio
async def test_a_resume_that_fails_to_activate_does_not_detach_the_live_attachment() -> None:
    """A rejected resume must roll back only what it owns.

    Two reconnects can both authenticate and both complete the engine resume;
    only one activates. The loser's activation raises, and the rollback used to
    detach the *session*, starting the engine's disconnect grace for the winner
    -- a grace ordinary heartbeats do not clear.

    The activation failure is injected rather than raced for: what is under test
    is the rollback's precondition, not the window that produces it.
    """
    omni = FakeOmni()
    handler = _handler(omni)
    ws, handle, task = await _open(handler, omni)
    token = ws.sent[0]["resume_token"]
    registry = handler._attachment_registry
    try:
        assert await registry.has_attachment(handle.session_id), "the opener is attached"
        detached_before = list(omni.detached)

        async def failing_resume(*args, **kwargs):
            raise RuntimeError("transport activation lost the race")

        original_resume = registry.resume
        registry.resume = failing_resume  # type: ignore[method-assign]
        try:
            ws_lose, task_lose = await _resume(handler, handle.session_id, token)
            error = await ws_lose.wait_for("error")
        finally:
            registry.resume = original_resume  # type: ignore[method-assign]

        assert error["error"]["code"] == "session_resume_conflict"
        assert omni.detached == detached_before, "the loser must not detach the winner"
        assert await registry.has_attachment(handle.session_id), "the winner is still attached"
    finally:
        for pending in (task, locals().get("task_lose")):
            if pending is not None:
                pending.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await pending


# --------------------------------------------------------------------------- #
# Cancelled resume and the engine disconnect grace (#7636 Issue 3)            #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_resume_cancelled_during_activation_puts_the_engine_lease_back_into_grace() -> None:
    """``_resume`` resumes the engine lease first; a cancel mid-activation must detach it again.

    The rollback used to catch ``Exception`` only. A handler task cancelled
    while sending ``session.resumed`` or a replay entry left the engine with
    ``detached_at=None`` and the outer handler with no attachment to clean up:
    the reaper reclaimed nothing after the grace, and with one slot the next
    open was refused with ``resource_exhausted``.
    """
    omni = FakeOmni()
    handler = _handler(omni)
    ws, handle, task = await _open(handler, omni)
    token = ws.sent[0]["resume_token"]
    ws.disconnect()
    await asyncio.wait_for(task, timeout=2.0)
    assert omni.detached == [handle.session_id]
    registry = handler._attachment_registry

    activating = asyncio.Event()

    async def parked_resume(*args, **kwargs):
        activating.set()
        await asyncio.Event().wait()

    original_resume = registry.resume
    registry.resume = parked_resume  # type: ignore[method-assign]
    try:
        ws2, task2 = await _resume(handler, handle.session_id, token)
        await asyncio.wait_for(activating.wait(), timeout=2.0)
        assert omni.resumed == [(handle.session_id, 0)], "the engine lease was resumed before activation"

        task2.cancel()
        with suppress(asyncio.CancelledError):
            await task2
    finally:
        registry.resume = original_resume  # type: ignore[method-assign]

    # The lease is detached again, so the disconnect grace runs for it.
    assert omni.detached == [handle.session_id, handle.session_id]
    assert not await registry.has_attachment(handle.session_id)
    assert handle.close_reasons == []
    # And the session is still resumable with the token the cancelled attempt presented.
    ws3, task3 = await _resume(handler, handle.session_id, token)
    resumed = await ws3.wait_for("session.resumed")
    assert resumed["session_id"] == handle.session_id
    ws3.disconnect()
    await asyncio.wait_for(task3, timeout=2.0)


@pytest.mark.asyncio
async def test_a_resume_cancelled_after_activation_detaches_the_attachment_it_made() -> None:
    """A cancel after the registry activated the new attachment must undo that attachment too.

    Otherwise the session stays attached to a socket nobody serves, with the
    engine lease resumed: no grace, no expiry, until the idle TTL.
    """
    omni = FakeOmni()
    handler = _handler(omni)
    ws, handle, task = await _open(handler, omni)
    token = ws.sent[0]["resume_token"]
    registry = handler._attachment_registry

    # The takeover notifies the replaced socket after activation; park there.
    notifying = asyncio.Event()

    async def parked_send(payload: dict[str, Any]) -> None:
        notifying.set()
        await asyncio.Event().wait()

    ws.send_json = parked_send  # type: ignore[method-assign]
    ws2, task2 = await _resume(handler, handle.session_id, token)
    await asyncio.wait_for(notifying.wait(), timeout=2.0)
    assert await registry.is_current_attachment(handle.session_id, 2), "the new socket is attached"

    task2.cancel()
    with suppress(asyncio.CancelledError):
        await task2

    assert not await registry.has_attachment(handle.session_id)
    assert omni.detached == [handle.session_id]
    assert handle.close_reasons == []

    task.cancel()
    with suppress(asyncio.CancelledError, Exception):
        await task


@pytest.mark.asyncio
async def test_an_abandoned_resume_never_detaches_the_lease_a_later_resume_owns() -> None:
    """Two connections resuming the same session: the loser's rollback must not touch the winner's lease.

    Interleaving: A has activated and is notifying the socket it replaced; B,
    holding A's rotated token, has completed its engine resume but its RPC
    result has not reached the handler yet, so nothing is attached for B.
    Cancelling A detaches A's registry generation, which used to make the
    engine detach unconditional: it landed on the lease B had just resumed,
    and the reaper expired B after the grace even though it heartbeats. The
    engine now refuses a detach fenced on a generation it has moved past.
    """
    omni = FakeOmni()
    handler = _handler(omni)
    ws, handle, task = await _open(handler, omni)
    token = ws.sent[0]["resume_token"]
    registry = handler._attachment_registry

    # A takes over and parks while notifying the replaced socket (after activation).
    notifying = asyncio.Event()

    async def parked_send(payload: dict[str, Any]) -> None:
        notifying.set()
        await asyncio.Event().wait()

    ws.send_json = parked_send  # type: ignore[method-assign]
    ws_a, task_a = await _resume(handler, handle.session_id, token)
    await asyncio.wait_for(notifying.wait(), timeout=2.0)
    resumed_a = await ws_a.wait_for("session.resumed")
    assert handle.lease_generation == 1

    # B resumes with A's rotated token: the engine applies it, the result is still in flight.
    omni.resume_gate = asyncio.Event()
    ws_b, task_b = await _resume(handler, handle.session_id, resumed_a["resume_token"])
    await asyncio.wait_for(omni.resume_started.wait(), timeout=2.0)
    assert handle.lease_generation == 2

    task_a.cancel()
    with suppress(asyncio.CancelledError):
        await task_a
    assert omni.detached == [], "A's rollback must not detach the lease B resumed"
    assert omni.detach_refused == [(handle.session_id, 1)]
    assert not await registry.has_attachment(handle.session_id), "A's attachment is gone"

    omni.resume_gate.set()
    resumed_b = await ws_b.wait_for("session.resumed")
    assert resumed_b["attachment_generation"] == 3
    assert await registry.is_current_attachment(handle.session_id, 3)
    assert omni.detached == []

    # B's own disconnect detaches the lease B holds.
    ws_b.disconnect()
    await asyncio.wait_for(task_b, timeout=2.0)
    assert omni.detached == [handle.session_id]
    assert omni.detach_refused == [(handle.session_id, 1)]
    task.cancel()
    with suppress(asyncio.CancelledError, Exception):
        await task


@pytest.mark.asyncio
async def test_an_abandoned_takeover_leaves_the_live_connection_serving_and_owning_the_lease() -> None:
    """A takeover cancelled mid-RPC must not put the connection it never replaced into disconnect grace.

    A is attached at generation 0. B's engine resume lands generation 1 and B
    is cancelled before it reaches the registry, so A stays the current
    attachment. Settling B's generation must hand it to A rather than detach
    it: A keeps serving, and A's own disconnect later detaches generation 1,
    which is the lease the engine actually holds.
    """
    omni = FakeOmni()
    handler = _handler(omni)
    ws, handle, task = await _open(handler, omni)
    token = ws.sent[0]["resume_token"]
    registry = handler._attachment_registry

    omni.resume_gate = asyncio.Event()
    ws_b, task_b = await _resume(handler, handle.session_id, token)
    await asyncio.wait_for(omni.resume_started.wait(), timeout=2.0)
    assert handle.lease_generation == 1

    task_b.cancel()
    with suppress(asyncio.CancelledError):
        await task_b
    omni.resume_gate.set()
    await asyncio.gather(*omni.compensations)

    assert omni.detached == [], "A is still serving: its lease must not enter disconnect grace"
    assert omni.detach_refused == []
    assert await registry.is_current_attachment(handle.session_id, 1), "A is still the attachment"
    handle.deliver(AudioDelta(session_id=handle.session_id, response_id="r1", delta="aGk="))
    await ws.wait_for("response.output_audio.delta")

    # A now owns generation 1: its disconnect detaches the lease the engine holds.
    ws.disconnect()
    await asyncio.wait_for(task, timeout=2.0)
    assert omni.detached == [handle.session_id]
    assert omni.detach_refused == []


@pytest.mark.asyncio
async def test_a_send_failure_detaches_the_failed_sockets_own_lease_not_a_pending_resumes() -> None:
    """The pump's detach is fenced on the lease of the socket whose send failed, captured with it.

    The pump holds the registry's outbound lock while sending to A. B's
    engine resume lands generation 1 meanwhile and B waits for that lock to
    activate. When A's send fails, the pump drops A and must detach A's
    generation 0, which the engine refuses, not B's generation 1, which would
    put B into disconnect grace the moment it activates.
    """
    omni = FakeOmni()
    handler = _handler(omni)
    ws, handle, task = await _open(handler, omni)
    token = ws.sent[0]["resume_token"]
    registry = handler._attachment_registry

    sending = asyncio.Event()
    fail_send = asyncio.Event()

    async def parked_failing_send(payload: dict[str, Any]) -> None:
        sending.set()
        await fail_send.wait()
        raise RuntimeError("socket closed")

    ws.send_json = parked_failing_send  # type: ignore[method-assign]
    handle.deliver(AudioDelta(session_id=handle.session_id, response_id="r1", delta="aGk="))
    await asyncio.wait_for(sending.wait(), timeout=2.0)

    # B: the engine resume lands; activation waits for the outbound lock the pump holds.
    ws_b, task_b = await _resume(handler, handle.session_id, token)
    await asyncio.sleep(0.05)
    assert omni.resumed == [(handle.session_id, 0)]
    assert handle.lease_generation == 1
    assert await registry.is_current_attachment(handle.session_id, 1), "B has not activated yet"

    fail_send.set()
    resumed_b = await ws_b.wait_for("session.resumed")
    assert resumed_b["attachment_generation"] == 2
    assert omni.detached == [], "A's lease (generation 0) is refused, B's lease is untouched"
    assert omni.detach_refused == [(handle.session_id, 0)]
    assert await registry.is_current_attachment(handle.session_id, 2)

    # B serves; its own disconnect detaches the lease it resumed.
    ws_b.disconnect()
    await asyncio.wait_for(task_b, timeout=2.0)
    assert omni.detached == [handle.session_id]
    ws.disconnect()
    with suppress(asyncio.CancelledError, Exception):
        await asyncio.wait_for(task, timeout=2.0)


@pytest.mark.asyncio
async def test_an_abandoned_resume_goes_to_the_resume_waiting_to_activate_not_the_socket_it_replaces() -> None:
    """Three connections: A sending, B resumed and waiting for the outbound lock, C abandoned mid-RPC.

    C's generation 2 must be parked for B's activation. If it were handed to
    A instead, A's send failure would detach generation 2 (accepted by the
    engine), and B would then activate on a lease already in disconnect grace
    and expire despite heartbeating.
    """
    omni = FakeOmni()
    handler = _handler(omni)
    ws, handle, task = await _open(handler, omni)
    token = ws.sent[0]["resume_token"]
    registry = handler._attachment_registry

    sending = asyncio.Event()
    fail_send = asyncio.Event()

    async def parked_failing_send(payload: dict[str, Any]) -> None:
        sending.set()
        await fail_send.wait()
        raise RuntimeError("socket closed")

    ws.send_json = parked_failing_send  # type: ignore[method-assign]
    handle.deliver(AudioDelta(session_id=handle.session_id, response_id="r1", delta="aGk="))
    await asyncio.wait_for(sending.wait(), timeout=2.0)

    # B: engine resume lands generation 1; activation waits for the lock the pump holds.
    ws_b, task_b = await _resume(handler, handle.session_id, token)
    await asyncio.sleep(0.05)
    assert handle.lease_generation == 1

    # C: same unrotated token, engine resume lands generation 2, cancelled mid-RPC.
    omni.resume_gate = asyncio.Event()
    ws_c, task_c = await _resume(handler, handle.session_id, token)
    await asyncio.wait_for(omni.resume_started.wait(), timeout=2.0)
    assert handle.lease_generation == 2
    task_c.cancel()
    with suppress(asyncio.CancelledError):
        await task_c
    omni.resume_gate.set()
    await asyncio.gather(*omni.compensations)
    assert omni.detached == [] and omni.detach_refused == [], "generation 2 is parked for B"

    # A's send fails: the pump drops A and detaches A's own generation 0, refused.
    fail_send.set()
    resumed_b = await ws_b.wait_for("session.resumed")
    assert resumed_b["attachment_generation"] == 2
    assert omni.detach_refused == [(handle.session_id, 0)]
    assert omni.detached == [], "the lease B serves is untouched"
    assert await registry.is_current_attachment(handle.session_id, 2)

    # B serves generation 2 (inherited from C): its disconnect detaches that lease.
    ws_b.disconnect()
    await asyncio.wait_for(task_b, timeout=2.0)
    assert omni.detached == [handle.session_id]
    assert omni.detach_refused == [(handle.session_id, 0)]
    ws.disconnect()
    with suppress(asyncio.CancelledError, Exception):
        await asyncio.wait_for(task, timeout=2.0)


@pytest.mark.asyncio
async def test_a_resume_cancelled_during_replay_settles_the_newer_lease_it_was_handed() -> None:
    """Activation rollback must give back the lease the provisional attachment actually owned.

    A resumed generation 1, delivered ``session.resumed`` and is blocked
    replaying events. B, with A's rotated token, resumed generation 2 and was
    cancelled mid-RPC; its settlement handed 2 to A. When A is then cancelled
    during replay, the registry rolls A back; detaching A's own generation 1
    would be refused and generation 2 would never enter disconnect grace.
    """
    omni = FakeOmni()
    handler = _handler(omni)
    ws, handle, task = await _open(handler, omni)
    token = ws.sent[0]["resume_token"]
    registry = handler._attachment_registry
    ws.disconnect()
    await asyncio.wait_for(task, timeout=2.0)
    assert omni.detached == [handle.session_id]
    # Journaled while detached: A's activation has something to replay.
    handle.deliver(AudioDelta(session_id=handle.session_id, response_id="r1", delta="one"))
    await asyncio.sleep(0.05)

    # A: session.resumed goes out, the replay entry parks.
    ws_a = FakeWebSocket({"duplex": "1", "resume": "1"})
    replaying = asyncio.Event()
    sends = 0
    original_send = ws_a.send_json

    async def send_parking_on_replay(payload: dict[str, Any]) -> None:
        nonlocal sends
        sends += 1
        if sends == 2:
            replaying.set()
            await asyncio.Event().wait()
        await original_send(payload)

    ws_a.send_json = send_parking_on_replay  # type: ignore[method-assign]
    task_a = asyncio.create_task(handler.handle_realtime_session(ws_a))
    ws_a.feed(
        {
            "type": "session.resume",
            "session_id": handle.session_id,
            "resume_token": token,
            "last_received_server_event_seq": 0,
        }
    )
    await asyncio.wait_for(replaying.wait(), timeout=2.0)
    resumed_a = await ws_a.wait_for("session.resumed")
    assert handle.lease_generation == 1

    # B: A's rotated token, engine resume lands generation 2, cancelled mid-RPC.
    omni.resume_gate = asyncio.Event()
    ws_b, task_b = await _resume(handler, handle.session_id, resumed_a["resume_token"])
    await asyncio.wait_for(omni.resume_started.wait(), timeout=2.0)
    assert handle.lease_generation == 2
    task_b.cancel()
    with suppress(asyncio.CancelledError):
        await task_b
    omni.resume_gate.set()
    await asyncio.gather(*omni.compensations)
    assert omni.detached == [handle.session_id], "generation 2 went to A's pending activation"

    # A is cancelled during replay: its rollback settles generation 2, the lease A owned.
    task_a.cancel()
    with suppress(asyncio.CancelledError):
        await task_a
    assert not await registry.has_attachment(handle.session_id)
    assert omni.detached == [handle.session_id, handle.session_id]
    assert omni.detach_refused == [], "generation 1 was never detached: the engine would refuse it"
