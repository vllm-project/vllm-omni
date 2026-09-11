# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The thin websocket handler: handshake, command translation, event pump, resume/takeover."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import pytest
from fastapi import WebSocketDisconnect

from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.config import DuplexCapabilities
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
        self._outbox: asyncio.Queue[DuplexEvent] = asyncio.Queue()

    def deliver(self, event: DuplexEvent) -> None:
        self._outbox.put_nowait(event)

    async def submit(self, command: commands.DuplexCommand) -> None:
        if self.closed:
            raise DuplexSessionError("closed", code="session_closed", session_id=self.session_id)
        self.commands.append(command)

    async def events(self):
        while True:
            event = await self._outbox.get()
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
        self.open_error: DuplexSessionError | None = None

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

    async def resume_session(self, session_id: str, *, expected_lease_generation: int) -> FakeHandle:
        self.resumed.append((session_id, expected_lease_generation))
        handle = self.handles[session_id]
        handle.lease_generation = expected_lease_generation + 1
        return handle

    async def detach_session(self, session_id: str) -> None:
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
