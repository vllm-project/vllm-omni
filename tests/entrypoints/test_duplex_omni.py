# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""``DuplexOmni`` / ``DuplexSessionHandle``: the thin Python API over a fake engine."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from tests.entrypoints.test_omni_entrypoints import FakeAsyncOmniEngine
from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
from vllm_omni.engine.duplex.events import (
    AudioDelta,
    DuplexEvent,
    SessionClosed,
    SessionCreated,
    SessionExpired,
)
from vllm_omni.engine.duplex.messages import (
    DuplexControlResultMessage,
    DuplexSessionError,
    DuplexSessionEventMessage,
)
from vllm_omni.engine.messages import ErrorMessage
from vllm_omni.entrypoints.duplex_omni import DuplexOmni, DuplexSessionHandle

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeDuplexEngine(FakeAsyncOmniEngine):
    """The engine surface ``DuplexOmni`` uses; every call is recorded."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.duplex_session_config = DuplexSessionRuntimeConfig(max_sessions=2)
        self.duplex_capabilities = DuplexCapabilities(supports_session_resume=True)
        self.opened: list[tuple[str, DuplexSessionConfig]] = []
        self.closed: list[tuple[str, str]] = []
        self.resumed: list[tuple[str, int]] = []
        self.touched: list[tuple[str, str]] = []
        self.commands: list[tuple[str, commands.DuplexCommand]] = []
        self.open_error: DuplexSessionError | None = None
        self.close_error: DuplexSessionError | None = None

    def _result(self, operation: str, session_id: str, **fields: Any) -> DuplexControlResultMessage:
        return DuplexControlResultMessage(control_id="c", operation=operation, session_id=session_id, ok=True, **fields)

    async def open_session_async(self, session_id, session_config, *, timeout=None):
        self.opened.append((session_id, session_config))
        if self.open_error is not None:
            raise self.open_error
        return self._result(
            "open",
            session_id,
            capabilities=self.duplex_capabilities,
            public_session={"id": session_id, "idle_timeout_s": 7},
            lease_generation=0,
        )

    async def close_session_async(self, session_id, *, reason="client_close", timeout=None):
        self.closed.append((session_id, reason))
        if self.close_error is not None:
            raise self.close_error
        # The runner emits session.closed before answering the RPC.
        self.emit(session_id, SessionClosed(reason=reason))
        return self._result("close", session_id)

    async def resume_session_async(self, session_id, *, expected_lease_generation, timeout=None):
        self.resumed.append((session_id, expected_lease_generation))
        return self._result("resume", session_id, lease_generation=expected_lease_generation + 1)

    async def touch_session_async(self, session_id, *, activity, timeout=None):
        self.touched.append((session_id, activity))
        return self._result("touch", session_id)

    async def submit_command_async(self, session_id, command):
        self.commands.append((session_id, command))

    def emit(self, session_id: str, event: DuplexEvent) -> None:
        # The session manager binds the session identity to every event it emits.
        event = replace(event, session_id=session_id)
        self.output_q.put(DuplexSessionEventMessage(session_id=session_id, event=event))


def _make_omni(monkeypatch: pytest.MonkeyPatch) -> tuple[DuplexOmni, FakeDuplexEngine]:
    engine = FakeDuplexEngine()
    monkeypatch.setattr("vllm_omni.entrypoints.duplex_omni.DuplexOmniEngine", lambda *args, **kwargs: engine)
    monkeypatch.setattr("vllm_omni.entrypoints.omni_base.omni_snapshot_download", lambda model: model)
    return DuplexOmni("dummy-model"), engine


async def _collect(handle: DuplexSessionHandle, count: int, *, timeout_s: float = 2.0) -> list[DuplexEvent]:
    events: list[DuplexEvent] = []

    async def consume() -> None:
        async for event in handle.events():
            events.append(event)
            if len(events) >= count:
                break

    await asyncio.wait_for(consume(), timeout=timeout_s)
    return events


@pytest.mark.asyncio
async def test_open_session_allocates_the_id_and_ignores_client_ids(monkeypatch) -> None:
    omni, engine = _make_omni(monkeypatch)
    try:
        handle = await omni.open_session({"session_id": "mine", "id": "mine-too", "instructions": "hi"})

        assert handle.session_id.startswith("duplex-") and len(handle.session_id) == len("duplex-") + 32
        assert handle.session_id != "mine"
        ((session_id, config),) = engine.opened
        assert session_id == handle.session_id
        assert isinstance(config, DuplexSessionConfig)
        assert config.model == "dummy-model" and config.instructions == "hi"
        assert handle.capabilities is engine.duplex_capabilities
        assert handle.public_session["id"] == handle.session_id
        assert omni.sessions == {handle.session_id: handle}
        assert omni.get_session(handle.session_id) is handle
        assert omni.active_session_count() == 1

        second = await omni.open_session(DuplexSessionConfig(instructions="typed"))
        assert second.session_id != handle.session_id
        assert engine.opened[1][1].model == "dummy-model"
        assert omni.duplex_session_config.max_sessions == 2
        assert omni.duplex_capabilities.supports_session_resume is True
    finally:
        omni.shutdown()


@pytest.mark.asyncio
async def test_open_failure_drops_the_pending_handle_and_raises_the_engine_code(monkeypatch) -> None:
    omni, engine = _make_omni(monkeypatch)
    engine.open_error = DuplexSessionError("full", code="resource_exhausted", retryable=True)
    try:
        with pytest.raises(DuplexSessionError) as excinfo:
            await omni.open_session()
        assert excinfo.value.code == "resource_exhausted"
        assert omni.sessions == {}
    finally:
        omni.shutdown()


@pytest.mark.asyncio
async def test_events_are_routed_to_their_handle_and_end_on_session_closed(monkeypatch) -> None:
    omni, engine = _make_omni(monkeypatch)
    try:
        handle = await omni.open_session()
        engine.emit("duplex-unknown", AudioDelta(delta="zzz"))  # dropped
        engine.emit(handle.session_id, SessionCreated(session={"id": handle.session_id}))
        engine.emit(handle.session_id, AudioDelta(response_id="r1", delta="abc"))
        engine.emit(handle.session_id, SessionClosed(reason="client_close"))

        events = await _collect(handle, 10)

        assert [event.type for event in events] == ["session.created", "response.output_audio.delta", "session.closed"]
        assert events[0].session_id == handle.session_id
        assert handle.closed and handle.close_reason == "client_close"
        assert await handle.wait_closed() == "client_close"
        assert handle.session_id not in omni.sessions
        with pytest.raises(DuplexSessionError) as excinfo:
            await handle.submit(commands.Heartbeat())
        assert excinfo.value.code == "session_closed"
    finally:
        omni.shutdown()


@pytest.mark.asyncio
async def test_events_is_single_consumer_but_may_be_reentered(monkeypatch) -> None:
    omni, engine = _make_omni(monkeypatch)
    try:
        handle = await omni.open_session()
        engine.emit(handle.session_id, AudioDelta(delta="one"))
        first = handle.events()
        assert (await first.__anext__()).type == "response.output_audio.delta"
        with pytest.raises(RuntimeError, match="active events"):
            await handle.events().__anext__()
        await first.aclose()

        engine.emit(handle.session_id, SessionExpired(reason="idle_ttl_expired"))
        events = await _collect(handle, 1)
        assert events[0].type == "session.expired" and events[0].is_terminal
        assert handle.closed and handle.close_reason == "idle_ttl_expired"
    finally:
        omni.shutdown()


@pytest.mark.asyncio
async def test_close_session_waits_for_the_typed_session_closed_event(monkeypatch) -> None:
    omni, engine = _make_omni(monkeypatch)
    try:
        handle = await omni.open_session()
        consumer = asyncio.create_task(_collect(handle, 1))
        await handle.close(reason="done")

        assert engine.closed == [(handle.session_id, "done")]
        assert handle.closed and handle.close_reason == "done"
        assert (await consumer)[0].type == "session.closed"
        # Idempotent: a second close is a no-op.
        await omni.close_session(handle.session_id)
        assert len(engine.closed) == 1
    finally:
        omni.shutdown()


@pytest.mark.asyncio
async def test_close_failure_marks_the_handle_closed_and_propagates(monkeypatch) -> None:
    omni, engine = _make_omni(monkeypatch)
    engine.close_error = DuplexSessionError("boom", code="engine_error")
    try:
        handle = await omni.open_session()
        with pytest.raises(DuplexSessionError, match="boom"):
            await omni.close_session(handle.session_id)
        assert handle.closed
        # A failed close must not leave a dead handle shadowing the id.
        assert omni.sessions == {}
    finally:
        omni.shutdown()


@pytest.mark.asyncio
async def test_engine_death_ends_every_session_handle(monkeypatch) -> None:
    omni, engine = _make_omni(monkeypatch)
    try:
        handle = await omni.open_session()
        engine.output_q.put(ErrorMessage(error="orchestrator died", fatal=True))

        await asyncio.wait_for(handle.wait_closed(), timeout=2.0)
        assert handle.closed
        assert "engine_dead" in (handle.close_reason or "")
        assert omni.sessions == {}
    finally:
        omni.shutdown()


@pytest.mark.asyncio
async def test_resume_detach_and_touch_forward_to_the_engine(monkeypatch) -> None:
    omni, engine = _make_omni(monkeypatch)
    try:
        handle = await omni.open_session()
        resumed = await omni.resume_session(handle.session_id, expected_lease_generation=0)
        assert resumed is handle and handle.lease_generation == 1
        assert engine.resumed == [(handle.session_id, 0)]

        await omni.detach_session(handle.session_id)
        await omni.touch_session(handle.session_id, activity="heartbeat")
        assert engine.touched == [(handle.session_id, "detach"), (handle.session_id, "heartbeat")]

        for call in (
            omni.resume_session("duplex-nope", expected_lease_generation=0),
            omni.detach_session("duplex-nope"),
        ):
            with pytest.raises(DuplexSessionError) as excinfo:
                await call
            assert excinfo.value.code == "unknown_session"
    finally:
        omni.shutdown()


@pytest.mark.asyncio
async def test_handle_wrappers_build_typed_commands_in_caller_order(monkeypatch) -> None:
    omni, engine = _make_omni(monkeypatch)
    try:
        handle = await omni.open_session()
        await handle.append_audio(b"\x00\x01", sample_rate_hz=16000, is_speech=True)
        await handle.commit(create_response=True)
        await handle.create_response({"instructions": "x"})
        await handle.cancel_response("resp-1")
        await handle.barge_in()
        await handle.ack_playback(120.7, response_id="resp-1", committed_ms=100)
        await handle.update({"temperature": 0.2})
        await handle.create_item({"id": "item_1", "type": "message"}, previous_item_id="root")
        await handle.delete_item("item_1")
        await handle.truncate_item("item_1", audio_end_ms=50)
        await handle.signal_turn("user_started", {"x": 1})
        await handle.heartbeat()

        submitted = [command for session_id, command in engine.commands if session_id == handle.session_id]
        assert [type(command) for command in submitted] == [
            commands.AppendAudio,
            commands.Commit,
            commands.CreateResponse,
            commands.CancelResponse,
            commands.BargeIn,
            commands.AckPlayback,
            commands.UpdateSession,
            commands.CreateItem,
            commands.DeleteItem,
            commands.TruncateItem,
            commands.SignalTurn,
            commands.Heartbeat,
        ]
        append, commit, create, cancel, _, ack, update, item, delete, truncate, signal, _ = submitted
        assert append.audio == b"\x00\x01" and append.sample_rate_hz == 16000 and append.is_speech is True
        assert commit.create_response is True and commit.final is True
        assert create.options == {"instructions": "x"}
        assert cancel.response_id == "resp-1"
        assert ack.played_ms == 120 and ack.committed_ms == 100 and ack.response_id == "resp-1"
        assert update.patch == {"temperature": 0.2}
        assert item.previous_item_id == "root" and item.item["id"] == "item_1"
        assert delete.item_id == "item_1"
        assert truncate.audio_end_ms == 50
        assert signal.event == "user_started" and signal.signal_payload == {"x": 1}
    finally:
        omni.shutdown()


@pytest.mark.asyncio
async def test_shutdown_closes_every_handle(monkeypatch) -> None:
    omni, engine = _make_omni(monkeypatch)
    first = await omni.open_session()
    second = await omni.open_session()

    omni.shutdown()

    assert first.closed and second.closed
    assert first.close_reason == "shutdown"
    assert omni.sessions == {}
    assert engine.shutdown_called
