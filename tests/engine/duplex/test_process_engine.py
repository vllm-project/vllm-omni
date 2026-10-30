# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Real spawn and IPC coverage without model weights or GPU execution."""

from __future__ import annotations

import asyncio
import os
import socket
from dataclasses import dataclass
from unittest.mock import AsyncMock, Mock

import msgspec
import pytest
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex.commands import AppendAudio, AppendText, ClearOutputAudio, UpdateSession
from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
from vllm_omni.engine.duplex.delivery import DuplexOutputBuffer
from vllm_omni.engine.duplex.events import AudioDelta, OutputAudioCleared, SessionClosed, SessionCreated
from vllm_omni.engine.duplex.messages import DuplexControlResultMessage, DuplexSessionError, DuplexSessionEventMessage
from vllm_omni.engine.duplex.process_engine import (
    _QUEUE_BYTES,
    ProcessDuplexEngineClient,
    ProcessDuplexStageMetadata,
    _DuplexProcessWorker,
    _MessageSocket,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeEngine:
    def __init__(self, small_buffers=False):
        self.duplex_session_config = DuplexSessionRuntimeConfig(
            max_pending_output_events_per_session=2 if small_buffers else 512,
        )
        self.duplex_capabilities = DuplexCapabilities(supports_chat_completions=True, supports_session_resume=True)
        self.stage_configs = [ProcessDuplexStageMetadata(i, "llm", {"max_model_len": 2048}) for i in range(4)]
        self.stage_vllm_configs = [None] * 4
        self.num_stages = 4  # Not restricted to Moshi's two/three-stage deployments.
        self.config_path = None
        self.endpoint_restrictions = ()
        self.duplex_frontend_group_count = 2
        self.duplex_frontend_max_sessions = 8
        self.buffers = {}
        self.output: asyncio.Queue[DuplexSessionEventMessage] = asyncio.Queue()
        self.lease = {}

    def result(self, session_id, operation):
        return DuplexControlResultMessage(
            control_id="test",
            operation=operation,
            session_id=session_id,
            ok=True,
            capabilities=self.duplex_capabilities,
            public_session={"worker_pid": os.getpid(), "worker_loop": type(asyncio.get_running_loop()).__module__},
            lease_generation=self.lease.get(session_id, 0),
        )

    async def open_session_async(self, session_id, config, *, output_buffer, **kwargs):
        if config.extra_body.get("reject"):
            raise DuplexSessionError("admission rejected", code="capacity", retryable=True, session_id=session_id)
        if config.extra_body.get("slow_open"):
            await asyncio.sleep(0.1)
        self.buffers[session_id] = output_buffer
        output_buffer.put(SessionCreated(session_id=session_id, session={"id": session_id}))
        return self.result(session_id, "open")

    async def submit_command_async(self, session_id, command):
        import pybase64 as base64

        buffer = self.buffers[session_id]
        if isinstance(command, AppendAudio):
            buffer.put(
                AudioDelta(
                    session_id=session_id, response_id="r", epoch=0, delta=base64.b64encode(command.audio).decode()
                )
            )
        elif isinstance(command, ClearOutputAudio):
            buffer.invalidate("r", 0)
            buffer.put(OutputAudioCleared(session_id=session_id, response_id="r", epoch=1))
            buffer.put(AudioDelta(session_id=session_id, response_id="r", epoch=0, delta="c3RhbGU="))
            buffer.put(AudioDelta(session_id=session_id, response_id="r", epoch=1, delta="ZnJlc2g="))
        elif isinstance(command, AppendText) and command.text == "crash":
            os._exit(17)
        elif isinstance(command, AppendText) and command.text == "bad_output":
            self.output.put_nowait("unexpected engine output")
        elif (isinstance(command, AppendText) and command.text == "reject") or (
            isinstance(command, UpdateSession) and command.patch.get("instructions") == "reject"
        ):
            raise DuplexSessionError("queue full", code="engine_backpressure", retryable=True)

    async def close_session_async(self, session_id, *, reason, **kwargs):
        await self.output.put(
            DuplexSessionEventMessage(session_id=session_id, event=SessionClosed(session_id=session_id, reason=reason))
        )
        return self.result(session_id, "close")

    async def resume_session_async(self, session_id, *, expected_lease_generation, control_id=None, **kwargs):
        key = (session_id, control_id)
        if key not in self.lease:
            self.lease[session_id] = expected_lease_generation + 1
            self.lease[key] = True
        return self.result(session_id, "resume")

    async def touch_session_async(self, session_id, **kwargs):
        if kwargs.get("activity") == "hang":
            await asyncio.Future()
        return self.result(session_id, "touch")

    async def get_outputs_async(self):
        return [await self.output.get()]


@dataclass
class FakeRuntime:
    model: str
    engine: FakeEngine
    process_warmup_audio: dict[str, object]
    health_error: bool = False
    shutdown_path: str | None = None

    async def check_health(self):
        if self.health_error:
            raise EngineDeadError("Stage-2 has no live replica")

    def shutdown(self):
        if self.shutdown_path is not None:
            with open(self.shutdown_path, "w") as marker:
                marker.write("closed")


def fake_factory(model, options):
    return FakeRuntime(
        model,
        FakeEngine(options.get("small_buffers", False)),
        {"audio": "AAAAAA==", "format": "pcm_f32le", "sample_rate_hz": 24000},
        health_error=options.get("health_error", False),
        shutdown_path=options.get("shutdown_path"),
    )


def failing_factory(model, options):
    raise RuntimeError("test startup failure")


def make_client(**options):
    return ProcessDuplexEngineClient("fake", {"init_timeout": 30, **options}, factory=fake_factory)


def test_transport_streams_both_directions():
    async def run():
        a, b = socket.socketpair()
        a.setblocking(False)
        b.setblocking(False)
        ar, aw = await asyncio.open_connection(sock=a)
        br, bw = await asyncio.open_connection(sock=b)
        left, right = _MessageSocket(ar, aw), _MessageSocket(br, bw)
        for i in range(64):
            await left.send(("command", i, b"pcm"))
            await right.send(("event", i, b"audio"))
        tasks = [asyncio.create_task(left.write()), asyncio.create_task(right.write())]
        try:
            assert [await right.receive() for _ in range(64)] == [["command", i, b"pcm"] for i in range(64)]
            assert [await left.receive() for _ in range(64)] == [["event", i, b"audio"] for i in range(64)]
            assert left.pending_bytes == right.pending_bytes == 0
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            aw.close()
            bw.close()
            await asyncio.gather(aw.wait_closed(), bw.wait_closed())

    asyncio.run(run())


@pytest.mark.parametrize("partial_eof", [False, True])
def test_transport_handles_partial_and_consecutive_messages(partial_eof):
    async def run():
        a, b = socket.socketpair()
        a.setblocking(False)
        b.setblocking(False)
        ar, aw = await asyncio.open_connection(sock=a)
        br, bw = await asyncio.open_connection(sock=b)
        receiver = _MessageSocket(br, bw)
        # A value larger than the read chunk must survive multiple feed() calls.
        first = ("event", b"audio" * 20000)
        payload = msgspec.msgpack.encode(first)
        receive = asyncio.create_task(receiver.receive())
        try:
            aw.write(payload[:-1])
            await aw.drain()
            await asyncio.sleep(0)
            assert not receive.done()
            if partial_eof:
                aw.close()
                with pytest.raises(EOFError):
                    await asyncio.wait_for(receive, 5)
            else:
                # Finish the partial value and append another in the same write.
                aw.write(payload[-1:] + msgspec.msgpack.encode(("result", 1, True, None)))
                await aw.drain()
                assert await asyncio.wait_for(receive, 5) == list(first)
                assert await asyncio.wait_for(receiver.receive(), 5) == ["result", 1, True, None]
                aw.close()
                with pytest.raises(EOFError):
                    await asyncio.wait_for(receiver.receive(), 5)
        finally:
            receive.cancel()
            await asyncio.gather(receive, return_exceptions=True)
            aw.close()
            bw.close()
            await asyncio.gather(aw.wait_closed(), bw.wait_closed())

    asyncio.run(run())


@pytest.mark.parametrize("limit", ["messages", "bytes"])
def test_transport_close_wakes_blocked_senders(monkeypatch, limit):
    async def run():
        import vllm_omni.engine.duplex.process_engine as transport

        message = ("command", b"pcm")
        monkeypatch.setattr(transport, "_QUEUE_SIZE", 1 if limit == "messages" else 1024)
        if limit == "bytes":
            monkeypatch.setattr(transport, "_QUEUE_BYTES", len(msgspec.msgpack.encode(message)))
        channel = _MessageSocket(asyncio.StreamReader(), Mock())
        await channel.send(message)
        blocked = [asyncio.create_task(channel.send(message)) for _ in range(2)]
        await asyncio.sleep(0)
        assert all(not task.done() for task in blocked)
        channel.close("worker exited")
        for task in blocked:
            with pytest.raises(DuplexSessionError, match="worker exited") as failed:
                await asyncio.wait_for(task, 1)
            assert failed.value.code == "engine_dead"
        with pytest.raises(DuplexSessionError):
            await channel.send(message)

    asyncio.run(run())


@pytest.mark.parametrize("cancel", [False, True], ids=["timeout", "cancellation"])
def test_transport_abandoned_send_is_not_delivered(monkeypatch, cancel):
    async def run():
        monkeypatch.setattr("vllm_omni.engine.duplex.process_engine._QUEUE_SIZE", 1)
        writer = Mock(drain=AsyncMock())
        channel = _MessageSocket(asyncio.StreamReader(), writer)
        await channel.send(("first",))
        blocked = asyncio.create_task(channel.send(("abandoned",), timeout=1 if cancel else 0.01))
        if cancel:
            await asyncio.sleep(0)
            blocked.cancel()
            with pytest.raises(asyncio.CancelledError):
                await blocked
        else:
            with pytest.raises(DuplexSessionError) as failed:
                await blocked
            assert failed.value.code == "engine_backpressure"
            assert failed.value.retryable
        pump = asyncio.create_task(channel.write())
        try:
            await asyncio.sleep(0)
            await channel.send(("last",))
            await asyncio.sleep(0)
            assert [msgspec.msgpack.decode(call.args[0]) for call in writer.write.call_args_list] == [
                ["first"],
                ["last"],
            ]
        finally:
            pump.cancel()
            await asyncio.gather(pump, return_exceptions=True)
            channel.close()

    asyncio.run(run())


def test_transport_rejects_oversized_message():
    with pytest.raises(ValueError, match="queue budget"):
        _MessageSocket.entry(("command", b"x" * _QUEUE_BYTES))


def test_native_values_survive_msgpack():
    command = AppendAudio(audio=b"pcm", video_frames=("frame",), hints={"model_hint": [1, True]})
    assert msgspec.msgpack.decode(msgspec.msgpack.encode(command), type=AppendAudio) == command
    config = DuplexSessionConfig(model="fake", extra_body={"custom": {"value": [1, 2]}}).normalized()
    assert msgspec.msgpack.decode(msgspec.msgpack.encode(config), type=DuplexSessionConfig) == config
    for capabilities in (None, DuplexCapabilities(supports_session_resume=True)):
        result = DuplexControlResultMessage(
            control_id="test", operation="open", session_id="s", ok=True, capabilities=capabilities
        )
        decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(result))
        assert msgspec.convert(decoded, type=DuplexControlResultMessage) == result


@pytest.mark.parametrize("operation", ["check_health_async", "get_outputs_async"])
def test_cancelled_startup_waiter_does_not_cancel_shared_connection(monkeypatch, operation):
    """An abandoned IPC caller must leave startup usable by later callers."""

    async def run():
        client = make_client()
        started, release = asyncio.Event(), asyncio.Event()
        connect = client._connect

        async def paused_connect():
            started.set()
            await release.wait()
            await connect()

        monkeypatch.setattr(client, "_connect", paused_connect)
        first = asyncio.create_task(getattr(client, operation)())
        try:
            await asyncio.wait_for(started.wait(), 5)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert client._start_task is not None and not client._start_task.cancelled()
            release.set()
            await asyncio.wait_for(client.check_health_async(), 5)
            buffer = DuplexOutputBuffer(max_bytes=1024, max_events=8)
            result = await asyncio.wait_for(
                client.open_session_async("s", DuplexSessionConfig(), output_buffer=buffer), 5
            )
            assert result.ok
            assert isinstance(await asyncio.wait_for(buffer.get(), 5), SessionCreated)
            await client.close_session_async("s", reason="done")
            outputs = await client.get_outputs_async(timeout=5)
            assert outputs[0].event.reason == "done"
        finally:
            release.set()
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)
            client.shutdown()

    asyncio.run(run())


def test_spawned_client_streams_audio_controls_and_invalidations():
    async def run():
        client = make_client()
        try:
            assert client.num_stages == 4
            assert client.metadata.warmup_audio["sample_rate_hz"] == 24000
            assert client.duplex_capabilities.supports_chat_completions  # Model facts are unchanged.
            await client.check_health_async()
            rejected = DuplexOutputBuffer(max_bytes=1024, max_events=8)
            with pytest.raises(DuplexSessionError) as failed:
                await client.open_session_async(
                    "rejected", DuplexSessionConfig(extra_body={"reject": True}), output_buffer=rejected
                )
            assert failed.value.code == "capacity"
            assert "rejected" not in client._buffers
            buffer = DuplexOutputBuffer(max_bytes=1024 * 1024, max_events=512)
            result = await client.open_session_async("s", DuplexSessionConfig(), output_buffer=buffer)
            assert result.public_session["worker_pid"] != os.getpid()
            assert result.public_session["worker_loop"] == "uvloop"
            assert isinstance(await buffer.get(), SessionCreated)
            for i in range(40):
                await client.submit_command_async("s", AppendAudio(audio=bytes([i]) * 1920))
            for i in range(40):
                event = await asyncio.wait_for(buffer.get(), 5)
                assert event.audio == bytes([i]) * 1920
            held = event
            await client.submit_command_async("s", ClearOutputAudio(response_id="r"))
            for _ in range(100):
                if not buffer.is_valid(held):
                    break
                await asyncio.sleep(0.01)
            with buffer.guard(held) as valid:
                assert not valid
            assert isinstance(await buffer.get(), OutputAudioCleared)
            assert (await buffer.get()).audio == b"fresh"
            await client.touch_session_async("s", activity="heartbeat")
            result = await client.resume_session_async("s", expected_lease_generation=0, control_id="resume")
            assert result.lease_generation == 1
            result = await client.resume_session_async("s", expected_lease_generation=0, control_id="resume")
            assert result.lease_generation == 1
            await client.close_session_async("s", reason="done")
            outputs = await client.get_outputs_async(timeout=5)
            assert outputs[0].event.reason == "done"
        finally:
            await client.aclose()
        assert not client._process.is_alive()

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["crash", "bad_output"])
def test_child_death_wakes_pending_controls_and_buffers(tmp_path, failure):
    async def run():
        marker = tmp_path / "shutdown"
        client = make_client(shutdown_path=str(marker))
        try:
            buffer = DuplexOutputBuffer(max_bytes=1024, max_events=8)
            await client.open_session_async("s", DuplexSessionConfig(), output_buffer=buffer)
            await buffer.get()
            pending = asyncio.create_task(client.touch_session_async("s", activity="hang"))
            await asyncio.sleep(0.05)
            await client.submit_command_async("s", AppendText(text=failure))
            assert await asyncio.wait_for(buffer.get(), 5) is None
            with pytest.raises(DuplexSessionError) as failed:
                await asyncio.wait_for(pending, 5)
            assert failed.value.code == "engine_dead"
        finally:
            client.shutdown()
        if failure == "bad_output":
            assert marker.read_text() == "closed"
            assert client._process.exitcode == 0

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["eof", "writer", "invalidation"])
def test_worker_stops_all_tasks_on_disconnect_or_overflow(monkeypatch, failure):
    async def run():
        monkeypatch.setattr("vllm_omni.engine.duplex.process_engine._QUEUE_SIZE", 1)
        reader = asyncio.StreamReader()
        writer = Mock(drain=AsyncMock(), close=Mock(side_effect=reader.feed_eof))
        channel = _MessageSocket(reader, writer)
        worker = _DuplexProcessWorker(fake_factory("fake", {}), channel)
        log = Mock()
        monkeypatch.setattr("vllm_omni.engine.duplex.process_engine.logger.error", log)
        if failure == "eof":
            reader.feed_eof()
        elif failure == "writer":
            monkeypatch.setattr(channel, "write", AsyncMock(side_effect=BrokenPipeError))
        else:
            # Hold the writer stalled so the queue stays full while the event
            # loop delivers an orchestrator-thread invalidation.
            monkeypatch.setattr(channel, "write", asyncio.Event().wait)
            await channel.send(("queued",))
            worker.notify_invalidation("s", "r", 0)
        tasks_before = asyncio.all_tasks()
        try:
            await asyncio.wait_for(worker.run(), 1)
            assert asyncio.all_tasks() == tasks_before
            assert not worker.buffers
            if failure == "invalidation":
                log.assert_called_once_with("Duplex IPC invalidation queue overflow")
                writer.close.assert_called()
            else:
                log.assert_not_called()
        finally:
            channel.close()

    asyncio.run(run())


def test_startup_failure_propagates():
    with pytest.raises(RuntimeError, match="test startup failure"):
        ProcessDuplexEngineClient("fake", {"init_timeout": 30}, factory=failing_factory)
