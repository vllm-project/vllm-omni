# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Real spawn and IPC coverage without model weights or GPU execution."""

from __future__ import annotations

import asyncio
import os
import socket
from dataclasses import dataclass

import msgspec
import pytest
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex.commands import AppendAudio, AppendText, ClearOutputAudio
from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
from vllm_omni.engine.duplex.delivery import DuplexOutputBuffer
from vllm_omni.engine.duplex.events import AudioDelta, OutputAudioCleared, SessionClosed, SessionCreated
from vllm_omni.engine.duplex.messages import DuplexControlResultMessage, DuplexSessionError, DuplexSessionEventMessage
from vllm_omni.engine.duplex.process_engine import (
    _QUEUE_BYTES,
    ProcessDuplexEngineClient,
    ProcessDuplexStageMetadata,
    _BatchedSocket,
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
        elif isinstance(command, AppendText) and command.text == "reject":
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

    async def check_health(self):
        if self.health_error:
            raise EngineDeadError("Stage-2 has no live replica")

    def shutdown(self):
        pass


def fake_factory(model, options):
    return FakeRuntime(
        model,
        FakeEngine(options.get("small_buffers", False)),
        {"audio": "AAAAAA==", "format": "pcm_f32le", "sample_rate_hz": 24000},
        health_error=options.get("health_error", False),
    )


def failing_factory(model, options):
    raise RuntimeError("test startup failure")


def make_client(**options):
    return ProcessDuplexEngineClient("fake", {"init_timeout": 30, **options}, factory=fake_factory)


def test_transport_batches_both_directions():
    async def run():
        a, b = socket.socketpair()
        a.setblocking(False)
        b.setblocking(False)
        ar, aw = await asyncio.open_connection(sock=a)
        br, bw = await asyncio.open_connection(sock=b)
        left, right = _BatchedSocket(ar, aw), _BatchedSocket(br, bw)
        for i in range(64):
            await left.send(("command", i, b"pcm"))
            await right.send(("event", i, b"audio"))
        tasks = [asyncio.create_task(left.write()), asyncio.create_task(right.write())]
        try:
            assert await right.receive() == [["command", i, b"pcm"] for i in range(64)]
            assert await left.receive() == [["event", i, b"audio"] for i in range(64)]
            assert left.frames_sent == right.frames_sent == 1
            assert left.pending_bytes == right.pending_bytes == 0
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            aw.close()
            bw.close()
            await asyncio.gather(aw.wait_closed(), bw.wait_closed())

    asyncio.run(run())


def test_transport_rejects_oversized_message():
    with pytest.raises(ValueError, match="queue budget"):
        _BatchedSocket.entry(("command", b"x" * _QUEUE_BYTES))


def test_native_values_survive_msgpack():
    command = AppendAudio(audio=b"pcm", video_frames=("frame",), hints={"model_hint": [1, True]})
    assert msgspec.msgpack.decode(msgspec.msgpack.encode(command), type=AppendAudio) == command
    config = DuplexSessionConfig(model="fake", extra_body={"custom": {"value": [1, 2]}}).normalized()
    assert msgspec.msgpack.decode(msgspec.msgpack.encode(config), type=DuplexSessionConfig) == config


def test_spawned_client_streams_audio_controls_and_invalidations():
    async def run():
        client = make_client()
        try:
            assert client.num_stages == 4
            assert client.metadata.warmup_audio["sample_rate_hz"] == 24000
            assert client.duplex_capabilities.supports_chat_completions  # Model facts are unchanged.
            await client.check_health_async()
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


def test_child_death_wakes_pending_controls_and_buffers():
    async def run():
        client = make_client()
        try:
            buffer = DuplexOutputBuffer(max_bytes=1024, max_events=8)
            await client.open_session_async("s", DuplexSessionConfig(), output_buffer=buffer)
            await buffer.get()
            pending = asyncio.create_task(client.touch_session_async("s", activity="hang"))
            await asyncio.sleep(0.05)
            await client.submit_command_async("s", AppendText(text="crash"))
            assert await asyncio.wait_for(buffer.get(), 5) is None
            with pytest.raises(DuplexSessionError) as failed:
                await asyncio.wait_for(pending, 5)
            assert failed.value.code == "engine_dead"
        finally:
            client.shutdown()

    asyncio.run(run())


def test_startup_failure_propagates():
    with pytest.raises(RuntimeError, match="test startup failure"):
        ProcessDuplexEngineClient("fake", {"init_timeout": 30}, factory=failing_factory)
