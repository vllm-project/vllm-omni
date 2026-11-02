# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Drain through native Janus queues, correlated RPC and a separate runner loop.

Only model/stage execution, encoding and the socket are doubles. Unlike the
direct manager tests, frontend commands and control replies cross the actual
queue/router boundary. This is not a GPU or live EngineCore deployment test.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TypeVar

import janus
import pytest
import pytest_asyncio
from pytest_mock import MockerFixture

from tests.engine.duplex.test_audio_drain import _generated, _until
from tests.engine.duplex.test_session_runner import close_harness
from tests.engine.duplex.test_session_runner_personaplex import code2wav_output, frame, open_personaplex_harness
from tests.entrypoints.duplex.test_audio_send_completion import _next_audio
from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.messages import DrainDuplexAudioMessage, DuplexSessionCommandMessage, DuplexSessionError
from vllm_omni.engine.duplex_omni_engine import DuplexOmniEngine
from vllm_omni.engine.messages import EngineQueueMessage
from vllm_omni.engine.rpc_result_router import CorrelatedRpcClient
from vllm_omni.entrypoints.duplex.serving import OmniDuplexSessionHandler
from vllm_omni.entrypoints.duplex_omni import DuplexOmni, DuplexSessionHandle
from vllm_omni.model_executor.models.personaplex.duplex import stage0

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
_T = TypeVar("_T")


class _ControlTransport:
    """Own all backend-loop mutations and join every thread on teardown."""

    engine: DuplexOmniEngine
    handle: DuplexSessionHandle
    handler: OmniDuplexSessionHandler

    def __init__(self, mocker: MockerFixture) -> None:
        self.mocker = mocker
        self.ready: Future[None] = Future()
        self.finished: Future[None] = Future()
        self.dispatched: list[EngineQueueMessage] = []
        self.held: EngineQueueMessage | None = None
        self.thread = threading.Thread(target=self._run, name="drain-test-runner", daemon=True)

    def _run(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as exc:
            if not self.ready.done():
                self.ready.set_exception(exc)
            self.finished.set_exception(exc)
        else:
            self.finished.set_result(None)

    async def _serve(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.h = await open_personaplex_harness(runtime_config=DuplexSessionRuntimeConfig(max_sessions=2))
        self.mocker.patch.object(
            self.h.runner.model, "_schedule_silence_continuation", new=self.mocker.AsyncMock(return_value=False)
        )
        self.requests: janus.Queue[EngineQueueMessage] = janus.Queue(maxsize=1)
        self.results: janus.Queue[EngineQueueMessage] = janus.Queue()
        self.h.manager._result_sink = self.results.async_q
        self.dispatch_gate = asyncio.Event()
        self.dispatch_gate.set()
        self.stop = asyncio.Event()

        async def pump() -> None:
            while True:
                message = await self.requests.async_q.get()
                self.held = message
                await self.dispatch_gate.wait()
                self.held = None
                self.dispatched.append(message)
                self.h.manager.dispatch(message)

        worker = asyncio.create_task(pump(), name="drain-test-request-pump")
        self.ready.set_result(None)
        try:
            await self.stop.wait()
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            await close_harness(self.h)
            for channel in (self.requests, self.results):
                channel.close()
                await channel.wait_closed()

    async def backend(self, operation: Coroutine[object, object, _T]) -> _T:
        return await asyncio.wait_for(asyncio.wrap_future(asyncio.run_coroutine_threadsafe(operation, self.loop)), 3)

    async def pending(self) -> None:
        async def check() -> None:
            await _until(lambda: self.h.runner._audio_drain_result is not None)

        await self.backend(check())


@pytest_asyncio.fixture
async def transport(mocker, monkeypatch):
    monkeypatch.setattr(stage0, "personaplex_prefill_slots", lambda *_: 4)
    native = _ControlTransport(mocker)
    native.thread.start()
    engine = None
    try:
        await asyncio.wait_for(asyncio.wrap_future(native.ready), 3)
        engine = DuplexOmniEngine.__new__(DuplexOmniEngine)
        engine.orchestrator_thread = native.thread
        engine.request_queue = native.requests
        engine.duplex_session_config = native.h.manager.runtime_config
        engine._correlated_rpc_client = CorrelatedRpcClient(native.requests.sync_q, native.results.sync_q)
        omni = DuplexOmni.__new__(DuplexOmni)
        omni.engine = engine
        native.engine = engine
        native.handle = DuplexSessionHandle(omni, native.h.session.session_id)
        native.handle._outbox = native.h.output_buffer
        native.handler = OmniDuplexSessionHandler(duplex_omni=omni)
        yield native
    finally:
        if engine is not None:
            await asyncio.to_thread(engine.rpc_client.close)
            assert not engine.rpc_client._router._thread.is_alive()
        if native.ready.done() and not native.ready.cancelled() and native.ready.exception() is None:
            native.loop.call_soon_threadsafe(native.stop.set)
        await asyncio.to_thread(native.thread.join, 5)
        assert not native.thread.is_alive()
        native.finished.result(timeout=0)


@pytest_asyncio.fixture
async def frontend_executor(transport, request, monkeypatch):
    """Restore the frontend pool before the native transport tears down."""
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=request.param) as executor, monkeypatch.context() as patcher:
        patcher.setattr(loop, "_default_executor", executor)
        yield executor


@pytest.mark.asyncio
@pytest.mark.parametrize("frontend_executor", [1, 2], indirect=True)
async def test_native_control_requires_successful_send_and_returns_exact_frozen_target(
    transport, frontend_executor, mocker
):
    native = transport

    async def prepare() -> None:
        mocker.patch.object(native.h.port, "flush_audio_prefix", new=mocker.AsyncMock(return_value=3))
        for _ in range(3):
            await native.h.run(frame())

    await native.backend(prepare())
    drain = asyncio.create_task(native.handle.drain_audio(timeout=2))
    started, release = asyncio.Event(), asyncio.Event()

    async def send(_payload) -> None:
        started.set()
        await release.wait()

    sending = None
    try:
        await native.pending()

        async def generate() -> None:
            for sequence in (1, 2, 3):
                _generated(native.h, sequence)
            native.h.deliver(code2wav_output(native.h.stage0_request_id(), samples=3840, text="hello"))

        await native.backend(generate())
        event = await _next_audio(native.h.output_buffer)
        await native.handler._attachment_registry.create(native.handle.session_id, send=send, close=mocker.AsyncMock())
        sending = asyncio.create_task(native.handler._send_event(native.handle.session_id, event, handle=native.handle))
        await asyncio.wait_for(started.wait(), 3)
        assert not drain.done()
        release.set()
        await sending
        target = await drain
        assert (target.epoch, target.accepted_seq, target.expected_samples) == (0, 3, 3840)
        assert target.request_id == native.h.stage0_request_id()
        assert any(isinstance(message, DrainDuplexAudioMessage) for message in native.dispatched)
        assert any(
            isinstance(message, DuplexSessionCommandMessage)
            and isinstance(message.command, commands.AudioSendCompleted)
            for message in native.dispatched
        )
        assert not native.engine.rpc_client._router._pending
    finally:
        release.set()
        if sending is not None:
            await asyncio.gather(sending, return_exceptions=True)
        drain.cancel()
        await asyncio.gather(drain, return_exceptions=True)


@pytest.mark.asyncio
async def test_native_close_cannot_wait_behind_an_acoustic_drain(transport, mocker):
    native = transport
    await native.backend(native.h.run(frame()))
    drain = asyncio.create_task(native.handle.drain_audio(timeout=2))
    try:
        await native.pending()
        result = await asyncio.wait_for(native.engine.close_session_async(native.handle.session_id), 1)
        assert result.ok and result.operation == "close"
        with pytest.raises(DuplexSessionError, match="invalidated|closed") as error:
            await drain
        assert error.value.code == "failed_precondition"
        assert not native.engine.rpc_client._router._pending
    finally:
        drain.cancel()
        await asyncio.gather(drain, return_exceptions=True)


@pytest.mark.asyncio
async def test_native_timeout_retires_its_rpc_and_allows_a_new_prefix_wait(transport, mocker):
    native = transport

    async def prepare() -> None:
        mocker.patch.object(native.h.port, "flush_audio_prefix", new=mocker.AsyncMock(return_value=1))
        await native.h.run(frame())

    await native.backend(prepare())
    with pytest.raises(DuplexSessionError) as error:
        await native.handle.drain_audio(timeout=0.05)
    assert error.value.code == "timeout" and error.value.retryable

    async def settled() -> None:
        await _until(lambda: native.h.runner._audio_drain_result is None and not native.h.runner._background_tasks)
        _generated(native.h, 1)

    await native.backend(settled())
    assert not native.engine.rpc_client._router._pending
    target = await native.handle.drain_audio(timeout=1)
    assert (target.accepted_seq, target.expected_samples) == (1, 0)
    drains = [message for message in native.dispatched if isinstance(message, DrainDuplexAudioMessage)]
    assert len(drains) == 2 and drains[0].control_id != drains[1].control_id


@pytest.mark.asyncio
async def test_full_native_request_queue_cannot_block_drain_past_its_timeout(transport):
    native = transport

    async def pause() -> None:
        native.dispatch_gate.clear()

    await native.backend(pause())
    heartbeat = DuplexSessionCommandMessage(session_id=native.handle.session_id, command=commands.Heartbeat())
    native.requests.sync_q.put_nowait(heartbeat)

    async def held() -> None:
        await _until(lambda: native.held is heartbeat)

    await native.backend(held())
    native.requests.sync_q.put_nowait(heartbeat)
    assert native.requests.sync_q.full()
    drain = asyncio.create_task(native.handle.drain_audio(timeout=0.01))
    try:
        await asyncio.sleep(0.15)
        assert drain.done(), "queue submission outlived the drain timeout before result waiting even started"
        with pytest.raises(DuplexSessionError) as error:
            await drain
        assert error.value.code == "engine_backpressure" and error.value.retryable
        assert not native.engine.rpc_client._router._pending
    finally:
        native.loop.call_soon_threadsafe(native.dispatch_gate.set)
        await asyncio.gather(drain, return_exceptions=True)


@pytest.mark.asyncio
async def test_other_session_controls_do_not_complete_or_cancel_the_pending_drain(transport):
    native = transport
    await native.backend(native.h.run(frame()))
    drain = asyncio.create_task(native.handle.drain_audio(timeout=2))
    try:
        await native.pending()
        second = DuplexSessionHandle(native.handle._omni, "independent-session")
        opened = await native.engine.open_session_async(
            second.session_id,
            DuplexSessionConfig(model="nvidia/personaplex-7b-v1", voice="NATF2.pt", modalities=["audio", "text"]),
            output_buffer=second._outbox,
            timeout=1,
        )
        assert opened.ok
        target = await second.drain_audio(timeout=1)
        assert (target.accepted_seq, target.expected_samples) == (0, 0)
        assert not drain.done()
        assert (await native.engine.close_session_async(second.session_id, timeout=1)).ok
        assert not drain.done()
        await native.engine.close_session_async(native.handle.session_id, timeout=1)
        with pytest.raises(DuplexSessionError, match="invalidated|closed"):
            await drain
        assert not native.engine.rpc_client._router._pending
    finally:
        drain.cancel()
        await asyncio.gather(drain, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_frontend_wait_never_claims_completion_and_close_releases_backend(transport):
    native = transport
    await native.backend(native.h.run(frame()))
    drain = asyncio.create_task(native.handle.drain_audio(timeout=2))
    try:
        await native.pending()
        drain.cancel()
        with pytest.raises(asyncio.CancelledError):
            await drain
        # Cancelling the frontend wait cannot retract an admitted queue message.
        # Exercise the actual lifecycle control, not direct Future.cancel().
        assert (await native.engine.close_session_async(native.handle.session_id, timeout=1)).ok
        await _until(lambda: not native.engine.rpc_client._router._pending)
    finally:
        drain.cancel()
        await asyncio.gather(drain, return_exceptions=True)
