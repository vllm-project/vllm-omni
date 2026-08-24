"""Unit tests for acknowledged abort on AsyncOmniEngine."""

from __future__ import annotations

import asyncio
import queue
import threading
from types import SimpleNamespace

import pytest
from pytest_mock import MockerFixture

from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.messages import AbortRequestMessage, AbortResultMessage
from vllm_omni.engine.rpc_result_router import CorrelatedRpcClient

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_engine(request_q: queue.Queue, rpc_q: queue.Queue) -> AsyncOmniEngine:
    engine = object.__new__(AsyncOmniEngine)
    engine.request_queue = SimpleNamespace(sync_q=request_q)
    engine.rpc_output_queue = SimpleNamespace(sync_q=rpc_q)
    engine._correlated_rpc_client = CorrelatedRpcClient(request_q, rpc_q)
    return engine


def test_abort_result_message_correlation_key():
    msg = AbortResultMessage(rpc_id="abc", success=True)
    assert msg.rpc_correlation_key == ("abort", "abc")


def test_sync_abort_is_fire_and_forget():
    request_q: queue.Queue = queue.Queue()
    rpc_q: queue.Queue = queue.Queue()
    engine = _make_engine(request_q, rpc_q)

    engine.abort(["req-1"])

    msg = request_q.get_nowait()
    assert isinstance(msg, AbortRequestMessage)
    assert msg.request_ids == ["req-1"]
    assert msg.rpc_id is None
    assert rpc_q.empty()
    engine._correlated_rpc_client.close()


def test_abort_async_waits_for_ack(mocker: MockerFixture):
    request_q: queue.Queue = queue.Queue()
    rpc_q: queue.Queue = queue.Queue()
    engine = _make_engine(request_q, rpc_q)
    mocker.patch(
        "vllm_omni.engine.async_omni_engine.uuid.uuid4",
        return_value=SimpleNamespace(hex="abort-rpc-1"),
    )

    async def _run() -> None:
        task = asyncio.create_task(engine.abort_async(["req-a", "req-b"], timeout=1))
        msg = await asyncio.to_thread(request_q.get, True, 1)
        assert isinstance(msg, AbortRequestMessage)
        assert msg.request_ids == ["req-a", "req-b"]
        assert msg.rpc_id == "abort-rpc-1"
        assert not task.done()

        rpc_q.put(AbortResultMessage(rpc_id="abort-rpc-1", success=True))
        await task

    try:
        asyncio.run(_run())
    finally:
        engine._correlated_rpc_client.close()


def test_abort_async_raises_on_orchestrator_error(mocker: MockerFixture):
    request_q: queue.Queue = queue.Queue()
    rpc_q: queue.Queue = queue.Queue()
    engine = _make_engine(request_q, rpc_q)
    mocker.patch(
        "vllm_omni.engine.async_omni_engine.uuid.uuid4",
        return_value=SimpleNamespace(hex="abort-rpc-err"),
    )

    async def _run() -> None:
        task = asyncio.create_task(engine.abort_async(["req-x"], timeout=1))
        await asyncio.to_thread(request_q.get, True, 1)
        rpc_q.put(
            AbortResultMessage(
                rpc_id="abort-rpc-err",
                success=False,
                error="stage abort failed",
            )
        )
        with pytest.raises(RuntimeError, match="stage abort failed"):
            await task

    try:
        asyncio.run(_run())
    finally:
        engine._correlated_rpc_client.close()


def test_abort_async_times_out_without_result(mocker: MockerFixture):
    request_q: queue.Queue = queue.Queue()
    rpc_q: queue.Queue = queue.Queue()
    engine = _make_engine(request_q, rpc_q)
    mocker.patch(
        "vllm_omni.engine.async_omni_engine.uuid.uuid4",
        return_value=SimpleNamespace(hex="abort-rpc-timeout"),
    )

    async def _run() -> None:
        with pytest.raises(TimeoutError, match="abort timed out"):
            await engine.abort_async(["req-timeout"], timeout=0.05)

    try:
        asyncio.run(_run())
    finally:
        engine._correlated_rpc_client.close()


def test_abort_async_preserves_request_queue_backpressure(mocker: MockerFixture):
    class SignallingQueue(queue.Queue):
        def __init__(self) -> None:
            super().__init__(maxsize=1)
            self.put_attempted = threading.Event()

        def put(self, item, block=True, timeout=None):
            self.put_attempted.set()
            return super().put(item, block=block, timeout=timeout)

    request_q = SignallingQueue()
    request_q.put("queue-is-full")
    request_q.put_attempted.clear()
    rpc_q: queue.Queue = queue.Queue()
    engine = _make_engine(request_q, rpc_q)
    mocker.patch(
        "vllm_omni.engine.async_omni_engine.uuid.uuid4",
        return_value=SimpleNamespace(hex="blocked-abort"),
    )

    async def _run() -> None:
        task = asyncio.create_task(engine.abort_async(["req-blocked"], timeout=1))
        assert await asyncio.to_thread(request_q.put_attempted.wait, 1)
        assert not task.done()

        assert request_q.get(timeout=1) == "queue-is-full"
        msg = await asyncio.to_thread(request_q.get, True, 1)
        assert isinstance(msg, AbortRequestMessage)
        assert msg.rpc_id == "blocked-abort"
        rpc_q.put(AbortResultMessage(rpc_id="blocked-abort", success=True))
        await task

    try:
        asyncio.run(_run())
    finally:
        engine._correlated_rpc_client.close()
