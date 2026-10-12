# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for AsyncOmniEngine.try_get_output and try_get_output_async.

Focuses on the critical behavior: when the orchestrator thread dies,
subsequent attempts to collect output raise RuntimeError.
"""

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from pytest_mock import MockerFixture

from vllm_omni.engine.async_engine_utils import weak_shutdown_async_omni_engine
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.messages import (
    CollectiveRPCResultMessage,
    ErrorMessage,
    OutputMessage,
)
from vllm_omni.engine.rpc_result_router import CorrelatedRpcClient
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_engine(output_queue, mocker: MockerFixture, *, thread_alive: bool = True) -> AsyncOmniEngine:
    """Create an AsyncOmniEngine bypassing __init__."""
    engine = object.__new__(AsyncOmniEngine)
    engine.output_queue = output_queue
    engine.orchestrator_thread = mocker.MagicMock(
        is_alive=mocker.MagicMock(return_value=thread_alive),
    )
    return engine


def test_weak_shutdown_closes_rpc_router_before_joining_orchestrator(mocker: MockerFixture):
    request_queue = mocker.MagicMock()
    output_queue = mocker.MagicMock()
    rpc_output_queue = mocker.MagicMock()
    router = mocker.MagicMock()
    orchestrator_thread = mocker.MagicMock()
    orchestrator_thread.is_alive.return_value = True

    def assert_router_closed_before_join(*args, **kwargs):
        router.close.assert_called_once_with()

    orchestrator_thread.join.side_effect = assert_router_closed_before_join

    weak_shutdown_async_omni_engine(
        orchestrator_thread,
        request_queue,
        output_queue,
        rpc_output_queue,
        router,
    )

    request_queue.sync_q.put.assert_called_once()
    assert request_queue.sync_q.put.call_args.kwargs["timeout"] > 0
    orchestrator_thread.join.assert_called_once()
    assert orchestrator_thread.join.call_args.kwargs["timeout"] > 0


def test_try_get_output_raises_after_orchestrator_dies(mocker: MockerFixture):
    """Draining remaining results then hitting an empty queue with a dead
    orchestrator must raise RuntimeError so callers know the pipeline is gone."""
    mock_queue = mocker.MagicMock()
    # First call succeeds; second call finds the queue empty.
    mock_queue.sync_q.get.side_effect = [
        OutputMessage(
            request_id="r1",
            stage_id=0,
            engine_outputs=OmniRequestOutput(request_id="r1"),
            finished=False,
        ),
        queue.Empty,
    ]

    engine = _make_engine(mock_queue, mocker, thread_alive=True)

    # Collect the one buffered result.
    assert engine.try_get_output().request_id == "r1"

    # Orchestrator thread crashes between polls.
    engine.orchestrator_thread.is_alive.return_value = False

    with pytest.raises(RuntimeError, match="Orchestrator died unexpectedly"):
        engine.try_get_output()


@pytest.mark.asyncio
async def test_try_get_output_async_raises_after_orchestrator_dies(mocker: MockerFixture):
    """Same scenario as above but for the async variant."""
    raw_queue = queue.Queue()
    raw_queue.put_nowait(
        OutputMessage(
            request_id="r1",
            stage_id=0,
            engine_outputs=OmniRequestOutput(request_id="r1"),
            finished=False,
        )
    )

    engine = _make_engine(SimpleNamespace(sync_q=raw_queue), mocker, thread_alive=True)

    assert (await engine.try_get_output_async()).request_id == "r1"

    engine.orchestrator_thread.is_alive.return_value = False

    with pytest.raises(RuntimeError, match="Orchestrator died unexpectedly"):
        await engine.try_get_output_async()


def test_fatal_error_message_surfaces_through_try_get_output(mocker: MockerFixture):
    """When the orchestrator thread crashes, it enqueues a fatal error message.

    ``try_get_output`` must return this message so the caller
    (``OmniBase._handle_output_message``) can detect the fatal flag.
    """
    fatal_msg = ErrorMessage(error="Orchestrator thread crashed", fatal=True)

    mock_queue = mocker.MagicMock()
    mock_queue.sync_q.get.return_value = fatal_msg

    engine = _make_engine(mock_queue, mocker, thread_alive=False)

    msg = engine.try_get_output()
    assert msg is not None
    assert msg.type == "error"
    assert msg.fatal is True
    assert "crashed" in msg.error


@pytest.mark.asyncio
async def test_fatal_error_message_surfaces_through_try_get_output_async(mocker: MockerFixture):
    """Async variant of the fatal error message test."""
    fatal_msg = ErrorMessage(error="Orchestrator thread crashed", fatal=True)

    raw_queue = queue.Queue()
    raw_queue.put_nowait(fatal_msg)

    engine = _make_engine(SimpleNamespace(sync_q=raw_queue), mocker, thread_alive=False)

    msg = await engine.try_get_output_async()
    assert msg is not None
    assert msg.type == "error"
    assert msg.fatal is True


def test_output_remains_on_shared_output_path(mocker: MockerFixture):
    raw_queue = queue.Queue()
    raw_queue.put_nowait(
        OutputMessage(
            request_id="shared-path-request",
            stage_id=0,
            engine_outputs=OmniRequestOutput(request_id="shared-path-request"),
            finished=False,
        )
    )
    engine = _make_engine(SimpleNamespace(sync_q=raw_queue), mocker)

    output = engine.try_get_output(timeout=0.01)

    assert output.request_id == "shared-path-request"


def test_collective_rpc_preserves_request_queue_backpressure(mocker: MockerFixture):
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
    rpc_q = queue.Queue()
    engine = object.__new__(AsyncOmniEngine)
    engine.request_queue = SimpleNamespace(sync_q=request_q)
    engine.rpc_output_queue = SimpleNamespace(sync_q=rpc_q)
    engine._correlated_rpc_client = CorrelatedRpcClient(request_q, rpc_q)
    mocker.patch("vllm_omni.engine.omni_engine_base.uuid.uuid4", return_value=SimpleNamespace(hex="blocked-rpc"))

    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(engine.collective_rpc, "health", timeout=1)
        assert request_q.put_attempted.wait(timeout=1)
        assert not pending.done()

        assert request_q.get(timeout=1) == "queue-is-full"
        assert request_q.get(timeout=1).rpc_id == "blocked-rpc"
        rpc_q.put(
            CollectiveRPCResultMessage(
                rpc_id="blocked-rpc",
                method="health",
                stage_ids=[0],
                results=["healthy"],
            )
        )
        assert pending.result(timeout=1) == ["healthy"]

    engine._correlated_rpc_client.close()


def test_shutdown_does_not_race_runtime_cleanup_with_live_orchestrator(mocker: MockerFixture):
    engine = object.__new__(AsyncOmniEngine)
    engine._shutdown_called = False
    engine._weak_finalizer = None
    engine.request_queue = mocker.MagicMock()
    engine.request_queue.sync_q.put.side_effect = queue.Full
    engine.output_queue = mocker.MagicMock()
    engine.rpc_output_queue = mocker.MagicMock()
    engine._correlated_rpc_client = mocker.MagicMock()
    engine.orchestrator_thread = mocker.MagicMock()
    engine.orchestrator_thread.is_alive.return_value = True
    engine._runtime = mocker.MagicMock()

    engine.shutdown()

    engine.request_queue.sync_q.put.assert_called_once()
    assert engine.request_queue.sync_q.put.call_args.kwargs["timeout"] > 0
    assert engine.orchestrator_thread.join.call_args_list[0].kwargs["timeout"] > 0
    engine._correlated_rpc_client.close.assert_called_once_with()
    engine.request_queue.close.assert_called_once_with()
    engine.output_queue.close.assert_called_once_with()
    engine.rpc_output_queue.close.assert_called_once_with()
    engine._runtime.shutdown.assert_not_called()


def test_shutdown_releases_runtime_after_orchestrator_stops(mocker: MockerFixture):
    engine = object.__new__(AsyncOmniEngine)
    engine._shutdown_called = False
    engine._weak_finalizer = None
    engine.request_queue = mocker.MagicMock()
    engine.output_queue = mocker.MagicMock()
    engine.rpc_output_queue = mocker.MagicMock()
    engine._correlated_rpc_client = mocker.MagicMock()
    engine.orchestrator_thread = mocker.MagicMock()
    engine.orchestrator_thread.is_alive.side_effect = [True, False]
    engine._runtime = mocker.MagicMock()

    engine.shutdown()

    engine.orchestrator_thread.join.assert_called_once()
    engine._runtime.shutdown.assert_called_once_with()


def test_shutdown_defers_runtime_release_until_live_orchestrator_stops(mocker: MockerFixture):
    orchestrator_stopped = threading.Event()
    runtime_released = threading.Event()

    class ControllableOrchestratorThread:
        def __init__(self) -> None:
            self.join_timeouts: list[float | None] = []

        def is_alive(self) -> bool:
            return not orchestrator_stopped.is_set()

        def join(self, timeout: float | None = None) -> None:
            self.join_timeouts.append(timeout)
            if timeout is None:
                orchestrator_stopped.wait()

    engine = object.__new__(AsyncOmniEngine)
    engine._shutdown_called = False
    engine._weak_finalizer = None
    engine.request_queue = mocker.MagicMock()
    engine.output_queue = mocker.MagicMock()
    engine.rpc_output_queue = mocker.MagicMock()
    engine._correlated_rpc_client = mocker.MagicMock()
    engine.orchestrator_thread = ControllableOrchestratorThread()
    engine._runtime = mocker.MagicMock()
    engine._runtime.shutdown.side_effect = runtime_released.set

    try:
        engine.shutdown()

        engine._runtime.shutdown.assert_not_called()
        orchestrator_stopped.set()
        assert runtime_released.wait(timeout=1)
        engine._runtime.shutdown.assert_called_once_with()
        assert engine.orchestrator_thread.join_timeouts[0] is not None
        assert engine.orchestrator_thread.join_timeouts[-1] is None
    finally:
        orchestrator_stopped.set()


def test_weak_shutdown_is_bounded_when_request_queue_is_full(mocker: MockerFixture):
    request_queue = mocker.MagicMock()
    request_queue.sync_q.put.side_effect = queue.Full
    output_queue = mocker.MagicMock()
    rpc_output_queue = mocker.MagicMock()
    router = mocker.MagicMock()
    orchestrator_thread = mocker.MagicMock()
    orchestrator_thread.is_alive.return_value = True

    weak_shutdown_async_omni_engine(
        orchestrator_thread,
        request_queue,
        output_queue,
        rpc_output_queue,
        router,
    )

    request_queue.sync_q.put.assert_called_once()
    assert request_queue.sync_q.put.call_args.kwargs["timeout"] > 0
    orchestrator_thread.join.assert_called_once()
    assert orchestrator_thread.join.call_args.kwargs["timeout"] > 0
    request_queue.close.assert_called_once_with()
    output_queue.close.assert_called_once_with()
    rpc_output_queue.close.assert_called_once_with()
