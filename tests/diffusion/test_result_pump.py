# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MultiprocDiffusionExecutor async result pump and wait_output_ready."""

import concurrent.futures
import queue
import threading
import time
from unittest.mock import MagicMock

import pytest

from vllm_omni.diffusion.data import AsyncDiffusionOutput, AsyncOutputKind, DiffusionOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_executor(step_execution=False):
    """Create a minimal MultiprocDiffusionExecutor-like object with pump state."""
    from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor

    od_config = MagicMock()
    od_config.step_execution = step_execution

    executor = object.__new__(MultiprocDiffusionExecutor)
    executor.od_config = od_config
    executor._rpc_id_counter = 0
    executor._rpc_id_lock = threading.Lock()
    executor._rpc_futures = {}
    executor._output_futures = {}
    executor._completed_outputs = {}
    executor._batch_split_map = {}
    executor._futures_lock = threading.RLock()
    executor._pump_running = False
    executor._pump_stop = threading.Event()
    executor._sync_result_buffer = queue.Queue()
    executor._result_mq = MagicMock()
    executor._broadcast_mq = MagicMock()
    executor._closed = False
    executor._is_failed = False
    executor._finalizer = MagicMock()  # no-op in tests
    executor._shutdown_cleaner = None
    executor._processes = []
    return executor


def _feed_one_msg_to_pump(executor, msg):
    """Run _result_pump in a daemon thread, feed one *msg*, then stop."""
    call_count = [0]

    def mock_dequeue(timeout=None):
        call_count[0] += 1
        if call_count[0] == 1:
            return msg
        executor._pump_stop.set()
        time.sleep(0.05)
        raise TimeoutError

    executor._result_mq.dequeue = mock_dequeue
    t = threading.Thread(target=executor._result_pump, daemon=True)
    t.start()
    t.join(timeout=2.0)


@pytest.fixture(autouse=True)
def _mock_unpack(mocker):
    """Real _result_pump calls unpack_diffusion_output_shm; mock it away."""
    mocker.patch(
        "vllm_omni.diffusion.executor.multiproc_executor.unpack_diffusion_output_shm",
    )


class TestNextRpcId:
    """Test _next_rpc_id counter."""

    def test_counter_increments(self):
        executor = _make_executor()
        id1 = executor._next_rpc_id()
        id2 = executor._next_rpc_id()
        id3 = executor._next_rpc_id()
        assert id1 == "1"
        assert id2 == "2"
        assert id3 == "3"

    def test_counter_is_threadsafe(self):
        executor = _make_executor()
        ids = []

        def get_id():
            ids.append(executor._next_rpc_id())

        threads = [threading.Thread(target=get_id) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All IDs should be unique
        assert len(set(ids)) == 10


class TestWaitOutputReady:
    """Test wait_output_ready future creation and caching."""

    def test_returns_new_future_when_not_cached(self):
        executor = _make_executor()
        fut = executor.wait_output_ready("abc123")
        assert isinstance(fut, concurrent.futures.Future)
        assert not fut.done()

    def test_future_resolves_when_output_arrives(self):
        executor = _make_executor()
        fut = executor.wait_output_ready("abc123")
        output = DiffusionOutput(output="data")

        # Simulate pump resolving the future
        with executor._futures_lock:
            executor._output_futures.pop("abc123")
        if not fut.done():
            fut.set_result(output)

        assert fut.result(timeout=1.0) is output

    def test_returns_cached_future_when_already_completed(self):
        executor = _make_executor()
        output = DiffusionOutput(output="cached_data")
        fut = concurrent.futures.Future()
        fut.set_result(output)
        with executor._futures_lock:
            executor._completed_outputs["abc123"] = fut

        fut = executor.wait_output_ready("abc123")
        assert fut.done()
        assert fut.result(timeout=1.0) is output

    def test_removes_from_cache_after_retrieval(self):
        executor = _make_executor()
        output = DiffusionOutput(output="cached_data")
        with executor._futures_lock:
            executor._completed_outputs["abc123"] = output

        executor.wait_output_ready("abc123")
        # Second call should not find cached result
        with executor._futures_lock:
            assert "abc123" not in executor._completed_outputs


class TestResultPumpDispatch:
    """Test _result_pump message routing (running the real pump in a thread)."""

    def test_non_async_message_placed_in_sync_buffer(self):
        executor = _make_executor()
        msg = DiffusionOutput(output="sync_result")
        _feed_one_msg_to_pump(executor, msg)

        assert not executor._sync_result_buffer.empty()
        retrieved = executor._sync_result_buffer.get_nowait()
        assert isinstance(retrieved, DiffusionOutput)

    def test_compute_done_routes_to_rpc_future(self):
        executor = _make_executor()
        rpc_id = "42"
        fut = concurrent.futures.Future()
        with executor._futures_lock:
            executor._rpc_futures[rpc_id] = fut

        msg = AsyncDiffusionOutput(
            kind=AsyncOutputKind.COMPUTE_DONE,
            rpc_id=rpc_id,
            async_output_id="abc",
        )
        _feed_one_msg_to_pump(executor, msg)

        assert fut.done()
        result = fut.result(timeout=1.0)
        assert result.kind == AsyncOutputKind.COMPUTE_DONE

    def test_output_ready_routes_to_output_future(self):
        executor = _make_executor()
        async_output_id = "abc123"
        output = DiffusionOutput(output="final")
        fut = concurrent.futures.Future()
        with executor._futures_lock:
            executor._output_futures[async_output_id] = fut

        msg = AsyncDiffusionOutput(
            kind=AsyncOutputKind.OUTPUT_READY,
            async_output_id=async_output_id,
            output=output,
        )
        _feed_one_msg_to_pump(executor, msg)

        assert fut.done()
        assert fut.result(timeout=1.0) is output

    def test_output_ready_with_error_routes_to_future_as_exception(self):
        executor = _make_executor()
        async_output_id = "abc123"
        fut = concurrent.futures.Future()
        with executor._futures_lock:
            executor._output_futures[async_output_id] = fut

        msg = AsyncDiffusionOutput(
            kind=AsyncOutputKind.OUTPUT_READY,
            async_output_id=async_output_id,
            error="Background D2H/SHM packing failed",
        )
        _feed_one_msg_to_pump(executor, msg)

        assert fut.done()
        with pytest.raises(RuntimeError, match="Background D2H/SHM packing failed"):
            fut.result(timeout=1.0)

    def test_output_ready_caches_when_no_future_waiting(self):
        """When OUTPUT_READY arrives but no future is waiting, result is cached."""
        executor = _make_executor()
        async_output_id = "abc123"
        output = DiffusionOutput(output="orphan")

        msg = AsyncDiffusionOutput(
            kind=AsyncOutputKind.OUTPUT_READY,
            async_output_id=async_output_id,
            output=output,
        )
        _feed_one_msg_to_pump(executor, msg)

        # Later call to wait_output_ready should find it cached
        fut = executor.wait_output_ready(async_output_id)
        assert fut.done()
        assert fut.result(timeout=1.0) is output


class TestShutdownCleansUpFutures:
    """Test that shutdown cancels pending futures."""

    def test_shutdown_sets_exception_on_pending_futures(self):
        executor = _make_executor()

        rpc_fut = concurrent.futures.Future()
        output_fut = concurrent.futures.Future()
        with executor._futures_lock:
            executor._rpc_futures["1"] = rpc_fut
            executor._output_futures["abc"] = output_fut

        executor.shutdown()

        assert rpc_fut.done()
        with pytest.raises(RuntimeError, match="Executor shut down"):
            rpc_fut.result(timeout=1.0)

        assert output_fut.done()
        with pytest.raises(RuntimeError, match="Executor shut down"):
            output_fut.result(timeout=1.0)

        assert len(executor._rpc_futures) == 0
        assert len(executor._output_futures) == 0
