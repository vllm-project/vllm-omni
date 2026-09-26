# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for MultiprocDiffusionExecutor async result pump and wait_output_ready."""

import concurrent.futures
import queue
import threading
import time
from collections import OrderedDict
from types import SimpleNamespace
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
    executor._dropped_output_ids = OrderedDict()
    executor._batch_split_map = {}
    executor._futures_lock = threading.RLock()
    executor._pump_running = False
    executor._pump_stop = threading.Event()
    executor._sync_result_buffer = queue.Queue()
    executor._result_mq = MagicMock()
    executor._result_mqs = []
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

    def test_start_result_pump_reads_every_worker_queue(self):
        executor = _make_executor()
        messages = [DiffusionOutput(output="rank0"), DiffusionOutput(output="rank1")]
        result_mqs = [MagicMock(), MagicMock()]

        def make_dequeue(message):
            emitted = False

            def dequeue(timeout=None):
                nonlocal emitted
                if not emitted:
                    emitted = True
                    return message
                executor._pump_stop.wait(0.01)
                raise TimeoutError

            return dequeue

        for result_mq, message in zip(result_mqs, messages, strict=True):
            result_mq.dequeue = make_dequeue(message)

        executor._result_mqs = result_mqs
        executor._result_mq = result_mqs[0]
        executor._start_result_pump()
        deadline = time.monotonic() + 2.0
        while executor._sync_result_buffer.qsize() < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        executor._pump_stop.set()
        for thread in executor._result_pump_threads:
            thread.join(timeout=2.0)

        assert len(executor._result_pump_threads) == 2
        received = [executor._sync_result_buffer.get_nowait().output for _ in range(2)]
        assert sorted(received) == ["rank0", "rank1"]

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
        assert async_output_id not in executor._output_futures

    def test_output_ready_atomic_resolution_when_future_already_waiting(self):
        """When OUTPUT_READY arrives and a future is already waiting, resolve it directly."""
        executor = _make_executor()
        async_output_id = "abc123"
        output = DiffusionOutput(output="waiting")
        fut = executor.wait_output_ready(async_output_id)
        assert not fut.done()

        msg = AsyncDiffusionOutput(
            kind=AsyncOutputKind.OUTPUT_READY,
            async_output_id=async_output_id,
            output=output,
        )
        _feed_one_msg_to_pump(executor, msg)

        assert fut.done()
        assert fut.result(timeout=1.0) is output
        assert async_output_id not in executor._output_futures
        assert async_output_id not in executor._completed_outputs


class _FakeBatchOutput:
    """Batch-level output exposing per-request results."""

    def __init__(self, results):
        self._results = results

    def get_request_output(self, req_id):
        result = self._results.get(req_id)
        if result is None:
            return None
        return SimpleNamespace(result=result)


def _make_scheduler_output(req_ids):
    return SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(request_id=rid, req=SimpleNamespace()) for rid in req_ids]
    )


class TestBatchSplitDelivery:
    """execute_batch must resolve per-request futures in either arrival order."""

    @staticmethod
    def _run(executor, req_ids, batch_id, deliver_early):
        # Keep execute_batch on the fused request-batch path (not DLO DP).
        executor.od_config.parallel_config.data_parallel_size = 1
        executor.od_config.enable_distributed_layerwise_offload = False
        executor._ensure_open = lambda: None

        outputs = {rid: DiffusionOutput(output=f"img-{rid}") for rid in req_ids}
        ready = AsyncDiffusionOutput(
            kind=AsyncOutputKind.OUTPUT_READY,
            async_output_id=batch_id,
            output=_FakeBatchOutput(outputs),
        )

        def fake_collective_rpc(*args, **kwargs):
            if deliver_early:
                # Worker's background D2H/SHM thread wins the race: OUTPUT_READY
                # is pumped before execute_batch registers the split map.
                _feed_one_msg_to_pump(executor, ready)
            return AsyncDiffusionOutput(
                kind=AsyncOutputKind.COMPUTE_DONE,
                rpc_id="1",
                async_output_id=batch_id,
            )

        executor.collective_rpc = fake_collective_rpc
        batch = executor.execute_batch(_make_scheduler_output(req_ids))
        if not deliver_early:
            _feed_one_msg_to_pump(executor, ready)
        return batch, outputs

    def test_output_ready_after_split_map(self):
        executor = _make_executor()
        req_ids = ["r0", "r1", "r2"]
        _, outputs = self._run(executor, req_ids, "batch-1", deliver_early=False)

        for rid in req_ids:
            fut = executor.wait_output_ready(f"batch-1/{rid}")
            assert fut.done()
            assert fut.result(timeout=1.0) is outputs[rid]

    def test_output_ready_before_split_map(self):
        """Regression: the whole batch used to be lost, hanging every request."""
        executor = _make_executor()
        req_ids = ["r0", "r1", "r2"]
        _, outputs = self._run(executor, req_ids, "batch-1", deliver_early=True)

        for rid in req_ids:
            fut = executor.wait_output_ready(f"batch-1/{rid}")
            assert fut.done(), f"request {rid} never resolved"
            assert fut.result(timeout=1.0) is outputs[rid]

        # No stale batch-level state left behind.
        assert executor._batch_split_map == {}
        assert executor._completed_outputs == {}

    def test_engine_waiter_before_output_is_resolved(self):
        """Consumers already blocked in wait_output_ready must be woken."""
        executor = _make_executor()
        req_ids = ["r0", "r1"]
        batch_id = "batch-2"
        futures = {rid: executor.wait_output_ready(f"{batch_id}/{rid}") for rid in req_ids}

        _, outputs = self._run(executor, req_ids, batch_id, deliver_early=True)

        for rid in req_ids:
            assert futures[rid].done(), f"request {rid} never resolved"
            assert futures[rid].result(timeout=1.0) is outputs[rid]


class TestShutdownCleansUpFutures:
    """Test that shutdown cancels pending futures."""

    def test_shutdown_joins_result_pump_threads(self):
        executor = _make_executor()
        pump = threading.Thread(
            target=executor._pump_stop.wait,
            name="test-result-pump",
        )
        executor._result_pump_threads = [pump]
        pump.start()

        executor.shutdown()

        assert not pump.is_alive()
        assert executor._result_pump_threads == []

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


class _RacyFuture(concurrent.futures.Future):
    # done() always lies and reports False, to deterministically force the
    # pump's check-then-act race window without needing real thread timing.
    def done(self) -> bool:
        return False


def _racy_cancelled_future() -> concurrent.futures.Future:
    fut = _RacyFuture()
    fut.cancel()
    assert fut.cancelled()
    return fut


class TestResultPumpCancelledFutureRace:
    """Regression for #5793: a future cancelled concurrently with the pump's
    resolve call (e.g. an asyncio.wait_for timeout, or a request abort) must
    be dropped, not raise InvalidStateError and kill the pump thread.
    """

    def test_compute_done_racing_cancel_does_not_crash_pump(self):
        executor = _make_executor()
        fut = _racy_cancelled_future()
        with executor._futures_lock:
            executor._rpc_futures["1"] = fut

        msg = AsyncDiffusionOutput(kind=AsyncOutputKind.COMPUTE_DONE, rpc_id="1", async_output_id="abc")
        _feed_one_msg_to_pump(executor, msg)

        assert fut.cancelled()

    def test_output_ready_racing_cancel_does_not_crash_pump(self):
        executor = _make_executor()
        fut = _racy_cancelled_future()
        with executor._futures_lock:
            executor._output_futures["abc123"] = fut

        msg = AsyncDiffusionOutput(
            kind=AsyncOutputKind.OUTPUT_READY,
            async_output_id="abc123",
            output=DiffusionOutput(output="late"),
        )
        _feed_one_msg_to_pump(executor, msg)

        assert fut.cancelled()
        assert executor._completed_outputs == {}

    def test_batch_split_racing_cancel_does_not_crash_pump(self):
        executor = _make_executor()
        cancelled_fut = _racy_cancelled_future()
        healthy_fut = concurrent.futures.Future()
        with executor._futures_lock:
            executor._output_futures["batch-1/r-aborted"] = cancelled_fut
            executor._output_futures["batch-1/r-healthy"] = healthy_fut
            executor._batch_split_map["batch-1"] = {
                "batch-1/r-aborted": "r-aborted",
                "batch-1/r-healthy": "r-healthy",
            }

        outputs = {
            "r-aborted": DiffusionOutput(output="late"),
            "r-healthy": DiffusionOutput(output="ok"),
        }
        msg = AsyncDiffusionOutput(
            kind=AsyncOutputKind.OUTPUT_READY,
            async_output_id="batch-1",
            output=_FakeBatchOutput(outputs),
        )
        _feed_one_msg_to_pump(executor, msg)

        assert cancelled_fut.cancelled()
        assert healthy_fut.done()
        assert healthy_fut.result(timeout=1.0) is outputs["r-healthy"]
        assert executor._completed_outputs == {}


class TestResultPumpDiscardsDoneFuture:
    """A registered future that is already cancelled/done must have its late
    OUTPUT_READY result discarded, not re-cached into _completed_outputs.

    rahul-steiger-nv (#6439 review): the old `else` branch cached the result
    whenever `pending.done()` was true, so a genuinely cancelled waiter (the
    common abort case) leaked the late result back into the cache. The pump
    must distinguish "no waiter registered" (cache) from "waiter registered
    but done/cancelled" (discard).
    """

    def test_single_output_discards_for_cancelled_future(self):
        executor = _make_executor()
        fut: concurrent.futures.Future = concurrent.futures.Future()
        fut.cancel()
        assert fut.cancelled() and fut.done()
        with executor._futures_lock:
            executor._output_futures["aid-x"] = fut

        msg = AsyncDiffusionOutput(
            kind=AsyncOutputKind.OUTPUT_READY,
            async_output_id="aid-x",
            output=DiffusionOutput(output="late"),
        )
        _feed_one_msg_to_pump(executor, msg)

        assert executor._completed_outputs == {}
        assert "aid-x" not in executor._output_futures

    def test_batch_split_discards_for_cancelled_member(self):
        executor = _make_executor()
        cancelled_fut: concurrent.futures.Future = concurrent.futures.Future()
        cancelled_fut.cancel()
        assert cancelled_fut.cancelled() and cancelled_fut.done()
        healthy_fut: concurrent.futures.Future = concurrent.futures.Future()
        with executor._futures_lock:
            executor._output_futures["batch-9/r-aborted"] = cancelled_fut
            executor._output_futures["batch-9/r-healthy"] = healthy_fut
            executor._batch_split_map["batch-9"] = {
                "batch-9/r-aborted": "r-aborted",
                "batch-9/r-healthy": "r-healthy",
            }

        outputs = {
            "r-aborted": DiffusionOutput(output="late"),
            "r-healthy": DiffusionOutput(output="ok"),
        }
        msg = AsyncDiffusionOutput(
            kind=AsyncOutputKind.OUTPUT_READY,
            async_output_id="batch-9",
            output=_FakeBatchOutput(outputs),
        )
        _feed_one_msg_to_pump(executor, msg)

        # Aborted member's result is discarded, healthy member still delivered.
        assert healthy_fut.done()
        assert healthy_fut.result(timeout=1.0) is outputs["r-healthy"]
        assert executor._completed_outputs == {}


class TestDropOutput:
    """drop_output drains an aborted request's async output so it cannot leak.

    An aborted request never calls wait_output_ready, so a late OUTPUT_READY is
    unpacked and cached in _completed_outputs forever (issue #6413). drop_output
    evicts it, handling both arrival orderings.
    """

    def test_drops_already_cached_output(self):
        executor = _make_executor()
        fut = concurrent.futures.Future()
        fut.set_result(DiffusionOutput(output="leaked"))
        with executor._futures_lock:
            executor._completed_outputs["aid-1"] = fut

        executor.drop_output("aid-1")

        with executor._futures_lock:
            assert "aid-1" not in executor._completed_outputs

    def test_prevents_caching_when_output_not_yet_arrived(self):
        executor = _make_executor()

        # Abort finalized before OUTPUT_READY: register a placeholder waiter.
        executor.drop_output("aid-2")

        # The late result now resolves into the placeholder instead of caching.
        output = DiffusionOutput(output="late")
        msg = AsyncDiffusionOutput(
            kind=AsyncOutputKind.OUTPUT_READY,
            async_output_id="aid-2",
            output=output,
        )
        _feed_one_msg_to_pump(executor, msg)

        with executor._futures_lock:
            assert "aid-2" not in executor._completed_outputs
            assert "aid-2" not in executor._output_futures

    def test_leaves_real_waiter_untouched(self):
        executor = _make_executor()
        real = executor.wait_output_ready("aid-3")

        executor.drop_output("aid-3")

        with executor._futures_lock:
            # A genuine waiter still drains via the normal path.
            assert executor._output_futures.get("aid-3") is real

    def test_drop_then_pump_then_late_wait_fails_fast_without_caching(self):
        """drop_output → OUTPUT_READY → late wait_output_ready.

        The aborted output's tensors must NOT be retained anywhere, and the
        late waiter must get a completed (failed) Future rather than a fresh
        one that never resolves — the verl-omni fully-async abort ordering.
        """
        executor = _make_executor()

        executor.drop_output("aid-dpw")
        msg = AsyncDiffusionOutput(
            kind=AsyncOutputKind.OUTPUT_READY,
            async_output_id="aid-dpw",
            output=DiffusionOutput(output="discarded"),
        )
        _feed_one_msg_to_pump(executor, msg)

        with executor._futures_lock:
            assert "aid-dpw" not in executor._completed_outputs
            assert "aid-dpw" not in executor._output_futures

        fut = executor.wait_output_ready("aid-dpw")
        assert fut.done(), "late wait after drop returned a never-completing future"
        with pytest.raises(RuntimeError, match="was dropped"):
            fut.result(timeout=0)
        # Terminal: the dropped-id LRU keeps the id so any later wait_output_ready
        # on the same id also fails fast instead of hanging on a fresh Future.
        with executor._futures_lock:
            assert "aid-dpw" in executor._dropped_output_ids
        second = executor.wait_output_ready("aid-dpw")
        assert second.done()
        with pytest.raises(RuntimeError, match="was dropped"):
            second.result(timeout=0)

    def test_drop_then_wait_share_placeholder_and_fail_on_delivery(self):
        """drop_output → wait_output_ready (before pump) share one Future, and
        delivery terminates it with the dropped error instead of the tensors."""
        executor = _make_executor()

        executor.drop_output("aid-dw")
        fut = executor.wait_output_ready("aid-dw")
        with executor._futures_lock:
            assert executor._output_futures.get("aid-dw") is fut
        assert not fut.done()

        msg = AsyncDiffusionOutput(
            kind=AsyncOutputKind.OUTPUT_READY,
            async_output_id="aid-dw",
            output=DiffusionOutput(output="discarded"),
        )
        _feed_one_msg_to_pump(executor, msg)

        assert fut.done()
        with pytest.raises(RuntimeError, match="was dropped"):
            fut.result(timeout=0)
        with executor._futures_lock:
            assert "aid-dw" not in executor._completed_outputs
            assert "aid-dw" not in executor._output_futures

    def test_drop_after_cached_then_late_wait_fails_fast(self):
        """Result already cached → drop_output → late wait must not hang."""
        executor = _make_executor()
        fut = concurrent.futures.Future()
        fut.set_result(DiffusionOutput(output="cached"))
        with executor._futures_lock:
            executor._completed_outputs["aid-cd"] = fut

        executor.drop_output("aid-cd")
        late = executor.wait_output_ready("aid-cd")
        assert late.done()
        with pytest.raises(RuntimeError, match="was dropped"):
            late.result(timeout=0)

    def test_dropped_ids_memory_is_bounded(self, monkeypatch):
        from vllm_omni.diffusion.executor import multiproc_executor as mpe

        cap = 8
        monkeypatch.setattr(mpe, "_DROPPED_OUTPUT_IDS_MAX", cap)
        executor = _make_executor()
        n = cap + 5
        for i in range(n):
            executor.drop_output(f"aid-b{i}")
            executor._pump_stop.clear()
            _feed_one_msg_to_pump(
                executor,
                AsyncDiffusionOutput(
                    kind=AsyncOutputKind.OUTPUT_READY,
                    async_output_id=f"aid-b{i}",
                    output=DiffusionOutput(output=i),
                ),
            )
        with executor._futures_lock:
            assert len(executor._dropped_output_ids) == cap
            # Oldest evicted, newest kept; no tensors retained anywhere.
            assert "aid-b0" not in executor._dropped_output_ids
            assert f"aid-b{n - 1}" in executor._dropped_output_ids
            assert executor._completed_outputs == {}
            assert executor._output_futures == {}

    def test_cancelled_waiter_then_pump_then_late_wait_fails_fast_twice(self):
        """Fully-async abort overlap: step_streaming registers a real waiter,
        the async abort cancels it, then the pump delivers OUTPUT_READY.

        Reproduces SamitHuang's 09-14 P1 chain:
          * _finish_output line 1029 discarded a cancelled waiter without
            _remember_dropped → later wait_output_ready allocated a fresh
            Future and hung.
          * wait_output_ready line 1097 returned the cancelled existing future
            as if it were live → abort-after-cancel never installed dropped
            state.
          * wait_output_ready line 1101 deleted the LRU entry on the first
            failed wait → a second late wait on the same id missed the record
            and hung.
        """
        executor = _make_executor()

        # step_streaming registers a real waiter.
        real = executor.wait_output_ready("aid-cx")
        assert not real.done()

        # asyncio.wrap_future(...).cancel() marks the underlying
        # concurrent.futures.Future cancelled.
        assert real.cancel()
        assert real.cancelled()
        assert real.done()

        # Pump delivers OUTPUT_READY for the same id after the cancel.
        msg = AsyncDiffusionOutput(
            kind=AsyncOutputKind.OUTPUT_READY,
            async_output_id="aid-cx",
            output=DiffusionOutput(output="discarded-after-cancel"),
        )
        _feed_one_msg_to_pump(executor, msg)

        with executor._futures_lock:
            # Tensors are not re-cached; the cancelled waiter is gone.
            assert "aid-cx" not in executor._completed_outputs
            assert "aid-cx" not in executor._output_futures
            # The id was recorded as dropped so late waits fail fast.
            assert "aid-cx" in executor._dropped_output_ids

        # Late wait returns an already-failed Future (not a fresh never-completing one).
        late = executor.wait_output_ready("aid-cx")
        assert late.done()
        with pytest.raises(RuntimeError, match="was dropped"):
            late.result(timeout=0)

        # A second late wait must ALSO fail fast: the LRU entry must persist.
        late2 = executor.wait_output_ready("aid-cx")
        assert late2.done()
        with pytest.raises(RuntimeError, match="was dropped"):
            late2.result(timeout=0)
        with executor._futures_lock:
            assert "aid-cx" in executor._dropped_output_ids
            assert "aid-cx" not in executor._output_futures
            assert "aid-cx" not in executor._completed_outputs

    def test_wait_output_ready_replaces_stale_cancelled_existing_future(self):
        """A cancelled non-placeholder Future in _output_futures is not a live
        waiter: the next wait_output_ready must evict it and install dropped
        state rather than returning the cancelled future.

        Reproduces SamitHuang's wait_output_ready:1135 finding: without this,
        abort-after-cancel skips _remember_dropped entirely and downstream
        pump delivery falls into the discard branch that used to lose the id.
        """
        executor = _make_executor()

        stale = executor.wait_output_ready("aid-stale")
        assert stale.cancel()
        assert stale.done() and stale.cancelled()

        replacement = executor.wait_output_ready("aid-stale")
        assert replacement is not stale
        assert replacement.done()
        with pytest.raises(RuntimeError, match="was dropped"):
            replacement.result(timeout=0)
        with executor._futures_lock:
            assert "aid-stale" not in executor._output_futures
            assert "aid-stale" in executor._dropped_output_ids

    def test_pump_after_cancelled_waiter_then_wait_stays_dropped(self):
        """Timeline-B: the cancelled-waiter has already been observed by a
        second wait_output_ready and evicted into the dropped-id LRU, THEN the
        pump delivers OUTPUT_READY.

        Without a dropped-id guard in _finish_output, the pump's ``pending``
        is ``None`` (the second wait already popped the stale entry), so the
        no-waiter branch caches the tensors in ``_completed_outputs``. A
        subsequent wait_output_ready then pops that cache and returns a
        successful Future — violating the "every late wait on a dropped id
        fails the same way" contract from wait_output_ready.
        """
        executor = _make_executor()

        # (1) step_streaming registers a real waiter.
        real = executor.wait_output_ready("aid-tb")
        assert not real.done()

        # (2) asyncio cancels it.
        assert real.cancel()

        # (3) Second wait_output_ready observes the stale entry, evicts it into
        # the dropped-id LRU and returns an already-failed Future.
        first_late = executor.wait_output_ready("aid-tb")
        assert first_late.done()
        with pytest.raises(RuntimeError, match="was dropped"):
            first_late.result(timeout=0)
        with executor._futures_lock:
            assert "aid-tb" not in executor._output_futures
            assert "aid-tb" in executor._dropped_output_ids

        # (4) Pump delivers OUTPUT_READY for the dropped id AFTER the LRU
        # record is already in place.
        msg = AsyncDiffusionOutput(
            kind=AsyncOutputKind.OUTPUT_READY,
            async_output_id="aid-tb",
            output=DiffusionOutput(output="late-after-evict"),
        )
        _feed_one_msg_to_pump(executor, msg)

        # (5) The tensors must NOT be cached, and the id must remain in the LRU.
        with executor._futures_lock:
            assert "aid-tb" not in executor._completed_outputs
            assert "aid-tb" not in executor._output_futures
            assert "aid-tb" in executor._dropped_output_ids

        # (6) A subsequent wait_output_ready must still fail fast, NOT pop a
        # successful cached Future.
        second_late = executor.wait_output_ready("aid-tb")
        assert second_late.done()
        with pytest.raises(RuntimeError, match="was dropped"):
            second_late.result(timeout=0)

    def test_cancel_between_done_check_and_set_result_records_dropped(self):
        """Race: consumer cancels the Future between the pump's ``.done()``
        check and the actual ``set_result``.

        Reproduces Gaohan123's 09-19 P2: ``try_set_result`` catches the
        ``InvalidStateError`` silently, but the waiter has already been
        popped from ``_output_futures`` by ``_finish_output``. Without
        recording the id as dropped, a subsequent ``wait_output_ready``
        allocates a fresh Future that can never complete.
        """

        class _RaceCancellingFuture(concurrent.futures.Future):
            """Future that mimics a consumer cancelling in the tiny window
            between ``pending.done()`` and ``pending.set_result(...)``."""

            def set_result(self, result):
                self.cancel()  # race lands here — mid-branch cancel
                super().set_result(result)  # raises InvalidStateError

            def set_exception(self, exc):
                self.cancel()
                super().set_exception(exc)

        executor = _make_executor()
        racy = _RaceCancellingFuture()
        with executor._futures_lock:
            executor._output_futures["aid-race"] = racy

        # Simulate the pump delivery: at entry .done() is False, then the
        # concurrent cancel lands, then set_result raises InvalidStateError.
        with executor._futures_lock:
            executor._finish_output("aid-race", DiffusionOutput(output="lost"), None)

        assert racy.cancelled()
        with executor._futures_lock:
            # The waiter has been popped and the id must be recorded as dropped.
            assert "aid-race" not in executor._output_futures
            assert "aid-race" not in executor._completed_outputs
            assert "aid-race" in executor._dropped_output_ids

        # Late wait_output_ready must fail fast, not allocate a fresh Future.
        late = executor.wait_output_ready("aid-race")
        assert late.done()
        with pytest.raises(RuntimeError, match="was dropped"):
            late.result(timeout=0)
        # And a second late wait must also fail fast (LRU stays populated).
        late2 = executor.wait_output_ready("aid-race")
        assert late2.done()
        with pytest.raises(RuntimeError, match="was dropped"):
            late2.result(timeout=0)

    def test_batch_split_drop_placeholder_discards_member(self):
        """_deliver_batch_split applies the same contract: a dropped member is
        discarded (never cached), a live member is resolved directly."""
        executor = _make_executor()
        live = executor.wait_output_ready("batch-d/r-live")
        executor.drop_output("batch-d/r-dropped")
        with executor._futures_lock:
            executor._batch_split_map["batch-d"] = {
                "batch-d/r-dropped": "r-dropped",
                "batch-d/r-live": "r-live",
            }
        outputs = {
            "r-dropped": DiffusionOutput(output="discarded"),
            "r-live": DiffusionOutput(output="ok"),
        }
        _feed_one_msg_to_pump(
            executor,
            AsyncDiffusionOutput(
                kind=AsyncOutputKind.OUTPUT_READY,
                async_output_id="batch-d",
                output=_FakeBatchOutput(outputs),
            ),
        )
        assert live.result(timeout=1.0) is outputs["r-live"]
        with executor._futures_lock:
            assert executor._completed_outputs == {}
            assert executor._output_futures == {}
        late = executor.wait_output_ready("batch-d/r-dropped")
        assert late.done()
        with pytest.raises(RuntimeError, match="was dropped"):
            late.result(timeout=0)


class TestShutdownClearsCompletedOutputs:
    """shutdown() must release cached async outputs (issue #6413)."""

    def test_shutdown_clears_completed_outputs(self):
        executor = _make_executor()
        fut = concurrent.futures.Future()
        fut.set_result(DiffusionOutput(output="cached"))
        with executor._futures_lock:
            executor._completed_outputs["aid-4"] = fut

        executor.shutdown()

        assert executor._completed_outputs == {}


def _feed_msgs_to_pump(executor, msgs):
    """Run _result_pump in a daemon thread, feed *msgs* in order, then stop."""
    pending = list(msgs)

    def mock_dequeue(timeout=None):
        if pending:
            return pending.pop(0)
        executor._pump_stop.set()
        time.sleep(0.05)
        raise TimeoutError

    executor._result_mq.dequeue = mock_dequeue
    t = threading.Thread(target=executor._result_pump, daemon=True)
    t.start()
    t.join(timeout=2.0)
    return t


class _ExplodingBatchOutput:
    """Batch output whose extraction fails for one request id."""

    def __init__(self, results, failing_req_id):
        self._results = results
        self._failing_req_id = failing_req_id

    def get_request_output(self, req_id):
        if req_id == self._failing_req_id:
            raise RuntimeError("corrupt per-request result")
        result = self._results.get(req_id)
        if result is None:
            return None
        return SimpleNamespace(result=result)


class TestResultPumpFaultContainment:
    """One bad message must not cost the queue its only reader."""

    def test_dispatch_failure_does_not_stop_the_pump(self, mocker):
        executor = _make_executor()
        mocker.patch.object(
            executor._sync_result_buffer,
            "put",
            side_effect=RuntimeError("sync buffer is gone"),
        )
        fut = concurrent.futures.Future()
        with executor._futures_lock:
            executor._rpc_futures["1"] = fut
        good = AsyncDiffusionOutput(kind=AsyncOutputKind.RPC_RESULT, rpc_id="1")

        thread = _feed_msgs_to_pump(executor, [object(), good])

        # The message after the failure was still delivered, so the pump lived.
        assert fut.done()
        assert fut.result(timeout=1.0) is good
        assert not thread.is_alive()

    def test_output_ready_without_id_is_logged_not_silent(self, mocker):
        from vllm_omni.diffusion.executor import multiproc_executor

        executor = _make_executor()
        log_error = mocker.patch.object(multiproc_executor.logger, "error")
        orphan = AsyncDiffusionOutput(
            kind=AsyncOutputKind.OUTPUT_READY,
            async_output_id=None,
            output=DiffusionOutput(output="img"),
        )

        _feed_msgs_to_pump(executor, [orphan])

        assert executor._completed_outputs == {}
        assert executor._output_futures == {}
        assert log_error.call_count == 1
        assert "async_output_id" in log_error.call_args.args[0]

    def test_one_corrupt_request_does_not_drop_its_siblings(self):
        executor = _make_executor()
        req_ids = ["r0", "r1", "r2"]
        batch_id = "batch-fault"
        outputs = {rid: DiffusionOutput(output=f"img-{rid}") for rid in req_ids}
        with executor._futures_lock:
            executor._batch_split_map[batch_id] = {f"{batch_id}/{rid}": rid for rid in req_ids}
        ready = AsyncDiffusionOutput(
            kind=AsyncOutputKind.OUTPUT_READY,
            async_output_id=batch_id,
            output=_ExplodingBatchOutput(outputs, failing_req_id="r1"),
        )

        _feed_msgs_to_pump(executor, [ready])

        for rid in ("r0", "r2"):
            fut = executor.wait_output_ready(f"{batch_id}/{rid}")
            assert fut.done(), f"request {rid} never resolved"
            assert fut.result(timeout=1.0) is outputs[rid]
        failed = executor.wait_output_ready(f"{batch_id}/r1")
        assert failed.done()
        assert "corrupt per-request result" in failed.result(timeout=1.0).error
