# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for MultiprocDiffusionExecutor async result pump and wait_output_ready."""

import concurrent.futures
import copy
import queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_omni.diffusion.data import AsyncDiffusionOutput, AsyncOutputKind, DiffusionOutput
from vllm_omni.errors import OmniClientError, client_error_metadata

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

    @pytest.mark.parametrize("client_error", [True, False])
    @pytest.mark.parametrize("collect_rank_status", [True, False])
    def test_worker_rpc_error_round_trip(self, mocker, client_error, collect_rank_status):
        from vllm_omni.diffusion.worker.diffusion_worker import WorkerProc

        executor = _make_executor()
        future = concurrent.futures.Future()
        executor._rpc_futures["1"] = future
        message = "guide[0] interval is outside the target; OmniClientError is not a type tag"
        original = (
            OmniClientError(message, error_type="GuideValidationError") if client_error else RuntimeError(message)
        )
        proc = object.__new__(WorkerProc)
        proc.gpu_id = 0
        proc.od_config = SimpleNamespace(step_execution=False)
        proc.worker = SimpleNamespace(execute_method=MagicMock(side_effect=original))
        proc.result_mq = MagicMock()
        proc._result_mq_lock = threading.Lock()
        proc._running = True
        proc.recv_message = MagicMock(
            side_effect=[
                {
                    "type": "rpc",
                    "method": "execute_model",
                    "rpc_id": "1",
                    "output_rank": 0,
                    "exec_all_ranks": True,
                    "collect_rank_status": collect_rank_status,
                },
                {"type": "shutdown"},
            ]
        )
        mocker.patch("vllm_omni.diffusion.worker.diffusion_worker._cleanup_after_execution_error")
        mocker.patch.object(proc, "_gather_rpc_rank_statuses", side_effect=lambda status: [status])
        proc._worker_busy_loop()

        proc.result_mq.enqueue.assert_called_once()
        # Real transport serialization is covered in test_diffusion_ipc.py.
        wire_message = copy.deepcopy(proc.result_mq.enqueue.call_args.args[0])
        _feed_one_msg_to_pump(executor, wire_message)
        with pytest.raises(OmniClientError if client_error else RuntimeError) as caught:
            future.result(timeout=1)
        output = DiffusionOutput.from_exception(caught.value)
        assert message in output.error
        assert output.error_status_code == (400 if client_error else None)
        assert output.error_type == ("GuideValidationError" if client_error else None)
        assert client_error_metadata(caught.value) == (output.error_status_code, output.error_type)
        assert (output.error_status_code or 500) == (400 if client_error else 500)

    @pytest.mark.parametrize("unknown_peer_error", [True, False])
    def test_async_rejection_waits_for_all_rank_terminal_statuses(self, mocker, unknown_peer_error):
        from vllm_omni.diffusion.worker.diffusion_worker import WorkerProc

        executor = _make_executor()
        entered = threading.Event()
        release = threading.Event()
        barrier = threading.Barrier(2)
        statuses = [None, None]
        workers = []
        messages = []
        mocker.patch("vllm_omni.diffusion.worker.diffusion_worker._cleanup_after_execution_error")

        def execute(rank):
            if rank == 1:
                entered.set()
                assert release.wait(5)
                if unknown_peer_error:
                    raise RuntimeError("unknown failure on rank 1")
            raise OmniClientError(f"invalid guide on rank {rank}")

        def gather(rank, status):
            statuses[rank] = status
            barrier.wait(timeout=5)
            return statuses

        def enqueue(message):
            messages.append(copy.deepcopy(message))
            _feed_one_msg_to_pump(executor, messages[-1])

        def broadcast(request):
            assert request["collect_rank_status"] is True
            for rank in range(2):
                proc = object.__new__(WorkerProc)
                proc.gpu_id = rank
                proc.result_mq = SimpleNamespace(enqueue=enqueue)
                proc._result_mq_lock = threading.Lock()
                proc.worker = SimpleNamespace(execute_method=lambda *args, r=rank, **kwargs: execute(r))
                proc._gather_rpc_rank_statuses = lambda status, r=rank: gather(r, status)
                proc._running = True
                proc.recv_message = MagicMock(side_effect=[request, {"type": "shutdown"}])
                thread = threading.Thread(target=proc._worker_busy_loop)
                workers.append(thread)
                thread.start()

        executor._broadcast_mq.enqueue = broadcast
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(
                executor.collective_rpc, "execute_model", timeout=5, unique_reply_rank=0, exec_all_ranks=True
            )
            try:
                assert entered.wait(2)
                assert not result.done()
                assert messages == []
            finally:
                release.set()
            with pytest.raises(RuntimeError if unknown_peer_error else OmniClientError) as caught:
                result.result(timeout=5)
            output = DiffusionOutput.from_exception(caught.value)
            assert output.error_status_code == (None if unknown_peer_error else 400)
            assert ("unknown failure on rank 1" if unknown_peer_error else "invalid guide on rank 1") in output.error
        for worker in workers:
            worker.join(timeout=2)
            assert not worker.is_alive()
        assert len(messages) == 1
        assert executor._rpc_futures == {}

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

    @pytest.mark.parametrize("deliver_early", [True, False])
    @pytest.mark.parametrize("client_error", [True, False])
    def test_error_metadata_survives_batch_split(self, deliver_early, client_error):
        executor = _make_executor()
        executor.od_config.parallel_config.data_parallel_size = 1
        message = AsyncDiffusionOutput(
            kind=AsyncOutputKind.OUTPUT_READY,
            async_output_id="batch-error",
            error="invalid guide interval" if client_error else "packing failed",
            error_status_code=400 if client_error else None,
            error_type="GuideValidationError" if client_error else None,
        )

        def collective_rpc(*args, **kwargs):
            if deliver_early:
                _feed_one_msg_to_pump(executor, message)
            return AsyncDiffusionOutput(kind=AsyncOutputKind.COMPUTE_DONE, async_output_id="batch-error")

        executor.collective_rpc = collective_rpc
        executor.execute_batch(_make_scheduler_output(["A", "B"]))
        if not deliver_early:
            _feed_one_msg_to_pump(executor, message)
        for request_id in ["A", "B"]:
            output = executor.wait_output_ready(f"batch-error/{request_id}").result(timeout=1)
            assert output.error == message.error
            assert output.error_status_code == message.error_status_code
            assert output.error_type == message.error_type

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
