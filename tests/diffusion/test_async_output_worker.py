# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for WorkerProc async output logic."""

import queue
import threading
import time
from unittest.mock import MagicMock

import pytest

from vllm_omni.diffusion.data import (
    AsyncDiffusionOutput,
    AsyncOutputKind,
    DiffusionOutput,
    OmniACK,
)
from vllm_omni.diffusion.worker.diffusion_worker import WorkerProc
from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_worker_proc(step_execution=False):
    """Create a minimal WorkerProc-like object with async output state."""
    proc = object.__new__(WorkerProc)
    proc.od_config = MagicMock()
    proc.od_config.step_execution = step_execution
    proc.gpu_id = 0
    proc.result_mq = MagicMock()
    proc._running = True
    proc._async_output_queue = queue.Queue() if not step_execution else None
    proc._async_output_thread = None
    proc._async_output_done = threading.Condition()
    proc._async_output_pending = 0
    proc._result_mq_lock = threading.Lock()
    return proc


class _OverlapDetectingQueue:
    """Fails if two threads are inside enqueue() at the same time.

    Mirrors vLLM's MessageQueue, whose acquire_write() reads current_idx,
    writes that block, then advances it. Overlapping writers can target the
    same ring block and silently drop a message.
    """

    def __init__(self):
        self._active = 0
        self._guard = threading.Lock()
        self.overlaps = 0
        self.messages = []

    def enqueue(self, msg, timeout=None):
        with self._guard:
            self._active += 1
            if self._active > 1:
                self.overlaps += 1
        time.sleep(0.001)  # widen the window a real ring-buffer write would have
        with self._guard:
            self.messages.append(msg)
            self._active -= 1


class TestResultQueueSingleWriter:
    """Worker must serialize writes to the single-writer result queue."""

    def test_concurrent_enqueues_do_not_overlap(self):
        proc = _make_worker_proc(step_execution=False)
        detector = _OverlapDetectingQueue()
        proc.result_mq = detector

        def writer(tag):
            for i in range(20):
                proc._enqueue_result(f"{tag}-{i}")

        threads = [threading.Thread(target=writer, args=(tag,)) for tag in ("compute", "output")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert detector.overlaps == 0, f"{detector.overlaps} overlapping writes to result_mq"
        assert len(detector.messages) == 40


class TestGenerateAsyncOutputId:
    """Test _generate_async_output_id uniqueness."""

    def test_returns_hex_string(self):
        id1 = WorkerProc._generate_async_output_id()
        assert isinstance(id1, str)
        assert len(id1) == 32  # uuid4 hex is 32 chars

    def test_ids_are_unique(self):
        ids = {WorkerProc._generate_async_output_id() for _ in range(100)}
        assert len(ids) == 100


class TestReturnResultAsyncPath:
    """Test _return_result in request-mode (step_execution=False)."""

    def test_omni_ack_bypasses_async_path(self):
        """OmniACK should always use sync path regardless of step_execution."""
        proc = _make_worker_proc(step_execution=False)
        ack = OmniACK(task_id="task1", status="done")

        proc._return_result(ack)
        proc.result_mq.enqueue.assert_called_once_with(ack)
        assert proc._async_output_queue.empty()  # nothing enqueued for bg thread

    def test_diffusion_output_triggers_async_path(self, mocker):
        """DiffusionOutput in request-mode should enqueue compute_done + bg work."""
        proc = _make_worker_proc(step_execution=False)

        mock_platform = mocker.MagicMock()
        mock_platform.record_device_event.return_value = MagicMock()
        mocker.patch(
            "vllm_omni.diffusion.worker.diffusion_worker.current_omni_platform",
            mock_platform,
        )

        output = DiffusionOutput(output="data")
        proc._return_result(output, rpc_id="1")

        # Should have enqueued COMPUTE_DONE to result_mq
        enqueued_msg = proc.result_mq.enqueue.call_args[0][0]
        assert isinstance(enqueued_msg, AsyncDiffusionOutput)
        assert enqueued_msg.kind == AsyncOutputKind.COMPUTE_DONE
        assert enqueued_msg.rpc_id == "1"
        assert enqueued_msg.async_output_id is not None

        # Should have placed work item in async_output_queue
        assert not proc._async_output_queue.empty()
        item = proc._async_output_queue.get_nowait()
        assert item[0] is output
        assert item[1] == enqueued_msg.async_output_id

    def test_batch_runner_output_triggers_async_path(self, mocker):
        """BatchRunnerOutput in request-mode should also go through async path."""
        proc = _make_worker_proc(step_execution=False)

        mock_platform = mocker.MagicMock()
        mock_platform.record_device_event.return_value = MagicMock()
        mocker.patch(
            "vllm_omni.diffusion.worker.diffusion_worker.current_omni_platform",
            mock_platform,
        )

        batch = BatchRunnerOutput(
            runner_outputs=[RunnerOutput(request_id="r1", finished=True)],
        )
        proc._return_result(batch)

        enqueued_msg = proc.result_mq.enqueue.call_args[0][0]
        assert isinstance(enqueued_msg, AsyncDiffusionOutput)
        assert enqueued_msg.kind == AsyncOutputKind.COMPUTE_DONE

    def test_none_result_mq_skips_all_paths(self):
        """If result_mq is None, _return_result should do nothing."""
        proc = _make_worker_proc(step_execution=False)
        proc.result_mq = None

        proc._return_result(DiffusionOutput())
        # No crash, no enqueue attempted


class TestReturnResultSyncPath:
    """Test _return_result in step-execution mode (sync)."""

    def test_diffusion_output_uses_sync_path_when_step_execution(self, mocker):
        """In step_execution mode, DiffusionOutput goes through sync SHM pack."""
        proc = _make_worker_proc(step_execution=True)
        proc._async_output_queue = None  # step mode doesn't have async queue

        mock_pack = mocker.patch(
            "vllm_omni.diffusion.worker.diffusion_worker.pack_diffusion_output_shm",
        )
        output = DiffusionOutput(output="data")
        proc._return_result(output)

        mock_pack.assert_called_once_with(output)
        proc.result_mq.enqueue.assert_called_once_with(output)

    def test_non_diffusion_output_skips_async_in_request_mode(self, mocker):
        """Non-DiffusionOutput/BatchRunnerOutput in request mode uses sync path."""
        proc = _make_worker_proc(step_execution=False)

        mock_pack = mocker.patch(
            "vllm_omni.diffusion.worker.diffusion_worker.pack_diffusion_output_shm",
        )
        # A plain dict is neither DiffusionOutput nor BatchRunnerOutput
        output = {"type": "other"}
        proc._return_result(output)

        mock_pack.assert_called_once_with(output)
        proc.result_mq.enqueue.assert_called_once_with(output)


class TestAsyncOutputLoopLogic:
    """Test _async_output_loop background thread logic (mocked, no device)."""

    def test_bg_thread_enqueues_output_ready_after_pack(self, mocker):
        """The real _async_output_loop packs via SHM with d2h_stream and
        enqueues an OUTPUT_READY message."""
        import threading
        import time

        proc = _make_worker_proc(step_execution=False)

        mock_pack = mocker.patch(
            "vllm_omni.diffusion.worker.diffusion_worker.pack_diffusion_output_shm",
        )
        # Mock torch to avoid device initialization
        mock_torch = mocker.MagicMock()
        mock_stream = mocker.MagicMock()
        mock_torch.Stream.return_value = mock_stream
        mock_torch.accelerator.current_accelerator.return_value.type = "cpu"
        mock_torch.device.return_value = mocker.MagicMock()
        mocker.patch(
            "vllm_omni.diffusion.worker.diffusion_worker.torch",
            mock_torch,
        )

        output = DiffusionOutput(output="data")
        gpu_event = mocker.MagicMock()
        proc._async_output_queue.put((output, "abc123", gpu_event))

        # Run the real _async_output_loop in a background thread.
        t = threading.Thread(target=proc._async_output_loop, daemon=True)
        t.start()

        # Give the loop time to process, then send the shutdown sentinel.
        time.sleep(0.5)
        proc._async_output_queue.put(None)
        t.join(timeout=2.0)

        mock_pack.assert_called()
        # Verify an OUTPUT_READY with the correct async_output_id was enqueued.
        output_ready_enqueued = any(
            c[0][0].kind == AsyncOutputKind.OUTPUT_READY and c[0][0].async_output_id == "abc123"
            for c in proc.result_mq.enqueue.call_args_list
        )
        assert output_ready_enqueued, "Expected OUTPUT_READY with async_output_id='abc123' in enqueue calls"


class TestDrainAsyncOutputs:
    """Async outputs must finish packing before device memory is released."""

    def test_drain_returns_immediately_when_nothing_in_flight(self):
        proc = _make_worker_proc(step_execution=False)

        assert proc.drain_async_outputs(timeout=0.01) is True

    def test_return_result_keeps_output_pending_until_loop_completes(self, mocker):
        """_return_result marks work pending; the bg loop clears it."""
        proc = _make_worker_proc(step_execution=False)

        mock_platform = mocker.MagicMock()
        mock_platform.record_device_event.return_value = MagicMock()
        mocker.patch(
            "vllm_omni.diffusion.worker.diffusion_worker.current_omni_platform",
            mock_platform,
        )
        mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.pack_diffusion_output_shm")
        mock_torch = mocker.MagicMock()
        mock_torch.accelerator.current_accelerator.return_value.type = "cpu"
        mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.torch", mock_torch)

        proc._return_result(DiffusionOutput(output="data"), rpc_id="1")

        # Packing has not run yet, so a sleep here would free tensors in flight.
        assert proc._async_output_pending == 1
        assert proc.drain_async_outputs(timeout=0.05) is False

        thread = threading.Thread(target=proc._async_output_loop, daemon=True)
        thread.start()
        try:
            assert proc.drain_async_outputs(timeout=5.0) is True
        finally:
            proc._async_output_queue.put(None)
            thread.join(timeout=2.0)

    def test_sleep_rpc_drains_before_releasing_memory(self, mocker):
        proc = _make_worker_proc(step_execution=False)
        proc.worker = MagicMock()

        call_order: list[str] = []
        mocker.patch.object(
            proc,
            "drain_async_outputs",
            side_effect=lambda *a, **kw: call_order.append("drain") or True,
        )
        proc.worker.execute_method.side_effect = lambda *a, **kw: call_order.append("sleep")

        proc._execute_rpc(
            {
                "method": "handle_sleep_task",
                "args": (),
                "kwargs": {},
                "output_rank": 0,
                "exec_all_ranks": False,
                "collect_rank_status": False,
            }
        )

        assert call_order == ["drain", "sleep"]

    def test_non_sleep_rpc_does_not_drain(self, mocker):
        proc = _make_worker_proc(step_execution=False)
        proc.worker = MagicMock()
        drain = mocker.patch.object(proc, "drain_async_outputs", return_value=True)

        proc._execute_rpc(
            {
                "method": "execute_model",
                "args": (),
                "kwargs": {},
                "output_rank": 0,
                "exec_all_ranks": False,
                "collect_rank_status": False,
            }
        )

        drain.assert_not_called()


class TestWorkerProcShutdown:
    def test_shutdown_stops_async_thread_and_releases_queues(self):
        import threading

        proc = _make_worker_proc(step_execution=False)
        proc.worker = MagicMock()
        proc.mq = MagicMock()
        broadcast_mq = proc.mq
        result_mq = proc.result_mq

        thread = threading.Thread(target=proc._async_output_queue.get)
        proc._async_output_thread = thread
        thread.start()

        proc.shutdown()

        assert not thread.is_alive()
        proc.worker.shutdown.assert_called_once_with()
        broadcast_mq.shutdown.assert_called_once_with()
        result_mq.shutdown.assert_called_once_with()
        assert proc.mq is None
        assert proc.result_mq is None

    def test_shutdown_releases_queues_when_worker_shutdown_fails(self):
        proc = _make_worker_proc(step_execution=True)
        proc.worker = MagicMock()
        proc.worker.shutdown.side_effect = RuntimeError("shutdown failed")
        proc.mq = MagicMock()
        broadcast_mq = proc.mq
        result_mq = proc.result_mq

        with pytest.raises(RuntimeError, match="shutdown failed"):
            proc.shutdown()

        broadcast_mq.shutdown.assert_called_once_with()
        result_mq.shutdown.assert_called_once_with()
        assert proc.mq is None
        assert proc.result_mq is None
