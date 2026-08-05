# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for WorkerProc async output logic."""

import queue
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
    return proc


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

        proc = _make_worker_proc(step_execution=False)

        mock_pack = mocker.patch(
            "vllm_omni.diffusion.worker.diffusion_worker.pack_diffusion_output_shm",
        )
        packed = threading.Event()
        mock_pack.side_effect = lambda *_args, **_kwargs: packed.set()
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

        assert packed.wait(timeout=2.0)
        proc._async_output_thread = t
        proc._stop_async_output_thread()

        mock_pack.assert_called()
        assert not t.is_alive()
        assert proc._async_output_thread is None
        assert proc._async_output_queue is None
        # Verify an OUTPUT_READY with the correct async_output_id was enqueued.
        output_ready_enqueued = any(
            c[0][0].kind == AsyncOutputKind.OUTPUT_READY and c[0][0].async_output_id == "abc123"
            for c in proc.result_mq.enqueue.call_args_list
        )
        assert output_ready_enqueued, "Expected OUTPUT_READY with async_output_id='abc123' in enqueue calls"

    def test_stop_async_output_thread_drains_pending_outputs(self, mocker):
        import threading

        proc = _make_worker_proc(step_execution=False)
        first_pack_started = threading.Event()
        release_first_pack = threading.Event()
        pack_count = 0

        def blocking_pack(*_args, **_kwargs):
            nonlocal pack_count
            pack_count += 1
            if pack_count == 1:
                first_pack_started.set()
                assert release_first_pack.wait(timeout=5.0)

        mocker.patch(
            "vllm_omni.diffusion.worker.diffusion_worker.pack_diffusion_output_shm",
            side_effect=blocking_pack,
        )
        mock_torch = mocker.MagicMock()
        mock_torch.Stream.return_value = mocker.MagicMock()
        mock_torch.accelerator.current_accelerator.return_value.type = "cpu"
        mock_torch.device.return_value = mocker.MagicMock()
        mocker.patch(
            "vllm_omni.diffusion.worker.diffusion_worker.torch",
            mock_torch,
        )

        proc._async_output_queue.put((DiffusionOutput(output="first"), "first-id", None))
        proc._async_output_queue.put((DiffusionOutput(output="second"), "second-id", None))
        output_thread = threading.Thread(target=proc._async_output_loop, daemon=True)
        proc._async_output_thread = output_thread
        output_thread.start()
        assert first_pack_started.wait(timeout=2.0)

        stop_started = threading.Event()

        def stop_output_thread():
            stop_started.set()
            proc._stop_async_output_thread()

        stop_thread = threading.Thread(target=stop_output_thread)
        stop_thread.start()
        assert stop_started.wait(timeout=2.0)
        assert stop_thread.is_alive()

        release_first_pack.set()
        stop_thread.join(timeout=2.0)

        assert not stop_thread.is_alive()
        assert not output_thread.is_alive()
        assert pack_count == 2
        ready_ids = {
            call.args[0].async_output_id
            for call in proc.result_mq.enqueue.call_args_list
            if call.args[0].kind == AsyncOutputKind.OUTPUT_READY
        }
        assert ready_ids == {"first-id", "second-id"}

    def test_stop_async_output_thread_is_noop_without_async_state(self):
        proc = _make_worker_proc(step_execution=True)

        assert proc._stop_async_output_thread()

        assert proc._running is False
        assert proc._async_output_thread is None
        assert proc._async_output_queue is None

    def test_stop_async_output_thread_timeout_is_bounded(self, mocker):
        proc = _make_worker_proc(step_execution=False)
        output_thread = mocker.MagicMock()
        output_thread.is_alive.return_value = True
        proc._async_output_thread = output_thread
        mocker.patch.object(WorkerProc, "_ASYNC_OUTPUT_JOIN_TIMEOUT_S", 0.01)

        assert not proc._stop_async_output_thread()

        output_thread.join.assert_called_once_with(timeout=0.01)
        assert proc._async_output_thread is output_thread
        assert proc._async_output_queue.get_nowait() is WorkerProc._ASYNC_OUTPUT_STOP
