# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for request-mode CHUNK_READY video output."""

from __future__ import annotations

import asyncio
import concurrent.futures
import queue
import threading
import time
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_omni.diffusion.data import AsyncDiffusionOutput, AsyncOutputKind, DiffusionOutput
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine, DiffusionExecutionMode, _RequestStreamState
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.sched.interface import DiffusionRequestStatus
from vllm_omni.diffusion.worker.diffusion_worker import WorkerProc
from vllm_omni.diffusion.worker.utils import RunnerOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _RequestStreamingPipeline:
    supports_request_streaming_output = True
    supports_request_batch = True


class _StepOnlyPipeline:
    supports_request_batch = True


def _make_executor() -> MultiprocDiffusionExecutor:
    executor = object.__new__(MultiprocDiffusionExecutor)
    executor.od_config = SimpleNamespace(step_execution=False)
    executor._rpc_id_counter = 0
    executor._rpc_id_lock = threading.Lock()
    executor._rpc_futures = {}
    executor._output_futures = {}
    executor._completed_outputs = {}
    executor._dropped_output_ids = OrderedDict()
    executor._batch_split_map = {}
    executor._futures_lock = threading.RLock()
    executor._chunk_ready_handler = None
    executor._pump_running = False
    executor._pump_stop = threading.Event()
    executor._sync_result_buffer = queue.Queue()
    executor._result_mq = MagicMock()
    executor._result_mqs = []
    executor._closed = False
    executor._is_failed = False
    return executor


def _feed_one_msg_to_pump(executor: MultiprocDiffusionExecutor, msg: AsyncDiffusionOutput) -> None:
    call_count = [0]

    def mock_dequeue(timeout=None):
        del timeout
        call_count[0] += 1
        if call_count[0] == 1:
            return msg
        executor._pump_stop.set()
        time.sleep(0.05)
        raise TimeoutError

    executor._result_mq.dequeue = mock_dequeue
    thread = threading.Thread(target=executor._result_pump, daemon=True)
    thread.start()
    thread.join(timeout=2.0)


def _engine_for_chunks() -> tuple[DiffusionEngine, asyncio.Queue]:
    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine._cv = threading.Condition(threading.RLock())
    engine._out_streams = {}
    engine._unclaimed_async_outputs = {}
    engine._shutdown_output_futures = {}
    engine._request_chunk_states = {}
    engine._streaming_playback = {}
    engine._request_cancellations = None
    engine.main_loop = None
    engine.executor = MagicMock()
    engine.scheduler = MagicMock()
    engine.od_config = SimpleNamespace(streaming_output=True)
    engine.execution_mode = DiffusionExecutionMode.REQUEST_BATCH
    engine._pipeline_supports_request_streaming_output = True
    output_queue: asyncio.Queue = asyncio.Queue()
    engine._out_streams["req"] = output_queue
    engine._request_chunk_states["req"] = _RequestStreamState(
        execution_id="rpc-1",
        max_output_bytes=1024,
        expect_streaming=True,
    )
    return engine, output_queue


def test_chunk_ready_kind_and_fields() -> None:
    assert AsyncOutputKind.CHUNK_READY.value == "chunk_ready"
    message = AsyncDiffusionOutput(
        kind=AsyncOutputKind.CHUNK_READY,
        request_id="req",
        execution_id="rpc-1",
        chunk_index=2,
        async_output_id="aid",
    )
    assert message.request_id == "req"
    assert message.execution_id == "rpc-1"
    assert message.chunk_index == 2
    assert message.rpc_id is None


def test_result_pump_routes_chunk_ready_without_completing_rpc(mocker) -> None:
    mocker.patch("vllm_omni.diffusion.executor.multiproc_executor.unpack_diffusion_output_shm")
    executor = _make_executor()
    rpc_future: concurrent.futures.Future = concurrent.futures.Future()
    executor._rpc_futures["7"] = rpc_future
    seen: list[tuple] = []
    executor.register_chunk_ready_handler(lambda *args: seen.append(args))
    payload = torch.zeros(2)
    output = DiffusionOutput(finished=False, chunk_index=0, total_chunks=3, output=payload, async_output_id="chunk-0")
    msg = AsyncDiffusionOutput(
        kind=AsyncOutputKind.CHUNK_READY,
        rpc_id="7",
        execution_id="7",
        request_id="req",
        chunk_index=0,
        async_output_id="chunk-0",
        output=output,
    )

    _feed_one_msg_to_pump(executor, msg)

    assert not rpc_future.done()
    assert "7" in executor._rpc_futures
    assert len(seen) == 1
    request_id, execution_id, chunk_index, delivered, error = seen[0]
    assert request_id == "req"
    assert execution_id == "7"
    assert chunk_index == 0
    assert delivered is output
    assert error is None
    ready = executor.wait_output_ready("chunk-0")
    assert ready.done()
    assert ready.result().output is payload


def test_result_pump_closed_chunk_ready_reclaims_shm(mocker) -> None:
    unpack = mocker.patch("vllm_omni.diffusion.executor.multiproc_executor.unpack_diffusion_output_shm")
    executor = _make_executor()
    executor._closed = True
    seen: list[tuple] = []
    executor.register_chunk_ready_handler(lambda *args: seen.append(args))
    output = DiffusionOutput(finished=False, chunk_index=0, output=torch.zeros(1))
    msg = AsyncDiffusionOutput(
        kind=AsyncOutputKind.CHUNK_READY,
        request_id="req",
        execution_id="rpc-1",
        chunk_index=0,
        async_output_id="chunk-0",
        output=output,
    )

    _feed_one_msg_to_pump(executor, msg)

    unpack.assert_called_once_with(output)
    assert seen == []
    assert executor._completed_outputs == {}


def test_engine_puts_intermediate_chunk_without_scheduler_update() -> None:
    engine, output_queue = _engine_for_chunks()
    payload = torch.zeros(4)
    output = DiffusionOutput(
        finished=False,
        chunk_index=0,
        total_chunks=2,
        output=payload,
        async_output_id="chunk-0",
    )

    engine._on_request_chunk_ready("req", "rpc-1", 0, output, None)

    stub = output_queue.get_nowait()
    assert stub.finished is False
    assert stub.chunk_index == 0
    assert stub.total_chunks == 2
    assert stub.async_output_id == "chunk-0"
    assert stub.output is None
    assert engine._request_chunk_states["req"].next_chunk_index == 1
    assert engine._request_chunk_states["req"].in_flight_bytes == payload.untyped_storage().nbytes()
    engine.scheduler.update_from_output.assert_not_called()
    engine.executor.drop_output.assert_not_called()


def test_engine_chunk_gap_is_protocol_error() -> None:
    engine, output_queue = _engine_for_chunks()
    output = DiffusionOutput(finished=False, chunk_index=2, output=torch.zeros(2), async_output_id="chunk-2")

    engine._on_request_chunk_ready("req", "rpc-1", 2, output, None)

    failed = output_queue.get_nowait()
    assert failed.finished is True
    assert failed.error is not None
    assert "expected chunk 0" in failed.error
    assert engine._request_chunk_states["req"].terminal_emitted is True
    engine.executor.drop_output.assert_called_once_with("chunk-2")
    engine.scheduler.update_from_output.assert_not_called()


def test_engine_duplicate_chunk_is_not_redelivered() -> None:
    engine, output_queue = _engine_for_chunks()
    first = DiffusionOutput(finished=False, chunk_index=0, output=torch.zeros(2), async_output_id="chunk-0")
    duplicate = DiffusionOutput(finished=False, chunk_index=0, output=torch.zeros(2), async_output_id="chunk-0b")

    engine._on_request_chunk_ready("req", "rpc-1", 0, first, None)
    engine._on_request_chunk_ready("req", "rpc-1", 0, duplicate, None)

    assert output_queue.qsize() == 1
    assert output_queue.get_nowait().async_output_id == "chunk-0"
    engine.executor.drop_output.assert_called_once_with("chunk-0b")
    assert engine._request_chunk_states["req"].chunk_emitted_count == 1


def test_engine_streaming_final_does_not_resend_video() -> None:
    engine, output_queue = _engine_for_chunks()
    state = engine._request_chunk_states["req"]
    state.chunk_emitted_count = 2
    state.next_chunk_index = 2
    scheduler_state = SimpleNamespace(status=DiffusionRequestStatus.FINISHED_COMPLETED, error=None)
    engine.scheduler.get_request_state.return_value = scheduler_state
    engine.scheduler.pop_request_state.return_value = scheduler_state
    runner_output = RunnerOutput(request_id="req", finished=True, async_output_id="final-video")

    engine._emit_finished_request("req", runner_output, "missing")

    final = output_queue.get_nowait()
    assert final.finished is True
    assert final.output is None
    assert final.media is None
    assert final.async_output_id is None
    assert final.total_chunks == 2
    assert final.error is None
    engine.executor.drop_output.assert_called_once_with("final-video")
    assert "req" not in engine._request_chunk_states


def test_engine_chunk_error_suppresses_later_success() -> None:
    engine, output_queue = _engine_for_chunks()
    state = engine._request_chunk_states["req"]
    state.chunk_emitted_count = 1
    state.terminal_emitted = True
    state.failed = True
    state.error = "request streaming protocol error"
    scheduler_state = SimpleNamespace(status=DiffusionRequestStatus.FINISHED_COMPLETED, error=None)
    engine.scheduler.get_request_state.return_value = scheduler_state
    engine.scheduler.pop_request_state.return_value = scheduler_state
    runner_output = RunnerOutput(request_id="req", finished=True, async_output_id="final-video")

    engine._emit_finished_request("req", runner_output, "missing")

    assert output_queue.empty()
    engine.executor.drop_output.assert_called_once_with("final-video")


def test_resolve_execution_mode_keeps_request_batch_for_request_streaming() -> None:
    od_config = SimpleNamespace(
        model_class_name="CausVidWanPipeline",
        custom_pipeline_args={"pipeline_class": _RequestStreamingPipeline},
        streaming_output=True,
        step_execution=False,
        max_num_seqs=1,
    )
    engine = DiffusionEngine.__new__(DiffusionEngine)

    mode = engine._resolve_execution_mode(od_config)

    assert mode is DiffusionExecutionMode.REQUEST_BATCH
    assert engine.step_execution is False
    assert od_config.step_execution is False
    assert engine._pipeline_supports_request_streaming_output is True


def test_resolve_execution_mode_still_forces_step_without_request_streaming() -> None:
    od_config = SimpleNamespace(
        model_class_name="StepPipeline",
        custom_pipeline_args={"pipeline_class": _StepOnlyPipeline},
        streaming_output=True,
        step_execution=False,
        max_num_seqs=1,
    )
    engine = DiffusionEngine.__new__(DiffusionEngine)

    mode = engine._resolve_execution_mode(od_config)

    assert mode is DiffusionExecutionMode.STEP_BATCH
    assert engine.step_execution is True
    assert od_config.step_execution is True


def test_worker_emit_chunk_does_not_send_compute_done(mocker) -> None:
    proc = object.__new__(WorkerProc)
    proc.gpu_id = 0
    proc.result_mq = MagicMock()
    proc._result_mq_lock = threading.Lock()
    proc._async_output_queue = queue.Queue()
    proc._async_output_done = threading.Condition()
    proc._async_output_pending = 0
    proc._chunk_output_pending = 0
    proc.od_config = SimpleNamespace(
        step_execution=False,
        video_output_transport=SimpleNamespace(enable_registered_shm=False),
    )
    mocker.patch(
        "vllm_omni.diffusion.worker.diffusion_worker.current_omni_platform.record_device_event",
        return_value=None,
    )
    mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.pack_diffusion_output_shm")
    mock_torch = mocker.MagicMock()
    mock_torch.accelerator.current_accelerator.return_value.type = "cpu"
    mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.torch", mock_torch)

    emit = proc._make_emit_chunk("req", "rpc-9")
    chunk = DiffusionOutput(finished=False, chunk_index=1, total_chunks=4, output=torch.zeros(2))
    emit(chunk)
    with pytest.raises(ValueError, match="finished=False"):
        emit(DiffusionOutput(finished=True, chunk_index=2))

    proc.result_mq.enqueue.assert_not_called()
    assert proc._async_output_pending == 1
    assert proc._chunk_output_pending == 1

    thread = threading.Thread(target=proc._async_output_loop, daemon=True)
    thread.start()
    assert proc.drain_async_outputs(timeout=2.0) is True
    proc._async_output_queue.put(None)
    thread.join(timeout=2.0)

    messages = [call.args[0] for call in proc.result_mq.enqueue.call_args_list]
    assert len(messages) == 1
    message = messages[0]
    assert message.kind is AsyncOutputKind.CHUNK_READY
    assert message.request_id == "req"
    assert message.execution_id == "rpc-9"
    assert message.chunk_index == 1
    assert message.rpc_id == "rpc-9"
    assert message.kind != AsyncOutputKind.COMPUTE_DONE
    assert proc._chunk_output_pending == 0
