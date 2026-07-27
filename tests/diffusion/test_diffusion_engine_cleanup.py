# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from vllm_omni.diffusion import diffusion_engine as diffusion_engine_module
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine, DiffusionExecutionMode
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import DiffusionRequestStatus, RequestScheduler
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _make_request(request_id: str) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompt=f"prompt_{request_id}",
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
        request_id=request_id,
    )


def _make_engine() -> DiffusionEngine:
    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine.scheduler = RequestScheduler()
    engine.scheduler.initialize(SimpleNamespace())
    engine.executor = SimpleNamespace(shutdown=Mock())
    engine._rpc_lock = threading.RLock()
    engine._cv = threading.Condition(engine._rpc_lock)
    engine._out_streams = {}
    engine._closed = False
    engine._shutdown_complete = False
    engine.abort_queue = queue.Queue()
    engine._loop_started = False
    engine.stop_event = None
    engine.worker_thread = None
    return engine


def test_close_completes_pending_output_streams() -> None:
    engine = _make_engine()
    event_loop = asyncio.new_event_loop()
    try:
        engine.main_loop = event_loop
        queue: asyncio.Queue[DiffusionOutput] = asyncio.Queue()
        engine._out_streams["pending-stream"] = queue

        engine.close()

        output = queue.get_nowait()
        assert output.error == "DiffusionEngine is closed."
        assert output.finished is True
    finally:
        event_loop.close()


def test_emit_finished_outputs_finalizes_already_drained_waiter() -> None:
    class RacingOutQueue(dict):
        def get(self, key, default=None):
            return default

    engine = _make_engine()
    request_id = engine.scheduler.add_request(_make_request("pending-req"))
    engine.scheduler.finish_requests(request_id, DiffusionRequestStatus.FINISHED_ABORTED)
    engine._out_streams = RacingOutQueue()

    engine._emit_finished_outputs({request_id})

    assert engine.scheduler.get_request_state(request_id) is None


def test_emit_step_outputs_finalizes_finished_request_without_stream() -> None:
    engine = _make_engine()
    engine.execution_mode = DiffusionExecutionMode.STEP_BATCH
    request_id = engine.scheduler.add_request(_make_request("step-drained"))
    engine.scheduler.finish_requests(request_id, DiffusionRequestStatus.FINISHED_ABORTED)

    engine._emit_outputs({request_id}, [request_id], SimpleNamespace(get_request_output=lambda _request_id: None))

    assert engine.scheduler.get_request_state(request_id) is None


def test_init_accepts_custom_scheduler(monkeypatch: pytest.MonkeyPatch) -> None:
    od_config = SimpleNamespace(
        custom_pipeline_args=None,
        model_class_name="CustomSchedulerPipeline",
        streaming_output=False,
    )
    custom_scheduler = RequestScheduler()
    fake_executor = SimpleNamespace(
        execute_request=Mock(),
        execute_batch=Mock(),
        execute_step=Mock(),
    )

    monkeypatch.setattr(
        "vllm_omni.diffusion.diffusion_engine.get_diffusion_post_process_func",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.diffusion_engine.get_diffusion_pre_process_func",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.diffusion_engine.DiffusionExecutor.get_class",
        lambda *args, **kwargs: Mock(return_value=fake_executor),
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.diffusion_engine.supports_request_batch",
        lambda *args, **kwargs: False,
    )

    engine = DiffusionEngine(od_config, scheduler=custom_scheduler)

    assert engine.scheduler is custom_scheduler


@pytest.mark.asyncio
async def test_step_compatibility_wrapper_returns_final_batch() -> None:
    engine = _make_engine()
    first = [OmniRequestOutput.from_diffusion(request_id="req", images=[], finished=False)]
    final = [OmniRequestOutput.from_diffusion(request_id="req", images=[], finished=True)]

    async def _step_streaming(_request):
        yield first
        yield final

    engine.step_streaming = _step_streaming  # type: ignore[method-assign]
    with patch.object(diffusion_engine_module.logger, "warning_once") as warning_once:
        output = await engine.step(_make_request("req"))

    assert output is final
    warning_once.assert_called_once()


@pytest.mark.asyncio
async def test_async_wait_compatibility_wrapper_returns_final_output() -> None:
    engine = _make_engine()
    first = DiffusionOutput(output="chunk", finished=False)
    final = DiffusionOutput(output="final", finished=True)

    async def _stream_response(_request):
        yield first
        yield final

    engine.async_add_req_and_stream_response = _stream_response  # type: ignore[method-assign]
    with patch.object(diffusion_engine_module.logger, "warning_once") as warning_once:
        output = await engine.async_add_req_and_wait_for_response(_make_request("req"))

    assert output is final
    warning_once.assert_called_once()


def test_abort_request_id_aborts_scheduler_request() -> None:
    engine = _make_engine()
    request = _make_request("batch-parent")
    request_id = engine.scheduler.add_request(request)

    engine.abort("batch-parent")
    engine._process_aborts_queue()

    state = engine.scheduler.get_request_state(request_id)
    assert state is not None
    assert state.status == DiffusionRequestStatus.FINISHED_ABORTED


def test_close_rejects_late_async_requests() -> None:
    engine = _make_engine()
    event_loop = asyncio.new_event_loop()
    try:
        engine.main_loop = event_loop
        engine.close()

        with pytest.raises(RuntimeError, match="closed"):
            engine.add_request(_make_request("late-req"))
    finally:
        event_loop.close()


def test_close_resets_loop_started_for_dead_worker_thread() -> None:
    engine = _make_engine()
    engine._loop_started = True
    engine.worker_thread = SimpleNamespace(is_alive=Mock(return_value=False))

    engine.close()

    assert engine._loop_started is False


def test_close_defers_resource_shutdown_until_worker_thread_stops() -> None:
    engine = _make_engine()
    engine.scheduler.close = Mock()
    engine._loop_started = True
    worker_thread = SimpleNamespace(
        is_alive=Mock(side_effect=[True, True, False, False]),
        join=Mock(),
    )
    engine.worker_thread = worker_thread

    engine.close()

    worker_thread.join.assert_called_once_with(timeout=10)
    engine.scheduler.close.assert_not_called()
    engine.executor.shutdown.assert_not_called()
    assert engine._shutdown_complete is False
    assert engine._loop_started is True

    engine.close()

    engine.scheduler.close.assert_called_once()
    engine.executor.shutdown.assert_called_once()
    assert engine._shutdown_complete is True
    assert engine._loop_started is False
