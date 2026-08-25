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
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
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


def _make_kv_cleanup_engine(mode: DiffusionKVCacheMode) -> tuple[DiffusionEngine, Mock]:
    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine.od_config = SimpleNamespace(diffusion_kv_mode=mode)
    cleanup = Mock()
    engine.executor = SimpleNamespace(remove_diffusion_kv_requests=cleanup)
    return engine, cleanup


def test_paged_terminal_output_clears_worker_rows() -> None:
    engine, cleanup = _make_kv_cleanup_engine(DiffusionKVCacheMode.PAGED_SCHEDULER)
    engine._finalize_finished_request = lambda request_id, *_args: request_id
    engine._put_output = lambda *_args: None

    engine._emit_finished_outputs({"req-0", "req-1"})

    cleanup.assert_called_once()
    assert set(cleanup.call_args.args[0]) == {"req-0", "req-1"}


def test_paged_abort_clears_worker_rows_before_scheduler_free() -> None:
    engine, cleanup = _make_kv_cleanup_engine(DiffusionKVCacheMode.PAGED_SCHEDULER)
    engine.scheduler = SimpleNamespace(
        get_request_state=lambda _request_id: None,
        finish_requests=Mock(side_effect=AssertionError("unknown request must not be logically freed")),
    )

    engine._abort_requests(["req-idle", "req-idle"])

    cleanup.assert_called_once_with(["req-idle"])
    engine.scheduler.finish_requests.assert_not_called()


def test_dense_terminal_and_abort_paths_skip_worker_row_cleanup() -> None:
    engine, cleanup = _make_kv_cleanup_engine(DiffusionKVCacheMode.DENSE_LEGACY)
    engine._finalize_finished_request = lambda request_id, *_args: request_id
    engine._put_output = lambda *_args: None
    engine.scheduler = SimpleNamespace(get_request_state=lambda _request_id: None)

    engine._emit_finished_outputs({"req-0"})
    engine._abort_requests(["req-idle"])

    cleanup.assert_not_called()


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


def test_scheduler_initialization_failure_closes_scheduler_and_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    od_config = SimpleNamespace(
        custom_pipeline_args=None,
        model_class_name="SchedulerInitializationFailurePipeline",
        streaming_output=False,
    )
    initialization_error = RuntimeError("Scheduler initialization failed")
    custom_scheduler = SimpleNamespace(
        initialize=Mock(side_effect=initialization_error),
        close=Mock(),
    )
    fake_executor = SimpleNamespace(
        shutdown=Mock(),
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

    with pytest.raises(RuntimeError) as exc_info:
        DiffusionEngine(od_config, scheduler=custom_scheduler)

    assert exc_info.value is initialization_error
    custom_scheduler.initialize.assert_called_once_with(od_config)
    custom_scheduler.close.assert_called_once_with()


def test_init_shuts_down_executor_when_kv_control_plane_initialization_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_executor = SimpleNamespace(shutdown=Mock())
    monkeypatch.setattr(DiffusionEngine, "_init_process_hooks", lambda self, config: None)
    monkeypatch.setattr(
        DiffusionEngine,
        "_resolve_execution_mode",
        lambda self, config: DiffusionExecutionMode.REQUEST_BATCH,
    )
    monkeypatch.setattr(
        DiffusionEngine, "_init_executor", lambda self, config: setattr(self, "executor", fake_executor)
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.diffusion_engine.initialize_diffusion_kv_control_plane",
        Mock(side_effect=RuntimeError("KV initialization failed")),
    )

    with pytest.raises(RuntimeError, match="KV initialization failed"):
        DiffusionEngine(SimpleNamespace())

    fake_executor.shutdown.assert_called_once_with()


def test_paged_init_profiles_before_scheduler_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    events = []
    expected_profile_requests = [object()]
    fake_executor = SimpleNamespace(shutdown=Mock())
    kv_cache_config = object()
    kv_vllm_config = object()
    od_config = SimpleNamespace(diffusion_kv_mode="paged_scheduler")

    monkeypatch.setattr(DiffusionEngine, "_init_process_hooks", lambda self, config: None)
    monkeypatch.setattr(
        DiffusionEngine,
        "_resolve_execution_mode",
        lambda self, config: DiffusionExecutionMode.REQUEST_BATCH,
    )
    monkeypatch.setattr(
        DiffusionEngine,
        "_init_executor",
        lambda self, config: setattr(self, "executor", fake_executor),
    )
    monkeypatch.setattr(
        DiffusionEngine,
        "_prepare_diffusion_kv_profile_requests",
        lambda self: events.append("prepare-profile") or expected_profile_requests,
    )

    def initialize(executor, config, *, profile_requests):
        events.append("profile-workers")
        assert executor is fake_executor
        assert config is od_config
        assert profile_requests is expected_profile_requests
        return kv_cache_config, 16, 16, kv_vllm_config

    monkeypatch.setattr(
        "vllm_omni.diffusion.diffusion_engine.initialize_diffusion_kv_control_plane",
        initialize,
    )

    def init_scheduler(self, config, scheduler, *args, **kwargs):
        events.append("initialize-scheduler")
        self.scheduler = SimpleNamespace(close=Mock())

    monkeypatch.setattr(DiffusionEngine, "_init_scheduler", init_scheduler)
    monkeypatch.setattr(DiffusionEngine, "_init_runtime_state", lambda self: None)
    monkeypatch.setattr(DiffusionEngine, "_init_execute_fn", lambda self: None)
    monkeypatch.setattr(DiffusionEngine, "_log_execution_mode", lambda self, config: None)

    DiffusionEngine(od_config)

    assert events == ["prepare-profile", "profile-workers", "initialize-scheduler"]


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
