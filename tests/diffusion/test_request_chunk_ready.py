# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for request-mode CHUNK_READY video output."""

from __future__ import annotations

import asyncio
import concurrent.futures
import queue
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_omni.diffusion.data import AsyncDiffusionOutput, AsyncOutputKind, DiffusionOutput
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine, DiffusionExecutionMode
from vllm_omni.diffusion.request_streaming import RequestStreamState
from vllm_omni.diffusion.sched.interface import DiffusionRequestStatus
from vllm_omni.diffusion.worker.diffusion_worker import WorkerProc
from vllm_omni.diffusion.worker.utils import RunnerOutput

from tests.diffusion.test_result_pump import _make_executor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _RequestStreamingPipeline:
    supports_request_batch = True


class _StepOnlyPipeline:
    supports_request_batch = True
    supports_step_execution = True

    def prepare_encode(self): ...

    def denoise_step(self): ...

    def step_scheduler(self): ...

    def post_decode(self): ...


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
    engine.step_execution = False
    engine.execution_mode = DiffusionExecutionMode.REQUEST_BATCH
    output_queue: asyncio.Queue = asyncio.Queue()
    engine._out_streams["req"] = output_queue
    engine._request_chunk_states["req"] = RequestStreamState(
        max_output_bytes=1024,
    )
    return engine, output_queue


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

    engine._on_request_chunk_ready("req", 0, output, None)

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


def test_engine_chunk_gap_is_protocol_error(mocker) -> None:
    engine, output_queue = _engine_for_chunks()
    abort = mocker.patch.object(engine, "abort")
    output = DiffusionOutput(finished=False, chunk_index=2, output=torch.zeros(2), async_output_id="chunk-2")

    engine._on_request_chunk_ready("req", 2, output, None)

    failed = output_queue.get_nowait()
    assert failed.finished is True
    assert failed.error is not None
    assert "expected chunk 0" in failed.error
    assert engine._request_chunk_states["req"].terminal_emitted is True
    engine.executor.drop_output.assert_called_once_with("chunk-2")
    abort.assert_called_once_with("req")
    engine.scheduler.update_from_output.assert_not_called()


def test_engine_duplicate_chunk_is_not_redelivered() -> None:
    engine, output_queue = _engine_for_chunks()
    first = DiffusionOutput(finished=False, chunk_index=0, output=torch.zeros(2), async_output_id="chunk-0")
    duplicate = DiffusionOutput(finished=False, chunk_index=0, output=torch.zeros(2), async_output_id="chunk-0b")

    engine._on_request_chunk_ready("req", 0, first, None)
    engine._on_request_chunk_ready("req", 0, duplicate, None)

    assert output_queue.qsize() == 1
    assert output_queue.get_nowait().async_output_id == "chunk-0"
    engine.executor.drop_output.assert_called_once_with("chunk-0b")
    assert engine._request_chunk_states["req"].next_chunk_index == 1


def test_engine_streaming_final_does_not_resend_video() -> None:
    engine, output_queue = _engine_for_chunks()
    state = engine._request_chunk_states["req"]
    state.next_chunk_index = 2
    scheduler_state = SimpleNamespace(status=DiffusionRequestStatus.FINISHED_COMPLETED, error=None)
    engine.scheduler.get_request_state.return_value = scheduler_state
    engine.scheduler.pop_request_state.return_value = scheduler_state
    packed = DiffusionOutput(finished=True, output=torch.ones(4), async_output_id="final-video")
    fut: concurrent.futures.Future[DiffusionOutput] = concurrent.futures.Future()
    fut.set_result(packed)
    engine.executor.wait_output_ready.return_value = fut
    runner_output = RunnerOutput(request_id="req", finished=True, async_output_id="final-video")

    engine._emit_finished_request("req", runner_output, "missing")

    final = output_queue.get_nowait()
    assert final.finished is True
    assert final.async_output_id is None
    assert final.total_chunks == 2
    assert final.error is None
    engine.executor.wait_output_ready.assert_called_once_with("final-video")
    engine.executor.drop_output.assert_not_called()
    assert "req" not in engine._request_chunk_states


def test_engine_streaming_final_preserves_async_pack_error() -> None:
    engine, output_queue = _engine_for_chunks()
    state = engine._request_chunk_states["req"]
    state.next_chunk_index = 1
    scheduler_state = SimpleNamespace(status=DiffusionRequestStatus.FINISHED_COMPLETED, error=None)
    engine.scheduler.get_request_state.return_value = scheduler_state
    engine.scheduler.pop_request_state.return_value = scheduler_state
    fut: concurrent.futures.Future[DiffusionOutput] = concurrent.futures.Future()
    fut.set_exception(RuntimeError("pack boom"))
    engine.executor.wait_output_ready.return_value = fut
    runner_output = RunnerOutput(request_id="req", finished=True, async_output_id="final-bad")

    engine._emit_finished_request("req", runner_output, "missing")

    final = output_queue.get_nowait()
    assert final.finished is True
    assert final.error is not None
    assert "pack boom" in final.error
    engine.executor.wait_output_ready.assert_called_once_with("final-bad")


def test_engine_chunk_error_suppresses_later_success() -> None:
    engine, output_queue = _engine_for_chunks()
    state = engine._request_chunk_states["req"]
    state.next_chunk_index = 1
    state.terminal_emitted = True
    scheduler_state = SimpleNamespace(status=DiffusionRequestStatus.FINISHED_COMPLETED, error=None)
    engine.scheduler.get_request_state.return_value = scheduler_state
    engine.scheduler.pop_request_state.return_value = scheduler_state
    runner_output = RunnerOutput(request_id="req", finished=True, async_output_id="final-video")

    engine._emit_finished_request("req", runner_output, "missing")

    assert output_queue.empty()
    engine.executor.drop_output.assert_called_once_with("final-video")


def test_resolve_execution_mode_keeps_request_batch_for_request_streaming(mocker) -> None:
    od_config = SimpleNamespace(
        model_class_name="CausVidWanPipeline",
        custom_pipeline_args={"pipeline_class": _RequestStreamingPipeline},
        streaming_output=True,
        step_execution=False,
        max_num_seqs=1,
    )
    warn_once = mocker.patch("vllm_omni.diffusion.diffusion_engine.logger.warning_once")
    engine = DiffusionEngine.__new__(DiffusionEngine)

    mode = engine._resolve_execution_mode(od_config)

    assert mode is DiffusionExecutionMode.REQUEST_BATCH
    assert engine.step_execution is False
    warn_once.assert_called_once()
    assert "request chunk emitter" in warn_once.call_args.args[0]


def test_resolve_execution_mode_auto_enables_existing_step_contract() -> None:
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


def test_runner_binds_request_identity_and_suppresses_batches() -> None:
    from vllm_omni.diffusion.request_streaming import (
        bind_emit_factory,
        get_emit_chunk,
        request_chunk_emitter,
        reset_emit_factory,
    )

    factory = MagicMock(return_value=MagicMock())
    token = bind_emit_factory(factory)
    req = SimpleNamespace(request_id="req-1")
    try:
        with request_chunk_emitter([req], enabled=True):
            assert get_emit_chunk() is factory.return_value
            factory.assert_called_once_with("req-1")
        assert get_emit_chunk() is None
        factory.reset_mock()
        with request_chunk_emitter([req, SimpleNamespace(request_id="req-2")], enabled=True):
            assert get_emit_chunk() is None
        with request_chunk_emitter([req], enabled=False):
            assert get_emit_chunk() is None
        factory.assert_not_called()
    finally:
        reset_emit_factory(token)


def test_chunk_delivered_while_pipeline_forward_is_blocked() -> None:
    """Engine pump delivers media before the producing forward returns.

    Mid-forward Worker emit/packer coverage is
    ``test_chunk_reaches_result_mq_while_request_chunk_transport_open``.
    """
    from vllm_omni.diffusion.request_streaming import (
        bind_emit_factory,
        get_emit_chunk,
        request_chunk_emitter,
        reset_emit_factory,
    )

    release = threading.Event()
    delivered = threading.Event()
    returned = threading.Event()
    executor = _make_executor()
    engine, output_queue = _engine_for_chunks()
    engine.executor = executor

    def deliver(*args):
        engine._on_request_chunk_ready(*args)
        delivered.set()

    executor.register_chunk_ready_handler(deliver)
    messages = queue.Queue()

    def receive(timeout=None):
        try:
            return messages.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError from None

    executor._result_mq.dequeue = receive
    pump = threading.Thread(target=executor._result_pump, daemon=True)
    pump.start()

    def pipeline_forward():
        emit = get_emit_chunk()
        assert emit is not None
        emit(DiffusionOutput(finished=False, chunk_index=0, output=torch.zeros(2)))
        assert release.wait(timeout=5)

    def run():
        token = bind_emit_factory(
            lambda request_id: (
                lambda output: messages.put(
                    AsyncDiffusionOutput(
                        kind=AsyncOutputKind.CHUNK_READY,
                        request_id=request_id,
                        chunk_index=0,
                        async_output_id="first-chunk",
                        output=output,
                    )
                )
            )
        )
        try:
            with request_chunk_emitter([SimpleNamespace(request_id="req")], enabled=True):
                pipeline_forward()
            returned.set()
        finally:
            reset_emit_factory(token)

    forward = threading.Thread(target=run, daemon=True)
    forward.start()
    try:
        assert delivered.wait(timeout=2)
        assert not returned.is_set()
        placeholder = output_queue.get_nowait()
        media = executor.wait_output_ready(placeholder.async_output_id).result(timeout=1)
        assert media.output is not None
        assert media.output.shape == (2,)
    finally:
        release.set()
        forward.join(timeout=2)
        executor._pump_stop.set()
        pump.join(timeout=2)
    assert returned.is_set()
    assert not forward.is_alive() and not pump.is_alive()


def test_engine_drops_chunk_after_request_retirement() -> None:
    engine, outputs = _engine_for_chunks()
    engine._request_chunk_states.pop("req")
    chunk = DiffusionOutput(finished=False, chunk_index=0, output=torch.zeros(2), async_output_id="late")
    engine._on_request_chunk_ready("req", 0, chunk, None)
    assert outputs.empty()
    engine.executor.drop_output.assert_called_once_with("late")


@pytest.mark.parametrize("fail_forward", [False, True])
def test_transport_closes_and_drains_before_rpc_returns(fail_forward) -> None:
    from vllm_omni.diffusion.request_streaming import get_emit_chunk, request_chunk_emitter

    proc = object.__new__(WorkerProc)
    proc.od_config = SimpleNamespace(step_execution=False)
    proc.result_mq = MagicMock()
    submitted = []
    proc._make_emit_chunk = lambda request_id: lambda output: submitted.append(request_id)
    proc._wait_chunk_outputs = MagicMock(return_value=True)
    captured = None
    try:
        with proc._request_chunk_transport("execute_model", True):
            with request_chunk_emitter([SimpleNamespace(request_id="req")], enabled=True):
                captured = get_emit_chunk()
                assert captured is not None
                captured(DiffusionOutput(finished=False))
                if fail_forward:
                    raise ValueError("forward failed")
    except ValueError:
        assert fail_forward
    assert captured is not None
    captured(DiffusionOutput(finished=False))
    assert submitted == ["req"]
    proc._wait_chunk_outputs.assert_called_once_with()


def test_chunk_drain_timeout_stops_worker_instead_of_returning_rpc(mocker) -> None:
    proc = object.__new__(WorkerProc)
    proc.od_config = SimpleNamespace(step_execution=False)
    proc.result_mq = None
    proc.gpu_id = 0
    proc._wait_chunk_outputs = MagicMock(return_value=False)
    log_error = mocker.patch("vllm_omni.diffusion.worker.diffusion_worker.logger.error")
    with pytest.raises(SystemExit, match="did not drain") as raised:
        with proc._request_chunk_transport("execute_model", False):
            raise RuntimeError("forward boom")
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert "CHUNK_READY drain timed out" in log_error.call_args.args[0]


def test_admission_refuses_unretired_cancelled_id() -> None:
    engine, _queue = _engine_for_chunks()
    engine._request_chunk_states["req"].cancelled = True
    with engine._cv:
        with pytest.raises(ValueError, match="not retired"):
            engine._ensure_request_stream_admitable("req")


def test_finalization_does_not_pop_reused_request_output_state() -> None:
    engine, outputs = _engine_for_chunks()
    old = engine._request_chunk_states["req"]
    old.next_chunk_index = 2
    replacement = RequestStreamState()
    scheduler_state = SimpleNamespace(status=DiffusionRequestStatus.FINISHED_COMPLETED, error=None)
    engine.scheduler.get_request_state.return_value = scheduler_state

    def retire(_request_id):
        engine._request_chunk_states["req"] = replacement
        return scheduler_state

    engine.scheduler.pop_request_state.side_effect = retire
    engine._emit_finished_request(
        "req",
        RunnerOutput(
            request_id="req",
            finished=True,
            result=DiffusionOutput(finished=True),
        ),
        "missing",
    )
    assert engine._request_chunk_states["req"] is replacement
    assert outputs.get_nowait().total_chunks == 2
