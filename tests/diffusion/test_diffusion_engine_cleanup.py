# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import concurrent.futures
import queue
import threading
import weakref
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from vllm_omni.diffusion import diffusion_engine as diffusion_engine_module
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine, DiffusionExecutionMode
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import DiffusionRequestStatus, RequestScheduler
from vllm_omni.diffusion.worker.utils import RunnerOutput
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
    engine.od_config = SimpleNamespace(distributed_executor_backend=None)
    engine.scheduler = RequestScheduler()
    engine.scheduler.initialize(SimpleNamespace())
    engine.execution_mode = DiffusionExecutionMode.REQUEST_BATCH
    engine.executor = SimpleNamespace(
        collective_rpc=Mock(),
        drop_output=Mock(),
        shutdown=Mock(),
        wait_output_ready=Mock(),
    )
    engine._rpc_lock = threading.RLock()
    engine._cv = threading.Condition(engine._rpc_lock)
    engine._out_streams = {}
    engine._unclaimed_async_outputs = {}
    engine._shutdown_output_futures = {}
    engine._closed = False
    engine._shutting_down = False
    engine._shutdown_complete = False
    engine.abort_queue = queue.Queue()
    engine._loop_started = False
    engine.stop_event = None
    engine.worker_thread = None
    return engine


@pytest.mark.asyncio
@pytest.mark.parametrize("started", [False, True])
async def test_closing_response_stream_releases_queue(started: bool) -> None:
    engine = _make_engine()
    request = _make_request("close-stream")
    stream = engine.async_add_req_and_stream_response(request)
    output_queue = engine._out_streams[request.request_id]
    queue_ref = weakref.ref(output_queue)
    assert engine.scheduler.get_request_state(request.request_id) is not None
    output_queue.put_nowait(DiffusionOutput(finished=False))
    if started:
        await anext(stream)

    await stream.aclose()
    await stream.aclose()

    assert request.request_id not in engine._out_streams
    # Keeping the closed stream must not retain its queue through the callback.
    del output_queue
    assert queue_ref() is None
    with pytest.raises(StopAsyncIteration):
        await anext(stream)


@pytest.mark.asyncio
async def test_unstarted_stream_close_preserves_reused_request_queue() -> None:
    engine = _make_engine()
    request = _make_request("reused")
    stream = engine.async_add_req_and_stream_response(request)
    replacement: asyncio.Queue[DiffusionOutput] = asyncio.Queue()
    engine._out_streams[request.request_id] = replacement

    await stream.aclose()

    assert engine._out_streams[request.request_id] is replacement


@pytest.mark.asyncio
async def test_delayed_stream_iteration_uses_original_queue() -> None:
    engine = _make_engine()
    request = _make_request("reused")
    stream = engine.async_add_req_and_stream_response(request)
    original_output = DiffusionOutput(output="original", finished=True)
    engine._out_streams[request.request_id].put_nowait(original_output)
    replacement: asyncio.Queue[DiffusionOutput] = asyncio.Queue()
    replacement_output = DiffusionOutput(output="replacement", finished=True)
    replacement.put_nowait(replacement_output)
    engine._out_streams[request.request_id] = replacement

    assert await anext(stream) is original_output
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    assert engine._out_streams[request.request_id] is replacement
    assert replacement.get_nowait() is replacement_output


@pytest.mark.asyncio
async def test_throw_into_unstarted_stream_releases_queue() -> None:
    engine = _make_engine()
    request = _make_request("throw-before-start")
    stream = engine.async_add_req_and_stream_response(request)

    with pytest.raises(RuntimeError, match="caller failed"):
        await stream.athrow(RuntimeError("caller failed"))

    assert request.request_id not in engine._out_streams


@pytest.mark.asyncio
async def test_close_rejected_during_iteration_keeps_queue() -> None:
    engine = _make_engine()
    request = _make_request("running")
    stream = engine.async_add_req_and_stream_response(request)
    next_output: asyncio.Future[DiffusionOutput] = asyncio.ensure_future(anext(stream))
    await asyncio.sleep(0)
    try:
        with pytest.raises(RuntimeError, match="already running"):
            await stream.aclose()
        assert request.request_id in engine._out_streams
    finally:
        next_output.cancel()
        with pytest.raises(asyncio.CancelledError):
            await next_output
        await stream.aclose()
    assert request.request_id not in engine._out_streams


def _make_output_executor() -> MultiprocDiffusionExecutor:
    executor = MultiprocDiffusionExecutor.__new__(MultiprocDiffusionExecutor)
    executor._futures_lock = threading.RLock()
    executor._output_futures = {}
    executor._completed_outputs = {}
    executor._dropped_output_ids = OrderedDict()
    executor._closed = False
    return executor


def _make_shutdown_output_executor() -> MultiprocDiffusionExecutor:
    executor = _make_output_executor()
    # Stub process infrastructure, but exercise the real shutdown/cache cleanup.
    executor._pump_stop = threading.Event()
    executor._shutdown_cleaner = None
    executor._finalizer = SimpleNamespace(alive=False)
    executor._result_pump_threads = []
    executor._rpc_futures = {}
    executor._batch_split_map = {}
    return executor


def test_wait_output_ready_after_shutdown_fails_immediately() -> None:
    executor = _make_shutdown_output_executor()
    executor.shutdown()

    ready = executor.wait_output_ready("aid-after-shutdown")

    try:
        assert ready.done(), "A stopped executor must not create a waiter that cannot complete"
        with pytest.raises(RuntimeError, match="^Executor shut down$"):
            ready.result()
        assert not executor._output_futures
    finally:
        ready.cancel()
        executor.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("ready_before_close", [False, True])
async def test_close_with_real_executor_resolves_queued_output(monkeypatch, ready_before_close: bool) -> None:
    engine = _make_engine()
    engine.main_loop = None
    executor = _make_shutdown_output_executor()
    engine.executor = executor
    monkeypatch.setattr(diffusion_engine_module, "_async_output_timeout", lambda: 0.05)
    engine._out_streams["live"] = asyncio.Queue()
    stream = engine.get_output_stream("live")
    next_output = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)

    materialized_output = DiffusionOutput(output="cached", finished=True)
    ready: concurrent.futures.Future[DiffusionOutput] = concurrent.futures.Future()
    ready.set_result(materialized_output)
    if ready_before_close:
        executor._completed_outputs["aid-live"] = ready
    # Wake the consumer, then close before it gets another event-loop turn.
    engine._put_output("live", DiffusionOutput(async_output_id="aid-live", finished=True))
    engine.close()

    try:
        if ready_before_close:
            assert await next_output is materialized_output
            with pytest.raises(StopAsyncIteration):
                await anext(stream)
        else:
            with pytest.raises(RuntimeError, match="Executor shut down"):
                await next_output
        assert not engine._unclaimed_async_outputs
        assert not engine._shutdown_output_futures
        assert not executor._completed_outputs
        assert not executor._output_futures
    finally:
        await stream.aclose()
        executor.shutdown()


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


@pytest.mark.asyncio
@pytest.mark.parametrize("finished", [False, True])
async def test_close_preserves_cached_output_for_live_consumer(finished: bool) -> None:
    engine = _make_engine()
    engine.main_loop = None
    executor = _make_output_executor()
    # Isolate stream ownership from executor teardown, which clears its cache.
    executor.shutdown = Mock()
    engine.executor = executor
    request_id = "live"
    output_queue: asyncio.Queue[DiffusionOutput] = asyncio.Queue()
    engine._out_streams[request_id] = output_queue
    stream = engine.get_output_stream(request_id)
    next_output = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)

    materialized_output = DiffusionOutput(output="cached", finished=finished)
    ready: concurrent.futures.Future[DiffusionOutput] = concurrent.futures.Future()
    ready.set_result(materialized_output)
    executor._completed_outputs["aid-live"] = ready
    engine._put_output(request_id, DiffusionOutput(async_output_id="aid-live", finished=finished))

    engine.close()

    assert engine._unclaimed_async_outputs == {output_queue: {"aid-live"}}
    assert await next_output is materialized_output
    if not finished:
        closed_output = await anext(stream)
        assert closed_output.error == "DiffusionEngine is closed."
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    assert not engine._unclaimed_async_outputs
    assert not executor._completed_outputs
    assert not executor._dropped_output_ids


def test_close_discards_only_outputs_without_registered_streams() -> None:
    engine = _make_engine()
    engine.main_loop = None
    live_queue: asyncio.Queue[DiffusionOutput] = asyncio.Queue()
    abandoned_queue: asyncio.Queue[DiffusionOutput] = asyncio.Queue()
    engine._out_streams["reused"] = live_queue
    engine._unclaimed_async_outputs = {
        live_queue: {"aid-live"},
        abandoned_queue: {"aid-abandoned"},
    }

    engine.close()
    engine.close()

    engine.executor.drop_output.assert_called_once_with("aid-abandoned")
    assert engine._unclaimed_async_outputs == {live_queue: {"aid-live"}}


def test_put_output_discards_async_output_when_stream_is_missing() -> None:
    engine = _make_engine()

    engine._put_output("abandoned", DiffusionOutput(async_output_id="aid-missing"))

    engine.executor.drop_output.assert_called_once_with("aid-missing")


@pytest.mark.asyncio
async def test_stream_admits_request_before_iteration() -> None:
    engine = _make_engine()
    request = _make_request("unstarted")

    stream = engine.async_add_req_and_stream_response(request)

    assert engine.scheduler.get_request_state(request.request_id) is not None
    assert request.request_id in engine._out_streams
    engine.abort(request.request_id)
    assert engine.abort_queue.get_nowait() == request.request_id
    await stream.aclose()


@pytest.mark.parametrize("closed", [False, True])
def test_stream_admission_errors_are_raised_at_call_site(closed: bool) -> None:
    engine = _make_engine()
    engine._closed = closed
    error_type: type[Exception]
    if closed:
        error_type = RuntimeError
        message = "DiffusionEngine is closed."
    else:
        error_type = ValueError
        message = "invalid request"
        engine.pre_process_func = Mock(side_effect=ValueError(message))

    with pytest.raises(error_type, match=message):
        engine.async_add_req_and_stream_response(_make_request("invalid"))
    assert not engine._out_streams


@pytest.mark.asyncio
@pytest.mark.parametrize("started", [False, True])
async def test_closing_response_stream_discards_async_output(started: bool) -> None:
    engine = _make_engine()
    engine.main_loop = None
    request = _make_request("close-wrapper")
    stream = engine.async_add_req_and_stream_response(request)
    if started:
        engine._put_output(request.request_id, DiffusionOutput(finished=False))
        await anext(stream)
    engine._put_output(request.request_id, DiffusionOutput(async_output_id="aid-unclaimed"))

    await stream.aclose()

    engine.executor.drop_output.assert_called_once_with("aid-unclaimed")
    assert not engine._out_streams
    assert not engine._unclaimed_async_outputs


@pytest.mark.asyncio
async def test_closing_stream_discards_unclaimed_async_output() -> None:
    engine = _make_engine()
    request_id = "abandoned"
    output_queue: asyncio.Queue[DiffusionOutput] = asyncio.Queue()
    engine._out_streams[request_id] = output_queue
    engine._unclaimed_async_outputs[output_queue] = {"aid-queued"}
    stream = engine.get_output_stream(request_id)
    next_output = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)

    next_output.cancel()
    with pytest.raises(asyncio.CancelledError):
        await next_output

    engine.executor.drop_output.assert_called_once_with("aid-queued")
    assert request_id not in engine._out_streams
    assert not engine._unclaimed_async_outputs


@pytest.mark.asyncio
async def test_old_stream_cleanup_preserves_reused_request_outputs() -> None:
    engine = _make_engine()
    engine.main_loop = None
    request_id = "reused"
    old_queue: asyncio.Queue[DiffusionOutput] = asyncio.Queue()
    engine._out_streams[request_id] = old_queue
    engine._put_output(request_id, DiffusionOutput(finished=True))
    old_stream = engine.get_output_stream(request_id)
    await anext(old_stream)
    # Leave an unclaimed output owned by the old stream to exercise cleanup.
    engine._put_output(request_id, DiffusionOutput(async_output_id="aid-old"))

    new_queue: asyncio.Queue[DiffusionOutput] = asyncio.Queue()
    engine._out_streams[request_id] = new_queue
    engine._put_output(request_id, DiffusionOutput(async_output_id="aid-new"))

    await old_stream.aclose()

    engine.executor.drop_output.assert_called_once_with("aid-old")
    assert engine._out_streams[request_id] is new_queue
    assert engine._unclaimed_async_outputs == {new_queue: {"aid-new"}}

    materialized_output = DiffusionOutput(output="new", finished=True)
    ready: concurrent.futures.Future[DiffusionOutput] = concurrent.futures.Future()
    ready.set_result(materialized_output)
    engine.executor.wait_output_ready.return_value = ready
    new_stream = engine.get_output_stream(request_id)
    assert await anext(new_stream) is materialized_output
    engine.executor.wait_output_ready.assert_called_once_with("aid-new")
    await new_stream.aclose()
    assert not engine._out_streams
    assert not engine._unclaimed_async_outputs
    engine.executor.drop_output.assert_called_once_with("aid-old")


@pytest.mark.asyncio
async def test_real_stream_records_output_wait_and_excludes_it_from_execution() -> None:
    engine = _make_engine()
    engine.main_loop = None
    request = _make_request("timed-materialization")
    clock = [0.0]
    materialized = DiffusionOutput(output="materialized", finished=True, stage_durations={"denoise": 10.0})
    ready: concurrent.futures.Future[DiffusionOutput] = concurrent.futures.Future()
    ready.set_result(materialized)
    formatted = SimpleNamespace(metrics={})
    engine._check_and_start_background_loop = AsyncMock()
    engine._prepare_request_for_admission = Mock(return_value=request)
    engine.postprocess_output = Mock(return_value=[formatted])

    def admit(req):
        engine._out_streams[req.request_id] = asyncio.Queue()
        engine._put_output(req.request_id, DiffusionOutput(async_output_id="aid-timed", finished=True))
        clock[0] = 10.0
        return req.request_id

    def wait_output_ready(output_id):
        assert output_id == "aid-timed"
        # Advance after the real stream starts its await, without wall-clock sleep.
        asyncio.get_running_loop().call_soon(lambda: clock.__setitem__(0, 14.0))
        return ready

    engine._add_prepared_request = Mock(side_effect=admit)
    engine.executor.wait_output_ready.side_effect = wait_output_ready
    with patch.object(diffusion_engine_module.time, "perf_counter", side_effect=lambda: clock[0]):
        results = [batch async for batch in engine.step_streaming(request)]

    assert results == [[formatted]]
    assert materialized.output_ready_wait_time == pytest.approx(4.0)
    assert materialized.stage_durations == {"denoise": 10.0}
    assert formatted.metrics["output_ready_wait_time_ms"] == pytest.approx(4000.0)
    assert formatted.metrics["diffusion_engine_exec_time_ms"] == pytest.approx(10000.0)
    engine.executor.drop_output.assert_not_called()
    assert not engine._unclaimed_async_outputs


@pytest.mark.asyncio
async def test_async_output_is_claimed_after_materialization() -> None:
    engine = _make_engine()
    request_id = "claimed"
    pending_output = DiffusionOutput(async_output_id="aid-claimed")
    materialized_output = DiffusionOutput(output="materialized", stage_durations={"denoise": 1.5})
    ready: concurrent.futures.Future[DiffusionOutput] = concurrent.futures.Future()
    ready.set_result(materialized_output)
    engine.executor.wait_output_ready.return_value = ready
    output_queue: asyncio.Queue[DiffusionOutput] = asyncio.Queue()
    output_queue.put_nowait(pending_output)
    engine._out_streams[request_id] = output_queue
    engine._unclaimed_async_outputs[output_queue] = {"aid-claimed"}
    stream = engine.get_output_stream(request_id)

    assert await anext(stream) is materialized_output
    assert materialized_output.output_ready_wait_time >= 0.0
    assert materialized_output.stage_durations == {"denoise": 1.5}
    assert not engine._unclaimed_async_outputs
    engine.executor.wait_output_ready.assert_called_once_with("aid-claimed")

    await stream.aclose()
    engine.executor.drop_output.assert_not_called()


@pytest.mark.asyncio
async def test_exceptional_materialization_retires_async_output() -> None:
    engine = _make_engine()
    executor = _make_output_executor()
    engine.executor = executor
    request_id = "failed"
    async_output_id = "aid-failed"
    output_queue: asyncio.Queue[DiffusionOutput] = asyncio.Queue()
    output_queue.put_nowait(DiffusionOutput(async_output_id=async_output_id))
    engine._out_streams[request_id] = output_queue
    engine._unclaimed_async_outputs[output_queue] = {async_output_id}
    stream = engine.get_output_stream(request_id)
    next_output = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)

    with executor._futures_lock:
        ready = executor._output_futures.pop(async_output_id)
    ready.set_exception(RuntimeError("materialization failed"))

    with pytest.raises(RuntimeError, match="materialization failed"):
        await next_output

    assert executor._output_futures == {}
    assert executor._completed_outputs == {}
    assert not engine._unclaimed_async_outputs


@pytest.mark.asyncio
@pytest.mark.parametrize("running", [False, True])
async def test_cancelling_materialization_discards_async_output(running: bool) -> None:
    engine = _make_engine()
    request_id = "cancelled"
    pending_output = DiffusionOutput(async_output_id="aid-cancelled")
    ready: concurrent.futures.Future[DiffusionOutput] = concurrent.futures.Future()
    if running:
        ready.set_running_or_notify_cancel()
    engine.executor.wait_output_ready.return_value = ready
    output_queue: asyncio.Queue[DiffusionOutput] = asyncio.Queue()
    output_queue.put_nowait(pending_output)
    engine._out_streams[request_id] = output_queue
    engine._unclaimed_async_outputs[output_queue] = {"aid-cancelled"}
    stream = engine.get_output_stream(request_id)
    next_output = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)

    next_output.cancel()
    with pytest.raises(asyncio.CancelledError):
        await next_output

    if running:
        engine.executor.drop_output.assert_called_once_with("aid-cancelled")
        assert not ready.done()
    else:
        # The cancelled waiter remains registered until the executor drains it.
        engine.executor.drop_output.assert_not_called()
        assert ready.cancelled()
    assert request_id not in engine._out_streams
    assert not engine._unclaimed_async_outputs


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_cancellation_after_delivery_does_not_recreate_waiter(failed: bool) -> None:
    engine = _make_engine()
    executor = _make_output_executor()
    engine.executor = executor
    request_id = "delivered"
    async_output_id = "aid-delivered"
    output_queue: asyncio.Queue[DiffusionOutput] = asyncio.Queue()
    output_queue.put_nowait(DiffusionOutput(async_output_id=async_output_id))
    engine._out_streams[request_id] = output_queue
    engine._unclaimed_async_outputs[output_queue] = {async_output_id}
    stream = engine.get_output_stream(request_id)
    next_output = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)

    # Match delivery: remove the waiter and resolve it before cancellation
    # reaches the coroutine awaiting its wrapped future.
    with executor._futures_lock:
        ready = executor._output_futures.pop(async_output_id)
    if failed:
        ready.set_exception(RuntimeError("materialization failed"))
    else:
        ready.set_result(DiffusionOutput(output="materialized", finished=True))
    next_output.cancel()
    with pytest.raises(asyncio.CancelledError):
        await next_output

    assert executor._output_futures == {}
    assert executor._completed_outputs == {}
    assert request_id not in engine._out_streams
    assert not engine._unclaimed_async_outputs


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


def test_emit_finished_outputs_discards_async_output_without_stream() -> None:
    engine = _make_engine()
    request_id = engine.scheduler.add_request(_make_request("completed-async"))
    engine.scheduler.finish_requests(request_id, DiffusionRequestStatus.FINISHED_COMPLETED)
    runner_output = RunnerOutput(
        request_id=request_id,
        finished=True,
        async_output_id="aid-completed",
    )

    engine._emit_finished_outputs({request_id}, runner_output)

    engine.executor.drop_output.assert_called_once_with("aid-completed")
    assert engine.scheduler.get_request_state(request_id) is None


def test_emit_step_outputs_finalizes_finished_request_without_stream() -> None:
    engine = _make_engine()
    engine.execution_mode = DiffusionExecutionMode.STEP_BATCH
    request_id = engine.scheduler.add_request(_make_request("step-drained"))
    engine.scheduler.finish_requests(request_id, DiffusionRequestStatus.FINISHED_ABORTED)

    engine._emit_outputs({request_id}, [request_id], SimpleNamespace(get_request_output=lambda _request_id: None))

    assert engine.scheduler.get_request_state(request_id) is None


@pytest.mark.parametrize(
    ("path", "async_output"),
    [("request", False), ("step", False), ("unscheduled_step", False), ("request", True), ("step", True)],
)
def test_terminal_output_keeps_original_queue_when_request_id_is_reused(monkeypatch, path, async_output) -> None:
    engine = _make_engine()
    engine.main_loop = None
    engine.execution_mode = (
        DiffusionExecutionMode.REQUEST_BATCH if path == "request" else DiffusionExecutionMode.STEP_BATCH
    )
    request_id = engine._add_prepared_request(_make_request("reused"))
    original_queue = engine._out_streams[request_id]
    status = DiffusionRequestStatus.FINISHED_COMPLETED if async_output else DiffusionRequestStatus.FINISHED_ABORTED
    engine.scheduler.finish_requests(request_id, status)
    finalize = engine._finalize_finished_request

    def admit_after_finalization(*args, **kwargs):
        output = finalize(*args, **kwargs)
        engine._add_prepared_request(_make_request(request_id))
        return output

    monkeypatch.setattr(engine, "_finalize_finished_request", admit_after_finalization)
    runner_output = RunnerOutput(
        request_id=request_id,
        step_index=None,
        finished=True,
        result=None,
        async_output_id="aid-original" if async_output else None,
    )
    engine._emit_outputs({request_id}, [] if path == "unscheduled_step" else [request_id], runner_output)

    assert original_queue.qsize() == 1, "the old terminal output must not be sent to the new lifecycle"
    output = original_queue.get_nowait()
    if async_output:
        assert output.async_output_id == "aid-original"
        assert engine._unclaimed_async_outputs == {original_queue: {"aid-original"}}
        engine.executor.drop_output.assert_not_called()
    else:
        assert output.aborted
    assert engine._out_streams[request_id].empty()
    assert engine.scheduler.get_request_state(request_id).status == DiffusionRequestStatus.WAITING


def _make_kv_cleanup_engine(mode: DiffusionKVCacheMode) -> tuple[DiffusionEngine, Mock]:
    engine = _make_engine()
    engine.od_config = SimpleNamespace(diffusion_kv_mode=mode)
    cleanup = Mock()
    engine.executor = SimpleNamespace(remove_diffusion_kv_requests=cleanup)
    return engine, cleanup


def test_paged_terminal_output_clears_worker_rows() -> None:
    engine, cleanup = _make_kv_cleanup_engine(DiffusionKVCacheMode.PAGED_SCHEDULER)
    for request_id in ("req-0", "req-1"):
        engine.scheduler.add_request(_make_request(request_id))
    engine.scheduler.finish_requests(["req-0", "req-1"], DiffusionRequestStatus.FINISHED_COMPLETED)

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
    engine.scheduler.add_request(_make_request("req-0"))
    engine.scheduler.finish_requests("req-0", DiffusionRequestStatus.FINISHED_COMPLETED)

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

    def prepare_profile_requests(self) -> list[object]:
        events.append("prepare-profile")
        return expected_profile_requests

    monkeypatch.setattr(DiffusionEngine, "_prepare_diffusion_kv_profile_requests", prepare_profile_requests)

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


def test_fail_engine_does_not_reenter_when_executor_shutdown_fails() -> None:
    engine = _make_engine()
    shutdown_error = RuntimeError("worker shutdown failed")
    engine.executor.shutdown.side_effect = shutdown_error
    engine.scheduler.close = Mock()
    engine._fail_pending_rpcs = Mock()

    with pytest.raises(RuntimeError, match="worker shutdown failed"):
        engine._fail_engine(RuntimeError("engine failed"))

    engine._fail_engine(RuntimeError("follow-up failure"))

    engine.executor.shutdown.assert_called_once_with()
    engine.scheduler.close.assert_not_called()
    assert engine._closed is True
    assert engine._shutting_down is True
    assert engine._shutdown_complete is False


def test_finalize_aborted_request_drops_pending_async_output() -> None:
    """An aborted request with a pending async output must be drained (#6413).

    The aborted branch of _finalize_finished_request returns before the
    async_output_id branch, so without draining, the worker's late OUTPUT_READY
    is cached in the executor's _completed_outputs forever. The engine must tell
    the executor to drop that output id.
    """
    engine = _make_engine()
    engine.executor = SimpleNamespace(drop_output=Mock())
    request_id = engine.scheduler.add_request(_make_request("aborted-async"))
    engine.scheduler.finish_requests(request_id, DiffusionRequestStatus.FINISHED_ABORTED)

    runner_output = SimpleNamespace(result=None, async_output_id="aid-async-1")
    output = engine._finalize_finished_request(request_id, runner_output)

    assert output.aborted is True
    engine.executor.drop_output.assert_called_once_with("aid-async-1")


def test_finalize_aborted_request_without_async_output_skips_drop() -> None:
    """No async output pending -> nothing to drop; drop_output not called."""
    engine = _make_engine()
    engine.executor = SimpleNamespace(drop_output=Mock())
    request_id = engine.scheduler.add_request(_make_request("aborted-plain"))
    engine.scheduler.finish_requests(request_id, DiffusionRequestStatus.FINISHED_ABORTED)

    output = engine._finalize_finished_request(request_id)

    assert output.aborted is True
    engine.executor.drop_output.assert_not_called()


@pytest.mark.parametrize("scheduler_fails", [False, True])
def test_fail_engine_releases_cancellation_signals_after_workers_stop(scheduler_fails):
    from multiprocessing.shared_memory import SharedMemory

    from vllm_omni.diffusion.cancellation import RequestCancellationRegistry

    engine = _make_engine()
    engine._fail_pending_rpcs = Mock()
    registry = RequestCancellationRegistry()
    engine._request_cancellations = registry
    name = registry.create("running")

    def shutdown():
        # Cancellation is visible before shutdown, and late readers can attach.
        reader = SharedMemory(name=name)
        try:
            assert reader.buf[0] == 1
        finally:
            reader.close()

    engine.executor.shutdown.side_effect = shutdown
    engine.scheduler.close = Mock(side_effect=RuntimeError("scheduler cleanup failed") if scheduler_fails else None)
    try:
        if scheduler_fails:
            with pytest.raises(RuntimeError, match="scheduler cleanup failed"):
                engine._fail_engine(RuntimeError("engine failed"))
        else:
            engine._fail_engine(RuntimeError("engine failed"))
            engine.close()
            engine.executor.shutdown.assert_called_once_with()
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=name)
        assert not registry._signals
    finally:
        registry.close()


def test_close_retains_cancellation_signal_until_stuck_worker_stops():
    from multiprocessing.shared_memory import SharedMemory

    from vllm_omni.diffusion.cancellation import RequestCancellationRegistry

    engine = _make_engine()
    registry = RequestCancellationRegistry()
    engine._request_cancellations = registry
    name = registry.create("running")
    engine.worker_thread = Mock()
    engine.worker_thread.is_alive.side_effect = [True, True, False, False]
    try:
        engine.close()
        engine.executor.shutdown.assert_not_called()
        reader = SharedMemory(name=name)
        try:
            assert reader.buf[0] == 1
        finally:
            reader.close()
        engine.close()
        engine.executor.shutdown.assert_called_once_with()
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=name)
        assert not registry._signals
    finally:
        registry.close()
