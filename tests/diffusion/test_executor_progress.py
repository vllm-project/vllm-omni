# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import queue
import threading
from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.data import AsyncDiffusionOutput, AsyncOutputKind, DiffusionOutput
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.executor.uniproc_executor import UniProcDiffusionExecutor
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker, WorkerProc

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class StepRunner(ProgressBarMixin):
    def execute_model(self, request, **kwargs):
        self.set_progress_bar_config(disable=True)
        with self.progress_bar(total=2) as bar:
            bar.update()
            bar.update()
        return DiffusionOutput()


class ResultQueue:
    def __init__(self):
        self.items = queue.Queue()
        self.progress_messages = []

    def enqueue(self, item):
        if isinstance(item, AsyncDiffusionOutput) and item.kind == AsyncOutputKind.PROGRESS:
            self.progress_messages.append(item)
        self.items.put(item)

    def dequeue(self, timeout):
        try:
            return self.items.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError from None


class WorkerQueue:
    """Replace IPC transport while retaining executor and worker RPC handling."""

    def __init__(self, worker):
        self.worker = worker

    def enqueue(self, request):
        result, should_reply = self.worker._execute_rpc(request)
        assert should_reply
        if "rpc_id" in request:
            result = AsyncDiffusionOutput(kind=AsyncOutputKind.RPC_RESULT, rpc_id=request["rpc_id"], output=result)
        self.worker.result_mq.enqueue(result)


def make_multiproc_executor(proc, config):
    executor = object.__new__(MultiprocDiffusionExecutor)
    executor.od_config = config
    executor._closed = False
    executor._is_failed = False
    executor._result_mq = proc.result_mq
    executor._broadcast_mq = WorkerQueue(proc)
    executor._rpc_wave_id = 0
    executor._rpc_id_counter = 0
    executor._rpc_id_lock = threading.Lock()
    executor._futures_lock = threading.Lock()
    executor._rpc_futures = {}
    executor._sync_result_buffer = queue.Queue()
    executor._pump_stop = threading.Event()
    return executor


@pytest.mark.parametrize("backend", ["uni", "mp"])
@pytest.mark.parametrize("synchronous", [False, True])
@pytest.mark.asyncio
async def test_executor_delivers_worker_progress(backend, synchronous, monkeypatch):
    worker = object.__new__(DiffusionWorker)
    worker.lora_manager = None
    worker.model_runner = StepRunner()
    wrapper = SimpleNamespace(execute_method=lambda method, *args: getattr(worker, method)(*args))
    config = SimpleNamespace(
        step_execution=False, streaming_output=False, custom_pipeline_args={"pipeline_class": StepRunner}
    )
    request = SimpleNamespace(request_id="video", sampling_params=SimpleNamespace(emit_request_lifecycle=False))
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

    outputs = asyncio.Queue()

    def connect_engine(executor):
        # Skip model loading, retain real engine callback registration.
        monkeypatch.setattr(DiffusionEngine, "_init_process_hooks", lambda self, config: None)
        monkeypatch.setattr(DiffusionEngine, "_init_executor", lambda self, config: setattr(self, "executor", executor))
        engine = DiffusionEngine(config)
        engine.main_loop = asyncio.get_running_loop()
        if not synchronous:
            engine._out_streams["video"] = outputs
        return engine

    def execute(executor, engine):
        if not synchronous:
            return executor.collective_rpc(
                "execute_model", timeout=2 if backend == "mp" else None, args=(request, config), unique_reply_rank=0
            )
        from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput

        scheduled = SimpleNamespace(is_empty=False, scheduled_request_ids=["video"])
        engine.scheduler = SimpleNamespace(
            add_request=lambda req: "video",
            schedule=lambda: scheduled,
            update_from_output=lambda plan, output: {"video"},
        )
        engine._prepare_request_for_admission = lambda req: req
        engine._process_aborts_queue = lambda: None
        engine._prepare_kv_for_forward = lambda plan: None
        engine._poll_native_kv = lambda: None
        engine._remove_diffusion_kv_requests = lambda ids: None
        engine._finalize_finished_request = lambda request_id, runner_output, **kwargs: runner_output.result

        def execute_plan(plan):
            executor.collective_rpc(
                "execute_model", timeout=2 if backend == "mp" else None, args=(request, config), unique_reply_rank=0
            )
            return BatchRunnerOutput.from_list(
                [RunnerOutput(request_id="video", finished=True, result=DiffusionOutput())]
            )

        engine.execute_fn = execute_plan
        output = engine.add_req_and_wait_for_response(request)
        assert output.error is None
        return output

    if backend == "uni":
        executor = object.__new__(UniProcDiffusionExecutor)
        executor._closed = False
        executor.driver_worker = wrapper
        engine = connect_engine(executor)
        execute(executor, engine)
    else:
        result_queue = ResultQueue()
        proc = object.__new__(WorkerProc)
        proc.gpu_id = 0
        proc.od_config = config
        proc.worker = wrapper
        proc.result_mq = result_queue
        proc._result_mq_lock = threading.Lock()
        executor = make_multiproc_executor(proc, config)
        engine = connect_engine(executor)
        pump = threading.Thread(target=executor._result_pump)
        pump.start()
        try:
            result = execute(executor, engine)
            assert [message.result.completed for message in result_queue.progress_messages] == [1, 2]
            assert isinstance(result, DiffusionOutput if synchronous else AsyncDiffusionOutput)
        finally:
            executor._pump_stop.set()
            pump.join(timeout=3)
    await asyncio.sleep(0)
    expected = [] if synchronous else [50, 100]
    assert [outputs.get_nowait().denoising_progress for _ in range(outputs.qsize())] == expected


@pytest.mark.parametrize("backend", ["uni", "mp"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.asyncio
async def test_step_worker_outputs_update_progress(streaming, backend):
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.sched.step_scheduler import StepScheduler
    from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    class Runner:
        step = 0

        def execute_stepwise(self, scheduled):
            self.step += 1
            return BatchRunnerOutput.from_list(
                [
                    RunnerOutput(
                        request_id=rid,
                        step_index=self.step,
                        finished=False,
                        result=DiffusionOutput(finished=False, chunk_index=0) if streaming and self.step == 2 else None,
                    )
                    for rid in scheduled.scheduled_request_ids
                ]
            )

    config = SimpleNamespace(step_execution=not streaming, streaming_output=streaming)
    engine = object.__new__(DiffusionEngine)
    engine.execution_mode = engine._resolve_execution_mode(config)
    engine.scheduler = StepScheduler()
    engine.main_loop = asyncio.get_running_loop()
    engine._out_streams_lock = threading.Lock()
    engine._cv = threading.Condition()
    outputs = asyncio.Queue()
    engine._out_streams = {"video": outputs}
    engine._remove_diffusion_kv_requests = lambda ids: None
    request = OmniDiffusionRequest(
        request_id="video",
        prompt="test",
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=4, emit_request_lifecycle=False),
    )
    engine.scheduler.add_request(request)
    worker = object.__new__(DiffusionWorker)
    worker.lora_manager = None
    worker._step_lora_state = {}
    worker.model_runner = Runner()
    executor = object.__new__(UniProcDiffusionExecutor)
    executor._closed = False
    executor.driver_worker = SimpleNamespace(execute_method=lambda method, *args: getattr(worker, method)(*args))
    engine.executor = executor
    proc = object.__new__(WorkerProc)
    proc.gpu_id = 0
    proc.od_config = config
    proc.worker = executor.driver_worker
    proc.result_mq = ResultQueue()
    if backend == "mp":
        executor = make_multiproc_executor(proc, config)
        engine.executor = executor
    pump = None
    if backend == "mp" and streaming:
        pump = threading.Thread(target=executor._result_pump)
        pump.start()
    try:
        for expected in (25, 50):
            scheduled = engine.scheduler.schedule()
            result = engine.executor.execute_step(scheduled)
            if backend == "mp":
                assert proc.result_mq.items.empty()  # RPC result was consumed; no extra progress messages.
            finished = engine.scheduler.update_from_output(scheduled, result)
            engine._emit_outputs(finished, scheduled.scheduled_request_ids, result, worker_execution_completed=True)
            await asyncio.sleep(0)
            assert not outputs.empty(), "Completed worker step did not emit progress"
            output = outputs.get_nowait()
            assert output.denoising_progress == expected
            assert not output.finished
            if streaming and expected == 50:
                chunk = outputs.get_nowait()
                assert chunk.denoising_progress is None
                assert chunk.chunk_index == 0
                assert not chunk.finished
    finally:
        if pump is not None:
            executor._pump_stop.set()
            pump.join(timeout=3)
            assert not pump.is_alive()


@pytest.mark.asyncio
@pytest.mark.parametrize("removed", [False, True])
async def test_progress_from_worker_thread_uses_event_loop(removed: bool) -> None:
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
    from vllm_omni.diffusion.models.progress_bar import DiffusionProgress

    engine = object.__new__(DiffusionEngine)
    engine.main_loop = asyncio.get_running_loop()
    engine._out_streams_lock = threading.Lock()
    outputs = asyncio.Queue()
    engine._out_streams = {"video": outputs}
    thread = threading.Thread(target=engine._on_progress, args=(DiffusionProgress("video", 1, 2),))
    thread.start()
    thread.join(timeout=1)
    assert not thread.is_alive()
    if removed:
        with engine._out_streams_lock:
            engine._out_streams.pop("video")
    await asyncio.sleep(0)
    if removed:
        assert outputs.empty()
    else:
        output = await asyncio.wait_for(outputs.get(), timeout=1)
        assert output.denoising_progress == 50
        assert not output.finished


@pytest.mark.parametrize("closes_during_handoff", [False, True])
def test_progress_after_event_loop_closed_is_ignored(
    closes_during_handoff: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
    from vllm_omni.diffusion.models.progress_bar import DiffusionProgress

    engine = object.__new__(DiffusionEngine)
    engine.main_loop = asyncio.new_event_loop()
    engine.main_loop.close()
    if closes_during_handoff:
        # The loop may close after the producer checks is_running().
        monkeypatch.setattr(engine.main_loop, "is_running", lambda: True)
    engine._out_streams_lock = threading.Lock()
    outputs = asyncio.Queue()
    engine._out_streams = {"video": outputs}
    engine._on_progress(DiffusionProgress("video", 1, 2))
    assert outputs.empty()


@pytest.mark.asyncio
async def test_progress_after_output_stream_cancelled_is_ignored() -> None:
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
    from vllm_omni.diffusion.models.progress_bar import DiffusionProgress

    engine = object.__new__(DiffusionEngine)
    engine.main_loop = asyncio.get_running_loop()
    engine._out_streams_lock = threading.Lock()
    outputs = asyncio.Queue()
    engine._out_streams = {"video": outputs}
    stream = engine.get_output_stream("video")
    consumer = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer
    engine._on_progress(DiffusionProgress("video", 1, 2))
    await asyncio.sleep(0)
    assert outputs.empty()
