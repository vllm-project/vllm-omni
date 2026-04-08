from __future__ import annotations

import asyncio
import concurrent.futures
import queue
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import janus
import pytest
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams

from vllm_omni.engine.orchestrator import Orchestrator
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class OrchestratorFixture:
    orchestrator: Orchestrator
    request_sync_q: Any
    output_sync_q: Any
    queues: tuple[janus.Queue, ...]
    thread: threading.Thread
    result_future: concurrent.futures.Future[None]


class FakeStageClient:
    def __init__(
        self,
        *,
        stage_type: str = "llm",
        final_output: bool = False,
        final_output_type: str = "text",
        next_inputs: list[dict] | None = None,
    ) -> None:
        self.stage_type = stage_type
        self.final_output = final_output
        self.final_output_type = final_output_type
        self.next_inputs = list(next_inputs or [])
        self.custom_process_input_func = None
        self.add_request_calls: list[tuple] = []
        self.abort_calls: list[list[str]] = []
        self.shutdown_calls = 0
        self._engine_core_outputs = queue.Queue()
        self._diffusion_outputs = queue.Queue()

    # Orchestrator-facing interface.
    async def add_request_async(self, *args, **_kwargs) -> None:
        self.add_request_calls.append(args)

    async def get_output_async(self):
        try:
            return self._engine_core_outputs.get_nowait()
        except queue.Empty:
            return SimpleNamespace(outputs=[])

    def get_diffusion_output_nowait(self):
        try:
            return self._diffusion_outputs.get_nowait()
        except queue.Empty:
            return None

    def set_engine_outputs(self, outputs) -> None:
        return None

    def process_engine_inputs(self, stage_list, prompt=None):
        return list(self.next_inputs)

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        self.abort_calls.append(list(request_ids))

    def shutdown(self) -> None:
        self.shutdown_calls += 1

    # Test helpers for seeding fake stage outputs.
    def push_engine_core_outputs(self, outputs) -> None:
        self._engine_core_outputs.put_nowait(outputs)

    def push_diffusion_output(self, output) -> None:
        self._diffusion_outputs.put_nowait(output)


class FakeOutputProcessor:
    def __init__(self, *, request_outputs: list[object] | None = None) -> None:
        self.request_outputs = list(request_outputs or [])

    def add_request(self, *_args, **_kwargs) -> None:
        return None

    def process_outputs(self, *_args, **_kwargs):
        return SimpleNamespace(
            request_outputs=list(self.request_outputs),
            reqs_to_abort=[],
        )

    def update_scheduler_stats(self, _scheduler_stats) -> None:
        return None


def _sampling_params(max_tokens: int = 4) -> SamplingParams:
    return SamplingParams(max_tokens=max_tokens)


def _engine_core_outputs(tag: str, timestamp: float) -> SimpleNamespace:
    return SimpleNamespace(outputs=[tag], timestamp=timestamp, scheduler_stats=None)


def _build_request_output(
    request_id: str,
    *,
    token_ids: list[int] | None = None,
    prompt_token_ids: list[int] | None = None,
    finished: bool = True,
    text: str = "test",
) -> RequestOutput:
    completion = CompletionOutput(
        index=0,
        text=text,
        token_ids=list(token_ids or [1, 2]),
        cumulative_logprob=0.0,
        logprobs=None,
        finish_reason="stop" if finished else None,
        stop_reason=None,
    )
    return RequestOutput(
        request_id=request_id,
        prompt="prompt",
        prompt_token_ids=list(prompt_token_ids or [10, 11]),
        prompt_logprobs=None,
        outputs=[completion],
        finished=finished,
        metrics=None,
        lora_request=None,
    )


def _build_harness(
    stage_clients: list[object],
    *,
    output_processors: list[object] | None = None,
    stage_vllm_configs: list[object] | None = None,
    async_chunk: bool = False,
) -> OrchestratorFixture:
    if output_processors is None:
        output_processors = [FakeOutputProcessor() for _ in stage_clients]
    if stage_vllm_configs is None:
        stage_vllm_configs = [SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)) for _ in stage_clients]

    ready_future: concurrent.futures.Future[tuple[Orchestrator, janus.Queue, janus.Queue, janus.Queue]] = (
        concurrent.futures.Future()
    )
    result_future: concurrent.futures.Future[None] = concurrent.futures.Future()

    def _runner() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _run() -> None:
            request_queue = janus.Queue()
            output_queue = janus.Queue()
            rpc_queue = janus.Queue()
            orchestrator = Orchestrator(
                request_async_queue=request_queue.async_q,
                output_async_queue=output_queue.async_q,
                rpc_async_queue=rpc_queue.async_q,
                stage_clients=stage_clients,
                output_processors=output_processors,
                stage_vllm_configs=stage_vllm_configs,
                async_chunk=async_chunk,
            )
            ready_future.set_result((orchestrator, request_queue, output_queue, rpc_queue))
            await orchestrator.run()

        try:
            loop.run_until_complete(_run())
            result_future.set_result(None)
        except Exception as exc:
            result_future.set_exception(exc)
        finally:
            try:
                pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                asyncio.set_event_loop(None)
                loop.close()

    thread = threading.Thread(target=_runner, daemon=True, name="test-orchestrator")
    thread.start()

    orchestrator, request_queue, output_queue, rpc_queue = ready_future.result(timeout=5)
    return OrchestratorFixture(
        orchestrator=orchestrator,
        request_sync_q=request_queue.sync_q,
        output_sync_q=output_queue.sync_q,
        queues=(request_queue, output_queue, rpc_queue),
        thread=thread,
        result_future=result_future,
    )


async def _shutdown_orchestrator(orchestrator_fixture: OrchestratorFixture) -> None:
    orchestrator_fixture.request_sync_q.put_nowait({"type": "shutdown"})
    await asyncio.to_thread(orchestrator_fixture.thread.join, 5)
    if orchestrator_fixture.thread.is_alive():
        raise AssertionError("Timed out waiting for orchestrator thread shutdown")
    orchestrator_fixture.result_future.result(timeout=0)


async def _wait_for(predicate, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for predicate")
        await asyncio.sleep(0.01)


async def _get_output_message(orchestrator_fixture: OrchestratorFixture, *, timeout: float = 2.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for orchestrator output")
        try:
            msg = orchestrator_fixture.output_sync_q.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.01)
            continue
        if msg.get("type") == "output":
            return msg


async def _enqueue_add_request(
    orchestrator_fixture: OrchestratorFixture,
    *,
    request_id: str,
    prompt,
    original_prompt,
    sampling_params_list,
    final_stage_id: int,
) -> None:
    orchestrator_fixture.request_sync_q.put_nowait(
        {
            "type": "add_request",
            "request_id": request_id,
            "prompt": prompt,
            "original_prompt": original_prompt,
            "sampling_params_list": sampling_params_list,
            "final_stage_id": final_stage_id,
        }
    )


async def _enqueue_abort_request(orchestrator_fixture: OrchestratorFixture, request_ids: list[str]) -> None:
    orchestrator_fixture.request_sync_q.put_nowait(
        {
            "type": "abort",
            "request_ids": request_ids,
        }
    )


@pytest.fixture
def orchestrator_factory():
    fixtures: list[OrchestratorFixture] = []

    def _factory(*args, **kwargs) -> OrchestratorFixture:
        fixture = _build_harness(*args, **kwargs)
        fixtures.append(fixture)
        return fixture

    yield _factory

    for fixture in fixtures:
        if fixture.thread.is_alive():
            fixture.request_sync_q.put_nowait({"type": "shutdown"})
            fixture.thread.join(timeout=5)
        for q in fixture.queues:
            q.close()


@pytest.mark.asyncio
async def test_run_two_stage_llm(orchestrator_factory) -> None:
    stage0 = FakeStageClient(stage_type="llm", final_output=False)
    stage1 = FakeStageClient(
        stage_type="llm",
        final_output=True,
        next_inputs=[{"prompt_token_ids": [7, 8, 9]}],
    )
    processors = [
        FakeOutputProcessor(request_outputs=[_build_request_output("req-llm", token_ids=[3, 4], finished=True)]),
        FakeOutputProcessor(request_outputs=[_build_request_output("req-llm", token_ids=[10, 11], finished=True)]),
    ]
    orchestrator_fixture = orchestrator_factory([stage0, stage1], output_processors=processors)
    request = SimpleNamespace(request_id="req-llm", prompt_token_ids=[1, 2, 3])

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-llm",
            prompt=request,
            original_prompt={"prompt": "hello"},
            sampling_params_list=[_sampling_params(), _sampling_params()],
            final_stage_id=1,
        )

        await _wait_for(lambda: len(stage0.add_request_calls) == 1)
        stage0.push_engine_core_outputs(_engine_core_outputs("stage0-raw", 1.0))

        await _wait_for(lambda: len(stage1.add_request_calls) == 1)
        stage1_request = stage1.add_request_calls[0][0]
        assert stage1_request.request_id == "req-llm"
        assert stage1_request.prompt_token_ids == [7, 8, 9]

        stage1.push_engine_core_outputs(_engine_core_outputs("stage1-raw", 2.0))

        output_msg = await _get_output_message(orchestrator_fixture)

        assert output_msg["request_id"] == "req-llm"
        assert output_msg["stage_id"] == 1
        assert output_msg["finished"] is True
        assert output_msg["engine_outputs"].request_id == "req-llm"
        assert "req-llm" not in orchestrator_fixture.orchestrator.request_states
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_run_single_stage_diffusion(orchestrator_factory) -> None:
    stage0 = FakeStageClient(stage_type="diffusion", final_output=True, final_output_type="image")
    orchestrator_fixture = orchestrator_factory([stage0])
    params = OmniDiffusionSamplingParams()

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-diff",
            prompt={"prompt": "draw a cat"},
            original_prompt={"prompt": "draw a cat"},
            sampling_params_list=[params],
            final_stage_id=0,
        )

        await _wait_for(lambda: len(stage0.add_request_calls) == 1)
        stage0.push_diffusion_output(
            OmniRequestOutput.from_diffusion(
                request_id="req-diff",
                images=[],
                final_output_type="image",
            )
        )

        output_msg = await _get_output_message(orchestrator_fixture)

        assert output_msg["request_id"] == "req-diff"
        assert output_msg["stage_id"] == 0
        assert output_msg["finished"] is True
        assert output_msg["engine_outputs"].request_id == "req-diff"
        assert "req-diff" not in orchestrator_fixture.orchestrator.request_states
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_run_llm_to_diffusion(orchestrator_factory) -> None:
    stage0 = FakeStageClient(stage_type="llm", final_output=False)
    stage1 = FakeStageClient(stage_type="diffusion", final_output=True, final_output_type="image")
    processors = [
        FakeOutputProcessor(request_outputs=[_build_request_output("req-img", token_ids=[3, 4], finished=True)]),
        FakeOutputProcessor(),
    ]
    orchestrator_fixture = orchestrator_factory([stage0, stage1], output_processors=processors)
    request = SimpleNamespace(request_id="req-img", prompt_token_ids=[1, 2, 3])
    params = OmniDiffusionSamplingParams()
    original_prompt = {"prompt": "draw a fox"}

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-img",
            prompt=request,
            original_prompt=original_prompt,
            sampling_params_list=[_sampling_params(), params],
            final_stage_id=1,
        )

        await _wait_for(lambda: len(stage0.add_request_calls) == 1)
        stage0.push_engine_core_outputs(_engine_core_outputs("stage0-raw", 1.0))

        await _wait_for(lambda: len(stage1.add_request_calls) == 1)
        assert stage1.add_request_calls[0] == ("req-img", original_prompt, params)

        stage1.push_diffusion_output(
            OmniRequestOutput.from_diffusion(
                request_id="req-img",
                images=[],
                final_output_type="image",
            )
        )

        output_msg = await _get_output_message(orchestrator_fixture)

        assert output_msg["request_id"] == "req-img"
        assert output_msg["stage_id"] == 1
        assert output_msg["finished"] is True
        assert output_msg["engine_outputs"].request_id == "req-img"
        assert "req-img" not in orchestrator_fixture.orchestrator.request_states
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_run_async_chunk(orchestrator_factory) -> None:
    stage0 = FakeStageClient(stage_type="llm", final_output=False)
    stage1 = FakeStageClient(stage_type="llm", final_output=True)
    processors = [
        FakeOutputProcessor(request_outputs=[_build_request_output("req-async", token_ids=[1], finished=True)]),
        FakeOutputProcessor(request_outputs=[_build_request_output("req-async", token_ids=[20, 21], finished=True)]),
    ]
    orchestrator_fixture = orchestrator_factory(
        [stage0, stage1],
        output_processors=processors,
        async_chunk=True,
    )
    request = SimpleNamespace(request_id="req-async", prompt_token_ids=[1, 2, 3, 4])

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-async",
            prompt=request,
            original_prompt={"prompt": "hello async"},
            sampling_params_list=[_sampling_params(), _sampling_params()],
            final_stage_id=1,
        )

        await _wait_for(lambda: len(stage1.add_request_calls) == 1)
        prewarmed_request = stage1.add_request_calls[0][0]
        assert prewarmed_request.request_id == "req-async"
        assert prewarmed_request.prompt_token_ids
        assert all(token_id == 0 for token_id in prewarmed_request.prompt_token_ids)

        stage1.push_engine_core_outputs(_engine_core_outputs("stage1-final", 3.0))

        output_msg = await _get_output_message(orchestrator_fixture)

        assert output_msg["request_id"] == "req-async"
        assert output_msg["stage_id"] == 1
        assert output_msg["finished"] is True
        assert "req-async" not in orchestrator_fixture.orchestrator.request_states
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_run_shutdown(orchestrator_factory) -> None:
    stages = [
        FakeStageClient(stage_type="llm", final_output=False),
        FakeStageClient(stage_type="diffusion", final_output=True, final_output_type="image"),
    ]
    orchestrator_fixture = orchestrator_factory(stages)

    await _shutdown_orchestrator(orchestrator_fixture)

    assert not orchestrator_fixture.thread.is_alive()
    for stage in stages:
        assert stage.shutdown_calls == 1


@pytest.mark.asyncio
async def test_run_abort(orchestrator_factory) -> None:
    stages = [
        FakeStageClient(stage_type="llm", final_output=False),
        FakeStageClient(stage_type="llm", final_output=True),
    ]
    processors = [
        FakeOutputProcessor(request_outputs=[_build_request_output("req-abort", token_ids=[1], finished=True)]),
        FakeOutputProcessor(request_outputs=[_build_request_output("req-abort", token_ids=[2], finished=True)]),
    ]
    orchestrator_fixture = orchestrator_factory(stages, output_processors=processors)
    request = SimpleNamespace(request_id="req-abort", prompt_token_ids=[1, 2, 3])

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-abort",
            prompt=request,
            original_prompt={"prompt": "cancel me"},
            sampling_params_list=[_sampling_params(), _sampling_params()],
            final_stage_id=1,
        )
        await _wait_for(lambda: len(stages[0].add_request_calls) == 1)

        await _enqueue_abort_request(orchestrator_fixture, ["req-abort"])
        await _wait_for(lambda: all(stage.abort_calls for stage in stages))

        for stage in stages:
            assert stage.abort_calls == [["req-abort"]]
        assert "req-abort" not in orchestrator_fixture.orchestrator.request_states
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)
