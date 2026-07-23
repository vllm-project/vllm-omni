from __future__ import annotations

import asyncio
import concurrent.futures
import logging
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
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.engine.messages import (
    AbortRequestMessage,
    AddCompanionRequestMessage,
    CollectiveRPCRequestMessage,
    CollectiveRPCResultMessage,
    ErrorMessage,
    OutputMessage,
    ShutdownRequestMessage,
    StageSubmissionMessage,
)
from vllm_omni.engine.orchestrator import (
    Orchestrator,
    OrchestratorRequestState,
    _build_terminal_empty_output,
    _infer_stage_audio_sample_rate,
)
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.experimental.fullduplex.engine.duplex_control_plane import DuplexControlPlane
from vllm_omni.experimental.fullduplex.engine.duplex_runtime import (
    DuplexInputMode,
    DuplexRuntimeCapabilities,
    DuplexSessionRuntimeState,
    duplex_resource_request_id,
)
from vllm_omni.experimental.fullduplex.engine.messages import (
    AppendDuplexInputMessage,
    CloseDuplexSessionMessage,
    DuplexFence,
    OpenDuplexSessionMessage,
    SignalDuplexTurnMessage,
)
from vllm_omni.experimental.fullduplex.minicpmo45.runtime import MiniCPMO45DuplexRuntimeExtension
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeRunningCounter:
    def __init__(self) -> None:
        self.value = 0

    def increment(self) -> None:
        self.value += 1

    def decrement(self) -> None:
        self.value -= 1


@pytest.mark.asyncio
async def test_engine_dead_broadcasts_fatal_to_rpc_waiters(monkeypatch: pytest.MonkeyPatch) -> None:
    rpc_queue: asyncio.Queue = asyncio.Queue()
    orchestrator = Orchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=asyncio.Queue(),
        rpc_async_queue=rpc_queue,
        stage_pools=[],
    )
    orchestrator._fatal_error = "stage engine died"
    orchestrator._fatal_error_stage_id = 2

    async def wait_for_requests() -> None:
        await asyncio.Event().wait()

    async def fail_outputs() -> None:
        raise EngineDeadError("stage engine died")

    monkeypatch.setattr(orchestrator, "_request_handler", wait_for_requests)
    monkeypatch.setattr(orchestrator, "_orchestration_output_handler", fail_outputs)

    await orchestrator.run()

    fatal = rpc_queue.get_nowait()
    assert isinstance(fatal, ErrorMessage)
    assert fatal.fatal is True
    assert fatal.error == "stage engine died"
    assert fatal.stage_id == 2


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
        engine_input_source: list[int] | None = None,
        is_comprehension: bool = False,
        model_stage: str | None = None,
        kv_sender_info: dict[str, Any] | None = None,
    ) -> None:
        self.stage_id = 0
        self.replica_id = 0
        self.stage_type = stage_type
        self.final_output = final_output
        self.final_output_type = final_output_type
        self.default_sampling_params = SamplingParams(max_tokens=1)
        self.requires_multimodal_data = False
        self.engine_input_source = list(engine_input_source or [0])
        self.is_comprehension = is_comprehension
        self.model_stage = model_stage
        self.next_inputs = list(next_inputs or [])
        self.custom_process_input_func = None
        self._kv_sender_info = dict(kv_sender_info) if kv_sender_info is not None else None
        self.add_request_calls: list[tuple] = []
        self.abort_calls: list[list[str]] = []
        self.collective_rpc_calls: list[tuple[str, float | None, tuple[Any, ...], dict[str, Any]]] = []
        self.shutdown_calls = 0
        self._engine_core_outputs = queue.Queue()
        self._diffusion_outputs = queue.Queue()

    # Orchestrator-facing interface.
    async def add_request_async(self, *args, **kwargs) -> None:
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

    def process_engine_inputs(self, source_outputs, prompt=None, streaming_context=None):
        return list(self.next_inputs)

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        self.abort_calls.append(list(request_ids))

    async def collective_rpc_async(
        self,
        *,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        normalized_kwargs = dict(kwargs or {})
        self.collective_rpc_calls.append((method, timeout, args, normalized_kwargs))
        return {
            "supported": False,
            "todo": True,
            "reason": f"{self.__class__.__name__}.collective_rpc_async is not implemented yet",
        }

    def get_kv_sender_info(self) -> dict[str, Any] | None:
        if self._kv_sender_info is None:
            return None
        return dict(self._kv_sender_info)

    def check_health(self) -> None:
        return None

    def shutdown(self) -> None:
        self.shutdown_calls += 1

    # Test helpers for seeding fake stage outputs.
    def push_engine_core_outputs(self, outputs) -> None:
        self._engine_core_outputs.put_nowait(outputs)

    def push_diffusion_output(self, output) -> None:
        self._diffusion_outputs.put_nowait(output)


def test_terminal_empty_audio_output_uses_stage_sample_rate() -> None:
    final_stage = FakeStageClient(final_output=True, final_output_type="audio")
    final_stage.sample_rate = 44100
    final_pool = SimpleNamespace(stage_client=final_stage, _stage_vllm_config=None)

    terminal_output = _build_terminal_empty_output(
        "req-1",
        final_output_type="audio",
        audio_sample_rate=_infer_stage_audio_sample_rate(final_pool),
    )

    assert terminal_output.outputs[0].multimodal_output["sr"] == 44100


class FakeCollectiveRpcStageClient(FakeStageClient):
    def __init__(self, *args, rpc_result: Any = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rpc_result = rpc_result

    async def collective_rpc_async(
        self,
        *,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        normalized_kwargs = dict(kwargs or {})
        self.collective_rpc_calls.append((method, timeout, args, normalized_kwargs))
        return self.rpc_result


class FakeOutputProcessor:
    def __init__(self, *, request_outputs: list[object] | None = None) -> None:
        self.request_outputs = list(request_outputs or [])
        self.add_request_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.abort_calls: list[list[str]] = []

    def add_request(self, *args, **kwargs) -> None:
        self.add_request_calls.append((args, kwargs))
        return None

    def process_outputs(self, *_args, **_kwargs):
        return SimpleNamespace(
            request_outputs=list(self.request_outputs),
            reqs_to_abort=[],
        )

    def abort_requests(self, request_ids, internal: bool = False):
        self.abort_calls.append(request_ids)
        return request_ids

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


def _build_stage_pools(
    stage_clients: list[list[FakeStageClient]],
    *,
    output_processors: list[FakeOutputProcessor] | None = None,
    stage_vllm_configs: list[object] | None = None,
) -> list[StagePool]:
    """Build StagePool list from per-stage replica lists.

    ``stage_clients[i]`` is the list of FakeStageClient replicas for stage i.
    """
    num_stages = len(stage_clients)
    if output_processors is None:
        output_processors = [FakeOutputProcessor() for _ in stage_clients]
    if stage_vllm_configs is None:
        stage_vllm_configs = [SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)) for _ in stage_clients]

    pools: list[StagePool] = []
    for stage_id in range(num_stages):
        clients = stage_clients[stage_id]
        if clients[0].stage_type == "diffusion":
            pools.append(StagePool(stage_id, clients[0]))
        else:
            pools.append(
                StagePool(
                    stage_id,
                    clients,
                    output_processor=output_processors[stage_id],
                    stage_vllm_config=stage_vllm_configs[stage_id],
                )
            )
    return pools


def _build_harness(
    stage_clients: list[object],
    *,
    output_processors: list[object] | None = None,
    stage_vllm_configs: list[object] | None = None,
    async_chunk: bool = False,
    stage_pools: list[StagePool] | None = None,
) -> OrchestratorFixture:
    """Build an Orchestrator test harness.

    Accepts either pre-built ``stage_pools`` or flat lists of single-replica
    clients/processors.
    """
    if stage_pools is None:
        # Wrap flat lists into per-stage single-replica lists.
        nested_clients = [[c] for c in stage_clients]
        stage_pools = _build_stage_pools(
            nested_clients,
            output_processors=output_processors,
            stage_vllm_configs=stage_vllm_configs,
        )

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
                stage_pools=stage_pools,
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
    orchestrator_fixture.request_sync_q.put_nowait(ShutdownRequestMessage())
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


async def _get_output_message(orchestrator_fixture: OrchestratorFixture, *, timeout: float = 2.0) -> OutputMessage:
    deadline = time.monotonic() + timeout
    while True:
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for orchestrator output")
        try:
            msg = orchestrator_fixture.output_sync_q.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.01)
            continue
        if isinstance(msg, OutputMessage):
            return msg


async def _get_rpc_message(
    orchestrator_fixture: OrchestratorFixture,
    *,
    timeout: float = 2.0,
) -> CollectiveRPCResultMessage:
    deadline = time.monotonic() + timeout
    rpc_sync_q = orchestrator_fixture.queues[2].sync_q
    while True:
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for orchestrator rpc output")
        try:
            return rpc_sync_q.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.01)


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
        StageSubmissionMessage(
            type="add_request",
            request_id=request_id,
            prompt=prompt,
            original_prompt=original_prompt,
            output_prompt_text=None,
            sampling_params_list=sampling_params_list,
            final_stage_id=final_stage_id,
            preprocess_ms=0.0,
            request_timestamp=time.time(),
            enqueue_ts=time.perf_counter(),
        )
    )


async def _enqueue_abort_request(orchestrator_fixture: OrchestratorFixture, request_ids: list[str]) -> None:
    orchestrator_fixture.request_sync_q.put_nowait(AbortRequestMessage(request_ids=request_ids))


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
            fixture.request_sync_q.put_nowait(ShutdownRequestMessage())
            fixture.thread.join(timeout=5)
        for q in fixture.queues:
            q.close()


# ---------------------------------------------------------------------------
# Existing single-replica tests (adapted to StagePool interface)
# ---------------------------------------------------------------------------


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

        assert output_msg.request_id == "req-llm"
        assert output_msg.stage_id == 1
        assert output_msg.finished is True
        assert output_msg.engine_outputs.request_id == "req-llm"
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

        assert output_msg.request_id == "req-diff"
        assert output_msg.stage_id == 0
        assert output_msg.finished is True
        assert output_msg.engine_outputs.request_id == "req-diff"
        assert "req-diff" not in orchestrator_fixture.orchestrator.request_states
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_run_single_stage_diffusion_streaming_forwards_intermediate_chunks(orchestrator_factory) -> None:
    """Intermediate diffusion chunks (finished=False) reach the frontend before the final chunk."""
    stage0 = FakeStageClient(stage_type="diffusion", final_output=True, final_output_type="image")
    orchestrator_fixture = orchestrator_factory([stage0])
    params = OmniDiffusionSamplingParams()

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-stream",
            prompt={"prompt": "draw a cat"},
            original_prompt={"prompt": "draw a cat"},
            sampling_params_list=[params],
            final_stage_id=0,
        )

        await _wait_for(lambda: len(stage0.add_request_calls) == 1)
        stage0.push_diffusion_output(
            OmniRequestOutput.from_diffusion(
                request_id="req-stream",
                images=[],
                final_output_type="image",
                custom_output={"chunk": 0},
                finished=False,
            )
        )
        stage0.push_diffusion_output(
            OmniRequestOutput.from_diffusion(
                request_id="req-stream",
                images=[],
                final_output_type="image",
                custom_output={"chunk": 1},
                finished=True,
            )
        )

        output_msgs: list[OutputMessage] = []
        deadline = time.monotonic() + 2.0
        while not output_msgs or not output_msgs[-1].finished:
            if time.monotonic() >= deadline:
                raise AssertionError(
                    f"Timed out waiting for finished orchestrator output, got {len(output_msgs)} message(s)"
                )
            try:
                msg = orchestrator_fixture.output_sync_q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue
            if isinstance(msg, OutputMessage):
                output_msgs.append(msg)

        assert [msg.request_id for msg in output_msgs] == ["req-stream", "req-stream"]
        assert [msg.finished for msg in output_msgs] == [False, True]
        assert [msg.engine_outputs.finished for msg in output_msgs] == [False, True]
        assert [msg.engine_outputs.custom_output["chunk"] for msg in output_msgs] == [0, 1]
        await _wait_for(lambda: "req-stream" not in orchestrator_fixture.orchestrator.request_states)
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

        assert output_msg.request_id == "req-img"
        assert output_msg.stage_id == 1
        assert output_msg.finished is True
        assert output_msg.engine_outputs.request_id == "req-img"
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

        assert output_msg.request_id == "req-async"
        assert output_msg.stage_id == 1
        assert output_msg.finished is True
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
        await _wait_for(lambda: bool(stages[0].abort_calls))

        assert stages[0].abort_calls == [["req-abort"]]
        assert stages[1].abort_calls == []
        assert "req-abort" not in orchestrator_fixture.orchestrator.request_states
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


def _duplex_open_message(
    session_id: str,
    *,
    incarnation: int = 0,
    session_config: dict[str, object] | None = None,
    runtime_config: dict[str, object] | None = None,
) -> OpenDuplexSessionMessage:
    return OpenDuplexSessionMessage(
        control_id=f"open-{session_id}",
        fence=DuplexFence(session_id, incarnation=incarnation),
        session_id=session_id,
        capabilities={
            "input_modes": [DuplexInputMode.APPEND_AUDIO_CHUNK.value],
            "implementation_level": "model_native_duplex",
        },
        session_config=session_config or {},
        runtime_config=runtime_config or {},
    )


async def _handle_duplex(orchestrator: Orchestrator, message: object) -> None:
    await orchestrator._require_duplex_control_plane().handle(message)


def _duplex_request_state(
    orchestrator: Orchestrator,
    session: DuplexSessionRuntimeState,
    *,
    stage_id: int,
) -> OrchestratorRequestState | None:
    context = orchestrator._require_duplex_control_plane().ensure_stage_request(
        session,
        stage_id=stage_id,
    )
    if context is None:
        return None
    return orchestrator.request_states.get(context.request_id)


@pytest.mark.asyncio
async def test_duplex_control_plane_keeps_public_and_runtime_config_separate() -> None:
    stage0 = FakeStageClient(stage_type="llm", final_output=True)
    rpc_q: asyncio.Queue = asyncio.Queue()
    orchestrator = Orchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=asyncio.Queue(),
        rpc_async_queue=rpc_q,
        stage_pools=_build_stage_pools([[stage0]]),
        duplex_runtime_extension=MiniCPMO45DuplexRuntimeExtension(),
        enable_duplex_control=True,
    )
    open_message = _duplex_open_message(
        "sid-config-channels",
        incarnation=3,
        session_config={
            "instructions": "public instructions",
            "extra_body": {"duplex_stage_max_tokens": {"0": 99}},
        },
        runtime_config={
            "duplex_stage_max_tokens": {"0": 3},
            "duplex_stage_sampling_params": {"0": {"stop_token_ids": [151705]}},
        },
    )

    await _handle_duplex(orchestrator, open_message)

    assert rpc_q.get_nowait().ok is True
    session = orchestrator.duplex_sessions.require(open_message.session_id)
    request_state = _duplex_request_state(orchestrator, session, stage_id=0)
    assert request_state.sampling_params_list[0].max_tokens == 3
    assert request_state.sampling_params_list[0].stop_token_ids == [151705]
    bridge = request_state.streaming.bridge_states["duplex"]
    assert bridge["incarnation"] == open_message.fence.incarnation
    assert bridge["session_config"] == open_message.session_config
    assert bridge["runtime_config"] == open_message.runtime_config


@pytest.mark.asyncio
async def test_duplex_control_plane_preserves_turn_commit_without_model_extension() -> None:
    rpc_q: asyncio.Queue = asyncio.Queue()
    orchestrator = Orchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=asyncio.Queue(),
        rpc_async_queue=rpc_q,
        stage_pools=[],
        duplex_runtime_extension=None,
        enable_duplex_control=True,
    )
    assert isinstance(orchestrator.duplex_control_plane, DuplexControlPlane)

    message = OpenDuplexSessionMessage(
        control_id="open-turn-commit",
        fence=DuplexFence("sid-turn-commit"),
        session_id="sid-turn-commit",
        capabilities={"input_modes": [DuplexInputMode.TURN_COMMIT_ONLY.value]},
        session_config={},
    )
    await _handle_duplex(orchestrator, message)

    result = rpc_q.get_nowait()
    assert result.ok is True
    assert result.stage_results[0]["result"]["scheduler_request_context"] is False


def test_ordinary_orchestrator_bypasses_duplex_control_plane() -> None:
    orchestrator = Orchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=asyncio.Queue(),
        rpc_async_queue=asyncio.Queue(),
        stage_pools=[],
        duplex_runtime_extension=MiniCPMO45DuplexRuntimeExtension(),
    )

    assert orchestrator.duplex_control_plane is None


@pytest.mark.asyncio
async def test_duplex_close_cleans_preregistered_request_without_append() -> None:
    stage0 = FakeStageClient(stage_type="llm", final_output=True)
    rpc_q: asyncio.Queue = asyncio.Queue()
    running_counter = FakeRunningCounter()
    orchestrator = Orchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=asyncio.Queue(),
        rpc_async_queue=rpc_q,
        stage_pools=_build_stage_pools([[stage0]]),
        duplex_runtime_extension=MiniCPMO45DuplexRuntimeExtension(),
        enable_duplex_control=True,
        running_counter=running_counter,
    )
    open_message = _duplex_open_message("sid-preregister-close")

    await _handle_duplex(orchestrator, open_message)
    open_result = rpc_q.get_nowait()
    request_id = open_result.stage_results[0]["result"]["request_id"]
    assert request_id in orchestrator.request_states

    await _handle_duplex(
        orchestrator,
        CloseDuplexSessionMessage(
            control_id="close-preregistered",
            fence=open_message.fence,
            session_id=open_message.session_id,
        ),
    )

    assert rpc_q.get_nowait().ok is True
    assert request_id not in orchestrator.request_states
    assert orchestrator.duplex_sessions.get(open_message.session_id) is None
    assert running_counter.value == 0


@pytest.mark.asyncio
async def test_duplex_open_failure_rolls_back_session_and_reserved_request() -> None:
    stage0 = FakeStageClient(stage_type="llm", final_output=True)
    rpc_q: asyncio.Queue = asyncio.Queue()
    orchestrator = Orchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=asyncio.Queue(),
        rpc_async_queue=rpc_q,
        stage_pools=_build_stage_pools([[stage0]]),
        duplex_runtime_extension=MiniCPMO45DuplexRuntimeExtension(),
        enable_duplex_control=True,
    )
    open_message = _duplex_open_message(
        "sid-open-rollback",
        runtime_config={
            "duplex_stage_sampling_params": {
                "0": {"stop_token_ids": 123},
            }
        },
    )

    await _handle_duplex(orchestrator, open_message)

    assert rpc_q.get_nowait().ok is False
    assert orchestrator.duplex_sessions.get(open_message.session_id) is None
    assert orchestrator.request_states == {}


@pytest.mark.asyncio
async def test_duplex_running_counter_tracks_only_submitted_request() -> None:
    stage0 = FakeStageClient(stage_type="llm", final_output=True)
    rpc_q: asyncio.Queue = asyncio.Queue()
    running_counter = FakeRunningCounter()
    orchestrator = Orchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=asyncio.Queue(),
        rpc_async_queue=rpc_q,
        stage_pools=_build_stage_pools([[stage0]]),
        duplex_runtime_extension=MiniCPMO45DuplexRuntimeExtension(),
        enable_duplex_control=True,
        running_counter=running_counter,
    )
    open_message = _duplex_open_message("sid-duplex-counter")
    await _handle_duplex(orchestrator, open_message)
    assert rpc_q.get_nowait().ok is True
    assert running_counter.value == 0

    await _handle_duplex(
        orchestrator,
        AppendDuplexInputMessage(
            control_id="append-duplex-counter",
            fence=open_message.fence,
            session_id=open_message.session_id,
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK.value,
            payload={"is_speech": True},
        ),
    )
    assert rpc_q.get_nowait().ok is True
    assert running_counter.value == 1

    await _handle_duplex(
        orchestrator,
        CloseDuplexSessionMessage(
            control_id="close-duplex-counter",
            fence=open_message.fence,
            session_id=open_message.session_id,
        ),
    )
    assert rpc_q.get_nowait().ok is True
    assert running_counter.value == 0


@pytest.mark.asyncio
async def test_duplex_failed_append_does_not_advance_sequence_or_fence() -> None:
    class FailingPlanExtension(MiniCPMO45DuplexRuntimeExtension):
        def plan_append(self, **kwargs):
            raise RuntimeError("planned append failure")

    stage0 = FakeStageClient(stage_type="llm", final_output=True)
    rpc_q: asyncio.Queue = asyncio.Queue()
    orchestrator = Orchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=asyncio.Queue(),
        rpc_async_queue=rpc_q,
        stage_pools=_build_stage_pools([[stage0]]),
        duplex_runtime_extension=FailingPlanExtension(),
        enable_duplex_control=True,
    )
    fence = DuplexFence("sid-append-rollback", turn_id=1)
    session = orchestrator.duplex_sessions.open_session(
        DuplexFence("sid-append-rollback"),
        capabilities=DuplexRuntimeCapabilities(
            input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK},
        ),
    )

    await _handle_duplex(
        orchestrator,
        AppendDuplexInputMessage(
            control_id="append-rollback",
            fence=fence,
            session_id=fence.session_id,
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK.value,
            payload={"is_speech": True},
        ),
    )

    result = rpc_q.get_nowait()
    assert result.ok is False
    assert session.fence == DuplexFence("sid-append-rollback")
    assert session.input_seq == 0
    assert session.input_turn_seq == 0
    assert stage0.add_request_calls == []


@pytest.mark.asyncio
async def test_duplex_duplicate_append_operation_is_submitted_once() -> None:
    stage0 = FakeStageClient(stage_type="llm", final_output=True)
    rpc_q: asyncio.Queue = asyncio.Queue()
    orchestrator = Orchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=asyncio.Queue(),
        rpc_async_queue=rpc_q,
        stage_pools=_build_stage_pools([[stage0]]),
        duplex_runtime_extension=MiniCPMO45DuplexRuntimeExtension(),
        enable_duplex_control=True,
    )
    fence = DuplexFence("sid-idempotent-append")
    session = orchestrator.duplex_sessions.open_session(
        fence,
        capabilities=DuplexRuntimeCapabilities(
            input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK},
        ),
    )

    for control_id in ("append-first", "append-timeout-retry"):
        await _handle_duplex(
            orchestrator,
            AppendDuplexInputMessage(
                control_id=control_id,
                operation_id="physical-input-1",
                fence=fence,
                session_id=fence.session_id,
                mode=DuplexInputMode.APPEND_AUDIO_CHUNK.value,
                payload={"is_speech": True},
            ),
        )

    first = rpc_q.get_nowait()
    retry = rpc_q.get_nowait()
    assert first.ok is True
    assert retry.ok is True
    assert first.stage_results == retry.stage_results
    assert len(stage0.add_request_calls) == 1
    assert session.input_seq == 1


@pytest.mark.asyncio
async def test_duplex_barge_in_aborts_bound_stage_requests_before_releasing_fence() -> None:
    stage0 = FakeStageClient(stage_type="llm", final_output=False)
    stage1 = FakeStageClient(stage_type="llm", final_output=True)
    stage_pools = _build_stage_pools(
        [[stage0], [stage1]],
        output_processors=[FakeOutputProcessor(), FakeOutputProcessor()],
        stage_vllm_configs=[
            SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
            SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
        ],
    )
    rpc_q: asyncio.Queue = asyncio.Queue()
    orchestrator = Orchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=asyncio.Queue(),
        rpc_async_queue=rpc_q,
        stage_pools=stage_pools,
        duplex_runtime_extension=MiniCPMO45DuplexRuntimeExtension(),
        enable_duplex_control=True,
    )
    session = orchestrator.duplex_sessions.open_session(
        DuplexFence("sid-stage-signal"),
        capabilities=DuplexRuntimeCapabilities(
            input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK},
        ),
    )
    session.bind_stage_request(0, "req-stage0", fence=session.fence)
    session.bind_stage_request(1, "req-stage1", fence=session.fence)
    assert stage_pools[0].select_replica_id("req-stage0") == 0
    assert stage_pools[1].select_replica_id("req-stage1") == 0
    session.append_input(
        mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
        fence=session.fence,
    )
    await _handle_duplex(
        orchestrator,
        SignalDuplexTurnMessage(
            control_id="ctrl-signal",
            fence=session.fence,
            next_fence=DuplexFence("sid-stage-signal", epoch=1),
            session_id="sid-stage-signal",
            event="barge_in",
        ),
    )

    assert stage0.abort_calls == [["req-stage0"]]
    assert stage1.abort_calls == [["req-stage1"]]
    assert stage0.collective_rpc_calls == []
    assert stage1.collective_rpc_calls == []
    assert session.stage_bindings == {}
    assert session.fence == DuplexFence("sid-stage-signal", epoch=1)
    result = rpc_q.get_nowait()
    assert result.ok is True


@pytest.mark.asyncio
async def test_duplex_late_barge_in_releases_only_cancelled_fence_bindings() -> None:
    stage0 = FakeStageClient(stage_type="llm", final_output=False)
    stage1 = FakeStageClient(stage_type="llm", final_output=True)
    stage_pools = _build_stage_pools(
        [[stage0], [stage1]],
        output_processors=[FakeOutputProcessor(), FakeOutputProcessor()],
        stage_vllm_configs=[
            SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
            SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
        ],
    )
    rpc_q: asyncio.Queue = asyncio.Queue()
    orchestrator = Orchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=asyncio.Queue(),
        rpc_async_queue=rpc_q,
        stage_pools=stage_pools,
        enable_duplex_control=True,
    )
    cancelled_fence = DuplexFence("sid-late-stage-signal")
    session = orchestrator.duplex_sessions.open_session(
        cancelled_fence,
        capabilities=DuplexRuntimeCapabilities(
            input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK},
        ),
    )
    session.bind_stage_request(0, "req-stage0", fence=cancelled_fence)
    session.bind_stage_request(1, "req-stage1", fence=cancelled_fence)
    assert stage_pools[0].select_replica_id("req-stage0") == 0
    assert stage_pools[1].select_replica_id("req-stage1") == 0
    current_fence = DuplexFence("sid-late-stage-signal", epoch=1)
    session.accept_fence(current_fence)

    await _handle_duplex(
        orchestrator,
        SignalDuplexTurnMessage(
            control_id="ctrl-late-signal",
            fence=cancelled_fence,
            next_fence=current_fence,
            session_id="sid-late-stage-signal",
            event="barge_in",
        ),
    )

    assert stage0.abort_calls == [["req-stage0"]]
    assert stage1.abort_calls == [["req-stage1"]]
    assert session.stage_bindings == {}
    assert session.fence == current_fence
    result = rpc_q.get_nowait()
    assert result.ok is True


@pytest.mark.asyncio
async def test_duplex_cancel_without_next_fence_is_rejected_without_releasing_bindings() -> None:
    stage0 = FakeStageClient(stage_type="llm", final_output=True)
    stage_pools = _build_stage_pools(
        [[stage0]],
        output_processors=[FakeOutputProcessor()],
        stage_vllm_configs=[SimpleNamespace(model_config=SimpleNamespace(max_model_len=64))],
    )
    rpc_q: asyncio.Queue = asyncio.Queue()
    orchestrator = Orchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=asyncio.Queue(),
        rpc_async_queue=rpc_q,
        stage_pools=stage_pools,
        enable_duplex_control=True,
    )
    fence = DuplexFence("sid-cancel-contract")
    session = orchestrator.duplex_sessions.open_session(fence)
    session.bind_stage_request(0, "req-live", fence=fence)

    await _handle_duplex(
        orchestrator,
        SignalDuplexTurnMessage(
            control_id="cancel-without-next",
            fence=fence,
            session_id=fence.session_id,
            event="input.cancel",
        ),
    )

    result = rpc_q.get_nowait()
    assert result.ok is False
    assert result.error_count == 1
    assert "next_fence" in result.stage_results[0]["result"]["error"]
    assert session.fence == fence
    assert session.stage_request_ids() == ["req-live"]
    assert stage0.abort_calls == []


@pytest.mark.asyncio
async def test_duplex_session_update_replaces_runtime_config() -> None:
    rpc_q: asyncio.Queue = asyncio.Queue()
    orchestrator = Orchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=asyncio.Queue(),
        rpc_async_queue=rpc_q,
        stage_pools=[],
        enable_duplex_control=True,
    )
    fence = DuplexFence("sid-update-config")
    session = orchestrator.duplex_sessions.open_session(
        fence,
        session_config={"temperature": 0.7},
    )

    await _handle_duplex(
        orchestrator,
        SignalDuplexTurnMessage(
            control_id="update-config",
            fence=fence,
            session_id=fence.session_id,
            event="session.update",
            session_config={"temperature": 0.0, "instructions": "updated"},
        ),
    )

    result = rpc_q.get_nowait()
    assert result.ok is True
    assert session.session_config == {"temperature": 0.0, "instructions": "updated"}


@pytest.mark.asyncio
async def test_duplex_session_update_refreshes_next_append_sampling_params() -> None:
    stage0 = FakeStageClient(stage_type="llm", final_output=True)
    stage_pools = _build_stage_pools(
        [[stage0]],
        output_processors=[FakeOutputProcessor()],
        stage_vllm_configs=[SimpleNamespace(model_config=SimpleNamespace(max_model_len=64))],
    )
    rpc_q: asyncio.Queue = asyncio.Queue()
    orchestrator = Orchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=asyncio.Queue(),
        rpc_async_queue=rpc_q,
        stage_pools=stage_pools,
        duplex_runtime_extension=MiniCPMO45DuplexRuntimeExtension(),
        enable_duplex_control=True,
    )
    fence = DuplexFence("sid-update-policy")
    session = orchestrator.duplex_sessions.open_session(
        fence,
        capabilities=DuplexRuntimeCapabilities(
            input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK},
        ),
        runtime_config={
            "duplex_stage_max_tokens": {"0": 2},
            "duplex_stage_sampling_params": {"0": {"stop_token_ids": [151645]}},
        },
    )
    req_state = _duplex_request_state(orchestrator, session, stage_id=0)
    assert req_state is not None
    assert req_state.sampling_params_list[0].max_tokens == 2

    await _handle_duplex(
        orchestrator,
        SignalDuplexTurnMessage(
            control_id="update-policy",
            fence=fence,
            session_id=fence.session_id,
            event="session.update",
            runtime_config={
                "duplex_stage_max_tokens": {"0": 7},
                "duplex_stage_sampling_params": {"0": {"stop_token_ids": [151705]}},
            },
        ),
    )
    assert rpc_q.get_nowait().ok is True

    await _handle_duplex(
        orchestrator,
        AppendDuplexInputMessage(
            control_id="append-updated-policy",
            fence=fence,
            session_id=fence.session_id,
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK.value,
            payload={"is_speech": True},
        ),
    )

    assert rpc_q.get_nowait().ok is True
    assert req_state.sampling_params_list[0].max_tokens == 7
    assert req_state.sampling_params_list[0].stop_token_ids == [151705]
    submitted_request = stage0.add_request_calls[0][0]
    assert submitted_request.sampling_params.stop_token_ids == [151705]
    assert submitted_request.sampling_params.max_tokens == 7


@pytest.mark.asyncio
async def test_duplex_invalid_session_update_preserves_previous_config() -> None:
    stage0 = FakeStageClient(stage_type="llm", final_output=True)
    rpc_q: asyncio.Queue = asyncio.Queue()
    orchestrator = Orchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=asyncio.Queue(),
        rpc_async_queue=rpc_q,
        stage_pools=_build_stage_pools([[stage0]]),
        duplex_runtime_extension=MiniCPMO45DuplexRuntimeExtension(),
        enable_duplex_control=True,
    )
    fence = DuplexFence("sid-invalid-update")
    original_runtime_config = {
        "duplex_stage_sampling_params": {"0": {"stop_token_ids": [151645]}},
    }
    session = orchestrator.duplex_sessions.open_session(
        fence,
        runtime_config=original_runtime_config,
    )

    await _handle_duplex(
        orchestrator,
        SignalDuplexTurnMessage(
            control_id="invalid-update",
            fence=fence,
            session_id=fence.session_id,
            event="session.update",
            runtime_config={
                "duplex_stage_sampling_params": {"0": {"stop_token_ids": 123}},
            },
        ),
    )

    result = rpc_q.get_nowait()
    assert result.ok is False
    assert session.runtime_config == original_runtime_config
    assert session.config_generation == 0


@pytest.mark.asyncio
async def test_duplex_arbitrary_non_cancel_signal_is_rejected() -> None:
    rpc_q: asyncio.Queue = asyncio.Queue()
    orchestrator = Orchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=asyncio.Queue(),
        rpc_async_queue=rpc_q,
        stage_pools=[],
        enable_duplex_control=True,
    )
    fence = DuplexFence("sid-unsupported-signal")
    orchestrator.duplex_sessions.open_session(fence)

    await _handle_duplex(
        orchestrator,
        SignalDuplexTurnMessage(
            control_id="unsupported-signal",
            fence=fence,
            session_id=fence.session_id,
            event="turn.end",
        ),
    )

    result = rpc_q.get_nowait()
    assert result.ok is False
    assert "unsupported duplex runtime signal" in result.stage_results[0]["result"]["error"]


@pytest.mark.asyncio
async def test_duplex_cancel_rejects_late_old_append_and_accepts_next_epoch() -> None:
    stage0 = FakeStageClient(stage_type="llm", final_output=True)
    stage_pools = _build_stage_pools(
        [[stage0]],
        output_processors=[FakeOutputProcessor()],
        stage_vllm_configs=[SimpleNamespace(model_config=SimpleNamespace(max_model_len=64))],
    )
    rpc_q: asyncio.Queue = asyncio.Queue()
    orchestrator = Orchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=asyncio.Queue(),
        rpc_async_queue=rpc_q,
        stage_pools=stage_pools,
        duplex_runtime_extension=MiniCPMO45DuplexRuntimeExtension(),
        enable_duplex_control=True,
    )
    cancelled_fence = DuplexFence("sid-cancel-late-append")
    next_fence = DuplexFence("sid-cancel-late-append", epoch=1)
    session = orchestrator.duplex_sessions.open_session(
        cancelled_fence,
        capabilities=DuplexRuntimeCapabilities(
            input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK},
        ),
    )

    await _handle_duplex(
        orchestrator,
        SignalDuplexTurnMessage(
            control_id="cancel-old-fence",
            fence=cancelled_fence,
            next_fence=next_fence,
            session_id=session.session_id,
            event="input.cancel",
        ),
    )

    cancel_result = rpc_q.get_nowait()
    assert cancel_result.ok is True
    assert session.fence == next_fence

    await _handle_duplex(
        orchestrator,
        AppendDuplexInputMessage(
            control_id="late-old-append",
            fence=cancelled_fence,
            session_id=session.session_id,
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK.value,
            payload={"is_speech": True},
        ),
    )

    stale_result = rpc_q.get_nowait()
    assert stale_result.ok is False
    assert stage0.add_request_calls == []
    assert session.stage_bindings == {}

    await _handle_duplex(
        orchestrator,
        AppendDuplexInputMessage(
            control_id="next-epoch-append",
            fence=next_fence,
            session_id=session.session_id,
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK.value,
            payload={"is_speech": True},
        ),
    )

    next_result = rpc_q.get_nowait()
    assert next_result.ok is True
    assert len(stage0.add_request_calls) == 1
    assert session.stage_bindings[0].fence == next_fence


@pytest.mark.asyncio
async def test_duplex_append_updates_bridge_turn_id_on_long_lived_stage0_request() -> None:
    stage0 = FakeStageClient(stage_type="llm", final_output=True)
    stage_pools = _build_stage_pools(
        [[stage0]],
        output_processors=[FakeOutputProcessor()],
        stage_vllm_configs=[SimpleNamespace(model_config=SimpleNamespace(max_model_len=64))],
    )
    orchestrator = Orchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=asyncio.Queue(),
        rpc_async_queue=asyncio.Queue(),
        stage_pools=stage_pools,
        duplex_runtime_extension=MiniCPMO45DuplexRuntimeExtension(),
        enable_duplex_control=True,
    )
    session = orchestrator.duplex_sessions.open_session(
        DuplexFence("sid-bridge-turn"),
        capabilities=DuplexRuntimeCapabilities(
            input_modes={DuplexInputMode.APPEND_AUDIO_CHUNK},
        ),
        session_config={
            "voice": "test",
        },
        runtime_config={
            "duplex_stage_sampling_params": {
                "0": {"stop_token_ids": [151645]},
            },
        },
    )

    await _handle_duplex(
        orchestrator,
        AppendDuplexInputMessage(
            control_id="append-1",
            fence=DuplexFence("sid-bridge-turn"),
            session_id="sid-bridge-turn",
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK.value,
            payload={"is_speech": True},
        ),
    )
    req_state1 = _duplex_request_state(orchestrator, session, stage_id=0)
    assert req_state1 is not None
    duplex_state1 = req_state1.streaming.bridge_states["duplex"]
    assert duplex_state1["session_id"] == "sid-bridge-turn"
    assert duplex_state1["epoch"] == 0
    assert duplex_state1["turn_id"] == 0
    assert duplex_state1["session_config"] == session.session_config
    assert req_state1.sampling_params_list[0].stop_token_ids == [151645]
    submitted_request = stage0.add_request_calls[0][0]
    assert submitted_request.sampling_params.stop_token_ids == [151645]
    assert submitted_request.sampling_params.max_tokens == 1

    await _handle_duplex(
        orchestrator,
        AppendDuplexInputMessage(
            control_id="append-2",
            fence=DuplexFence("sid-bridge-turn", turn_id=1, response_seq=1),
            session_id="sid-bridge-turn",
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK.value,
            payload={"is_speech": True, "new_user_turn": True},
        ),
    )
    req_state2 = _duplex_request_state(orchestrator, session, stage_id=0)
    assert req_state2 is req_state1
    duplex_state2 = req_state2.streaming.bridge_states["duplex"]
    assert duplex_state2["turn_id"] == 1
    assert duplex_state2["epoch"] == 0
    expected_request_id = duplex_resource_request_id(DuplexFence("sid-bridge-turn"), "stage0")
    assert [call[0].request_id for call in stage0.add_request_calls] == [
        expected_request_id,
        expected_request_id,
    ]


# ---------------------------------------------------------------------------
# Multi-replica tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_replica_round_robin_distribution(orchestrator_factory) -> None:
    """Two replicas at stage-0, single replica at stage-1.

    Send two requests — they should land on different stage-0 replicas
    (round-robin), then both forward to the single stage-1 replica.
    """
    stage0_r0 = FakeStageClient(stage_type="llm", final_output=False)
    stage0_r1 = FakeStageClient(stage_type="llm", final_output=False)
    stage1 = FakeStageClient(
        stage_type="llm",
        final_output=True,
        next_inputs=[{"prompt_token_ids": [7, 8]}],
    )

    proc0 = FakeOutputProcessor(request_outputs=[_build_request_output("req-0", token_ids=[3], finished=True)])
    proc1 = FakeOutputProcessor(request_outputs=[_build_request_output("req-0", token_ids=[10], finished=True)])

    default_vllm_cfg = SimpleNamespace(model_config=SimpleNamespace(max_model_len=64))
    stage_pools = _build_stage_pools(
        [[stage0_r0, stage0_r1], [stage1]],
        output_processors=[proc0, proc1],
        stage_vllm_configs=[default_vllm_cfg, default_vllm_cfg],
    )

    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)

    try:
        # Request 0 → should land on replica 0 (RR starts at 0)
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-0",
            prompt=SimpleNamespace(request_id="req-0", prompt_token_ids=[1, 2]),
            original_prompt={"prompt": "hello 0"},
            sampling_params_list=[_sampling_params(), _sampling_params()],
            final_stage_id=1,
        )
        await _wait_for(lambda: len(stage0_r0.add_request_calls) == 1)
        assert len(stage0_r1.add_request_calls) == 0

        # Request 1 → should land on replica 1 (RR advances)
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-1",
            prompt=SimpleNamespace(request_id="req-1", prompt_token_ids=[5, 6]),
            original_prompt={"prompt": "hello 1"},
            sampling_params_list=[_sampling_params(), _sampling_params()],
            final_stage_id=1,
        )
        await _wait_for(lambda: len(stage0_r1.add_request_calls) == 1)
        assert len(stage0_r0.add_request_calls) == 1  # unchanged

        # Complete req-0 at stage-0 replica-0 → should forward to stage-1
        stage0_r0.push_engine_core_outputs(_engine_core_outputs("s0r0-raw", 1.0))
        await _wait_for(lambda: len(stage1.add_request_calls) == 1)
        assert stage1.add_request_calls[0][0].request_id == "req-0"

        # Complete req-0 at stage-1 → final output
        proc1.request_outputs = [_build_request_output("req-0", token_ids=[10], finished=True)]
        stage1.push_engine_core_outputs(_engine_core_outputs("s1-raw", 2.0))
        output_msg = await _get_output_message(orchestrator_fixture)

        assert output_msg.request_id == "req-0"
        assert output_msg.stage_id == 1
        assert output_msg.finished is True
        assert "req-0" not in orchestrator_fixture.orchestrator.request_states
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_multi_replica_abort_broadcasts_to_all_replicas(orchestrator_factory) -> None:
    """Abort must be sent to every replica across all stages."""
    stage0_r0 = FakeStageClient(stage_type="llm", final_output=False)
    stage0_r1 = FakeStageClient(stage_type="llm", final_output=False)
    stage1 = FakeStageClient(stage_type="llm", final_output=True)

    proc0 = FakeOutputProcessor()
    proc1 = FakeOutputProcessor()

    default_vllm_cfg = SimpleNamespace(model_config=SimpleNamespace(max_model_len=64))
    stage_pools = _build_stage_pools(
        [[stage0_r0, stage0_r1], [stage1]],
        output_processors=[proc0, proc1],
        stage_vllm_configs=[default_vllm_cfg, default_vllm_cfg],
    )
    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-abort-mr",
            prompt=SimpleNamespace(request_id="req-abort-mr", prompt_token_ids=[1]),
            original_prompt={"prompt": "cancel"},
            sampling_params_list=[_sampling_params(), _sampling_params()],
            final_stage_id=1,
        )
        await _wait_for(lambda: len(stage0_r0.add_request_calls) == 1)

        await _enqueue_abort_request(orchestrator_fixture, ["req-abort-mr"])
        await _wait_for(lambda: bool(stage0_r0.abort_calls))

        assert stage0_r0.abort_calls == [["req-abort-mr"]]
        assert stage0_r1.abort_calls == []
        assert stage1.abort_calls == []
        assert "req-abort-mr" not in orchestrator_fixture.orchestrator.request_states
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_multi_replica_shutdown_all_replicas(orchestrator_factory) -> None:
    """Shutdown must shut down every replica across all stages."""
    stage0_r0 = FakeStageClient(stage_type="llm", final_output=False)
    stage0_r1 = FakeStageClient(stage_type="llm", final_output=False)
    stage1 = FakeStageClient(stage_type="llm", final_output=True)

    default_vllm_cfg = SimpleNamespace(model_config=SimpleNamespace(max_model_len=64))
    stage_pools = _build_stage_pools(
        [[stage0_r0, stage0_r1], [stage1]],
        stage_vllm_configs=[default_vllm_cfg, default_vllm_cfg],
    )
    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)

    await _shutdown_orchestrator(orchestrator_fixture)

    assert not orchestrator_fixture.thread.is_alive()
    for client in [stage0_r0, stage0_r1, stage1]:
        assert client.shutdown_calls == 1


@pytest.mark.asyncio
async def test_stage_pool_submit_update_reuses_existing_binding() -> None:
    """A request admitted to one replica must keep using that replica on updates."""
    stage0_r0 = FakeStageClient(stage_type="llm", final_output=False)
    stage0_r1 = FakeStageClient(stage_type="llm", final_output=False)
    pool = StagePool(
        0,
        [stage0_r0, stage0_r1],
        output_processor=FakeOutputProcessor(),
        stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
    )

    req0_state = OrchestratorRequestState(
        request_id="req-0",
        sampling_params_list=[_sampling_params()],
        final_stage_id=0,
    )
    req1_state = OrchestratorRequestState(
        request_id="req-1",
        sampling_params_list=[_sampling_params()],
        final_stage_id=0,
    )

    await pool.submit_initial("req-0", req0_state, SimpleNamespace(request_id="req-0", prompt_token_ids=[1, 2]))
    await pool.submit_update("req-0", req0_state, SimpleNamespace(request_id="req-0", prompt_token_ids=[3]))
    await pool.submit_initial("req-1", req1_state, SimpleNamespace(request_id="req-1", prompt_token_ids=[4, 5]))
    await pool.submit_update("req-1", req1_state, SimpleNamespace(request_id="req-1", prompt_token_ids=[6]))

    assert pool.get_bound_replica_id("req-0") == 0
    assert pool.get_bound_replica_id("req-1") == 1
    assert len(stage0_r0.add_request_calls) == 2
    assert len(stage0_r1.add_request_calls) == 2
    assert stage0_r0.add_request_calls[0][0].request_id == "req-0"
    assert stage0_r0.add_request_calls[1][0].request_id == "req-0"
    assert stage0_r1.add_request_calls[0][0].request_id == "req-1"
    assert stage0_r1.add_request_calls[1][0].request_id == "req-1"


@pytest.mark.asyncio
async def test_stage_pool_failed_replica_releases_distributed_affinity_and_stops_polling() -> None:
    stage0 = FakeStageClient(stage_type="llm", final_output=False)
    pool = StagePool(
        0,
        [stage0],
        output_processor=FakeOutputProcessor(),
        stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
    )
    pool._addr_to_replica_id["tcp://replica-0"] = 0
    pool.bind("distributed-request", "tcp://replica-0")
    pool._request_bindings["legacy-request"] = 0

    affected = pool.mark_replica_unavailable(0)

    assert set(affected) == {"distributed-request", "legacy-request"}
    assert pool.get_bound_replica_id("distributed-request") is None
    assert pool.get_bound_replica_id("legacy-request") is None
    assert await pool.poll_llm_raw_output(0) is None


def test_stage_pool_reattached_replica_becomes_available_again() -> None:
    failed_client = FakeStageClient(stage_type="llm", final_output=False)
    replacement_client = FakeStageClient(stage_type="llm", final_output=False)
    pool = StagePool(
        0,
        [failed_client],
        output_processor=FakeOutputProcessor(),
        stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
    )
    input_addr = "tcp://replica-0"
    pool._addr_to_replica_id[input_addr] = 0
    pool.mark_replica_unavailable(0)
    removed = pool.remove_client(input_addr)

    replica_id = pool.add_client(input_addr, replacement_client)

    assert removed is failed_client
    assert replica_id == 0
    assert len(pool.clients) == 1
    assert pool.clients[0] is replacement_client
    assert pool.available_replica_ids() == [0]


@pytest.mark.asyncio
async def test_stage_pool_submit_update_refreshes_output_processor_state() -> None:
    output_processor = FakeOutputProcessor()

    class AssertingStageClient(FakeStageClient):
        async def add_request_async(self, *args, **kwargs) -> None:
            if len(self.add_request_calls) == 1:
                prompts = [call_kwargs["prompt"] for _, call_kwargs in output_processor.add_request_calls]
                assert prompts == ["seg-1", "seg-2"]
            await super().add_request_async(*args, **kwargs)

    stage0 = AssertingStageClient(stage_type="llm", final_output=False)
    pool = StagePool(
        0,
        [stage0],
        output_processor=output_processor,
        stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
    )
    req_state = OrchestratorRequestState(
        request_id="req-0",
        sampling_params_list=[_sampling_params()],
        final_stage_id=0,
    )

    await pool.submit_initial(
        "req-0",
        req_state,
        SimpleNamespace(request_id="req-0", prompt_token_ids=[1, 2]),
        prompt_text="seg-1",
    )
    await pool.submit_update(
        "req-0",
        req_state,
        SimpleNamespace(request_id="req-0", prompt_token_ids=[3], resumable=True),
        prompt_text="seg-2",
    )

    assert len(output_processor.add_request_calls) == 2
    assert output_processor.add_request_calls[1][1]["prompt"] == "seg-2"


@pytest.mark.asyncio
async def test_handle_streaming_update_passes_prompt_text_to_stage_pool() -> None:
    class RecordingPool:
        def __init__(self) -> None:
            self.calls: list[tuple[str, Any]] = []

        async def submit_update(self, request_id, req_state, request, *, prompt_text=None) -> int:
            self.calls.append((request_id, prompt_text))
            return 0

    pool = RecordingPool()
    orchestrator = object.__new__(Orchestrator)
    orchestrator.async_chunk = False
    orchestrator.request_states = {
        "req-stream": OrchestratorRequestState(
            request_id="req-stream",
            sampling_params_list=[_sampling_params()],
            final_stage_id=0,
        )
    }
    orchestrator.stage_pools = [pool]

    await orchestrator._handle_streaming_update(
        StageSubmissionMessage(
            type="streaming_update",
            request_id="req-stream",
            prompt=SimpleNamespace(request_id="req-stream", prompt_token_ids=[1], resumable=True),
            original_prompt={"prompt": "segment-2"},
            output_prompt_text="segment-2",
            sampling_params_list=[_sampling_params()],
            final_stage_id=0,
            preprocess_ms=0.0,
            request_timestamp=time.time(),
            enqueue_ts=time.perf_counter(),
        )
    )

    assert pool.calls == [("req-stream", "segment-2")]
    assert orchestrator.request_states["req-stream"].streaming.enabled is True


@pytest.mark.asyncio
async def test_resumable_segment_boundary_builds_stage_metrics() -> None:
    built_metrics = SimpleNamespace(pipeline_timings={})

    class RecordingPool:
        def __init__(self) -> None:
            self.calls: list[list[Any]] = []

        def build_stage_metrics(self, outputs, **_kwargs):
            self.calls.append(outputs)
            return built_metrics

    pool = RecordingPool()
    orchestrator = object.__new__(Orchestrator)
    req_state = OrchestratorRequestState(
        request_id="req-stream",
        sampling_params_list=[_sampling_params()],
        final_stage_id=0,
    )
    req_state.streaming.enabled = True
    req_state.streaming.segment_finished = True
    req_state.stage_submit_ts[0] = time.time()
    orchestrator.request_states = {"req-stream": req_state}
    orchestrator.stage_pools = [pool]
    routed: list[Any] = []

    async def record_route(_stage_id, _replica_id, _output, _req_state, stage_metrics):
        routed.append(stage_metrics)

    orchestrator._route_output = record_route
    output = SimpleNamespace(request_id="req-stream", error=None, finished=False)

    await orchestrator._handle_processed_outputs(0, 0, [output])

    assert pool.calls == [[output]]
    assert routed == [built_metrics]


def test_stage_pool_metrics_use_resumable_segment_token_count() -> None:
    class SegmentMetricsOutputProcessor(FakeOutputProcessor):
        def pop_native_text_metrics(self, request_id: str) -> dict[str, Any]:
            assert request_id == "req-stream"
            return {"num_generation_tokens": 3}

    stage0 = FakeStageClient(stage_type="llm", final_output=False)
    pool = StagePool(
        0,
        [stage0],
        output_processor=SegmentMetricsOutputProcessor(),
        stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
    )
    output = SimpleNamespace(
        request_id="req-stream",
        outputs=[SimpleNamespace(cumulative_token_ids=list(range(11)))],
    )

    metrics = pool.build_stage_metrics(
        [output],
        submit_ts=time.time(),
        request_timestamp=time.time(),
        replica_id=0,
    )

    assert metrics.num_tokens_out == 3
    assert metrics.output_unit_count == 3


@pytest.mark.asyncio
async def test_stage_pool_submit_initial_rolls_back_output_processor_when_client_submit_fails() -> None:
    class FailingStageClient(FakeStageClient):
        async def add_request_async(self, *args, **kwargs) -> None:
            raise RuntimeError("submit failed")

    class TrackingOutputProcessor(FakeOutputProcessor):
        def __init__(self) -> None:
            super().__init__()
            self.added_request_ids: list[str] = []
            self.removed_request_ids: list[str] = []

        def add_request(self, request, *_args, **_kwargs) -> None:
            self.added_request_ids.append(request.request_id)

        def remove_request(self, request_id: str) -> None:
            self.removed_request_ids.append(request_id)

    client = FailingStageClient(stage_type="llm", final_output=False)
    output_processor = TrackingOutputProcessor()
    pool = StagePool(
        0,
        [client],
        output_processor=output_processor,
        stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
    )
    req_state = OrchestratorRequestState(
        request_id="req-0",
        sampling_params_list=[_sampling_params()],
        final_stage_id=0,
    )

    with pytest.raises(RuntimeError, match="submit failed"):
        await pool.submit_initial("req-0", req_state, SimpleNamespace(request_id="req-0", prompt_token_ids=[1, 2]))

    assert output_processor.added_request_ids == ["req-0"]
    assert output_processor.removed_request_ids == ["req-0"]
    assert pool.get_bound_replica_id("req-0") is None


@pytest.mark.asyncio
async def test_stage_pool_abort_requests_logs_when_binding_is_missing(caplog) -> None:
    stage0 = FakeStageClient(stage_type="llm", final_output=False)
    pool = StagePool(
        0,
        [stage0],
        output_processor=FakeOutputProcessor(),
        stage_vllm_config=SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
    )

    target_logger = logging.getLogger("vllm_omni.engine.stage_pool")
    target_logger.addHandler(caplog.handler)
    prev_level = target_logger.level
    target_logger.setLevel(logging.DEBUG)
    try:
        await pool.abort_requests(["missing-req"])
    finally:
        target_logger.removeHandler(caplog.handler)
        target_logger.setLevel(prev_level)

    assert not stage0.abort_calls
    assert "abort: no live binding for req=missing-req in stage-0" in caplog.text


@pytest.mark.asyncio
async def test_collective_rpc_ignores_invalid_stage_ids(orchestrator_factory, caplog) -> None:
    stage0 = FakeCollectiveRpcStageClient(stage_type="llm", final_output=True, rpc_result={"stage": 0})
    stage1 = FakeCollectiveRpcStageClient(stage_type="llm", final_output=True, rpc_result={"stage": 1})
    stage_pools = _build_stage_pools(
        [[stage0], [stage1]],
        output_processors=[FakeOutputProcessor(), FakeOutputProcessor()],
        stage_vllm_configs=[
            SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
            SimpleNamespace(model_config=SimpleNamespace(max_model_len=64)),
        ],
    )
    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)

    try:
        target_logger = logging.getLogger("vllm_omni.engine.orchestrator")
        target_logger.addHandler(caplog.handler)
        prev_level = target_logger.level
        target_logger.setLevel(logging.WARNING)
        try:
            orchestrator_fixture.request_sync_q.put_nowait(
                CollectiveRPCRequestMessage(
                    rpc_id="rpc-1",
                    method="list_loras",
                    timeout=None,
                    args=(),
                    kwargs={},
                    stage_ids=[99, 1],
                )
            )

            msg = await _get_rpc_message(orchestrator_fixture)
        finally:
            target_logger.removeHandler(caplog.handler)
            target_logger.setLevel(prev_level)

        assert msg.type == "collective_rpc_result"
        assert msg.rpc_id == "rpc-1"
        assert msg.stage_ids == [1]
        assert msg.results == [{"stage": 1}]
        assert not stage0.collective_rpc_calls
        assert len(stage1.collective_rpc_calls) == 1
        assert "collective_rpc: ignoring invalid stage_id 99" in caplog.text
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


@pytest.mark.asyncio
async def test_multi_replica_cfg_companion_inherits_parent_affinity(orchestrator_factory) -> None:
    """CFG companions should be routed to the same stage-0 replica as their parent."""
    stage0_r0 = FakeStageClient(stage_type="llm", final_output=False)
    stage0_r1 = FakeStageClient(stage_type="llm", final_output=False)
    default_vllm_cfg = SimpleNamespace(model_config=SimpleNamespace(max_model_len=64))
    stage_pools = _build_stage_pools(
        [[stage0_r0, stage0_r1]],
        output_processors=[FakeOutputProcessor()],
        stage_vllm_configs=[default_vllm_cfg],
    )
    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)

    try:
        # Consume replica-0 first so the parent request binds to replica-1.
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="warmup",
            prompt=SimpleNamespace(request_id="warmup", prompt_token_ids=[0]),
            original_prompt={"prompt": "warmup"},
            sampling_params_list=[_sampling_params()],
            final_stage_id=0,
        )
        await _wait_for(lambda: len(stage0_r0.add_request_calls) == 1)

        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="parent",
            prompt=SimpleNamespace(request_id="parent", prompt_token_ids=[1, 2]),
            original_prompt={"prompt": "parent"},
            sampling_params_list=[_sampling_params()],
            final_stage_id=0,
        )
        await _wait_for(lambda: len(stage0_r1.add_request_calls) == 1)

        orchestrator_fixture.request_sync_q.put_nowait(
            AddCompanionRequestMessage(
                companion_id="parent-neg",
                parent_id="parent",
                role="negative",
                prompt=SimpleNamespace(request_id="parent-neg", prompt_token_ids=[9]),
                companion_prompt_text={"prompt": "negative"},
                sampling_params_list=[_sampling_params()],
            )
        )
        await _wait_for(lambda: len(stage0_r1.add_request_calls) == 2)

        assert stage_pools[0].get_bound_replica_id("parent") == 1
        assert stage_pools[0].get_bound_replica_id("parent-neg") == 1
        assert len(stage0_r0.add_request_calls) == 1
        assert stage0_r1.add_request_calls[0][0].request_id == "parent"
        assert stage0_r1.add_request_calls[1][0].request_id == "parent-neg"
    finally:
        await _shutdown_orchestrator(orchestrator_fixture)


def test_orchestrator_does_not_re_introduce_global_stats_throttle() -> None:
    """Regression: each (stage, replica) must independently publish its wrapped
    vllm:* stats when its scheduler emits non-None scheduler_stats.

    A previous version of Orchestrator carried a global self._last_stats_ts /
    _stats_interval_s gate around _stat_logger.record(). Because
    OmniSchedulerMixin.make_stats() already throttles at 1 Hz per scheduler
    (one per (stage, replica)), the extra global gate starved every replica
    other than the first to emit within each second — their {stage, replica}
    gauges/counters went stale.

    The fix removed the global gate entirely; the only signal needed is
    'this replica's scheduler emitted non-None scheduler_stats'. This test
    fails loudly if someone reintroduces the global throttle.
    """
    import inspect

    from vllm_omni.engine.orchestrator import Orchestrator

    source = inspect.getsource(Orchestrator)
    assert "_last_stats_ts" not in source, (
        "Orchestrator must not gate stat recording on a global timestamp. "
        "OmniSchedulerMixin.make_stats() already throttles per scheduler "
        "(per (stage, replica)); an outer global gate starves all but the "
        "first replica to emit within each 1s window."
    )
    assert "_stats_interval_s" not in source
    assert "raw_outputs.scheduler_stats is not None" in source, (
        "Orchestrator must gate stat recording solely on "
        "raw_outputs.scheduler_stats being non-None — the per-scheduler 1Hz "
        "throttle in OmniSchedulerMixin.make_stats() is the only gate needed."
    )


@pytest.mark.asyncio
async def test_duplex_reaper_loop_waits_between_ticks():
    class _Plane:
        def __init__(self) -> None:
            self.calls = 0

        async def reap_expired(self) -> int:
            self.calls += 1
            return 0

    orchestrator = object.__new__(Orchestrator)
    orchestrator.duplex_control_plane = _Plane()
    orchestrator._duplex_reaper_interval_s = 0.01
    orchestrator._shutdown_event = asyncio.Event()

    task = asyncio.create_task(orchestrator._duplex_reaper_loop())
    await asyncio.sleep(0.035)
    orchestrator._shutdown_event.set()
    await task

    assert 2 <= orchestrator.duplex_control_plane.calls <= 5


@pytest.mark.asyncio
async def test_duplex_reaper_loop_survives_one_cleanup_failure():
    class _Plane:
        def __init__(self) -> None:
            self.calls = 0

        async def reap_expired(self) -> int:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient cleanup failure")
            return 0

    orchestrator = object.__new__(Orchestrator)
    orchestrator.duplex_control_plane = _Plane()
    orchestrator._duplex_reaper_interval_s = 0.01
    orchestrator._shutdown_event = asyncio.Event()

    task = asyncio.create_task(orchestrator._duplex_reaper_loop())
    await asyncio.sleep(0.035)
    orchestrator._shutdown_event.set()
    await task

    assert orchestrator.duplex_control_plane.calls >= 2


@pytest.mark.asyncio
async def test_abort_retry_does_not_repeat_successful_stage_abort():
    class _Pool:
        def __init__(self, *, fail_once: bool = False) -> None:
            self.bound = {"req-a"}
            self.fail_once = fail_once
            self.physical_abort_calls = 0

        async def abort_requests(self, request_ids: list[str]) -> None:
            active = self.bound.intersection(request_ids)
            if not active:
                return
            self.physical_abort_calls += 1
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("stage abort failed")

        def release_bindings(self, request_ids: list[str]) -> None:
            self.bound.difference_update(request_ids)

    first = _Pool()
    second = _Pool(fail_once=True)
    orchestrator = object.__new__(Orchestrator)
    orchestrator.stage_pools = [first, second]

    with pytest.raises(RuntimeError, match="stage abort failed"):
        await orchestrator._abort_request_ids(["req-a"])

    assert first.bound == set()
    assert first.physical_abort_calls == 1
    assert second.bound == {"req-a"}

    await orchestrator._abort_request_ids(["req-a"])

    assert first.physical_abort_calls == 1
    assert second.physical_abort_calls == 2
    assert second.bound == set()


@pytest.mark.asyncio
async def test_request_cleanup_failure_is_deferred_to_control_plane():
    class _FailingPool:
        async def abort_requests(self, request_ids: list[str]) -> None:
            raise RuntimeError("stage abort failed")

        def release_bindings(self, request_ids: list[str]) -> None:
            pass

    class _Plane:
        def __init__(self) -> None:
            self.deferred: list[str] = []
            self.finalized: list[str] = []

        def close_sessions_for_request_ids(self, request_ids: list[str], **kwargs):
            assert kwargs == {"abort": True, "cleanup_in_progress": True}
            return {"sid-cleanup": ["req-a"]}

        def defer_request_cleanups(self, session_ids: list[str]) -> None:
            self.deferred.extend(session_ids)

        def finalize_closed_sessions(self, session_ids: list[str]) -> None:
            self.finalized.extend(session_ids)

    orchestrator = object.__new__(Orchestrator)
    orchestrator.stage_pools = [_FailingPool()]
    orchestrator.duplex_control_plane = _Plane()
    orchestrator._pd_kv_params = {}
    orchestrator.request_states = {}
    orchestrator._running_counter = None

    with pytest.raises(RuntimeError, match="stage abort failed"):
        await orchestrator._cleanup_request_ids(
            ["req-a"],
            abort=True,
            close_duplex_sessions=True,
        )

    assert orchestrator.duplex_control_plane.deferred == ["sid-cleanup"]
    assert orchestrator.duplex_control_plane.finalized == []
