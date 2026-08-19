# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for error propagation paths within the Orchestrator.

Covers:
- EngineDeadError from an LLM stage poll → fatal error broadcast + replica eviction
- Diffusion stage error output (OmniRequestOutput.from_error) → routed correctly
"""

from __future__ import annotations

import asyncio
import queue
import time
from types import SimpleNamespace

import janus
import pytest
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.engine.messages import (
    AddCompanionRequestMessage,
    EngineQueueMessage,
    ErrorMessage,
    ShutdownRequestMessage,
    StageSubmissionMessage,
)
from vllm_omni.engine.orchestrator import Orchestrator, OrchestratorRequestState
from vllm_omni.engine.stage_pool import StageUnavailableError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

from .test_orchestrator import (
    FakeOutputProcessor,
    FakeStageClient,
    OrchestratorFixture,
    _build_harness,
    _build_request_output,
    _build_stage_pools,
    _engine_core_outputs,
    _enqueue_add_request,
    _wait_for,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _sampling_params(max_tokens: int = 4):
    from vllm.sampling_params import SamplingParams

    return SamplingParams(max_tokens=max_tokens)


async def _get_any_output_message(fixture: OrchestratorFixture, *, timeout: float = 2.0) -> EngineQueueMessage:
    """Like _get_output_message but returns any message type (including errors)."""
    deadline = time.monotonic() + timeout
    while True:
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for orchestrator output")
        try:
            return fixture.output_sync_q.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.01)


async def _wait_for_error_message(
    fixture: OrchestratorFixture,
    *,
    request_id: str | None = None,
    max_messages: int = 50,
) -> ErrorMessage:
    """Drain output messages until the ErrorMessage arrives, skipping the
    non-error outputs (e.g. stage metrics, forwarded stage results) that a
    driven pipeline can interleave ahead of the failure."""
    for _ in range(max_messages):
        msg = await _get_any_output_message(fixture)
        if isinstance(msg, ErrorMessage) and (request_id is None or msg.request_id == request_id):
            return msg
    raise AssertionError(f"No ErrorMessage for req={request_id} within {max_messages} messages")


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


# ───────── EngineDeadError from LLM stage poll ─────────


@pytest.mark.asyncio
async def test_engine_dead_error_evicts_replica_and_keeps_running(orchestrator_factory) -> None:
    """When a stage replica raises EngineDeadError during poll, the orchestrator must:
    1. Enqueue a fatal error message for each affected request
    2. Evict the dead replica and keep running so the API server stays up
       (per-replica fault isolation, #4285)
    """
    stage0 = _DieOnDemandLLMStageClient(stage_type="llm", final_output=True)
    orchestrator_fixture = orchestrator_factory([stage0])
    request = SimpleNamespace(request_id="req-dead", prompt_token_ids=[1, 2])

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-dead",
            prompt=request,
            original_prompt={"prompt": "hello"},
            sampling_params_list=[_sampling_params()],
            final_stage_id=0,
        )
        # Let the request dispatch before the replica dies; a replica dead on
        # its very first poll can be evicted before dispatch, which routes the
        # failure through the add-request guard instead (that ordering is
        # covered by test_add_request_to_dead_stage_fails_request_not_server).
        await _wait_for(lambda: len(stage0.add_request_calls) == 1)
        stage0.die = True

        # Collect the fatal error message.
        msg = await _get_any_output_message(orchestrator_fixture)

        assert isinstance(msg, ErrorMessage)
        assert msg.type == "error"
        assert msg.fatal is True
        assert msg.request_id == "req-dead"
        assert "Stage-0 replica-0 is dead" in msg.error

        # The dead replica is evicted; the orchestrator keeps running.
        pool = orchestrator_fixture.orchestrator.stage_pools[0]
        deadline = time.monotonic() + 5.0
        while pool.live_replica_ids() and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        assert pool.live_replica_ids() == []
        assert orchestrator_fixture.thread.is_alive()

        # Affected request state should be cleaned up.
        assert "req-dead" not in orchestrator_fixture.orchestrator.request_states
    finally:
        if orchestrator_fixture.thread.is_alive():
            orchestrator_fixture.request_sync_q.put_nowait(ShutdownRequestMessage())
            orchestrator_fixture.thread.join(timeout=5)


class _DieOnDemandLLMStageClient(FakeStageClient):
    """LLM stage replica that raises EngineDeadError on poll once ``die`` is set."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.die = False

    async def get_output_async(self):
        if self.die:
            raise EngineDeadError(f"Stage-{self.stage_id} replica-{self.replica_id} is dead")
        return await super().get_output_async()


class _DieOnDemandDiffusionStageClient(FakeStageClient):
    """Diffusion stage replica that raises EngineDeadError on poll once ``die`` is set."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.die = False

    def get_diffusion_output_nowait(self):
        if self.die:
            raise EngineDeadError(f"Stage-{self.stage_id} diffusion replica-{self.replica_id} is dead")
        return super().get_diffusion_output_nowait()


class _UnavailableDiffusionStageClient(FakeStageClient):
    """Diffusion replica that stays live (so the pool keeps stage_type) but whose
    dispatch fails with StageUnavailableError, as a downstream prewarm would when
    its stage has no capacity."""

    async def add_request_async(self, *args, **kwargs) -> None:
        raise StageUnavailableError(f"stage {self.stage_id} has no live replica")


@pytest.mark.asyncio
async def test_engine_dead_error_fails_only_dead_replica_requests(orchestrator_factory) -> None:
    """Multi-replica stage: one replica dying must fail only the requests bound
    to it. Requests bound to a surviving replica keep running, and the stage
    (still having a live replica) is not torn down (per-replica fault isolation,
    #4285).
    """
    r0 = _DieOnDemandLLMStageClient(stage_type="llm", final_output=True)
    r1 = FakeStageClient(stage_type="llm", final_output=True)
    stage_pools = _build_stage_pools([[r0, r1]])
    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)
    pool = orchestrator_fixture.orchestrator.stage_pools[0]

    try:
        # req-0 → replica 0 (round-robin starts at 0)
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-0",
            prompt=SimpleNamespace(request_id="req-0", prompt_token_ids=[1, 2]),
            original_prompt={"prompt": "a"},
            sampling_params_list=[_sampling_params()],
            final_stage_id=0,
        )
        await _wait_for(lambda: len(r0.add_request_calls) == 1)

        # req-1 → replica 1
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-1",
            prompt=SimpleNamespace(request_id="req-1", prompt_token_ids=[3, 4]),
            original_prompt={"prompt": "b"},
            sampling_params_list=[_sampling_params()],
            final_stage_id=0,
        )
        await _wait_for(lambda: len(r1.add_request_calls) == 1)
        assert pool.get_bound_replica_id("req-0") == 0
        assert pool.get_bound_replica_id("req-1") == 1

        # Kill replica 0 on its next poll.
        r0.die = True

        msg = await _get_any_output_message(orchestrator_fixture)
        assert isinstance(msg, ErrorMessage)
        assert msg.fatal is True
        assert msg.request_id == "req-0"

        # Replica 0 evicted, replica 1 still serving; req-1 untouched.
        await _wait_for(lambda: pool.live_replica_ids() == [1])
        assert orchestrator_fixture.thread.is_alive()
        assert "req-0" not in orchestrator_fixture.orchestrator.request_states
        assert "req-1" in orchestrator_fixture.orchestrator.request_states
    finally:
        if orchestrator_fixture.thread.is_alive():
            orchestrator_fixture.request_sync_q.put_nowait(ShutdownRequestMessage())
            orchestrator_fixture.thread.join(timeout=5)


@pytest.mark.asyncio
async def test_forward_to_dead_downstream_stage_fails_request_not_server(orchestrator_factory) -> None:
    """A request mid-pipeline whose downstream stage has lost all replicas must
    fail that request, not tear down the orchestrator. Forwarding to a stage with
    no live replica is handled gracefully (per-replica fault isolation, #4285).
    """
    stage0 = FakeStageClient(stage_type="llm", final_output=False, next_inputs=[{"prompt_token_ids": [7, 8]}])
    stage1 = _DieOnDemandLLMStageClient(stage_type="llm", final_output=True)
    proc0 = FakeOutputProcessor(request_outputs=[_build_request_output("req-x", token_ids=[3], finished=True)])
    proc1 = FakeOutputProcessor()
    stage_pools = _build_stage_pools([[stage0], [stage1]], output_processors=[proc0, proc1])
    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)
    pool1 = orchestrator_fixture.orchestrator.stage_pools[1]

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-x",
            prompt=SimpleNamespace(request_id="req-x", prompt_token_ids=[1, 2]),
            original_prompt={"prompt": "hello"},
            sampling_params_list=[_sampling_params(), _sampling_params()],
            final_stage_id=1,
        )
        await _wait_for(lambda: len(stage0.add_request_calls) == 1)

        # Kill stage-1's only replica before stage-0 output is forwarded.
        stage1.die = True
        await _wait_for(lambda: pool1.live_replica_ids() == [])

        # Completing stage 0 triggers a forward to the now-empty stage 1.
        stage0.push_engine_core_outputs(_engine_core_outputs("s0-raw", 1.0))

        # Completing stage 0 also emits its normal outputs; skip past them to
        # the downstream-forward failure.
        error_msg = await _wait_for_error_message(orchestrator_fixture, request_id="req-x")
        assert error_msg.fatal is True
        assert error_msg.stage_id == 1

        assert orchestrator_fixture.thread.is_alive()
        assert "req-x" not in orchestrator_fixture.orchestrator.request_states
    finally:
        if orchestrator_fixture.thread.is_alive():
            orchestrator_fixture.request_sync_q.put_nowait(ShutdownRequestMessage())
            orchestrator_fixture.thread.join(timeout=5)


@pytest.mark.asyncio
async def test_add_request_to_dead_stage_fails_request_not_server(orchestrator_factory) -> None:
    """A new request entering a stage that already lost all replicas fails that
    request instead of raising out of the request handler and tearing the server
    down. The guard runs before request state is registered, so nothing leaks.
    """
    r0 = _DieOnDemandLLMStageClient(stage_type="llm", final_output=True)
    stage_pools = _build_stage_pools([[r0]])
    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)
    pool = orchestrator_fixture.orchestrator.stage_pools[0]

    try:
        # Send a first request so the replica is polled and dies → evicted.
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-0",
            prompt=SimpleNamespace(request_id="req-0", prompt_token_ids=[1, 2]),
            original_prompt={"prompt": "a"},
            sampling_params_list=[_sampling_params()],
            final_stage_id=0,
        )
        r0.die = True
        await _wait_for(lambda: pool.live_replica_ids() == [])

        # A new request now enters a stage with no live replica.
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-after",
            prompt=SimpleNamespace(request_id="req-after", prompt_token_ids=[3, 4]),
            original_prompt={"prompt": "b"},
            sampling_params_list=[_sampling_params()],
            final_stage_id=0,
        )

        error_msg = await _wait_for_error_message(orchestrator_fixture, request_id="req-after")
        assert error_msg.fatal is True
        assert error_msg.stage_id == 0

        assert orchestrator_fixture.thread.is_alive()
        # Guard runs before registration, so the failed request never enters state.
        assert "req-after" not in orchestrator_fixture.orchestrator.request_states
    finally:
        if orchestrator_fixture.thread.is_alive():
            orchestrator_fixture.request_sync_q.put_nowait(ShutdownRequestMessage())
            orchestrator_fixture.thread.join(timeout=5)


@pytest.mark.asyncio
async def test_diffusion_replica_death_on_poll_keeps_server(orchestrator_factory) -> None:
    """A diffusion stage replica raising EngineDeadError during poll must be
    evicted while the orchestrator keeps serving. The per-replica catch has to
    cover the diffusion polling path, not just the LLM one (#4285).
    """
    stage0 = _DieOnDemandDiffusionStageClient(stage_type="diffusion", final_output=True, final_output_type="image")
    stage_pools = _build_stage_pools([[stage0]])
    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools)
    pool = orchestrator_fixture.orchestrator.stage_pools[0]

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-dit",
            prompt={"prompt": "a cat"},
            original_prompt={"prompt": "a cat"},
            sampling_params_list=[OmniDiffusionSamplingParams()],
            final_stage_id=0,
        )
        await _wait_for(lambda: len(stage0.add_request_calls) == 1)

        stage0.die = True

        error_msg = await _wait_for_error_message(orchestrator_fixture, request_id="req-dit")
        assert error_msg.fatal is True
        assert error_msg.stage_id == 0

        await _wait_for(lambda: pool.live_replica_ids() == [])
        assert orchestrator_fixture.thread.is_alive()
        assert "req-dit" not in orchestrator_fixture.orchestrator.request_states
    finally:
        if orchestrator_fixture.thread.is_alive():
            orchestrator_fixture.request_sync_q.put_nowait(ShutdownRequestMessage())
            orchestrator_fixture.thread.join(timeout=5)


@pytest.mark.asyncio
async def test_async_chunk_prewarm_failure_reports_failing_downstream_stage(orchestrator_factory) -> None:
    """When an async-chunk prewarm dispatch to a downstream stage fails, the
    failure must be attributed to that stage, not stage 0 (the prewarm's own
    dispatch stage). Regression for the guard using stage_id=0.
    """
    stage0 = FakeStageClient(stage_type="llm", final_output=False)
    stage1 = FakeStageClient(stage_type="diffusion", final_output=False, final_output_type="image")
    stage2 = _UnavailableDiffusionStageClient(stage_type="diffusion", final_output=True, final_output_type="image")
    stage_pools = _build_stage_pools([[stage0], [stage1], [stage2]])
    orchestrator_fixture = orchestrator_factory([], stage_pools=stage_pools, async_chunk=True)

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-x",
            prompt=SimpleNamespace(request_id="req-x", prompt_token_ids=[1, 2, 3]),
            original_prompt={"prompt": "a cat"},
            sampling_params_list=[_sampling_params(), OmniDiffusionSamplingParams(), OmniDiffusionSamplingParams()],
            final_stage_id=2,
        )

        error_msg = await _wait_for_error_message(orchestrator_fixture, request_id="req-x")
        assert error_msg.fatal is True
        assert error_msg.stage_id == 2
        assert "Stage-2" in error_msg.error

        # Stage 1 prewarm ran before the stage-2 failure; server stays up.
        assert len(stage1.add_request_calls) == 1
        assert orchestrator_fixture.thread.is_alive()
    finally:
        if orchestrator_fixture.thread.is_alive():
            orchestrator_fixture.request_sync_q.put_nowait(ShutdownRequestMessage())
            orchestrator_fixture.thread.join(timeout=5)


@pytest.mark.asyncio
async def test_async_chunk_prewarm_without_prompt_token_ids_fails_only_that_request(
    orchestrator_factory,
) -> None:
    """R1.3 of #4855: an async-chunk request whose stage-0 prompt carries no
    token ids cannot prewarm anything, so it must fail with a client error
    instead of stalling.

    Non-fatal is the point: ``fatal=True`` reads as "the engine died" and the
    entrypoints act on it (``OmniLLM.generate`` raises out of its batch loop,
    the serving layer calls ``terminate_if_errored``). One client's embeds-only
    prompt must not reach other requests.
    """
    stage0 = FakeStageClient(stage_type="llm", final_output=False)
    stage1 = FakeStageClient(stage_type="llm", final_output=True)
    orchestrator_fixture = orchestrator_factory([stage0, stage1], async_chunk=True)

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-embeds",
            # No prompt_token_ids: what an embeds-only prompt looks like here.
            prompt=SimpleNamespace(request_id="req-embeds"),
            original_prompt={"prompt": "embeds only"},
            sampling_params_list=[_sampling_params(), _sampling_params()],
            final_stage_id=1,
        )

        error_msg = await _wait_for_error_message(orchestrator_fixture, request_id="req-embeds")
        assert error_msg.fatal is False
        assert error_msg.status_code == 400
        assert error_msg.error_type == "BadRequestError"
        assert error_msg.stage_id == 0
        assert "prompt_token_ids" in error_msg.error

        # Nothing was prewarmed downstream, and the request is gone.
        assert stage1.add_request_calls == []
        await _wait_for(lambda: "req-embeds" not in orchestrator_fixture.orchestrator.request_states)
        assert orchestrator_fixture.thread.is_alive()
    finally:
        if orchestrator_fixture.thread.is_alive():
            orchestrator_fixture.request_sync_q.put_nowait(ShutdownRequestMessage())
            orchestrator_fixture.thread.join(timeout=5)


# ───────── Direct unit tests for the fault-isolation helpers ─────────


def _build_bare_orchestrator(stage_pools) -> tuple[Orchestrator, tuple[janus.Queue, ...]]:
    """Construct an Orchestrator on the current event loop without running it.

    Lets tests await the fault-isolation helpers directly instead of driving
    them through the orchestration loop.
    """
    queues = (janus.Queue(), janus.Queue(), janus.Queue())
    orchestrator = Orchestrator(
        request_async_queue=queues[0].async_q,
        output_async_queue=queues[1].async_q,
        rpc_async_queue=queues[2].async_q,
        stage_pools=stage_pools,
    )
    return orchestrator, queues


def _register_request(orchestrator: Orchestrator, request_id: str, *, submitted_stage: int | None = 0) -> None:
    state = OrchestratorRequestState(request_id=request_id)
    if submitted_stage is not None:
        state.stage_submit_ts[submitted_stage] = time.time()
    orchestrator.request_states[request_id] = state


@pytest.mark.asyncio
async def test_handle_dead_replica_fails_only_bound_requests() -> None:
    """Direct helper test: with a surviving replica, only requests bound to the
    dead replica fail; the survivor's requests and bindings are untouched.
    """
    r0 = FakeStageClient(stage_type="llm", final_output=True)
    r1 = FakeStageClient(stage_type="llm", final_output=True)
    orchestrator, queues = _build_bare_orchestrator(_build_stage_pools([[r0, r1]]))
    try:
        pool = orchestrator.stage_pools[0]
        _register_request(orchestrator, "req-0")
        _register_request(orchestrator, "req-1")
        assert pool.select_replica_id("req-0") == 0
        assert pool.select_replica_id("req-1") == 1

        await orchestrator._handle_dead_replica(0, 0, EngineDeadError("replica-0 dead"))

        assert pool.live_replica_ids() == [1]
        assert r0.shutdown_calls == 1
        msg = queues[1].async_q.get_nowait()
        assert isinstance(msg, ErrorMessage)
        assert msg.fatal is True
        assert msg.request_id == "req-0"
        assert msg.stage_id == 0
        assert "replica-0 dead" in msg.error
        assert queues[1].async_q.empty()
        assert "req-0" not in orchestrator.request_states
        assert "req-1" in orchestrator.request_states
        assert pool.get_bound_replica_id("req-0") is None
        assert pool.get_bound_replica_id("req-1") == 1
    finally:
        for q in queues:
            q.close()


@pytest.mark.asyncio
async def test_handle_dead_replica_last_replica_fails_all_stage_requests() -> None:
    """Direct helper test: when the dead replica was the stage's last one, every
    request routed through the stage fails — bound or not — while requests that
    never entered the stage survive.
    """
    orchestrator, queues = _build_bare_orchestrator(_build_stage_pools([[FakeStageClient(stage_type="llm")]]))
    try:
        pool = orchestrator.stage_pools[0]
        _register_request(orchestrator, "req-bound")
        _register_request(orchestrator, "req-unbound")
        _register_request(orchestrator, "req-elsewhere", submitted_stage=None)
        assert pool.select_replica_id("req-bound") == 0

        await orchestrator._handle_dead_replica(0, 0, EngineDeadError("last replica dead"))

        assert pool.live_replica_ids() == []
        failed = {queues[1].async_q.get_nowait().request_id for _ in range(2)}
        assert failed == {"req-bound", "req-unbound"}
        assert queues[1].async_q.empty()
        assert set(orchestrator.request_states) == {"req-elsewhere"}
    finally:
        for q in queues:
            q.close()


@pytest.mark.asyncio
async def test_fail_request_dead_stage_cleans_registered_request() -> None:
    """Direct helper test: a registered request failed against a dead stage gets
    a fatal ErrorMessage, its state and bindings are released, and its
    in-flight work on live replicas is aborted.
    """
    r0 = FakeStageClient(stage_type="llm")
    orchestrator, queues = _build_bare_orchestrator(_build_stage_pools([[r0]]))
    try:
        pool = orchestrator.stage_pools[0]
        _register_request(orchestrator, "req-a")
        assert pool.select_replica_id("req-a") == 0

        await orchestrator._fail_request_dead_stage("req-a", 0)

        msg = queues[1].async_q.get_nowait()
        assert isinstance(msg, ErrorMessage)
        assert msg.error == "Stage-0 has no live replica"
        assert msg.fatal is True
        assert msg.request_id == "req-a"
        assert msg.stage_id == 0
        assert "req-a" not in orchestrator.request_states
        assert pool.get_bound_replica_id("req-a") is None
        # abort=True: the failed request's work on the still-live replica is
        # stopped, not left to run to completion.
        assert r0.abort_calls == [["req-a"]]
    finally:
        for q in queues:
            q.close()


@pytest.mark.asyncio
async def test_streaming_update_racing_eviction_fails_request_not_server() -> None:
    """A streaming_update can find its request state still present while the
    stage has already lost all replicas (_handle_dead_replica awaits between
    eviction and cleanup). Submitting then raises from replica selection; the
    handler must fail that request instead of letting the raise escape
    _request_handler and shut the server down (#4285).
    """
    orchestrator, queues = _build_bare_orchestrator(_build_stage_pools([[FakeStageClient(stage_type="llm")]]))
    try:
        pool = orchestrator.stage_pools[0]
        _register_request(orchestrator, "req-s")
        orchestrator.request_states["req-s"].sampling_params_list = [_sampling_params()]
        assert pool.select_replica_id("req-s") == 0
        pool.evict_replica(0)

        msg = StageSubmissionMessage(
            type="streaming_update",
            request_id="req-s",
            prompt=SimpleNamespace(request_id="req-s", prompt_token_ids=[1]),
            original_prompt={"prompt": "hi"},
            output_prompt_text=None,
            sampling_params_list=[],
            final_stage_id=0,
            preprocess_ms=0.0,
            request_timestamp=time.time(),
            enqueue_ts=time.time(),
        )
        await orchestrator._handle_streaming_update(msg)

        err = queues[1].async_q.get_nowait()
        assert isinstance(err, ErrorMessage)
        assert err.fatal is True
        assert err.request_id == "req-s"
        assert "req-s" not in orchestrator.request_states
    finally:
        for q in queues:
            q.close()


@pytest.mark.asyncio
async def test_streaming_update_unrelated_error_is_not_masked() -> None:
    """The dispatch guard converts only StageUnavailableError/EngineDeadError;
    a genuine bug surfacing as a plain RuntimeError must propagate (fail
    fast), not be misreported as a dead-stage request failure.
    """

    class _BuggyClient(FakeStageClient):
        async def add_request_async(self, *args, **kwargs):
            raise RuntimeError("boom: unrelated bug")

    orchestrator, queues = _build_bare_orchestrator(
        _build_stage_pools([[_BuggyClient(stage_type="llm", final_output=True)]])
    )
    try:
        _register_request(orchestrator, "req-b")
        orchestrator.request_states["req-b"].sampling_params_list = [_sampling_params()]

        msg = StageSubmissionMessage(
            type="streaming_update",
            request_id="req-b",
            prompt=SimpleNamespace(request_id="req-b", prompt_token_ids=[1]),
            original_prompt={"prompt": "hi"},
            output_prompt_text=None,
            sampling_params_list=[],
            final_stage_id=0,
            preprocess_ms=0.0,
            request_timestamp=time.time(),
            enqueue_ts=time.time(),
        )
        with pytest.raises(RuntimeError, match="unrelated bug"):
            await orchestrator._handle_streaming_update(msg)
    finally:
        for q in queues:
            q.close()


@pytest.mark.asyncio
async def test_add_companion_racing_eviction_fails_parent_not_server() -> None:
    """A CFG companion submission hitting a fully dead stage must fail the
    parent request (whose cleanup also releases the companion), not raise out
    of _request_handler (#4285).
    """
    orchestrator, queues = _build_bare_orchestrator(_build_stage_pools([[FakeStageClient(stage_type="llm")]]))
    try:
        pool = orchestrator.stage_pools[0]
        _register_request(orchestrator, "req-p")
        orchestrator.request_states["req-p"].sampling_params_list = [_sampling_params()]
        pool.evict_replica(0)

        msg = AddCompanionRequestMessage(
            companion_id="req-p-cfg",
            parent_id="req-p",
            role="negative",
            prompt=SimpleNamespace(request_id="req-p-cfg", prompt_token_ids=[2]),
            companion_prompt_text=None,
            sampling_params_list=[_sampling_params()],
        )
        await orchestrator._handle_add_companion(msg)

        err = queues[1].async_q.get_nowait()
        assert isinstance(err, ErrorMessage)
        assert err.fatal is True
        assert err.request_id == "req-p"
        assert "req-p" not in orchestrator.request_states
        assert "req-p-cfg" not in orchestrator.request_states
    finally:
        for q in queues:
            q.close()


@pytest.mark.asyncio
async def test_fail_request_dead_stage_is_noop_safe_for_unregistered_request() -> None:
    """Direct helper test: the stage-0 dispatch guard calls the helper before the
    request is registered; the cleanup must be a safe no-op and still emit the
    fatal ErrorMessage.
    """
    orchestrator, queues = _build_bare_orchestrator(_build_stage_pools([[FakeStageClient(stage_type="llm")]]))
    try:
        await orchestrator._fail_request_dead_stage("req-unknown", 0)

        msg = queues[1].async_q.get_nowait()
        assert isinstance(msg, ErrorMessage)
        assert msg.error == "Stage-0 has no live replica"
        assert msg.fatal is True
        assert msg.request_id == "req-unknown"
        assert orchestrator.request_states == {}
    finally:
        for q in queues:
            q.close()


@pytest.mark.asyncio
async def test_dispatch_death_evicts_so_subsequent_requests_reroute_to_survivor() -> None:
    """A replica dying mid-submit (EngineDeadError from dispatch, before the poll
    loop evicts it) is evicted by the guard so subsequent requests reroute to the
    survivor instead of round-robining back onto the dead slot until the poll
    catches up. The survivor and its request are untouched (#4285).
    """
    r0 = FakeStageClient(stage_type="llm")
    r1 = FakeStageClient(stage_type="llm")
    orchestrator, queues = _build_bare_orchestrator(_build_stage_pools([[r0, r1]]))
    try:
        pool = orchestrator.stage_pools[0]
        _register_request(orchestrator, "req-dead")
        _register_request(orchestrator, "req-live")
        assert pool.select_replica_id("req-dead") == 0
        assert pool.select_replica_id("req-live") == 1

        async def _die() -> None:
            raise EngineDeadError("stage-0 replica-0 died on submit")

        failed = await orchestrator._dispatch_or_fail_request(
            _die, req_id="req-dead", stage_id=0, operation="add_request"
        )
        assert failed is False

        # The dead replica is evicted; only the request on it fails, the survivor
        # and its request are untouched.
        assert pool.live_replica_ids() == [1]
        assert r0.shutdown_calls == 1
        assert r1.shutdown_calls == 0
        msg = queues[1].async_q.get_nowait()
        assert isinstance(msg, ErrorMessage) and msg.fatal is True and msg.request_id == "req-dead"
        assert queues[1].async_q.empty()
        assert "req-dead" not in orchestrator.request_states
        assert "req-live" in orchestrator.request_states and pool.get_bound_replica_id("req-live") == 1

        # The point of eviction: a burst of subsequent requests all route to the
        # survivor and never land back on the evicted replica.
        for rid in ("next-1", "next-2", "next-3"):
            assert pool.select_replica_id(rid) == 1
    finally:
        for q in queues:
            q.close()


@pytest.mark.asyncio
async def test_dispatch_death_evicts_so_subsequent_requests_fast_fail() -> None:
    """When the replica dying on submit was the stage's only one, the guard evicts
    it (leaving the stage empty). A subsequent request then fast-fails at routing —
    the ``_handle_add_request`` entry guard turns ``StageUnavailableError`` into a
    fatal 5xx immediately, rather than dispatching onto the dead replica (#4285).
    """
    r0 = FakeStageClient(stage_type="llm")
    orchestrator, queues = _build_bare_orchestrator(_build_stage_pools([[r0]]))
    try:
        pool = orchestrator.stage_pools[0]
        _register_request(orchestrator, "req-a")
        assert pool.select_replica_id("req-a") == 0

        async def _die() -> None:
            raise EngineDeadError("stage-0 replica-0 died on submit")

        failed = await orchestrator._dispatch_or_fail_request(_die, req_id="req-a", stage_id=0, operation="add_request")
        assert failed is False

        assert pool.live_replica_ids() == []
        assert r0.shutdown_calls == 1
        msg = queues[1].async_q.get_nowait()
        assert isinstance(msg, ErrorMessage) and msg.request_id == "req-a"
        assert queues[1].async_q.empty()
        assert "req-a" not in orchestrator.request_states

        # A subsequent request cannot be routed at all: the stage has no live
        # replica, so selection raises fast instead of dispatching onto the dead
        # slot. _handle_add_request converts this to a fatal 5xx at its entry guard.
        with pytest.raises(StageUnavailableError):
            pool.select_replica_id("req-next")
    finally:
        for q in queues:
            q.close()


# ───────── Diffusion stage error output routing ─────────


@pytest.mark.asyncio
async def test_diffusion_error_output_routed_as_finished(orchestrator_factory) -> None:
    """When a diffusion stage returns an OmniRequestOutput with a non-None
    error, the orchestrator must route it as an error message and clean up
    the request state.
    """
    stage0 = FakeStageClient(stage_type="diffusion", final_output=True, final_output_type="image")
    orchestrator_fixture = orchestrator_factory([stage0])
    params = OmniDiffusionSamplingParams()

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-err",
            prompt={"prompt": "draw a cat"},
            original_prompt={"prompt": "draw a cat"},
            sampling_params_list=[params],
            final_stage_id=0,
        )

        await _wait_for(lambda: len(stage0.add_request_calls) == 1)

        # Push an error output from the diffusion stage.
        stage0.push_diffusion_output(OmniRequestOutput.from_error("req-err", "gpu fault"))

        msg = await _get_any_output_message(orchestrator_fixture)

        assert isinstance(msg, ErrorMessage)
        assert msg.type == "error"
        assert msg.request_id == "req-err"
        assert msg.stage_id == 0
        assert msg.error == "gpu fault"

        # Request state should be cleaned up.
        await _wait_for(lambda: "req-err" not in orchestrator_fixture.orchestrator.request_states)
    finally:
        orchestrator_fixture.request_sync_q.put_nowait(ShutdownRequestMessage())
        orchestrator_fixture.thread.join(timeout=5)


@pytest.mark.asyncio
async def test_diffusion_client_error_output_propagates_status_code(orchestrator_factory) -> None:
    """A client-error OmniRequestOutput (e.g. a guardrail 4xx) from a diffusion
    stage must surface as an ErrorMessage that carries status_code/error_type,
    so the frontend can map it to the correct HTTP response instead of a 500.
    """
    stage0 = FakeStageClient(stage_type="diffusion", final_output=True, final_output_type="image")
    orchestrator_fixture = orchestrator_factory([stage0])
    params = OmniDiffusionSamplingParams()

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-blocked",
            prompt={"prompt": "blocked"},
            original_prompt={"prompt": "blocked"},
            sampling_params_list=[params],
            final_stage_id=0,
        )

        await _wait_for(lambda: len(stage0.add_request_calls) == 1)

        # Push a client-error output from the diffusion stage.
        stage0.push_diffusion_output(
            OmniRequestOutput.from_error(
                "req-blocked",
                "Input was blocked by Cosmos3 guardrails.",
                status_code=400,
                error_type="BadRequestError",
            )
        )

        msg = await _get_any_output_message(orchestrator_fixture)

        assert isinstance(msg, ErrorMessage)
        assert msg.request_id == "req-blocked"
        assert msg.stage_id == 0
        assert msg.fatal is False
        assert msg.status_code == 400
        assert msg.error_type == "BadRequestError"
        assert msg.error == "Input was blocked by Cosmos3 guardrails."

        # Request state should be cleaned up.
        await _wait_for(lambda: "req-blocked" not in orchestrator_fixture.orchestrator.request_states)
    finally:
        orchestrator_fixture.request_sync_q.put_nowait(ShutdownRequestMessage())
        orchestrator_fixture.thread.join(timeout=5)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [
        pytest.param(429, "RateLimitError", id="429-too-many-requests"),
        pytest.param(413, "PayloadTooLargeError", id="413-payload-too-large"),
        pytest.param(422, "UnprocessableEntityError", id="422-unprocessable-entity"),
        pytest.param(403, "PermissionDeniedError", id="403-forbidden"),
    ],
)
async def test_diffusion_client_error_output_propagates_non_400_status(
    orchestrator_factory,
    status_code: int,
    error_type: str,
) -> None:
    """A diffusion-stage client error with a non-400 4xx status must be serialized
    into an ErrorMessage that preserves that exact status_code/error_type.
    """
    stage0 = FakeStageClient(stage_type="diffusion", final_output=True, final_output_type="image")
    orchestrator_fixture = orchestrator_factory([stage0])
    params = OmniDiffusionSamplingParams()

    try:
        await _enqueue_add_request(
            orchestrator_fixture,
            request_id="req-blocked",
            prompt={"prompt": "blocked"},
            original_prompt={"prompt": "blocked"},
            sampling_params_list=[params],
            final_stage_id=0,
        )

        await _wait_for(lambda: len(stage0.add_request_calls) == 1)

        # Push a non-400 client-error output from the diffusion stage.
        stage0.push_diffusion_output(
            OmniRequestOutput.from_error(
                "req-blocked",
                "client error from diffusion stage",
                status_code=status_code,
                error_type=error_type,
            )
        )

        msg = await _get_any_output_message(orchestrator_fixture)

        assert isinstance(msg, ErrorMessage)
        assert msg.request_id == "req-blocked"
        assert msg.stage_id == 0
        assert msg.fatal is False
        assert msg.status_code == status_code
        assert msg.error_type == error_type
        assert msg.error == "client error from diffusion stage"

        # Request state should be cleaned up.
        await _wait_for(lambda: "req-blocked" not in orchestrator_fixture.orchestrator.request_states)
    finally:
        orchestrator_fixture.request_sync_q.put_nowait(ShutdownRequestMessage())
        orchestrator_fixture.thread.join(timeout=5)
