# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Local acceptance validates after asynchronous preparation and routing."""

import asyncio
import threading
from types import SimpleNamespace

import pytest
from opentelemetry.trace import get_tracer
from vllm.sampling_params import SamplingParams

from tests.engine.duplex.test_memory_card_retrieval import _card
from tests.engine.test_duplex_orchestrator import SESSION_ID, _build, _open, _stage0_request_id
from tests.engine.test_orchestrator import FakeRunningCounter, FakeStageClient, _build_stage_pools, _processed_request
from vllm_omni.engine.duplex.contracts import DuplexStageRequestContext, DuplexStageSubmission
from vllm_omni.engine.orchestrator import OrchestratorRequestState
from vllm_omni.tracing import RequestTrace

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.asyncio]


class _Transport(FakeStageClient):
    def __init__(self):
        super().__init__()
        self.kwargs: list[dict[str, object]] = []
        self.enqueued = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()

    async def add_request_async(self, *args, **kwargs):
        self.kwargs.append(kwargs)
        await super().add_request_async(*args, **kwargs)
        self.enqueued.set()
        await self.release.wait()


@pytest.mark.parametrize("valid", [False, True])
async def test_admission_guard_precedes_native_stage_trace(valid):
    client = _Transport()
    pool = _build_stage_pools([[client]])[0]
    request = _processed_request("traced-guard", SamplingParams(max_tokens=1))
    trace = RequestTrace(get_tracer(__name__), request.request_id, None)
    state = OrchestratorRequestState(
        request_id=request.request_id, sampling_params_list=[request.sampling_params], trace=trace
    )

    def guard(materialized):
        assert materialized is request
        assert not trace.stages
        if not valid:
            raise ValueError("stale_selection")

    try:
        if valid:
            assert await pool.submit_initial(request.request_id, state, request, before_accept=guard) == 0
            assert set(trace.stages) == {0}
        else:
            with pytest.raises(ValueError, match="stale_selection"):
                await pool.submit_initial(request.request_id, state, request, before_accept=guard)
            assert not trace.stages
            assert not pool.output_processor.add_request_calls
            assert not client.add_request_calls
    finally:
        trace.end("success" if valid else "cancelled")


@pytest.mark.parametrize("update", [False, True])
@pytest.mark.parametrize("valid", [False, True])
async def test_guard_runs_after_routing_and_before_output_registration_or_send(monkeypatch, update, valid):
    client = _Transport()
    pool = _build_stage_pools([[client]])[0]
    request = _processed_request("guard", SamplingParams(max_tokens=1))
    state = OrchestratorRequestState(request_id=request.request_id, sampling_params_list=[request.sampling_params])
    started, release = asyncio.Event(), asyncio.Event()
    eligibility = {"current": True}
    calls: list[object] = []

    async def select(*args, **kwargs):
        started.set()
        await release.wait()
        return 0

    def guard(materialized):
        calls.append(materialized)
        assert not pool.output_processor.add_request_calls
        assert not client.add_request_calls
        if not eligibility["current"]:
            raise ValueError("stale_selection")

    monkeypatch.setattr(pool, "_pick_or_select", select)
    submit = pool.submit_update if update else pool.submit_initial
    task = asyncio.create_task(submit("guard", state, request, before_accept=guard))
    await asyncio.wait_for(started.wait(), 2)
    assert not calls
    eligibility["current"] = valid
    release.set()
    if valid:
        assert await task == 0
        assert len(pool.output_processor.add_request_calls) == len(client.add_request_calls) == 1
        assert client.kwargs == [{}]
        assert not hasattr(request, "before_accept")
    else:
        with pytest.raises(ValueError, match="stale_selection"):
            await task
        assert not pool.output_processor.add_request_calls and not client.add_request_calls
    assert calls == [request]


async def test_changes_after_local_acceptance_use_existing_transport_lifecycle():
    client = _Transport()
    client.release.clear()
    pool = _build_stage_pools([[client]])[0]
    request = _processed_request("accepted", SamplingParams(max_tokens=1))
    state = OrchestratorRequestState(request_id=request.request_id, sampling_params_list=[request.sampling_params])
    current = {"valid": True}
    accepted: list[str] = []

    def guard(request):
        assert current["valid"]
        accepted.append(request.request_id)

    task = asyncio.create_task(pool.submit_initial("accepted", state, request, before_accept=guard))
    await asyncio.wait_for(client.enqueued.wait(), 2)
    current["valid"] = False
    client.release.set()
    assert await task == 0
    assert accepted == ["accepted"] and len(client.add_request_calls) == 1


async def test_async_guard_is_rejected_without_running_or_leaking_a_coroutine():
    client = _Transport()
    pool = _build_stage_pools([[client]])[0]
    request = _processed_request("async-guard", SamplingParams(max_tokens=1))
    state = OrchestratorRequestState(request_id=request.request_id, sampling_params_list=[request.sampling_params])

    async def guard(request):
        pytest.fail("an async admission guard ran")

    with pytest.raises(TypeError, match="synchronous"):
        await pool.submit_initial("async-guard", state, request, before_accept=guard)
    assert not client.add_request_calls and not pool.output_processor.add_request_calls


async def test_orchestrator_validates_after_actual_input_preprocessing(monkeypatch):
    orchestrator, clients, rpc, _ = _build()
    entered, release = threading.Event(), threading.Event()
    checked: list[tuple[int, ...]] = []

    def process_inputs(**kwargs):
        entered.set()
        assert release.wait(2)
        return _processed_request(kwargs["request_id"], kwargs["params"])

    monkeypatch.setattr(
        orchestrator, "_get_stage_input_processor", lambda stage: SimpleNamespace(process_inputs=process_inputs)
    )
    try:
        assert (await _open(orchestrator, rpc)).ok
        session = orchestrator.session_manager.get(SESSION_ID)
        node_id = _card(session, "city", "I live in Hangzhou.")
        receipt = session.capture_memory_prompt(query="Hangzhou city")
        context = DuplexStageRequestContext(
            request_id=_stage0_request_id(),
            session_id=SESSION_ID,
            fence=session.fence,
            stage_id=0,
            final_stage_id=0,
            config_generation=0,
            sampling_params=(clients[0].default_sampling_params,),
        )
        orchestrator.ensure_request(context)

        def guard(request):
            checked.append(tuple(request.prompt_token_ids))
            if not session.memory_prompt_matches(receipt, (node_id,)):
                raise ValueError("stale_selection")

        submission = DuplexStageSubmission(context, {"prompt": "historical card"}, False, resumable=False)
        task = asyncio.create_task(orchestrator.submit(submission, before_accept=guard))
        assert await asyncio.to_thread(entered.wait, 2)
        session.register_history_item("city", {"role": "user", "content": "I live in Shanghai."})
        release.set()
        with pytest.raises(ValueError, match="stale_selection"):
            await task
        assert checked == [(4, 5, 6)]
        assert not clients[0].add_request_calls
        assert not orchestrator.stage_pools[0].output_processor.add_request_calls
    finally:
        release.set()
        await orchestrator.session_manager.shutdown()


@pytest.mark.parametrize("replace_request", [False, True])
async def test_cancelled_or_replaced_request_cannot_be_accepted_after_routing_wait(monkeypatch, replace_request):
    counter = FakeRunningCounter()
    orchestrator, clients, rpc, _ = _build(stages=2, running_counter=counter)
    orchestrator.async_chunk = True
    started, release = asyncio.Event(), asyncio.Event()
    called: list[object] = []
    task = None
    try:
        assert (await _open(orchestrator, rpc)).ok
        session = orchestrator.session_manager.get(SESSION_ID)
        node_id = _card(session, "city", "I live in Hangzhou.")
        receipt = session.capture_memory_prompt(query="Hangzhou")
        context = DuplexStageRequestContext(
            request_id=_stage0_request_id(),
            session_id=SESSION_ID,
            fence=session.fence,
            stage_id=0,
            final_stage_id=1,
            config_generation=0,
            sampling_params=tuple(client.default_sampling_params for client in clients),
        )
        orchestrator.ensure_request(context)
        old_state = orchestrator.request_states[context.request_id]
        pool = orchestrator.stage_pools[0]

        async def select(*args, **kwargs):
            started.set()
            await release.wait()
            return 0

        def guard(request):
            called.append(request)
            assert session.memory_prompt_matches(receipt, (node_id,))

        monkeypatch.setattr(pool, "_pick_or_select", select)
        submission = DuplexStageSubmission(context, {"prompt_token_ids": [1, 2]}, False)
        task = asyncio.create_task(orchestrator.submit(submission, before_accept=guard))
        await asyncio.wait_for(started.wait(), 2)
        await orchestrator.cleanup([context.request_id], abort=False)
        assert context.request_id not in orchestrator.request_states
        assert session.memory_prompt_matches(receipt, (node_id,))
        replacement = None
        if replace_request:
            orchestrator.ensure_request(context)
            replacement = orchestrator.request_states[context.request_id]
            assert replacement is not old_state
            pool.bind_request_to_replica(context.request_id, 0)
        release.set()
        with pytest.raises(RuntimeError, match="cancelled before stage acceptance"):
            await task
        assert not called
        assert all(not client.add_request_calls for client in clients)
        assert all(not stage.output_processor.add_request_calls for stage in orchestrator.stage_pools)
        assert counter.value == 0 and not old_state.running_counter_registered
        if replacement is not None:
            assert orchestrator.request_states[context.request_id] is replacement
            assert pool.get_bound_replica_id(context.request_id) == 0
            assert not replacement.running_counter_registered
    finally:
        release.set()
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await orchestrator.session_manager.shutdown()


@pytest.mark.parametrize("wait_point", ["input_lock", "routing"])
@pytest.mark.parametrize("change", ["unchanged", "edit", "delete", "supersede", "close"])
async def test_owner_receipt_is_checked_after_wait_and_rejection_allows_explicit_clean_retry(
    monkeypatch, wait_point, change
):
    counter = FakeRunningCounter()
    orchestrator, clients, rpc, _ = _build(running_counter=counter)
    entered, release = asyncio.Event(), asyncio.Event()
    checked: list[tuple[int, ...]] = []
    task = None
    held_lock = None
    try:
        assert (await _open(orchestrator, rpc)).ok
        session = orchestrator.session_manager.get(SESSION_ID)
        node_id = _card(session, "city", "Hangzhou")
        receipt = session.capture_memory_prompt(query="Hangzhou")
        assert session.memory_prompt_matches(receipt, (node_id,))
        context = DuplexStageRequestContext(
            request_id=_stage0_request_id(),
            session_id=SESSION_ID,
            fence=session.fence,
            stage_id=0,
            final_stage_id=0,
            config_generation=0,
            sampling_params=(clients[0].default_sampling_params,),
        )
        orchestrator.ensure_request(context)
        original_state = orchestrator.request_states[context.request_id]
        pool = orchestrator.stage_pools[0]
        if wait_point == "input_lock":
            held_lock = orchestrator._stage_input_submission_locks.setdefault(0, asyncio.Lock())
            await held_lock.acquire()
        else:
            original_select = pool._pick_or_select

            async def select(*args, **kwargs):
                entered.set()
                await release.wait()
                return await original_select(*args, **kwargs)

            monkeypatch.setattr(pool, "_pick_or_select", select)

        def guard(request):
            checked.append(tuple(request.prompt_token_ids))
            if not session.memory_prompt_matches(receipt, (node_id,)):
                raise ValueError("stale_selection")

        async def submit():
            if wait_point == "input_lock":
                entered.set()
            return await orchestrator.submit(
                DuplexStageSubmission(context, {"prompt_token_ids": [11, 12]}, False, resumable=False),
                before_accept=guard,
            )

        task = asyncio.create_task(submit())
        await asyncio.wait_for(entered.wait(), 2)
        assert not task.done() and not checked
        assert not clients[0].add_request_calls and not pool.output_processor.add_request_calls
        if change == "edit":
            session.register_history_item("city", {"role": "user", "content": "Shanghai"})
        elif change == "delete":
            assert session.delete_history_item("city")
        elif change == "supersede":
            _card(session, "replacement", "Shanghai", generation=1)
        elif change == "close":
            session.begin_close(reason="test")
        if held_lock is not None:
            held_lock.release()
            held_lock = None
        release.set()
        if change == "unchanged":
            await task
            assert [call[0].prompt_token_ids for call in clients[0].add_request_calls] == [[11, 12]]
            assert len(pool.output_processor.add_request_calls) == 1
            assert counter.value == 1
        else:
            with pytest.raises(ValueError, match="stale_selection"):
                await task
            assert not clients[0].add_request_calls and not pool.output_processor.add_request_calls
            assert counter.value == 0 and not original_state.running_counter_registered
            assert not original_state.stage_submit_ts and not original_state.stage_fences
        assert checked == [(11, 12)]
        assert not orchestrator._stage_input_submission_locks[0].locked()
        await orchestrator.cleanup([context.request_id], abort=False)
        assert context.request_id not in orchestrator.request_states
        assert pool.get_bound_replica_id(context.request_id) is None
        assert counter.value == 0
        if change == "edit":
            # The consumer decides when to retry. This is an explicit caller
            # retry after cleanup, not an automatic fallback implementation.
            orchestrator.ensure_request(context)
            assert orchestrator.request_states[context.request_id] is not original_state
            await orchestrator.submit(
                DuplexStageSubmission(context, {"prompt_token_ids": [7, 8]}, False, resumable=False)
            )
            assert [call[0].prompt_token_ids for call in clients[0].add_request_calls] == [[7, 8]]
            assert len(pool.output_processor.add_request_calls) == 1
            assert counter.value == 1
            await orchestrator.cleanup([context.request_id], abort=False)
            assert counter.value == 0
    finally:
        if held_lock is not None:
            held_lock.release()
        release.set()
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await orchestrator.session_manager.shutdown()
