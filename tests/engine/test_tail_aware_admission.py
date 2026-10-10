# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exercise central admission through the real orchestrator and pool submit path."""

import asyncio
from types import SimpleNamespace

import pytest
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.engine.messages import StageSubmissionMessage
from vllm_omni.engine.orchestrator import Orchestrator
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.cpu, pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.asyncio]


class Backend:
    stage_type = "diffusion"
    final_output = True

    def __init__(self):
        self.started = asyncio.Queue()
        self.finish_submit = asyncio.Event()
        self.error = None
        self.submitted = []

    async def add_request_async(self, request_id, *args, **kwargs):
        self.submitted.append(request_id)
        self.started.put_nowait(request_id)
        await self.finish_submit.wait()
        if self.error is not None:
            raise self.error

    async def abort_requests_async(self, request_ids):
        await asyncio.sleep(0)

    def shutdown(self):
        pass


class Counter:
    value = 0

    def increment(self):
        self.value += 1

    def decrement(self):
        self.value -= 1


def _message(request_id):
    return StageSubmissionMessage(
        type="add_request",
        request_id=request_id,
        prompt={"prompt": "test"},
        original_prompt={"prompt": "test"},
        output_prompt_text=None,
        sampling_params_list=[OmniDiffusionSamplingParams()],
        final_stage_id=0,
        preprocess_ms=0.0,
        request_timestamp=0.0,
        enqueue_ts=0.0,
    )


@pytest.fixture
async def runtime():
    instances = []

    def create(replicas=1, limit=1):
        clients = [Backend() for _ in range(replicas)]
        pool = StagePool(0, clients)
        pool.configure_tail_aware_scheduling(
            {"enabled": True, "max_pending_requests": limit, "hardware_profile": "910B2"},
            model_class_name="QwenImagePipeline",
        )
        stage_waiting = {}
        orch = Orchestrator(
            request_async_queue=asyncio.Queue(),
            output_async_queue=asyncio.Queue(),
            rpc_async_queue=asyncio.Queue(),
            stage_pools=[pool],
            running_counter=Counter(),
            engines_waiting_counter=Counter(),
            prom_metrics=SimpleNamespace(set_stage_waiting_requests=stage_waiting.__setitem__),
        )
        instances.append(orch)
        return orch, pool, clients, stage_waiting

    yield create
    for orch in instances:
        await orch._cleanup_request_ids(list(orch.request_states), abort=True)
        for pool in orch.stage_pools:
            pool.close_tail_aware_scheduling()
        assert not orch._tail_aware_admission_tasks
        assert orch._engines_waiting_counter.value == 0


async def test_admission_capacity_waiting_metrics_and_cancellation(runtime, monkeypatch):
    orch, pool, (backend,), stage_waiting = runtime()
    await orch._handle_add_request(_message("A"))
    assert await asyncio.wait_for(backend.started.get(), 1) == "A"
    first = orch._tail_aware_admission_tasks["A"]
    assert not first.done() and pool._tail_aware_controller.pending_count == 0

    # Patch module clock references without changing asyncio's event-loop clock.
    now = [10.0]
    clock = SimpleNamespace(perf_counter=lambda: now[0], time=lambda: 1000.0 + now[0])
    monkeypatch.setattr("vllm_omni.engine.orchestrator._time", clock)
    monkeypatch.setattr("vllm_omni.engine.stage_pool._time", clock)
    msg = _message("B")
    msg.enqueue_ts = 9.999
    await orch._handle_add_request(msg)
    now[0] += 0.02  # Waiting before the admission task starts also counts.
    await asyncio.sleep(0)
    assert orch.output_async_queue.empty()
    assert pool._tail_aware_controller.pending_count == 1
    assert orch._engines_waiting_counter.value == stage_waiting[0] == 1
    assert orch._running_counter.value - orch._engines_waiting_counter.value == 1
    second = orch._tail_aware_admission_tasks["B"]
    await orch._handle_add_request(_message("overflow"))
    error = await asyncio.wait_for(orch.output_async_queue.get(), 1)
    assert (error.request_id, error.status_code) == ("overflow", 429)

    # Replica snapshots add to central waiting instead of overwriting it.
    orch._update_stage_replica_waiting(0, 0, 2)
    assert orch._engines_waiting_counter.value == stage_waiting[0] == 3
    orch._update_stage_replica_waiting(0, 0, 0)
    backend.finish_submit.set()
    await asyncio.wait_for(first, 1)
    assert backend.submitted == ["A"]
    # The controller still enforces B's quota after A's submit task is gone.
    await orch._handle_add_request(_message("still-full"))
    error = await asyncio.wait_for(orch.output_async_queue.get(), 1)
    assert (error.request_id, error.status_code) == ("still-full", 429)
    now[0] += 0.05
    pool._tail_aware_controller.complete("A")
    await asyncio.wait_for(second, 1)
    assert backend.submitted == ["A", "B"]
    assert orch.request_states["B"].pipeline_timings["queue_wait_ms"] == pytest.approx(71.0)
    assert orch.request_states["B"].stage_submit_ts[0] == pytest.approx(1010.07)
    assert orch._engines_waiting_counter.value == stage_waiting[0] == 0

    # Cancelling the next waiter releases both its quota and its metric count.
    await orch._handle_add_request(_message("C"))
    await asyncio.sleep(0)
    assert orch._engines_waiting_counter.value == 1
    await orch._cleanup_request_ids(["C"], abort=True)
    assert pool._tail_aware_controller.pending_count == 0
    assert orch._engines_waiting_counter.value == stage_waiting[0] == 0
    assert backend.submitted == ["A", "B"]


async def test_same_turn_burst_uses_replicas_and_one_waiting_slot(runtime):
    orch, pool, clients, _ = runtime(replicas=2)
    for request_id in ("A", "B", "C", "overflow"):
        await orch._handle_add_request(_message(request_id))
    tasks = list(orch._tail_aware_admission_tasks.values())
    assert len(tasks) == 3
    assert await asyncio.wait_for(clients[0].started.get(), 1) == "A"
    assert await asyncio.wait_for(clients[1].started.get(), 1) == "B"
    error = await asyncio.wait_for(orch.output_async_queue.get(), 1)
    assert (error.request_id, error.status_code) == ("overflow", 429)
    assert orch.output_async_queue.empty()
    assert pool._tail_aware_controller.pending_count == 1
    assert pool.get_bound_replica_id("C") is None
    for client in clients:
        client.finish_submit.set()
    pool._tail_aware_controller.complete("A")
    await asyncio.wait_for(asyncio.gather(*tasks), 1)
    assert clients[0].submitted == ["A", "C"]
    assert clients[1].submitted == ["B"]


async def test_dead_submit_cannot_admit_waiter_to_failed_replica(runtime, monkeypatch):
    orch, pool, clients, _ = runtime(replicas=2, limit=3)
    for request_id, client in zip(("A", "C"), clients):
        await orch._handle_add_request(_message(request_id))
        assert await asyncio.wait_for(client.started.get(), 1) == request_id
    await orch._handle_add_request(_message("B"))
    await asyncio.sleep(0)
    tasks = list(orch._tail_aware_admission_tasks.values())
    failure_seen, allow_cleanup = asyncio.Event(), asyncio.Event()
    original = orch._fail_request_dead_stage

    async def delayed_cleanup(request_id, stage_id):
        failure_seen.set()
        await allow_cleanup.wait()
        await original(request_id, stage_id)

    monkeypatch.setattr(orch, "_fail_request_dead_stage", delayed_cleanup)
    clients[0].error = EngineDeadError()
    clients[0].finish_submit.set()
    try:
        await asyncio.wait_for(failure_seen.wait(), 1)
        await asyncio.sleep(0)
        assert pool.available_replica_ids() == [1]
        assert pool._tail_aware_controller.pending_count == 1
        assert pool.get_bound_replica_id("B") is None
        clients[1].finish_submit.set()
        pool._tail_aware_controller.complete("C")
        assert await asyncio.wait_for(clients[1].started.get(), 1) == "B"
    finally:
        allow_cleanup.set()
    await asyncio.wait_for(asyncio.gather(*tasks), 1)
    assert clients[0].submitted == ["A"]
    assert pool.get_bound_replica_id("B") == 1
    error = orch.output_async_queue.get_nowait()
    assert error.request_id == "A"
    assert orch.output_async_queue.empty()
