# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native generation admission must serve chunks fairly across live streams."""

from types import SimpleNamespace

import pytest
from vllm import SamplingParams
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.request import Request, RequestStatus

from tests.core.sched.test_generation_scheduler_restore import _scheduler_with_parked_generation_request

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _request(name: str, *, in_flight: int = 0, computed: int = 0) -> Request:
    request = Request(name, [1, 2], SamplingParams(max_tokens=4), pooling_params=None)
    request.num_in_flight_tokens = in_flight
    request.num_computed_tokens = computed
    request.external_req_id = name
    return request


@pytest.fixture
def scheduler(monkeypatch):
    scheduler, _ = _scheduler_with_parked_generation_request(monkeypatch, use_v2_model_runner=True)
    monkeypatch.setattr("vllm_omni.core.sched.omni_generation_scheduler.create_request_queue", create_request_queue)
    scheduler._native_data_plane = True
    scheduler._native_chunk_started = set()
    scheduler.chunk_transfer_adapter = None
    scheduler.input_coordinator = SimpleNamespace(finished_requests=set())
    scheduler.policy = SchedulingPolicy.FCFS
    scheduler.running = []
    scheduler.waiting = create_request_queue(scheduler.policy)
    scheduler.skipped_waiting = create_request_queue(scheduler.policy)
    scheduler.requests = {}
    scheduler._process_pending_omni_inputs = lambda model_mode: None
    scheduler._postprocess_omni_schedule_output = lambda output: None
    scheduler._restore_omni_wait_queues = lambda: None

    def advance(output):
        for rid, count in output.num_scheduled_tokens.items():
            req = scheduler.requests[rid]
            req.status = RequestStatus.RUNNING
            req.num_computed_tokens += count
            req.num_in_flight_tokens += count

    scheduler._update_after_schedule = advance
    return scheduler


def test_ready_first_chunk_runs_while_another_stream_is_in_flight(scheduler):
    running = _request("running", in_flight=2)
    running.status = RequestStatus.RUNNING
    first = _request("first")
    scheduler.running = [running]
    scheduler.waiting.add_request(first)
    scheduler.requests = {r.request_id: r for r in (running, first)}

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {"first": 2}
    assert running.num_in_flight_tokens == 2
    assert len(output.scheduled_new_reqs) == scheduler.max_num_running_reqs == 1


def test_completed_chunk_yields_to_waiting_stream_then_makes_progress(scheduler):
    continuation = _request("continuation")
    continuation.status = RequestStatus.RUNNING
    scheduler._native_chunk_started.add("continuation")
    first = _request("first")
    scheduler.running = [continuation]
    scheduler.waiting.add_request(first)
    scheduler.requests = {r.request_id: r for r in (continuation, first)}

    first_output = scheduler.schedule()
    next_output = scheduler.schedule()

    assert first_output.num_scheduled_tokens == {"first": 2}
    assert next_output.num_scheduled_tokens == {"continuation": 2}
    assert {r.request_id for r in scheduler.running} == {"first", "continuation"}


def test_waiting_in_flight_entry_does_not_block_or_repeat(scheduler):
    pending = _request("pending", in_flight=2)
    ready = _request("ready")
    scheduler.requests = {r.request_id: r for r in (pending, ready)}
    scheduler.waiting.add_request(pending)
    scheduler.waiting.add_request(ready)

    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {"ready": 2}
    assert list(scheduler.waiting) == [pending]


@pytest.mark.parametrize("terminal", [False, True])
def test_completed_payload_is_not_executed_again_without_new_chunk(scheduler, terminal):
    completed = _request("completed", computed=2)
    completed.status = RequestStatus.RUNNING
    scheduler.running = [completed]
    scheduler.requests = {"completed": completed}
    if terminal:
        scheduler.input_coordinator.finished_requests.add("completed")

    output = scheduler.schedule()

    assert not output.num_scheduled_tokens
    assert scheduler._pending_finish_reqs == ([completed] if terminal else [])


def test_chunk_wait_status_and_request_state_survive_requeue(scheduler):
    completed = _request("completed", computed=2)
    completed.status = RequestStatus.WAITING_FOR_CHUNK
    scheduler.running = [completed]
    scheduler._requeue_completed_native_chunks()

    assert completed.status == RequestStatus.WAITING_FOR_CHUNK
    assert completed.num_computed_tokens == 2
    assert list(scheduler.waiting) == [completed]
    assert not scheduler.running


def test_legacy_generation_keeps_lifetime_admission(scheduler):
    scheduler._native_data_plane = False
    request = _request("legacy")
    request.status = RequestStatus.RUNNING
    scheduler.running = [request]

    scheduler._requeue_completed_native_chunks()

    assert scheduler.running == [request]
    assert not scheduler.waiting


def test_first_chunks_share_a_batch_with_waiting_continuations(scheduler):
    scheduler.max_num_running_reqs = 4
    scheduler.max_num_scheduled_tokens = 32
    continued = [_request(f"continued-{i}") for i in range(4)]
    fresh = [_request(f"fresh-{i}") for i in range(4)]
    scheduler._native_chunk_started.update(r.request_id for r in continued)
    for request in [*continued, *fresh]:
        scheduler.requests[request.request_id] = request
        scheduler.waiting.add_request(request)

    output = scheduler.schedule()

    assert list(output.num_scheduled_tokens) == ["fresh-0", "fresh-1", "continued-0", "continued-1"]


@pytest.mark.parametrize("first_count", [0, 1, 4])
def test_unused_first_chunk_slots_are_borrowed(scheduler, first_count):
    scheduler.max_num_running_reqs = 4
    scheduler.max_num_scheduled_tokens = 32
    requests = [_request(f"req-{i}") for i in range(4)]
    scheduler._native_chunk_started.update(r.request_id for r in requests[first_count:])
    for request in requests:
        scheduler.requests[request.request_id] = request
        scheduler.waiting.add_request(request)

    output = scheduler.schedule()

    assert len(output.num_scheduled_tokens) == 4


def test_single_slot_alternates_first_chunks_and_continuations(scheduler):
    continuation = _request("continuation")
    first = _request("first")
    another = _request("another")
    scheduler._native_chunk_started.add("continuation")
    scheduler.requests = {r.request_id: r for r in (continuation, first, another)}
    for request in (continuation, first, another):
        scheduler.waiting.add_request(request)

    output = scheduler.schedule()
    next_output = scheduler.schedule()

    assert list(output.num_scheduled_tokens) == ["first"]
    assert list(next_output.num_scheduled_tokens) == ["continuation"]
    assert list(scheduler.waiting) == [another]


def test_first_chunk_reservation_preserves_explicit_priority_policy(scheduler):
    scheduler.policy = SchedulingPolicy.PRIORITY
    scheduler.waiting = create_request_queue(scheduler.policy)
    high_priority = _request("continuation")
    high_priority.priority = -1
    first = _request("first")
    scheduler._native_chunk_started.add(high_priority.request_id)
    scheduler.waiting.add_request(first)
    scheduler.waiting.add_request(high_priority)
    scheduler.requests = {r.request_id: r for r in (first, high_priority)}

    output = scheduler.schedule()

    assert list(output.num_scheduled_tokens) == ["continuation"]


def test_request_cleanup_releases_first_chunk_admission_state(scheduler, monkeypatch):
    from vllm.v1.core.sched.scheduler import Scheduler

    request = _request("reused")
    scheduler._native_chunk_started.add(request.request_id)
    monkeypatch.setattr(Scheduler, "_free_request", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(scheduler, "_free_input_coordinator_request", lambda req_id: None)

    scheduler._free_request(request)

    assert request.request_id not in scheduler._native_chunk_started


@pytest.mark.parametrize("limit", [1, 2, 4])
def test_execution_budget_is_independent_of_request_capacity(scheduler, limit):
    scheduler.max_num_running_reqs = 10
    scheduler.max_num_scheduled_tokens = 100
    scheduler._generation_execution_batch_size = limit
    continued = [_request(f"continued-{i}") for i in range(8)]
    fresh = [_request(f"fresh-{i}") for i in range(8)]
    scheduler._native_chunk_started.update(r.request_id for r in continued)
    for req in [*continued, *fresh]:
        scheduler.requests[req.request_id] = req
        scheduler.waiting.add_request(req)
    first = scheduler.schedule()
    second = scheduler.schedule()
    assert len(first.num_scheduled_tokens) == limit
    assert len(second.num_scheduled_tokens) == limit
    assert not (first.num_scheduled_tokens.keys() & second.num_scheduled_tokens.keys())
    assert scheduler.max_num_running_reqs == 10
    if limit == 1:
        assert list(first.num_scheduled_tokens) == ["fresh-0"]
        assert list(second.num_scheduled_tokens) == ["continued-0"]
    else:
        assert sum(r.startswith("fresh") for r in first.num_scheduled_tokens) == limit // 2
        assert sum(r.startswith("continued") for r in first.num_scheduled_tokens) == limit - limit // 2


@pytest.mark.parametrize("native,retained", [(False, False), (True, True)])
def test_execution_budget_leaves_other_admission_contracts_unchanged(scheduler, native, retained):
    scheduler.max_num_running_reqs = 10
    scheduler._generation_execution_batch_size = 2
    scheduler._native_data_plane = native
    scheduler._retains_state_across_chunks = retained
    assert scheduler._execution_batch_limit() == 10


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "4"])
def test_execution_budget_rejects_invalid_values(value):
    from vllm_omni.core.sched.omni_generation_scheduler import _resolve_generation_execution_batch_size

    config = SimpleNamespace(stage_connector_config={"extra": {"generation_execution_batch_size": value}})
    with pytest.raises(ValueError, match="positive integer"):
        _resolve_generation_execution_batch_size(config, 10)


@pytest.mark.parametrize("value,expected", [(None, 10), (4, 4), (20, 10)])
def test_execution_budget_respects_worker_capacity(value, expected):
    from vllm_omni.core.sched.omni_generation_scheduler import _resolve_generation_execution_batch_size

    config = SimpleNamespace(stage_connector_config={"extra": {"generation_execution_batch_size": value}})
    assert _resolve_generation_execution_batch_size(config, 10) == expected
