# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Scheduler-level admission tests for seeded-request isolation.

The pure decision-table tests in ``test_seeded_isolation.py`` do not exercise
upstream admission. These tests drive the real ``OmniARScheduler.schedule()``
wrapper with the vLLM admission condition copied from
``vllm/v1/core/sched/scheduler.py`` (0.33, L1062-1082):

    # Paused streaming sessions (WAITING_FOR_STREAMING_REQ) are not
    # in `running` but still hold a model-runner request slot.
    num_running = len(self.running) + self.num_waiting_for_streaming_input
    if num_running >= self.max_num_active_reqs:
        break

A paused streaming session therefore counts against the solo cap; if the cap is
``1`` the seeded request is never admitted while the unrelated session stays
parked (review P2 on #9937).
"""

from __future__ import annotations

import pytest
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.request import RequestStatus

import vllm_omni.core.sched.omni_ar_scheduler as ar_sched_mod
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Params:
    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed


class _StubRequest:
    """Request stub carrying the fields the scheduler and the queues read."""

    def __init__(
        self,
        request_id: str,
        seed: int | None = None,
        status: RequestStatus = RequestStatus.WAITING,
    ) -> None:
        self.request_id = request_id
        self.status = status
        self.priority = 0
        self.arrival_time = 0.0
        self.sampling_params = _Params(seed)

    def __lt__(self, other: _StubRequest) -> bool:
        return (self.priority, self.arrival_time) < (other.priority, other.arrival_time)


def _make_scheduler(
    *, parked: int, waiting, kv_holding=(), reserved_slots: int = 0, adapter_parked_running=()
) -> OmniARScheduler:
    """Real scheduler wrapper with the surface ``schedule()`` reads."""
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler.policy = SchedulingPolicy.FCFS
    scheduler.waiting = create_request_queue(SchedulingPolicy.FCFS)
    for request in waiting:
        scheduler.waiting.add_request(request)
    scheduler.kv_holding_waiting = create_request_queue(SchedulingPolicy.FCFS)
    for request in kv_holding:
        scheduler.kv_holding_waiting.add_request(request)
    scheduler.deferred_waiting = set()
    scheduler.running = []
    scheduler.requests = {request.request_id: request for request in (*waiting, *kv_holding)}
    scheduler.num_waiting_for_streaming_input = parked
    scheduler.max_num_active_reqs = 8
    scheduler.max_num_running_reqs = 8
    scheduler.input_coordinator = None
    from types import SimpleNamespace

    scheduler.chunk_transfer_adapter = SimpleNamespace(
        waiting_for_chunk_running_requests=list(adapter_parked_running),
        _held_non_active=[],
    )
    scheduler.use_v2_model_runner = reserved_slots > 0
    for name in (
        "_drop_aborted_queued_requests",
        "_process_pending_omni_inputs",
        "_resync_streaming_input_counter",
        "_drop_talker_drafts_if_prefill_pending",
        "_restore_omni_wait_queues",
    ):
        setattr(scheduler, name, lambda *args, **kwargs: None)
    scheduler._should_defer_waiting_admission = lambda: False
    scheduler._async_chunk_transport_enabled = lambda: reserved_slots > 0
    scheduler._get_async_chunk_reserved_running_slots = lambda: reserved_slots
    scheduler._enforce_kstep_span_uniformity = lambda output: None
    scheduler._postprocess_omni_schedule_output = lambda *args, **kwargs: None
    scheduler.get_finished_requests_needing_kv_transfer = lambda: {}
    scheduler._wrap_omni_scheduler_output = lambda output, **kwargs: output
    return scheduler


def _run_schedule(scheduler: OmniARScheduler, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Run ``OmniARScheduler.schedule`` with the upstream admission condition."""
    seen: dict = {"limit": None, "admitted": None, "waiting_len": None, "kv_len": None}

    def upstream_schedule(_self, _throttle_prefills: bool = False):
        seen["limit"] = scheduler.max_num_active_reqs
        seen["waiting_len"] = len(scheduler.waiting)
        seen["kv_len"] = len(scheduler.kv_holding_waiting)
        # The ``not len(scheduler.waiting)`` guard is a test-only divergence: the
        # real upstream breaks out of its admission loop when the queue is empty,
        # while this stub has to return before popping. Keep it when editing.
        num_running = len(scheduler.running) + scheduler.num_waiting_for_streaming_input
        if num_running >= scheduler.max_num_active_reqs or not len(scheduler.waiting):
            seen["admitted"] = None
            return seen
        request = scheduler.waiting.pop_request()
        scheduler.running.append(request)
        seen["admitted"] = request.request_id
        return seen

    monkeypatch.setattr(ar_sched_mod.VLLMScheduler, "schedule", upstream_schedule)
    OmniARScheduler.schedule(scheduler)
    return seen


def test_seeded_request_is_admitted_while_a_streaming_session_is_parked(monkeypatch) -> None:
    """Regression for review P2: a parked session must not consume the solo slot."""
    seeded = _StubRequest("seeded", seed=42)
    scheduler = _make_scheduler(parked=1, waiting=[seeded])

    seen = _run_schedule(scheduler, monkeypatch)

    assert seen["admitted"] == "seeded"
    assert [request.request_id for request in scheduler.running] == ["seeded"]
    assert seen["limit"] == 2  # one executing slot on top of one parked session


def test_solo_step_without_parked_sessions_keeps_the_single_slot(monkeypatch) -> None:
    seeded = _StubRequest("seeded", seed=42)
    scheduler = _make_scheduler(parked=0, waiting=[seeded])

    seen = _run_schedule(scheduler, monkeypatch)

    assert seen["admitted"] == "seeded"
    assert seen["limit"] == 1


def test_parked_slots_are_added_per_session(monkeypatch) -> None:
    seeded = _StubRequest("seeded", seed=0)  # seed 0 is an explicit seed
    scheduler = _make_scheduler(parked=3, waiting=[seeded])

    seen = _run_schedule(scheduler, monkeypatch)

    assert seen["admitted"] == "seeded"
    assert seen["limit"] == 4


def test_async_reservation_bounds_the_parked_aware_limit(monkeypatch) -> None:
    """The parked-aware solo budget is capped by the async-reserved capacity."""
    seeded = _StubRequest("seeded", seed=42)
    scheduler = _make_scheduler(parked=1, waiting=[seeded], reserved_slots=6)

    seen = _run_schedule(scheduler, monkeypatch)

    # original 8 - 6 reserved = 2, and 1 + 1 parked = 2: both fit, so the
    # seeded request gets its executing slot.
    assert seen["limit"] == 2
    assert seen["admitted"] == "seeded"


def test_async_reservation_can_still_block_the_solo_step(monkeypatch) -> None:
    """When reservations leave one slot, the parked session legitimately takes it."""
    seeded = _StubRequest("seeded", seed=42)
    scheduler = _make_scheduler(parked=1, waiting=[seeded], reserved_slots=7)

    seen = _run_schedule(scheduler, monkeypatch)

    # original 8 - 7 reserved = 1, below 1 + 1 parked: capacity wins and the
    # seeded request waits for the reservation to be released.
    assert seen["limit"] == 1
    assert seen["admitted"] is None


def test_parked_sessions_at_capacity_keep_the_solo_step_blocked(monkeypatch) -> None:
    """A full parked set must not let the solo budget exceed the worker capacity."""
    seeded = _StubRequest("seeded", seed=42)
    scheduler = _make_scheduler(parked=8, waiting=[seeded])

    seen = _run_schedule(scheduler, monkeypatch)

    # 8 parked sessions fill the configured 8 active requests: the parked-aware
    # budget is capped at the original limit and nothing else is admitted.
    assert seen["limit"] == 8
    assert seen["admitted"] is None
    assert scheduler.running == []


def test_idle_seeded_session_does_not_block_unrelated_work(monkeypatch) -> None:
    """A parked seeded session awaits client input; it must not hide the waiting queue."""
    idle_seeded = _StubRequest("seeded-session", seed=42, status=RequestStatus.WAITING_FOR_STREAMING_REQ)
    plain = _StubRequest("plain")
    scheduler = _make_scheduler(parked=1, waiting=[plain])
    scheduler.kv_holding_waiting.add_request(idle_seeded)

    seen = _run_schedule(scheduler, monkeypatch)

    # The idle session is not runnable, so the step is a normal admission and
    # the unrelated request runs while the seeded client stays idle.
    assert seen["admitted"] == "plain"
    assert seen["limit"] == 8


def test_seeded_request_parked_in_the_adapter_keeps_isolation(monkeypatch) -> None:
    """A seeded running request parked for chunk processing still owns the step."""
    seeded = _StubRequest("seeded", seed=42, status=RequestStatus.RUNNING)
    plain = _StubRequest("plain")
    scheduler = _make_scheduler(parked=0, waiting=[plain], adapter_parked_running=[seeded])

    seen = _run_schedule(scheduler, monkeypatch)

    # Admission is deferred: the seeded request is restored at the end of the
    # step, and admitting the plain request now would co-batch them.
    assert seen["admitted"] is None
    assert scheduler.running == []


def test_unseeded_adapter_park_still_counts_as_busy(monkeypatch) -> None:
    """An unseeded runner parked for the step must block the seeded solo admission."""
    parked_plain = _StubRequest("parked-plain", status=RequestStatus.RUNNING)
    seeded = _StubRequest("seeded", seed=42)
    scheduler = _make_scheduler(parked=0, waiting=[seeded], adapter_parked_running=[parked_plain])

    seen = _run_schedule(scheduler, monkeypatch)

    # The engine is busy: solo-admitting now would co-batch the seeded request
    # with the parked one once the step restores it.
    assert seen["admitted"] is None
    assert seen["limit"] == 8


def test_seeded_request_parked_in_the_coordinator_keeps_isolation(monkeypatch) -> None:
    """The native-coordinator park has the same contract as the adapter park."""
    from types import SimpleNamespace

    seeded = _StubRequest("seeded", seed=42, status=RequestStatus.RUNNING)
    plain = _StubRequest("plain")
    scheduler = _make_scheduler(parked=0, waiting=[plain])
    scheduler.chunk_transfer_adapter = None
    scheduler.input_coordinator = SimpleNamespace(_waiting_for_chunk_running=[seeded])

    seen = _run_schedule(scheduler, monkeypatch)

    assert seen["admitted"] is None


def test_resumed_seeded_session_waits_for_running_work(monkeypatch) -> None:
    """Once the session resumes it is runnable again and waits for the engine to drain."""
    running_plain = _StubRequest("plain", status=RequestStatus.RUNNING)
    resumed_seeded = _StubRequest("seeded-session", seed=42)
    scheduler = _make_scheduler(parked=0, waiting=[resumed_seeded])
    scheduler.running.append(running_plain)

    seen = _run_schedule(scheduler, monkeypatch)

    assert seen["admitted"] is None
    assert [request.request_id for request in scheduler.running] == ["plain"]


def test_unseeded_traffic_keeps_the_original_limit(monkeypatch) -> None:
    plain = _StubRequest("plain")
    scheduler = _make_scheduler(parked=2, waiting=[plain])

    seen = _run_schedule(scheduler, monkeypatch)

    assert seen["admitted"] == "plain"
    assert seen["limit"] == 8


def test_kv_holding_seeded_request_is_promoted_before_a_fresh_one() -> None:
    """Review P2: upstream drains KV-holding requests first, so must the policy."""
    fresh = _StubRequest("seeded-fresh", seed=7)
    resumed = _StubRequest("seeded-kv", seed=9)
    scheduler = _make_scheduler(parked=0, waiting=[fresh], kv_holding=[resumed])

    assert scheduler._promote_seeded_request() == "kv_holding"
    assert scheduler.kv_holding_waiting.peek_request() is resumed
    assert scheduler.waiting.peek_request() is fresh


def test_solo_step_hides_the_waiting_queue_when_promoting_kv_holding(monkeypatch) -> None:
    """A fresh request that cannot allocate must not hide a runnable KV-holding one.

    The promoted KV-holding request keeps its queue; the fresh ``waiting`` queue
    is the one emptied for the solo step, so the engine cannot livelock on a
    request whose prefill allocation keeps failing.
    """
    fresh = _StubRequest("seeded-fresh", seed=7)
    resumed = _StubRequest("seeded-kv", seed=9)
    scheduler = _make_scheduler(parked=0, waiting=[fresh], kv_holding=[resumed])

    seen = _run_schedule(scheduler, monkeypatch)

    assert seen["waiting_len"] == 0
    assert seen["kv_len"] == 1
    # The stub only admits from ``waiting``; the point is that the fresh queue
    # is hidden and the KV-holding request stays eligible.
    assert seen["admitted"] is None
