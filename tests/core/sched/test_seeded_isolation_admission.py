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
    def __init__(self, seed: int | None = None, extra_args: dict | None = None) -> None:
        self.seed = seed
        self.extra_args = extra_args


class _StubRequest:
    """Request stub carrying the fields the scheduler and the queues read.

    ``seed`` defaults to a client-marked seed; pass ``stage_default=True`` for a
    request whose seed came from ``default_sampling_params`` (unmarked).
    """

    def __init__(
        self,
        request_id: str,
        seed: int | None = None,
        status: RequestStatus = RequestStatus.WAITING,
        *,
        stage_default: bool = False,
    ) -> None:
        self.request_id = request_id
        self.status = status
        self.priority = 0
        self.arrival_time = 0.0
        extra_args = None
        if seed is not None and not stage_default:
            extra_args = {"omni_user_seed": True}
        self.sampling_params = _Params(seed, extra_args)

    def __lt__(self, other: _StubRequest) -> bool:
        return (self.priority, self.arrival_time) < (other.priority, other.arrival_time)


def _make_scheduler(
    *, parked: int, waiting, kv_holding=(), reserved_slots: int = 0, adapter_parked_running=(), cls=OmniARScheduler
) -> OmniARScheduler:
    """Real scheduler wrapper with the surface ``schedule()`` reads."""
    scheduler = cls.__new__(cls)
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
    seen: dict = {
        "limit": None,
        "admitted": None,
        "waiting_len": None,
        "kv_len": None,
        "waiting_ids": None,
        "kv_ids": None,
    }

    def upstream_schedule(_self, _throttle_prefills: bool = False):
        seen["limit"] = scheduler.max_num_active_reqs
        seen["waiting_len"] = len(scheduler.waiting)
        seen["kv_len"] = len(scheduler.kv_holding_waiting)
        seen["waiting_ids"] = [request.request_id for request in scheduler.waiting]
        seen["kv_ids"] = [request.request_id for request in scheduler.kv_holding_waiting]
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

    promoted = scheduler._promote_seeded_request()
    assert promoted is not None and promoted[0] == "kv_holding"
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


def test_stage_default_seed_keeps_normal_admission(monkeypatch) -> None:
    """Review P1: a deploy-level ``default_sampling_params.seed`` is not an opt-in.

    Those seeds are copied into every request by the serving layer, so treating
    them as seeded would serialize the whole AR engine and drop ``max_num_seqs``
    to one on the shipped configs that set them.
    """
    plain = _StubRequest("plain")
    default_seeded = _StubRequest("default-seeded", seed=42, stage_default=True)
    scheduler = _make_scheduler(parked=0, waiting=[plain, default_seeded])

    seen = _run_schedule(scheduler, monkeypatch)

    # Normal admission: the full active-request budget, no solo step.
    assert seen["limit"] == 8
    assert seen["admitted"] is not None


def test_unseeded_kv_holder_drains_before_a_fresh_seeded_request(monkeypatch) -> None:
    """A fresh seeded request must not hide an unseeded KV holder that can free blocks."""
    seeded = _StubRequest("seeded-fresh", seed=7)
    holder = _StubRequest("holder")
    scheduler = _make_scheduler(parked=0, waiting=[seeded], kv_holding=[holder])

    seen = _run_schedule(scheduler, monkeypatch)

    # Drain step: the holder keeps its queue and the fresh seeded request waits.
    assert seen["waiting_len"] == 0
    assert seen["kv_len"] == 1
    assert seen["limit"] == 8


def test_cfg_pair_is_one_isolation_unit() -> None:
    from vllm_omni.core.sched.omni_cfg_ar_scheduler import OmniCFGARScheduler

    scheduler = OmniCFGARScheduler.__new__(OmniCFGARScheduler)
    scheduler._request_pairs = {"a": "pair", "b": "pair"}

    class _Req:
        def __init__(self, request_id):
            self.request_id = request_id

    cond, uncond, single = _Req("a"), _Req("b"), _Req("c")
    scheduler.requests = {"a": cond, "b": uncond, "c": single}
    scheduler._pairs = {"pair": {"cond": "a", "uncond": "b"}}

    assert scheduler._isolation_unit(cond) == [cond, uncond]
    assert scheduler._isolation_group_size(cond) == 2
    assert scheduler._isolation_group_size(single) == 1

    scheduler._pairs["pair"] = {"cond": "a"}
    assert scheduler._isolation_unit(cond) is None
    assert scheduler._isolation_group_size(cond) is None


class _PairScheduler(OmniARScheduler):
    """Isolation unit of two branches for a request that names its partner."""

    def _isolation_unit(self, request):
        if getattr(request, "request_id", None) == "cond":
            return [request, self._partner]
        return [request]


def test_solo_step_restricts_the_queue_to_the_pair(monkeypatch) -> None:
    """A cap wide enough for a pair must not admit a request between its branches."""
    other = _StubRequest("other")
    cond = _StubRequest("cond", seed=7)
    uncond = _StubRequest("uncond", seed=7)
    scheduler = _make_scheduler(cls=_PairScheduler, parked=0, waiting=[other, cond, uncond])
    scheduler._partner = uncond

    seen = _run_schedule(scheduler, monkeypatch)

    assert seen["limit"] == 2
    assert seen["waiting_len"] == 2
    assert seen["admitted"] == "cond"


def test_incomplete_pair_defers_instead_of_splitting(monkeypatch) -> None:
    class _IncompletePairScheduler(OmniARScheduler):
        def _isolation_unit(self, request):
            return None if getattr(request, "request_id", None) == "cond" else [request]

    cond = _StubRequest("cond", seed=7)
    scheduler = _make_scheduler(cls=_IncompletePairScheduler, parked=0, waiting=[cond])

    seen = _run_schedule(scheduler, monkeypatch)

    assert seen["waiting_len"] == 0
    assert seen["kv_len"] == 0
    assert seen["admitted"] is None


def test_common_cfg_pair_is_one_isolation_unit() -> None:
    """Audex registers its CFG pair on the ordinary AR scheduler."""
    cond = _StubRequest("cond", seed=7)
    uncond = _StubRequest("uncond")  # the companion is not marked
    scheduler = _make_scheduler(parked=0, waiting=[cond, uncond])
    scheduler._cfg_pairs = {"pair": {"cond": "cond", "uncond": "uncond"}}
    scheduler._cfg_req_to_pair = {"cond": "pair", "uncond": "pair"}
    scheduler.requests = {"cond": cond, "uncond": uncond}

    assert scheduler._isolation_unit(cond) == [cond, uncond]
    assert scheduler._isolation_group_size(cond) == 2

    scheduler._cfg_pairs["pair"] = {"cond": "cond"}
    assert scheduler._isolation_unit(cond) is None


def test_unseeded_holder_drains_before_a_seeded_holder(monkeypatch) -> None:
    """A seeded holder that needs a block must not hide a holder that can finish."""
    fresh = _StubRequest("seeded-fresh", seed=7)
    seeded_holder = _StubRequest("seeded-holder", seed=9)
    unseeded_holder = _StubRequest("unseeded-holder")
    scheduler = _make_scheduler(parked=0, waiting=[fresh], kv_holding=[unseeded_holder, seeded_holder])

    seen = _run_schedule(scheduler, monkeypatch)

    # Only the unseeded holder is admitted so it can release its blocks; the
    # seeded holder stays queued for its own solo step.
    assert seen["waiting_len"] == 0
    assert seen["kv_len"] == 1
    assert seen["limit"] == 8


def test_solo_step_admits_a_common_cfg_pair_together(monkeypatch) -> None:
    """Audex-shaped pair: the companion does not carry the marker but must run too."""
    other = _StubRequest("other")
    cond = _StubRequest("cond", seed=7)
    uncond = _StubRequest("uncond")
    scheduler = _make_scheduler(parked=0, waiting=[other, cond, uncond])
    scheduler._cfg_pairs = {"pair": {"cond": "cond", "uncond": "uncond"}}
    scheduler._cfg_req_to_pair = {"cond": "pair", "uncond": "pair"}
    scheduler.requests = {"other": other, "cond": cond, "uncond": uncond}

    seen = _run_schedule(scheduler, monkeypatch)

    assert seen["limit"] == 2
    assert seen["waiting_len"] == 2
    assert seen["admitted"] == "cond"


def _audex_pair(scheduler, cond, uncond):
    scheduler._cfg_pairs = {"pair": {"cond": cond.request_id, "uncond": uncond.request_id}}
    scheduler._cfg_req_to_pair = {cond.request_id: "pair", uncond.request_id: "pair"}
    scheduler.requests = {cond.request_id: cond, uncond.request_id: uncond}


def test_paired_companion_is_not_drained_alone(monkeypatch) -> None:
    """A CFG companion must not drain without its seeded partner."""
    cond = _StubRequest("cond", seed=7)
    uncond = _StubRequest("uncond")
    scheduler = _make_scheduler(parked=0, waiting=[], kv_holding=[uncond, cond])
    _audex_pair(scheduler, cond, uncond)

    seen = _run_schedule(scheduler, monkeypatch)

    # The unit is restricted together from the holder queue, not drained apart.
    assert seen["limit"] == 2
    assert seen["waiting_len"] == 0
    assert seen["kv_ids"] == ["cond", "uncond"]


def test_standalone_holder_still_drains_before_a_paired_unit(monkeypatch) -> None:
    loose = _StubRequest("loose")
    cond = _StubRequest("cond", seed=7)
    uncond = _StubRequest("uncond")
    scheduler = _make_scheduler(parked=0, waiting=[], kv_holding=[loose, uncond, cond])
    _audex_pair(scheduler, cond, uncond)

    seen = _run_schedule(scheduler, monkeypatch)

    # Only the standalone unseeded holder drains; the pair stays queued.
    assert seen["kv_ids"] == ["loose"]
    assert seen["limit"] == 8
