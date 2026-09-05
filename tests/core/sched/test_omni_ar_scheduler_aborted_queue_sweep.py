"""Tests for ``OmniARScheduler._drop_aborted_queued_requests``.

``schedule()`` sweeps ``FINISHED_ABORTED`` requests out of the queues before
handing control to upstream, because upstream ``Scheduler.schedule()`` raises
``RuntimeError: Invalid request status: FINISHED_ABORTED`` for any request it
admits in a finished state -- which kills the stage's engine core, not just
the request.

Regression for the sweep missing ``skipped_waiting``, the third queue
upstream admits from. An aborted duplex session parked there was re-selected
by ``_select_waiting_queue_for_scheduling`` on a later tick and crashed the
stage.
"""

from __future__ import annotations

import pytest
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.request import RequestStatus

import vllm_omni.core.sched.omni_ar_scheduler as ar_sched_mod
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_POLICIES = [
    pytest.param(SchedulingPolicy.FCFS, id="fcfs"),
    pytest.param(SchedulingPolicy.PRIORITY, id="priority"),
]


class _StubRequest:
    """Minimal ``Request`` stub with the surface the sweep exercises.

    Carries the ``(priority, arrival_time)`` ordering ``PriorityRequestQueue``
    relies on so the same stub works under both scheduling policies.
    """

    def __init__(self, request_id: str, status: RequestStatus, arrival_time: float = 0.0) -> None:
        self.request_id = request_id
        self.status = status
        self.priority = 0
        self.arrival_time = arrival_time

    def __lt__(self, other: _StubRequest) -> bool:
        return (self.priority, self.arrival_time) < (other.priority, other.arrival_time)


def _make_scheduler(policy: SchedulingPolicy, *, waiting=(), skipped_waiting=(), running=()):
    """Bare scheduler with just the surface the sweep reads."""
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler.waiting = create_request_queue(policy)
    for req in waiting:
        scheduler.waiting.add_request(req)
    scheduler.skipped_waiting = create_request_queue(policy)
    for req in skipped_waiting:
        scheduler.skipped_waiting.add_request(req)
    scheduler.running = list(running)
    return scheduler


@pytest.mark.parametrize("policy", _POLICIES)
def test_schedule_sweeps_skipped_waiting_before_upstream_selection(
    monkeypatch: pytest.MonkeyPatch,
    policy: SchedulingPolicy,
) -> None:
    """The public schedule path must remove aborted skipped requests first."""
    aborted = _StubRequest("req-aborted", RequestStatus.FINISHED_ABORTED)
    scheduler = _make_scheduler(policy, skipped_waiting=[aborted])
    scheduler.chunk_transfer_adapter = None
    scheduler.input_coordinator = None
    scheduler.num_waiting_for_streaming_input = 0
    scheduler._process_pending_omni_inputs = lambda model_mode: None
    scheduler._resync_streaming_input_counter = lambda: None
    scheduler._should_defer_waiting_admission = lambda: False
    scheduler._restore_omni_wait_queues = lambda: None
    scheduler._postprocess_omni_schedule_output = lambda *args, **kwargs: None
    scheduler.get_finished_requests_needing_kv_transfer = lambda: {}
    scheduler._wrap_omni_scheduler_output = lambda output, **kwargs: output

    sentinel = object()

    def upstream_schedule(_self, _throttle_prefills: bool = False):
        assert list(scheduler.skipped_waiting) == []
        return sentinel

    monkeypatch.setattr(ar_sched_mod.VLLMScheduler, "schedule", upstream_schedule)

    assert OmniARScheduler.schedule(scheduler) is sentinel
