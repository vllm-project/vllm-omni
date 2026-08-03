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
def test_sweep_drops_aborted_request_from_skipped_waiting(policy: SchedulingPolicy) -> None:
    """The sweep must cover ``skipped_waiting``.

    Upstream ``schedule()`` admits from ``skipped_waiting`` as readily as from
    ``waiting``, so an aborted request left there is re-selected and raises.
    """
    aborted = _StubRequest("req-aborted", RequestStatus.FINISHED_ABORTED)
    scheduler = _make_scheduler(policy, skipped_waiting=[aborted])

    scheduler._drop_aborted_queued_requests()

    assert list(scheduler.skipped_waiting) == []


@pytest.mark.parametrize("policy", _POLICIES)
def test_sweep_drops_aborted_requests_from_waiting_and_running(policy: SchedulingPolicy) -> None:
    """``waiting`` and ``running`` stay covered alongside the new queue."""
    aborted_waiting = _StubRequest("req-waiting", RequestStatus.FINISHED_ABORTED)
    aborted_running = _StubRequest("req-running", RequestStatus.FINISHED_ABORTED)
    scheduler = _make_scheduler(
        policy,
        waiting=[aborted_waiting],
        running=[aborted_running],
    )

    scheduler._drop_aborted_queued_requests()

    assert list(scheduler.waiting) == []
    assert scheduler.running == []


@pytest.mark.parametrize("policy", _POLICIES)
def test_sweep_leaves_live_requests_intact(policy: SchedulingPolicy) -> None:
    """The sweep must not over-sweep.

    A request parked in ``skipped_waiting`` on a blocked waiting status is one
    upstream still intends to schedule; only the aborted entry may go.
    """
    blocked = _StubRequest("req-blocked", RequestStatus.WAITING_FOR_STREAMING_REQ, arrival_time=0.0)
    aborted = _StubRequest("req-aborted", RequestStatus.FINISHED_ABORTED, arrival_time=1.0)
    live_waiting = _StubRequest("req-live-waiting", RequestStatus.WAITING)
    live_running = _StubRequest("req-live-running", RequestStatus.RUNNING)
    scheduler = _make_scheduler(
        policy,
        waiting=[live_waiting],
        skipped_waiting=[blocked, aborted],
        running=[live_running],
    )

    scheduler._drop_aborted_queued_requests()

    assert list(scheduler.skipped_waiting) == [blocked]
    assert list(scheduler.waiting) == [live_waiting]
    assert scheduler.running == [live_running]
