# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The 1 Hz SchedulerStats throttle must not swallow the report of a stage going idle.

EngineCore steps again only while ``has_requests()`` is True, so a throttled
report on the step that drains the stage is never retried:
``vllm:num_requests_running`` / ``vllm:num_requests_waiting`` /
``vllm:kv_cache_usage_perc`` would stay at their last busy value until the next
request arrives. Short requests (e.g. TTS) hit this on almost every burst.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.v1.metrics.stats import SchedulerStats

from vllm_omni.core.sched import omni_scheduler_mixin
from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _UpstreamScheduler:
    """Stands in for the vLLM ``Scheduler`` half of the MRO."""

    log_stats = True

    def __init__(self) -> None:
        self.num_running = 0
        self.num_waiting = 0
        # Parked streaming sessions: counted as waiting, but not stepped.
        self.num_waiting_for_streaming_input = 0
        # Finished requests whose cleanup (e.g. a KV handoff) keeps the engine stepping.
        self.pending_cleanup = False
        # Read by OmniSchedulerMixin.has_requests().
        self._outputs_awaiting_stage_payload: dict[str, object] = {}

    def has_unfinished_requests(self) -> bool:
        return self.num_running + self.num_waiting - self.num_waiting_for_streaming_input > 0

    def has_requests(self) -> bool:
        return self.has_unfinished_requests() or self.pending_cleanup

    def get_request_counts(self) -> tuple[int, int]:
        return self.num_running, self.num_waiting

    def make_stats(self, *args, **kwargs) -> SchedulerStats | None:
        if not self.log_stats:
            return None
        return SchedulerStats(num_running_reqs=self.num_running, num_waiting_reqs=self.num_waiting)


class _Scheduler(OmniSchedulerMixin, _UpstreamScheduler):
    pass


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    now = [1000.0]
    monkeypatch.setattr(omni_scheduler_mixin, "time", SimpleNamespace(monotonic=lambda: now[0]))
    return now


def _busy(sched: _Scheduler, running: int = 1) -> None:
    sched.num_running = running
    stats = sched.make_stats()
    assert stats is not None and stats.num_running_reqs == running


def test_busy_ticks_stay_throttled(clock):
    sched = _Scheduler()
    sched.num_running, sched.num_waiting = 2, 1
    assert sched.make_stats() is not None
    clock[0] += 0.2
    sched.num_running, sched.num_waiting = 1, 0
    assert sched.make_stats() is None
    clock[0] += 1.0
    later = sched.make_stats()
    assert later is not None and later.num_running_reqs == 1


def test_drain_bypasses_throttle(clock):
    sched = _Scheduler()
    _busy(sched)
    # The request finishes well inside the throttle window.
    clock[0] += 0.3
    sched.num_running = 0
    idle = sched.make_stats()
    assert idle is not None
    assert (idle.num_running_reqs, idle.num_waiting_reqs) == (0, 0)


def test_draining_the_waiting_queue_bypasses_throttle(clock):
    sched = _Scheduler()
    sched.num_waiting = 2
    assert sched.make_stats() is not None
    clock[0] += 0.2
    sched.num_waiting = 0
    idle = sched.make_stats()
    assert idle is not None and idle.num_waiting_reqs == 0


def test_parked_streaming_session_flushes_running(clock):
    sched = _Scheduler()
    _busy(sched)
    clock[0] += 0.3
    # The last session finished its chunk and parked: upstream still counts it
    # in get_request_counts(), but EngineCore stops stepping.
    sched.num_running, sched.num_waiting = 0, 1
    sched.num_waiting_for_streaming_input = 1
    idle = sched.make_stats()
    assert idle is not None and idle.num_running_reqs == 0


def test_session_ending_soon_after_parking_still_reports(clock):
    sched = _Scheduler()
    _busy(sched)
    clock[0] += 1.5
    sched.num_running, sched.num_waiting = 0, 1
    sched.num_waiting_for_streaming_input = 1
    assert sched.make_stats() is not None
    # The final chunk arrives, runs and ends inside the window that the parked
    # report opened, so none of its busy steps are reported.
    clock[0] += 0.1
    sched.num_running, sched.num_waiting = 1, 0
    sched.num_waiting_for_streaming_input = 0
    assert sched.make_stats() is None
    clock[0] += 0.1
    sched.num_running = 0
    idle = sched.make_stats()
    assert idle is not None
    assert (idle.num_running_reqs, idle.num_waiting_reqs) == (0, 0)


def test_cleanup_after_the_last_request_still_reports(clock):
    sched = _Scheduler()
    _busy(sched)
    clock[0] += 0.2
    # The request finished, but a KV connector still holds its blocks, so
    # EngineCore keeps stepping although nothing is unfinished.
    sched.num_running = 0
    sched.pending_cleanup = True
    sched.make_stats()
    clock[0] += 0.2
    # The connector releases the blocks; EngineCore stops stepping after this.
    sched.pending_cleanup = False
    assert sched.make_stats() is not None


def test_no_stats_when_disabled(clock):
    sched = _Scheduler()
    sched.log_stats = False
    sched.num_running = 1
    assert sched.make_stats() is None
    clock[0] += 0.3
    sched.num_running = 0
    assert sched.make_stats() is None
