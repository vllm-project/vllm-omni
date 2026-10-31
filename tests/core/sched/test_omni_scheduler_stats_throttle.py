# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The 1 Hz SchedulerStats throttle must not swallow the busy -> idle tick.

EngineCore stops stepping once the scheduler is empty, so a throttled idle tick
is never retried and ``vllm:num_requests_running`` / ``vllm:num_requests_waiting``
/ ``vllm:kv_cache_usage_perc`` stay frozen at their last busy value until the
next request arrives. Short requests (e.g. TTS) hit this on almost every burst.
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


def _busy_then_finish(clock: list[float], sched: _Scheduler) -> SchedulerStats | None:
    sched.num_running = 1
    busy = sched.make_stats()
    assert busy is not None and busy.num_running_reqs == 1
    # The request finishes well inside the throttle window.
    clock[0] += 0.3
    sched.num_running = 0
    return sched.make_stats()


def test_idle_transition_bypasses_throttle(clock):
    idle = _busy_then_finish(clock, _Scheduler())
    assert idle is not None
    assert (idle.num_running_reqs, idle.num_waiting_reqs) == (0, 0)


def test_idle_flush_is_emitted_once(clock):
    sched = _Scheduler()
    assert _busy_then_finish(clock, sched) is not None
    clock[0] += 0.1
    assert sched.make_stats() is None


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


def test_waiting_requests_count_as_busy(clock):
    sched = _Scheduler()
    sched.num_waiting = 2
    assert sched.make_stats() is not None
    clock[0] += 0.2
    sched.num_waiting = 0
    idle = sched.make_stats()
    assert idle is not None and idle.num_waiting_reqs == 0


def test_no_flush_when_stats_disabled(clock):
    sched = _Scheduler()
    sched.log_stats = False
    sched.num_running = 1
    assert sched.make_stats() is None
    clock[0] += 0.3
    sched.num_running = 0
    assert sched.make_stats() is None
