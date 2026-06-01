# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit coverage for _process_pending_input_timeouts.

Verifies that the mixin correctly *delegates* timed-out requests to the
base scheduler's ``finish_requests`` API with ``RequestStatus.FINISHED_ERROR``.
The end-to-end effect (queue removal + status set + per-request cleanup +
client-facing FINISHED_ERROR emission) is the responsibility of upstream
vLLM's ``finish_requests`` implementation and is covered by upstream tests;
this file only asserts the wiring from the mixin to that API.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin


class _FakeCoordinator:
    def __init__(self, timed_out_ids):
        self._timed_out_ids = set(timed_out_ids)
        self.calls = []

    def collect_timed_out_request_ids(self, timeout_s):
        self.calls.append(timeout_s)
        return set(self._timed_out_ids)


class _FakeScheduler(OmniSchedulerMixin):
    def __init__(self, requests, coordinator):
        self.requests = requests
        self.input_coordinator = coordinator
        self.finish_calls = []

    def finish_requests(self, req_ids, status):
        self.finish_calls.append((set(req_ids), status))


def test_process_pending_input_timeouts_delegates_to_finish_requests():
    """Timed-out request present in self.requests is forwarded to finish_requests."""
    req_id = "stuck-req"
    requests = {req_id: SimpleNamespace(request_id=req_id)}
    coord = _FakeCoordinator(timed_out_ids={req_id})
    scheduler = _FakeScheduler(requests, coord)

    scheduler._process_pending_input_timeouts()

    assert len(coord.calls) == 1, "coordinator should be polled once"
    assert coord.calls[0] > 0, "timeout must be positive when enabled"

    assert len(scheduler.finish_calls) == 1
    finished_ids, status = scheduler.finish_calls[0]
    assert finished_ids == {req_id}
    # RequestStatus is the upstream enum; the mixin imports it as
    # RequestStatus.FINISHED_ERROR.  Check by name to avoid hard import here.
    assert getattr(status, "name", str(status)).endswith("FINISHED_ERROR")


def test_process_pending_input_timeouts_skips_already_freed_request():
    """Timed-out id no longer in self.requests must not be forwarded."""
    coord = _FakeCoordinator(timed_out_ids={"already-freed"})
    scheduler = _FakeScheduler(requests={}, coordinator=coord)

    scheduler._process_pending_input_timeouts()

    assert coord.calls == [coord.calls[0]] and coord.calls[0] > 0
    assert scheduler.finish_calls == []


def test_process_pending_input_timeouts_noop_without_coordinator():
    """No coordinator => no finish_requests call, no crash."""

    class _NoCoord(OmniSchedulerMixin):
        def __init__(self):
            self.requests = {}
            self.input_coordinator = None
            self.finish_calls = []

        def finish_requests(self, req_ids, status):
            self.finish_calls.append((set(req_ids), status))

    scheduler = _NoCoord()
    scheduler._process_pending_input_timeouts()
    assert scheduler.finish_calls == []


def test_process_pending_input_timeouts_disabled_when_timeout_zero(monkeypatch):
    """Setting DEFAULT_INPUT_WAIT_TIMEOUT_S <= 0 disables the safety net."""
    from vllm_omni.core.sched import omni_scheduler_mixin

    monkeypatch.setattr(omni_scheduler_mixin, "DEFAULT_INPUT_WAIT_TIMEOUT_S", 0.0)

    coord = _FakeCoordinator(timed_out_ids={"r1"})
    scheduler = _FakeScheduler(requests={"r1": SimpleNamespace(request_id="r1")}, coordinator=coord)
    scheduler._process_pending_input_timeouts()
    assert coord.calls == [], "coordinator must not be polled when timeout is disabled"
    assert scheduler.finish_calls == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
