# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit coverage for the two stage-input timeout hooks.

``_process_pending_input_timeouts`` covers the full-payload transport;
``_process_pending_chunk_timeouts`` covers async-chunk. Both share
``VLLM_OMNI_INPUT_WAIT_TIMEOUT_S``.

Verifies that the mixin correctly *delegates* timed-out requests to the
base scheduler's ``finish_requests`` API with ``RequestStatus.FINISHED_ERROR``.
The end-to-end effect (queue removal + status set + per-request cleanup +
client-facing FINISHED_ERROR emission) is the responsibility of upstream
vLLM's ``finish_requests`` implementation and is covered by upstream tests;
this file only asserts the wiring from the mixin to that API.
"""

from __future__ import annotations

import importlib
import logging
import math
import os
from types import SimpleNamespace

import pytest

from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


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


class _FakeChunkAdapter:
    """Stands in for OmniChunkTransferAdapter's timeout-collection surface."""

    def __init__(self, timed_out_ids, *, receives_chunks=True):
        self._timed_out_ids = set(timed_out_ids)
        self.receives_chunks = receives_chunks
        self.calls = []

    def collect_timed_out_request_ids(self, timeout_s):
        self.calls.append(timeout_s)
        return set(self._timed_out_ids)


class _FakeChunkScheduler(OmniSchedulerMixin):
    def __init__(self, requests, adapter):
        self.requests = requests
        self.chunk_transfer_adapter = adapter
        self.finish_calls = []

    def finish_requests(self, req_ids, status):
        self.finish_calls.append((set(req_ids), status))


def test_process_pending_chunk_timeouts_delegates_to_finish_requests():
    """A request stalled in WAITING_FOR_CHUNK is failed, not left parked.

    This is the async-chunk half of the safety net (RFC #4855 R1.1); before it
    existed, the chunk path had no deadline at all and issue #3833 could park a
    request forever.
    """
    req_id = "stalled-stream"
    adapter = _FakeChunkAdapter(timed_out_ids={req_id})
    scheduler = _FakeChunkScheduler({req_id: SimpleNamespace(request_id=req_id)}, adapter)

    scheduler._process_pending_chunk_timeouts()

    assert len(adapter.calls) == 1
    assert adapter.calls[0] > 0
    assert len(scheduler.finish_calls) == 1
    finished_ids, status = scheduler.finish_calls[0]
    assert finished_ids == {req_id}
    assert getattr(status, "name", str(status)).endswith("FINISHED_ERROR")


def test_process_pending_chunk_timeouts_skips_already_freed_request():
    adapter = _FakeChunkAdapter(timed_out_ids={"already-freed"})
    scheduler = _FakeChunkScheduler(requests={}, adapter=adapter)

    scheduler._process_pending_chunk_timeouts()

    assert scheduler.finish_calls == []


def test_process_pending_chunk_timeouts_noop_on_a_producer_stage():
    """Stage 0 does not receive chunks, so it has nothing to expire."""
    adapter = _FakeChunkAdapter(timed_out_ids={"r1"}, receives_chunks=False)
    scheduler = _FakeChunkScheduler({"r1": SimpleNamespace(request_id="r1")}, adapter)

    scheduler._process_pending_chunk_timeouts()

    assert adapter.calls == []
    assert scheduler.finish_calls == []


def test_process_pending_chunk_timeouts_noop_without_adapter():
    class _NoAdapter(OmniSchedulerMixin):
        def __init__(self):
            self.requests = {}
            self.chunk_transfer_adapter = None
            self.finish_calls = []

        def finish_requests(self, req_ids, status):
            self.finish_calls.append((set(req_ids), status))

    scheduler = _NoAdapter()
    scheduler._process_pending_chunk_timeouts()
    assert scheduler.finish_calls == []


def test_process_pending_chunk_timeouts_disabled_when_timeout_zero(monkeypatch):
    """The shared knob disables both transports' nets together."""
    from vllm_omni.core.sched import omni_scheduler_mixin

    monkeypatch.setattr(omni_scheduler_mixin, "DEFAULT_INPUT_WAIT_TIMEOUT_S", 0.0)

    adapter = _FakeChunkAdapter(timed_out_ids={"r1"})
    scheduler = _FakeChunkScheduler({"r1": SimpleNamespace(request_id="r1")}, adapter)
    scheduler._process_pending_chunk_timeouts()

    assert adapter.calls == []
    assert scheduler.finish_calls == []


class _FakeSendFailureAdapter:
    """Stands in for the adapter's drained send-failure map (R1.2)."""

    def __init__(self, failures):
        self._failures = dict(failures)
        self.collect_calls = 0

    def collect_failed_send_request_ids(self):
        self.collect_calls += 1
        failures, self._failures = self._failures, {}
        return failures


class _FakeSendFailureScheduler(OmniSchedulerMixin):
    def __init__(self, requests, adapter):
        self.requests = requests
        self.chunk_transfer_adapter = adapter


def test_log_failed_chunk_sends_reports_a_live_request(caplog):
    adapter = _FakeSendFailureAdapter({"req-live": "OSError: No space left on device"})
    scheduler = _FakeSendFailureScheduler({"req-live": SimpleNamespace(request_id="req-live")}, adapter)

    with caplog.at_level(logging.ERROR):
        scheduler._log_failed_chunk_sends()

    assert "req-live" in caplog.text
    assert "No space left on device" in caplog.text


def test_log_failed_chunk_sends_still_reports_a_departed_request(caplog):
    """``collect_*`` drains the map, so a failure for a request that already left
    this scheduler is the one place a drop could still go unrecorded."""
    adapter = _FakeSendFailureAdapter({"req-gone": "OSError: No space left on device"})
    scheduler = _FakeSendFailureScheduler({}, adapter)

    with caplog.at_level(logging.WARNING):
        scheduler._log_failed_chunk_sends()

    assert adapter.collect_calls == 1
    assert "req-gone" in caplog.text
    assert "No space left on device" in caplog.text
    assert "finished or aborted" in caplog.text


def test_log_failed_chunk_sends_noop_without_adapter():
    scheduler = _FakeSendFailureScheduler({}, None)
    scheduler._log_failed_chunk_sends()


class _FakeReceiveFailureAdapter:
    def __init__(self, failures):
        self._failures = dict(failures)

    def collect_failed_receive_request_ids(self):
        failures, self._failures = self._failures, {}
        return failures


class _FakeReceiveFailureScheduler(OmniSchedulerMixin):
    def __init__(self, requests, adapter):
        self.requests = requests
        self.chunk_transfer_adapter = adapter
        self.finish_calls = []

    def finish_requests(self, req_ids, status):
        self.finish_calls.append((set(req_ids), status))


def test_process_chunk_receive_failures_delegates_to_finish_requests():
    request_id = "invalid-chunk"
    adapter = _FakeReceiveFailureAdapter({request_id: "prompt exceeds limit"})
    scheduler = _FakeReceiveFailureScheduler({request_id: SimpleNamespace(request_id=request_id)}, adapter)

    scheduler._process_chunk_receive_failures()

    assert len(scheduler.finish_calls) == 1
    finished_ids, status = scheduler.finish_calls[0]
    assert finished_ids == {request_id}
    assert getattr(status, "name", str(status)).endswith("FINISHED_ERROR")


# --- R1.4: VLLM_OMNI_INPUT_WAIT_TIMEOUT_S is parsed at import time, so each of
# these reloads the module under a different environment. One knob now gates the
# deadline on both transports, which is why a silent 0 matters more than it did
# when only the full-payload path used it.


@pytest.fixture
def reload_with(monkeypatch):
    """Reload the mixin module under a given env value, then put it back.

    ``importlib.reload`` re-executes into the live module dict, so whatever the
    last test parsed is what every later test in the session sees -- ``-k
    zero_warns`` alone would leave ``DEFAULT_INPUT_WAIT_TIMEOUT_S = 0.0`` and
    silently disable both deadlines. The rejection cases are worse: the
    ValueError fires *after* the rebind, so ``nan`` lingers in the module.
    ``monkeypatch`` restores the environment but not the module, so reload once
    more on the way out.
    """
    import vllm_omni.core.sched.omni_scheduler_mixin as mod

    def _reload(raw):
        if raw is None:
            monkeypatch.delenv("VLLM_OMNI_INPUT_WAIT_TIMEOUT_S", raising=False)
        else:
            monkeypatch.setenv("VLLM_OMNI_INPUT_WAIT_TIMEOUT_S", raw)
        return importlib.reload(mod)

    yield _reload

    monkeypatch.undo()
    importlib.reload(mod)


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, 600.0),
        ("30", 30.0),
        ("0.5", 0.5),
        ("0", 0.0),  # explicit opt-out is still honoured
    ],
)
def test_timeout_accepts_valid_values(reload_with, raw, expected):
    mod = reload_with(raw)
    assert mod.DEFAULT_INPUT_WAIT_TIMEOUT_S == expected


@pytest.mark.parametrize("raw", ["-1", "-600", "-0.5"])
def test_negative_timeout_is_rejected(reload_with, raw):
    """`-1` is a common "no limit" idiom; substituting 600 would give an operator
    who meant "never time out" mysterious failures ten minutes in. Fail loudly."""
    with pytest.raises(ValueError, match="negative"):
        reload_with(raw)


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "1e999"])
def test_non_finite_timeout_is_rejected(reload_with, raw):
    """nan compares false against every deadline check and inf is unreachable;
    both disable the net while looking like a number."""
    with pytest.raises(ValueError, match="not finite"):
        reload_with(raw)


def test_zero_warns_and_names_both_transports(reload_with, caplog):
    with caplog.at_level(logging.WARNING):
        mod = reload_with("0")
    assert mod.DEFAULT_INPUT_WAIT_TIMEOUT_S == 0.0
    text = caplog.text
    assert "DISABLED" in text
    assert "WAITING_FOR_INPUT" in text and "WAITING_FOR_CHUNK" in text


def test_unparsable_timeout_falls_back(reload_with):
    mod = reload_with("not-a-number")
    assert mod.DEFAULT_INPUT_WAIT_TIMEOUT_S == 600.0


def test_a_rejected_value_does_not_outlive_its_test(monkeypatch):
    """The reason ``reload_with`` reloads on the way out, done here by hand.

    ``nan`` is bound before the ValueError fires, so the reload that rejects it
    still leaves it in the module -- and nan compares false against every
    deadline check, which is exactly the silent-disable this item exists to
    stop. Restoring the environment is not enough; the module has to be reloaded
    too.
    """
    import vllm_omni.core.sched.omni_scheduler_mixin as mod

    ambient = float(os.environ.get("VLLM_OMNI_INPUT_WAIT_TIMEOUT_S", "600"))

    monkeypatch.setenv("VLLM_OMNI_INPUT_WAIT_TIMEOUT_S", "nan")
    with pytest.raises(ValueError, match="not finite"):
        importlib.reload(mod)
    assert math.isnan(mod.DEFAULT_INPUT_WAIT_TIMEOUT_S)

    monkeypatch.undo()
    importlib.reload(mod)
    assert mod.DEFAULT_INPUT_WAIT_TIMEOUT_S == ambient


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
