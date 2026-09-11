# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Lease semantics of the engine-resident duplex session (idle TTL, disconnect grace, resume CAS)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexFence
from vllm_omni.engine.duplex.lease import (
    DuplexLeaseActivity,
    DuplexLeaseConfig,
    DuplexLeaseState,
)
from vllm_omni.engine.duplex.session import DuplexEngineSession, DuplexFenceMismatchError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeMonotonicClock:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _lease_config(*, idle_ttl_s: float | None = 300.0) -> DuplexLeaseConfig:
    return DuplexLeaseConfig(idle_ttl_s=idle_ttl_s, disconnect_grace_s=30.0)


def _session(session_id: str, clock: FakeMonotonicClock, *, idle_ttl_s: float | None = 300.0) -> DuplexEngineSession:
    return DuplexEngineSession(
        session_id=session_id,
        config=DuplexSessionConfig(),
        lease=DuplexLeaseState(config=_lease_config(idle_ttl_s=idle_ttl_s), generation=0, last_activity=clock()),
        _clock=clock,
    )


def test_open_touch_detach_and_expiry_use_monotonic_time() -> None:
    clock = FakeMonotonicClock(100.0)
    session = _session("sid-expiry", clock)

    assert session.lease.last_activity == 100.0
    clock.advance(10.0)
    session.touch_lease(DuplexLeaseActivity.HEARTBEAT)
    assert session.lease.last_activity == 110.0
    session.detach_lease()
    clock.advance(29.0)
    assert session.lease.disconnect_grace_expired(clock()) is False
    assert session.lease.idle_expired(clock()) is False

    clock.advance(1.0)
    assert session.lease.disconnect_grace_expired(clock()) is True

    clock.advance(271.0)
    assert session.lease.idle_expired(clock()) is True


def test_detach_grace_and_resume_advance_the_lease_generation() -> None:
    clock = FakeMonotonicClock(10.0)
    session = _session("sid-resume", clock)

    session.detach_lease()
    clock.advance(29.0)
    assert session.lease.disconnect_grace_expired(clock()) is False
    clock.advance(2.0)
    assert session.lease.disconnect_grace_expired(clock()) is True

    generation = session.resume_lease(expected_lease_generation=0)

    assert generation == 1
    assert session.lease_generation == 1
    assert session.lease.detached_at is None
    assert session.lease.disconnect_grace_expired(clock()) is False
    with pytest.raises(ValueError, match="lease generation mismatch"):
        session.resume_lease(expected_lease_generation=0)


def test_active_operation_prevents_mid_transaction_expiry() -> None:
    clock = FakeMonotonicClock(0.0)
    session = _session("sid-operation", clock, idle_ttl_s=5.0)

    session.begin_lease_operation(session.fence, "append-1")
    clock.advance(10.0)
    assert session.lease.idle_expired(clock()) is False

    session.end_lease_operation("append-1")
    assert session.lease.idle_expired(clock()) is False
    clock.advance(6.0)
    assert session.lease.idle_expired(clock()) is True


def test_sessions_have_independent_activity_deadlines_and_resources() -> None:
    clock = FakeMonotonicClock(0.0)
    session_a = _session("sid-a", clock, idle_ttl_s=10.0)
    session_b = _session("sid-b", clock, idle_ttl_s=10.0)
    session_a.reserve_stage_request(0, "req-a-reserved", fence=session_a.fence)
    session_a.bind_stage_request(1, "req-a-submitted", fence=session_a.fence)
    session_b.bind_stage_request(0, "req-b", fence=session_b.fence)

    clock.advance(6.0)
    session_b.touch_lease(DuplexLeaseActivity.MODEL_OUTPUT)
    clock.advance(5.0)

    assert session_a.lease.idle_expired(clock()) is True
    assert session_b.lease.idle_expired(clock()) is False
    assert session_a.resource_request_ids(submitted=False) == ["req-a-reserved"]
    assert session_a.resource_request_ids(submitted=True) == ["req-a-submitted"]
    assert session_b.resource_request_ids() == ["req-b"]


def test_close_is_a_single_terminal_transition() -> None:
    clock = FakeMonotonicClock(0.0)
    session = _session("sid-race", clock, idle_ttl_s=1.0)
    clock.advance(2.0)
    assert session.lease.idle_expired(clock()) is True

    assert session.begin_close(reason="explicit_close") is True
    assert session.lease.terminal_reason == "explicit_close"
    # A terminal lease is never reaped again and a second close is a no-op.
    assert session.lease.idle_expired(clock()) is False
    assert session.lease.disconnect_grace_expired(clock()) is False
    assert session.begin_close(reason="idle_ttl_expired") is True
    assert session.lease.terminal_reason == "explicit_close"
    with pytest.raises(RuntimeError, match="terminal"):
        session.touch_lease(DuplexLeaseActivity.HEARTBEAT)


def test_stale_fence_cannot_start_a_lease_operation() -> None:
    clock = FakeMonotonicClock(0.0)
    session = _session("sid-fence", clock)
    session.epoch = 1
    session.sync_fence()
    stale = DuplexFence("sid-fence", epoch=0)

    with pytest.raises(DuplexFenceMismatchError, match="fence mismatch"):
        session.begin_lease_operation(stale, "append-1")
    with pytest.raises(DuplexFenceMismatchError, match="fence mismatch"):
        session.begin_lease_operation(DuplexFence("other-session", epoch=1), "append-2")


def test_disabled_idle_expiry_never_expires_session() -> None:
    clock = FakeMonotonicClock(0.0)
    session = _session("sid-no-expiry", clock, idle_ttl_s=None)

    clock.advance(1_000_000.0)

    assert session.lease.expires_at is None
    assert session.lease.idle_expired(clock()) is False


def test_detached_session_expires_at_disconnect_grace_when_idle_ttl_is_disabled() -> None:
    clock = FakeMonotonicClock(0.0)
    session = _session("sid-disconnect-grace", clock, idle_ttl_s=None)

    session.detach_lease()
    clock.advance(29.0)
    assert session.lease.disconnect_grace_expired(clock()) is False

    clock.advance(1.0)
    assert session.lease.disconnect_grace_expired(clock()) is True


def test_lease_config_is_immutable_and_validated() -> None:
    config = _lease_config()
    with pytest.raises(FrozenInstanceError):
        config.idle_ttl_s = 1.0  # type: ignore[misc]
    with pytest.raises(ValueError, match="idle_ttl_s"):
        DuplexLeaseConfig(idle_ttl_s=0.0)
    with pytest.raises(ValueError, match="disconnect_grace_s"):
        DuplexLeaseConfig(disconnect_grace_s=0.0)
