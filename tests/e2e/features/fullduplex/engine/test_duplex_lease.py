# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from vllm_omni.experimental.fullduplex.engine.duplex_runtime import DuplexInputMode
from vllm_omni.experimental.fullduplex.engine.duplex_session import DuplexSessionRuntimeManager
from vllm_omni.experimental.fullduplex.engine.lease import (
    DuplexLeaseActivity,
    DuplexLeaseConfig,
)
from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeMonotonicClock:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _lease_config(*, idle_ttl_s: float | None = 300.0) -> DuplexLeaseConfig:
    return DuplexLeaseConfig(
        idle_ttl_s=idle_ttl_s,
        disconnect_grace_s=30.0,
    )


def test_open_touch_detach_and_idle_expiry_use_monotonic_time() -> None:
    clock = FakeMonotonicClock(100.0)
    manager = DuplexSessionRuntimeManager(clock=clock)
    fence = DuplexFence("sid-expiry")
    session = manager.open_session(fence, lease_config=_lease_config())

    assert session.lease.last_activity == 100.0
    clock.advance(10.0)
    session.touch(fence, DuplexLeaseActivity.HEARTBEAT)
    session.detach(fence)
    clock.advance(29.0)
    assert session.lease.disconnect_grace_expired(clock()) is False
    assert manager.collect_expired() == []

    clock.advance(272.0)
    expired = manager.collect_expired()

    assert [item.session_id for item in expired] == ["sid-expiry"]
    assert expired[0].reason == "disconnect_grace_expired"
    assert manager.get("sid-expiry") is session
    manager.finalize_close_session(session)
    assert manager.get("sid-expiry") is None


def test_detach_grace_and_resume_advance_token_independent_lease_generation() -> None:
    clock = FakeMonotonicClock(10.0)
    manager = DuplexSessionRuntimeManager(clock=clock)
    fence = DuplexFence("sid-resume")
    session = manager.open_session(fence, lease_config=_lease_config())

    session.detach(fence)
    clock.advance(29.0)
    assert session.lease.disconnect_grace_expired(clock()) is False
    clock.advance(2.0)
    assert session.lease.disconnect_grace_expired(clock()) is True

    generation = session.resume(fence, expected_lease_generation=0)

    assert generation == 1
    assert session.lease.detached_at is None
    assert session.lease.disconnect_grace_expired(clock()) is False
    with pytest.raises(ValueError, match="lease generation mismatch"):
        session.resume(fence, expected_lease_generation=0)


def test_active_operation_prevents_mid_transaction_expiry() -> None:
    clock = FakeMonotonicClock(0.0)
    manager = DuplexSessionRuntimeManager(clock=clock)
    fence = DuplexFence("sid-operation")
    session = manager.open_session(fence, lease_config=_lease_config(idle_ttl_s=5.0))

    session.begin_operation(fence, "append-1")
    clock.advance(10.0)
    assert manager.collect_expired() == []

    session.end_operation(fence, "append-1")
    assert manager.collect_expired() == []
    clock.advance(6.0)
    assert [item.session_id for item in manager.collect_expired()] == ["sid-operation"]


def test_sessions_have_independent_activity_deadlines_and_resources() -> None:
    clock = FakeMonotonicClock(0.0)
    manager = DuplexSessionRuntimeManager(clock=clock)
    fence_a = DuplexFence("sid-a")
    fence_b = DuplexFence("sid-b")
    session_a = manager.open_session(fence_a, lease_config=_lease_config(idle_ttl_s=10.0))
    session_b = manager.open_session(fence_b, lease_config=_lease_config(idle_ttl_s=10.0))
    session_a.reserve_stage_request(0, "req-a-reserved", fence=fence_a)
    session_a.bind_stage_request(1, "req-a-submitted", fence=fence_a)
    session_b.bind_stage_request(0, "req-b", fence=fence_b)

    clock.advance(6.0)
    session_b.touch(fence_b, DuplexLeaseActivity.MODEL_OUTPUT)
    clock.advance(5.0)
    expired = manager.collect_expired()

    assert len(expired) == 1
    assert expired[0].session_id == "sid-a"
    assert expired[0].reserved_request_ids == ("req-a-reserved",)
    assert expired[0].submitted_request_ids == ("req-a-submitted",)
    assert manager.get("sid-b") is session_b
    assert session_b.resource_request_ids() == ["req-b"]


def test_close_and_reaper_select_exactly_one_terminal_transition() -> None:
    clock = FakeMonotonicClock(0.0)
    manager = DuplexSessionRuntimeManager(clock=clock)
    fence = DuplexFence("sid-race")
    manager.open_session(fence, lease_config=_lease_config(idle_ttl_s=1.0))
    clock.advance(2.0)

    assert len(manager.collect_expired()) == 1
    assert manager.close_session(fence, reason="explicit_close") is None
    assert manager.collect_expired() == []

    fence_2 = DuplexFence("sid-race-2")
    manager.open_session(fence_2, lease_config=_lease_config(idle_ttl_s=1.0))
    assert manager.close_session(fence_2, reason="explicit_close") is not None
    clock.advance(2.0)
    assert manager.collect_expired() == []


def test_stale_fence_cannot_touch_detach_or_resume_lease() -> None:
    clock = FakeMonotonicClock(0.0)
    manager = DuplexSessionRuntimeManager(clock=clock)
    current = DuplexFence("sid-fence", epoch=1)
    stale = DuplexFence("sid-fence", epoch=0)
    session = manager.open_session(current, lease_config=_lease_config())

    with pytest.raises(RuntimeError, match="fence mismatch"):
        session.touch(stale, DuplexLeaseActivity.APPEND)
    with pytest.raises(RuntimeError, match="fence mismatch"):
        session.detach(stale)
    with pytest.raises(RuntimeError, match="fence mismatch"):
        session.resume(stale, expected_lease_generation=0)


def test_disabled_idle_expiry_never_collects_session() -> None:
    clock = FakeMonotonicClock(0.0)
    manager = DuplexSessionRuntimeManager(clock=clock)
    manager.open_session(DuplexFence("sid-no-expiry"), lease_config=_lease_config(idle_ttl_s=None))

    clock.advance(1_000_000.0)

    assert manager.collect_expired() == []


def test_detached_session_expires_at_disconnect_grace_when_idle_ttl_is_disabled() -> None:
    clock = FakeMonotonicClock(0.0)
    manager = DuplexSessionRuntimeManager(clock=clock)
    fence = DuplexFence("sid-disconnect-grace")
    session = manager.open_session(fence, lease_config=_lease_config(idle_ttl_s=None))

    session.detach(fence)
    clock.advance(29.0)
    assert manager.collect_expired() == []

    clock.advance(1.0)
    expired = manager.collect_expired()

    assert [item.session_id for item in expired] == [fence.session_id]
    assert expired[0].reason == "disconnect_grace_expired"


def test_runtime_manager_enforces_server_owned_session_admission() -> None:
    manager = DuplexSessionRuntimeManager(max_sessions=2)
    manager.open_session(DuplexFence("sid-admission-a"))
    manager.open_session(DuplexFence("sid-admission-b"))

    with pytest.raises(RuntimeError, match="duplex_session_capacity_exhausted"):
        manager.open_session(DuplexFence("sid-admission-c"))


def test_completed_append_cache_is_bounded() -> None:
    manager = DuplexSessionRuntimeManager(completed_append_limit=2)
    fence = DuplexFence("sid-completed-cache")
    session = manager.open_session(fence)

    for index in range(3):
        session.record_completed_append(
            f"operation-{index}",
            fence=fence,
            mode=DuplexInputMode.TURN_COMMIT_ONLY,
            final=True,
            stage_results=[{"index": index}],
        )

    assert list(session.completed_appends) == ["operation-1", "operation-2"]

    session.accept_fence(DuplexFence(fence.session_id, epoch=1))
    assert session.completed_appends == {}


def test_lease_config_and_expiry_record_are_immutable() -> None:
    config = _lease_config()
    with pytest.raises(FrozenInstanceError):
        config.idle_ttl_s = 1.0  # type: ignore[misc]

    clock = FakeMonotonicClock(0.0)
    manager = DuplexSessionRuntimeManager(clock=clock)
    manager.open_session(DuplexFence("sid-record"), lease_config=_lease_config(idle_ttl_s=1.0))
    clock.advance(2.0)
    record = manager.collect_expired()[0]
    with pytest.raises(FrozenInstanceError):
        record.reason = "changed"  # type: ignore[misc]
