# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from vllm_omni.engine.duplex.lease import (
    DuplexLeaseActivity,
    DuplexLeaseConfig,
)
from vllm_omni.engine.duplex.messages import DuplexFence
from vllm_omni.engine.duplex.runtime import DuplexInputMode
from vllm_omni.engine.duplex.session import (
    DuplexSessionRuntimeManager,
    duplex_append_fingerprint,
)

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


@pytest.mark.parametrize("retained", [4, 5])
def test_session_validates_rollover_budget_before_creating_generation_state(retained: int) -> None:
    manager = DuplexSessionRuntimeManager(recovery_max_replay_tokens=4, rollover_retain_tokens=retained)
    with pytest.raises(ValueError, match="retain tokens must be smaller"):
        manager.open_session(DuplexFence("bad-rollover"), lease_config=_lease_config())
    assert manager.session_count == 0


def test_disabled_rollover_does_not_require_a_smaller_retained_window() -> None:
    manager = DuplexSessionRuntimeManager(
        recovery_max_replay_tokens=4, rollover_retain_tokens=4, rollover_trigger_fraction=0.0
    )
    assert manager.open_session(DuplexFence("no-rollover"), lease_config=_lease_config()) is not None


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


def test_completed_append_fingerprint_is_order_stable_and_rejects_payload_reuse() -> None:
    manager = DuplexSessionRuntimeManager()
    fence = DuplexFence("sid-fingerprint")
    session = manager.open_session(fence)
    first = duplex_append_fingerprint(
        mode=DuplexInputMode.TURN_COMMIT_ONLY,
        payload={"audio": "abc", "nested": {"b": 2, "a": 1}},
        final=True,
        config_generation=0,
    )
    reordered = duplex_append_fingerprint(
        mode=DuplexInputMode.TURN_COMMIT_ONLY,
        payload={"nested": {"a": 1, "b": 2}, "audio": "abc"},
        final=True,
        config_generation=0,
    )
    changed = duplex_append_fingerprint(
        mode=DuplexInputMode.TURN_COMMIT_ONLY,
        payload={"audio": "different", "nested": {"a": 1, "b": 2}},
        final=True,
        config_generation=0,
    )
    assert first == reordered
    assert first != changed
    session.record_completed_append(
        "operation",
        fence=fence,
        mode=DuplexInputMode.TURN_COMMIT_ONLY,
        final=True,
        stage_results=[{"ok": True}],
        operation_fingerprint=first,
        config_generation=0,
    )

    assert session.completed_append(
        "operation",
        fence=fence,
        mode=DuplexInputMode.TURN_COMMIT_ONLY,
        final=True,
        operation_fingerprint=reordered,
        config_generation=0,
    ) == [{"ok": True}]
    with pytest.raises(ValueError, match="reused with different metadata"):
        session.completed_append(
            "operation",
            fence=fence,
            mode=DuplexInputMode.TURN_COMMIT_ONLY,
            final=True,
            operation_fingerprint=changed,
            config_generation=0,
        )


def test_append_fingerprint_covers_materialized_model_metadata() -> None:
    common = {
        "mode": DuplexInputMode.APPEND_AUDIO_CHUNK,
        "payload": {"audio": b"same-pcm", "is_speech": True},
        "final": False,
        "config_generation": 3,
    }
    first = duplex_append_fingerprint(
        **common,
        request_metadata={
            "prompt": {
                "prompt_token_ids": [10, 11],
                "model_intermediate_buffer": {"audio_features": b"features-a"},
            },
            "sampling_params": {"temperature": 0.0, "stop_token_ids": {151645, 151646}},
        },
    )
    reordered = duplex_append_fingerprint(
        **common,
        request_metadata={
            "sampling_params": {"stop_token_ids": {151646, 151645}, "temperature": 0.0},
            "prompt": {
                "model_intermediate_buffer": {"audio_features": bytearray(b"features-a")},
                "prompt_token_ids": [10, 11],
            },
        },
    )
    changed_metadata = duplex_append_fingerprint(
        **common,
        request_metadata={
            "prompt": {
                "prompt_token_ids": [10, 11],
                "model_intermediate_buffer": {"audio_features": b"features-b"},
            },
            "sampling_params": {"temperature": 0.0, "stop_token_ids": {151645, 151646}},
        },
    )

    assert first == reordered
    assert first != changed_metadata


def test_recovery_journal_enforces_token_and_byte_hard_limits() -> None:
    manager = DuplexSessionRuntimeManager(
        recovery_max_replay_tokens=3,
        recovery_max_replay_bytes=1024,
        rollover_trigger_fraction=0,
        rollover_retain_tokens=1,
    )
    session = manager.open_session(DuplexFence("sid-journal-limits"))
    fingerprint = b"f" * 32
    first = session.prepare_replay_append(
        operation_id="op-1",
        operation_fingerprint=fingerprint,
        prompt={"prompt_token_ids": [1, 2], "model_intermediate_buffer": {}},
    )
    session.record_replay_append(first)

    second = session.prepare_replay_append(
        operation_id="op-2",
        operation_fingerprint=fingerprint,
        prompt={"prompt_token_ids": [3, 4], "model_intermediate_buffer": {}},
    )
    assert session.replay_append_would_overflow(second) is True
    with pytest.raises(RuntimeError, match="duplex_recovery_journal_capacity_exhausted"):
        session.record_replay_append(second)

    with pytest.raises(RuntimeError, match="duplex_recovery_journal_entry_too_large"):
        session.prepare_replay_append(
            operation_id="op-large",
            operation_fingerprint=fingerprint,
            prompt={"prompt_token_ids": [1], "audio": b"x" * 2048},
        )


def test_recovery_journal_is_snapshot_bounded_and_cleared_by_epoch_change() -> None:
    manager = DuplexSessionRuntimeManager(
        recovery_max_replay_tokens=8,
        recovery_max_replay_bytes=4096,
        rollover_trigger_fraction=0,
        rollover_retain_tokens=3,
    )
    fence = DuplexFence("sid-journal-snapshot")
    session = manager.open_session(fence)
    source_prompt = {
        "prompt_token_ids": [1, 2],
        "model_intermediate_buffer": {"duplex": {"payload": {"audio": b"pcm"}}},
    }
    append = session.prepare_replay_append(
        operation_id="op-snapshot",
        operation_fingerprint=b"fingerprint",
        prompt=source_prompt,
    )
    assert isinstance(source_prompt["prompt_token_ids"], list)
    assert isinstance(source_prompt["model_intermediate_buffer"], dict)
    source_prompt["prompt_token_ids"].append(99)
    source_prompt["model_intermediate_buffer"]["duplex"]["payload"]["audio"] = b"changed"
    session.record_replay_append(append)

    assert append.prompt["prompt_token_ids"] == [1, 2]
    with pytest.raises(TypeError):
        append.prompt["new"] = "forbidden"  # type: ignore[index]
    assert manager.iter_sessions() == (session,)

    session.accept_fence(DuplexFence(fence.session_id, epoch=1))
    assert session.replay_appends == []
    assert session.replay_token_count == 0
    assert session.replay_byte_count == 0


def test_recovery_journal_rejects_reference_cycles_instead_of_undercounting() -> None:
    manager = DuplexSessionRuntimeManager()
    session = manager.open_session(DuplexFence("sid-journal-cycle"))
    cyclic: dict[str, object] = {"prompt_token_ids": [1]}
    cyclic["cycle"] = cyclic

    with pytest.raises(ValueError, match="reference cycle"):
        session.prepare_replay_append(
            operation_id="op-cycle",
            operation_fingerprint=b"fingerprint",
            prompt=cyclic,
        )


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
