# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The session-owned tool-call ledger and its deadline, without any sleeping.

The ledger is pure bookkeeping behind two seams -- a monotonic clock and a
timer scheduler -- so every ordering it decides (a result that beats its
deadline versus one that misses it, a cancellation racing an acceptance, a
timer firing after the call already settled) is enumerated here by hand.
``FakeClock``/``FakeTimerScheduler`` are those two seams; the runner-level
tests in ``test_session_runner_nemotron_voicechat.py`` drive the same decisions
through the real mailbox and the real plugin.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest

from tests.engine.duplex.test_session_manager import FakeClock
from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex.session.tool_calls import (
    TERMINAL_TOOL_CALL_STATES,
    ToolCallAcceptance,
    ToolCallLedger,
    ToolCallOrigin,
    ToolCallRegistrationError,
    ToolCallRejection,
    ToolCallState,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# --------------------------------------------------------------------------- #
# Seams                                                                       #
# --------------------------------------------------------------------------- #


class _FakeTimer:
    def __init__(self, entry: list[object]) -> None:
        self._entry = entry

    def cancel(self) -> None:
        self._entry[1] = True


class FakeTimerScheduler:
    """Records armed deadlines instead of waiting for them.

    ``fire_all`` delivers the callbacks that are still armed, which is what a
    real ``asyncio`` timer does when its loop reaches the deadline.
    """

    def __init__(self) -> None:
        #: [delay_s, cancelled, callback]
        self.timers: list[list[object]] = []

    def call_later(self, delay_s: float, callback: Callable[[], None]) -> _FakeTimer:
        entry: list[object] = [delay_s, False, callback]
        self.timers.append(entry)
        return _FakeTimer(entry)

    def pending_delays(self) -> list[float]:
        return [cast(float, entry[0]) for entry in self.timers if not entry[1]]

    def fire_all(self) -> None:
        armed = [entry for entry in self.timers if not entry[1]]
        for entry in armed:
            entry[1] = True
            callback = entry[2]
            assert callable(callback)
            callback()


def _origin(call_id: str, *, epoch: int = 0, turn_id: int = 0, response_id: str = "resp_1") -> ToolCallOrigin:
    return ToolCallOrigin(
        call_id=call_id,
        epoch=epoch,
        turn_id=turn_id,
        response_id=response_id,
        item_id=f"item_{call_id}",
    )


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(100.0)


@pytest.fixture
def scheduler() -> FakeTimerScheduler:
    return FakeTimerScheduler()


@pytest.fixture
def expired_calls() -> list[str]:
    return []


@pytest.fixture
def ledger(clock: FakeClock, scheduler: FakeTimerScheduler, expired_calls: list[str]) -> ToolCallLedger:
    led = ToolCallLedger(
        max_calls=8,
        clock=clock,
        timeout_s=5.0,
        on_expire=expired_calls.append,
    )
    led.set_scheduler(scheduler)
    return led


# --------------------------------------------------------------------------- #
# Exposure                                                                    #
# --------------------------------------------------------------------------- #


def test_registration_records_the_origin_and_arms_one_deadline(ledger: ToolCallLedger, scheduler: FakeTimerScheduler):
    call = ledger.register(_origin("call_a", epoch=3, turn_id=7))

    assert call.state is ToolCallState.PENDING
    assert call.deadline == 105.0
    assert scheduler.pending_delays() == [5.0]
    assert ledger.record("call_a") is call
    assert ledger.record("call_missing") is None
    assert ledger.record(None) is None


def test_duplicate_registration_never_rebinds_the_record(ledger: ToolCallLedger):
    ledger.register(_origin("call_a", response_id="resp_first"))

    with pytest.raises(ToolCallRegistrationError) as excinfo:
        ledger.register(_origin("call_a", response_id="resp_second"))

    assert excinfo.value.code == "tool_call_registration_failed"
    assert ledger.record("call_a").origin.response_id == "resp_first"
    assert len(ledger) == 1


def test_budget_is_exhausted_explicitly_and_never_evicts(clock: FakeClock):
    ledger = ToolCallLedger(max_calls=2, clock=clock)
    ledger.register(_origin("call_a"))
    ledger.register(_origin("call_b"))

    with pytest.raises(ToolCallRegistrationError) as excinfo:
        ledger.register(_origin("call_c"))

    assert excinfo.value.code == "resource_exhausted"
    assert len(ledger) == 2
    assert "call_c" not in ledger


def test_terminal_records_are_tombstones_until_close(ledger: ToolCallLedger):
    ledger.register(_origin("call_a"))
    acceptance = ledger.begin_accept("call_a")
    assert isinstance(acceptance, ToolCallAcceptance)
    assert ledger.commit_accept(acceptance, epoch=0) is True

    # A *new* call that reuses the id is refused rather than rebinding, so a
    # late result can never answer the newer call.
    with pytest.raises(ToolCallRegistrationError):
        ledger.register(_origin("call_a", response_id="resp_2"))

    assert ledger.begin_accept("call_a") is ToolCallRejection.NOT_PENDING
    assert ledger.record("call_a").state is ToolCallState.ACCEPTED
    assert ledger.record("call_a").state in TERMINAL_TOOL_CALL_STATES


# --------------------------------------------------------------------------- #
# Deadline                                                                    #
# --------------------------------------------------------------------------- #


def test_a_result_strictly_before_the_deadline_is_accepted(ledger: ToolCallLedger, clock: FakeClock):
    ledger.register(_origin("call_a"))
    clock.advance(4.999)

    acceptance = ledger.begin_accept("call_a")

    assert isinstance(acceptance, ToolCallAcceptance)
    assert acceptance.epoch == 0 and acceptance.turn_id == 0
    assert ledger.commit_accept(acceptance, epoch=0) is True
    assert ledger.record("call_a").state is ToolCallState.ACCEPTED
    assert ledger.record("call_a").settled_epoch == 0


def test_a_result_at_the_deadline_is_timed_out(ledger: ToolCallLedger, clock: FakeClock):
    ledger.register(_origin("call_a"))
    clock.advance(5.0)

    assert ledger.begin_accept("call_a") is ToolCallRejection.EXPIRED
    assert ledger.record("call_a").state is ToolCallState.TIMED_OUT


def test_a_disabled_deadline_never_expires(clock: FakeClock, scheduler: FakeTimerScheduler):
    ledger = ToolCallLedger(max_calls=4, clock=clock, timeout_s=None)
    ledger.set_scheduler(scheduler)
    ledger.register(_origin("call_a"))
    clock.advance(10_000.0)

    assert scheduler.pending_delays() == []
    assert ledger.expire("call_a") is False
    assert isinstance(ledger.begin_accept("call_a"), ToolCallAcceptance)


def test_the_timer_callback_only_reports_the_identity(
    ledger: ToolCallLedger, scheduler: FakeTimerScheduler, expired_calls: list[str], clock: FakeClock
):
    ledger.register(_origin("call_a"))

    scheduler.fire_all()

    # Firing enqueues; the mailbox decides. Nothing moved on its own.
    assert expired_calls == ["call_a"]
    assert ledger.record("call_a").state is ToolCallState.PENDING
    assert ledger.expire("call_a") is False  # the clock has not reached the deadline
    clock.advance(5.0)
    assert ledger.expire("call_a") is True
    assert ledger.record("call_a").state is ToolCallState.TIMED_OUT


def test_expiry_is_idempotent_and_a_late_callback_is_inert(ledger: ToolCallLedger, clock: FakeClock):
    ledger.register(_origin("call_a"))
    clock.advance(5.0)

    assert ledger.expire("call_a") is True
    assert ledger.expire("call_a") is False
    assert ledger.expire("call_a", now=clock.value + 60.0) is False
    assert ledger.commit_accept(ToolCallAcceptance("call_a", 0, 0, None), epoch=0) is False
    assert ledger.record("call_a").state is ToolCallState.TIMED_OUT


def test_settlement_cancels_the_armed_timer(ledger: ToolCallLedger, scheduler: FakeTimerScheduler):
    ledger.register(_origin("call_a"))

    acceptance = ledger.begin_accept("call_a")
    assert isinstance(acceptance, ToolCallAcceptance)
    assert ledger.commit_accept(acceptance, epoch=0) is True

    assert scheduler.pending_delays() == []
    scheduler.fire_all()  # nothing was armed, so nothing fires


# --------------------------------------------------------------------------- #
# Cancellation                                                                #
# --------------------------------------------------------------------------- #


def test_epoch_invalidation_cancels_pending_calls_and_keeps_tombstones(
    ledger: ToolCallLedger, scheduler: FakeTimerScheduler
):
    ledger.register(_origin("call_old", epoch=0))
    settled = ledger.begin_accept("call_old")
    assert isinstance(settled, ToolCallAcceptance)
    ledger.commit_accept(settled, epoch=0)
    ledger.register(_origin("call_pending", epoch=0))
    ledger.register(_origin("call_next_epoch", epoch=1))

    invalidated = ledger.invalidate_epoch(1)

    assert [origin.call_id for origin in invalidated] == ["call_pending"]
    assert ledger.record("call_pending").state is ToolCallState.CANCELLED
    # The older epoch's accepted record is a tombstone, not an eviction.
    assert ledger.record("call_old").state is ToolCallState.ACCEPTED
    assert ledger.record("call_next_epoch").state is ToolCallState.PENDING
    assert scheduler.pending_delays() == [5.0]
    assert ledger.begin_accept("call_pending") is ToolCallRejection.NOT_PENDING


def test_revalidation_refuses_a_call_whose_epoch_advanced_while_awaiting(ledger: ToolCallLedger):
    ledger.register(_origin("call_a", epoch=0, turn_id=2))
    acceptance = ledger.begin_accept("call_a")
    assert isinstance(acceptance, ToolCallAcceptance)

    assert ledger.revalidate(acceptance, epoch=1) is ToolCallRejection.NOT_PENDING
    assert ledger.record("call_a").state is ToolCallState.CANCELLED
    assert ledger.commit_accept(acceptance, epoch=1) is False


def test_revalidation_refuses_a_deadline_crossed_while_awaiting(ledger: ToolCallLedger, clock: FakeClock):
    ledger.register(_origin("call_a"))
    acceptance = ledger.begin_accept("call_a")
    assert isinstance(acceptance, ToolCallAcceptance)

    clock.advance(5.0)

    assert ledger.revalidate(acceptance, epoch=0) is ToolCallRejection.EXPIRED
    assert ledger.record("call_a").state is ToolCallState.TIMED_OUT
    assert ledger.commit_accept(acceptance, epoch=0) is False


def test_revalidation_is_clean_before_the_deadline(ledger: ToolCallLedger):
    ledger.register(_origin("call_a", epoch=4, turn_id=1))
    acceptance = ledger.begin_accept("call_a")
    assert isinstance(acceptance, ToolCallAcceptance)

    assert ledger.revalidate(acceptance, epoch=4) is None
    assert ledger.commit_accept(acceptance, epoch=4) is True


# --------------------------------------------------------------------------- #
# Teardown and budget config                                                  #
# --------------------------------------------------------------------------- #


def test_close_drops_records_and_disarms_timers(
    ledger: ToolCallLedger, scheduler: FakeTimerScheduler, expired_calls: list[str]
):
    ledger.register(_origin("call_a"))
    ledger.close()

    assert len(ledger) == 0
    assert scheduler.pending_delays() == []
    scheduler.fire_all()
    assert expired_calls == []
    assert ledger.expire("call_a") is False
    assert ledger.begin_accept("call_a") is ToolCallRejection.UNKNOWN
    with pytest.raises(ToolCallRegistrationError):
        ledger.register(_origin("call_b"))


def test_close_is_idempotent(ledger: ToolCallLedger, scheduler: FakeTimerScheduler):
    ledger.register(_origin("call_a"))
    ledger.close()
    ledger.close()

    assert len(ledger) == 0
    assert scheduler.pending_delays() == []


def test_runtime_config_requires_a_positive_integer_tool_call_budget():
    assert DuplexSessionRuntimeConfig().max_tool_calls_per_session == 4096

    with pytest.raises(ValueError, match="max_tool_calls_per_session must be positive"):
        DuplexSessionRuntimeConfig(max_tool_calls_per_session=0)
    with pytest.raises(ValueError, match="max_tool_calls_per_session must be positive"):
        DuplexSessionRuntimeConfig(max_tool_calls_per_session=-64)

    # A budget is a count: ``True <= 0`` and ``0.5 <= 0`` are both false, and
    # ``nan`` passes every comparison, so only an integer shape may reach the
    # ledger's ``int(max_calls)``.
    for budget in (True, 0.5, 4096.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="max_tool_calls_per_session must be an integer"):
            DuplexSessionRuntimeConfig(max_tool_calls_per_session=budget)
