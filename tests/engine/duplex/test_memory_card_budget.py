# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Resource mechanics with explicit clocks; no live policy or model requests."""

import math

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def budget(*, now=0.0, **limits):
    from vllm_omni.engine.duplex.session.auxiliary_budget import DuplexAuxiliaryBudget, DuplexAuxiliaryBudgetLimits

    return DuplexAuxiliaryBudget(DuplexAuxiliaryBudgetLimits(**limits), now=now)


def pool(capacity=1):
    from vllm_omni.engine.duplex.session.auxiliary_budget import DuplexAuxiliaryPermitPool

    return DuplexAuxiliaryPermitPool(capacity=capacity)


def submit(account, inputs=2048, outputs=512, *, now):
    reservation = account.try_reserve(inputs, outputs, now=now)
    assert reservation is not None
    assert account.settle(reservation, submitted=True, now=now)
    return reservation


def test_idle_and_cooldown_boundaries_use_submitted_start():
    account = budget()
    assert account.next_ready_at(2048, 512, now=0) == 1
    assert account.try_reserve(2048, 512, now=0.999) is None
    first = submit(account, now=1)
    assert account.usage(now=1) == (2048, 512)
    assert account.next_ready_at(2048, 512, now=1) == 6
    assert account.try_reserve(2048, 512, now=5.999) is None
    submit(account, now=6)
    assert account.usage(now=6) == (4096, 1024)
    assert not account.settle(first, submitted=False, now=6)
    assert account.next_ready_at(1, 1, now=6) == 61


def test_window_is_charged_at_submit_and_expires_at_exact_boundary():
    account = budget(cooldown_s=0)
    reservation = account.try_reserve(4096, 1024, now=1)
    assert reservation is not None
    # A held pre-submit reservation cannot expire and overbook capacity.
    assert account.try_reserve(1, 1, now=100) is None
    assert account.next_ready_at(1, 1, now=100) is None
    assert account.settle(reservation, submitted=True, now=100)
    assert account.next_ready_at(1, 1, now=100) == 160
    assert account.try_reserve(1, 1, now=159.999) is None
    assert account.usage(now=160) == (0, 0)
    assert account.try_reserve(4096, 1024, now=160) is not None


def test_pre_submit_refund_is_once_and_does_not_start_cooldown():
    account = budget()
    first = account.try_reserve(4096, 1024, now=1)
    assert account.usage(now=1) == (4096, 1024)
    assert account.settle(first, submitted=False, now=1)
    assert not account.settle(first, submitted=False, now=1)
    assert account.usage(now=1) == (0, 0)
    second = account.try_reserve(4096, 1024, now=1)
    assert second is not None and second is not first
    assert not account.settle(first, submitted=True, now=1)
    assert account.try_reserve(1, 1, now=1) is None
    assert account.settle(second, submitted=True, now=1)
    assert not account.settle(second, submitted=False, now=1)
    assert not account.settle(second, submitted=True, now=1)
    assert account.usage(now=1) == (4096, 1024)


def test_cancelled_failed_and_retry_submissions_share_one_allowance():
    account = budget()
    failed = submit(account, now=1)
    assert not account.settle(failed, submitted=False, now=2)
    retry = submit(account, now=6)
    assert not account.settle(retry, submitted=False, now=7)
    assert account.try_reserve(1, 1, now=11) is None
    assert account.usage(now=11) == (4096, 1024)


@pytest.mark.parametrize("costs", [((3000, 1), (1096, 1)), ((1, 900), (1, 124))])
def test_next_ready_waits_for_both_dimensions(costs):
    account = budget(cooldown_s=0)
    submit(account, *costs[0], now=1)
    submit(account, *costs[1], now=2)
    assert account.next_ready_at(4096, 1024, now=2) == 62
    assert account.try_reserve(4096, 1024, now=61) is None
    assert account.try_reserve(4096, 1024, now=62) is not None


def test_activity_changes_only_idle_grace_and_never_refunds_compute():
    account = budget()
    submit(account, now=1)
    account.note_activity(now=5.5)
    assert account.next_ready_at(2048, 512, now=5.5) == 6.5
    assert account.try_reserve(2048, 512, now=6) is None
    submit(account, now=6.5)
    account.note_activity(now=10)
    assert account.usage(now=11) == (4096, 1024)
    assert account.next_ready_at(1, 1, now=11) == 61


def test_foreign_or_unknown_handles_cannot_change_a_budget():
    first, second = budget(), budget()
    handle = first.try_reserve(2048, 512, now=1)
    for foreign in (handle, object(), None):
        with pytest.raises(ValueError, match="invalid_auxiliary_reservation"):
            second.settle(foreign, submitted=False, now=1)
    assert first.usage(now=1) == (2048, 512)
    assert second.usage(now=1) == (0, 0)


def test_close_cancels_unsubmitted_reservation_but_preserves_spent_tokens():
    account = budget(cooldown_s=0)
    submit(account, now=1)
    held = account.try_reserve(2048, 512, now=1)
    account.close(now=1)
    account.close(now=1)
    assert account.usage(now=1) == (2048, 512)
    assert account.next_ready_at(1, 1, now=1) is None
    assert account.try_reserve(1, 1, now=1) is None
    assert not account.settle(held, submitted=True, now=1)


def test_unfittable_cost_is_deferred_without_reservation_or_timer():
    account = budget()
    for inputs, outputs in ((4097, 1), (1, 1025)):
        assert account.next_ready_at(inputs, outputs, now=1) is None
        assert account.try_reserve(inputs, outputs, now=1) is None
    assert account.usage(now=1) == (0, 0)
    assert account.try_reserve(4096, 1024, now=1) is not None


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.0, "1", None, 2**31])
@pytest.mark.parametrize("dimension", ["input", "output"])
def test_costs_are_strict_bounded_positive_integers(value, dimension):
    account = budget()
    costs = (value, 1) if dimension == "input" else (1, value)
    for method in (account.try_reserve, account.next_ready_at):
        with pytest.raises(ValueError, match="invalid_auxiliary_token_cost"):
            method(*costs, now=1)
    assert account.usage(now=1) == (0, 0)


@pytest.mark.parametrize(
    "limits",
    [
        {"max_input_tokens": True},
        {"max_output_tokens": 0},
        {"max_input_tokens": 2**31},
        {"window_s": 0},
        {"window_s": math.inf},
        {"window_s": math.nan},
        {"cooldown_s": -1},
        {"idle_s": True},
        {"idle_s": "1"},
    ],
)
def test_invalid_limits_are_rejected(limits):
    with pytest.raises(ValueError, match="invalid_auxiliary_budget_limits"):
        budget(**limits)


@pytest.mark.parametrize("bad_now", [True, -1, math.nan, math.inf, "1", None])
def test_invalid_clocks_cannot_mutate_a_reservation(bad_now):
    account = budget()
    handle = account.try_reserve(4096, 1024, now=1)
    with pytest.raises(ValueError, match="invalid_auxiliary_clock"):
        account.settle(handle, submitted=False, now=bad_now)
    assert account.usage(now=1) == (4096, 1024)
    assert account.settle(handle, submitted=False, now=1)


def test_backwards_clock_is_rejected_without_refunding_or_rewinding():
    account = budget()
    submit(account, now=10)
    with pytest.raises(ValueError, match="invalid_auxiliary_clock"):
        account.try_reserve(1, 1, now=9)
    assert account.usage(now=10) == (2048, 512)
    assert account.next_ready_at(2048, 512, now=10) == 15


def test_submitted_debit_metadata_stays_bounded_even_with_large_token_limits():
    account = budget(max_input_tokens=10000, max_output_tokens=10000, cooldown_s=0, idle_s=0)
    for _ in range(4096):
        submit(account, 1, 1, now=0)
    assert account.try_reserve(1, 1, now=0) is None
    assert account.next_ready_at(1, 1, now=0) == 60
    assert account.usage(now=60) == (0, 0)
    assert account.try_reserve(1, 1, now=60) is not None


def test_shared_permit_is_nonblocking_and_cleanup_failure_retains_it():
    gate = pool()
    first = gate.try_acquire()
    assert first is not None and gate.active_count == 1
    assert gate.try_acquire() is None  # Another session can defer immediately.
    assert gate.mark_submitted(first)
    assert not gate.mark_submitted(first)
    assert not gate.release(first, cleanup_succeeded=False)
    assert gate.try_acquire() is None
    assert gate.release(first, cleanup_succeeded=True)
    assert gate.active_count == 0
    second = gate.try_acquire()
    assert second is not None and second is not first
    assert not gate.release(first, cleanup_succeeded=True)
    assert not gate.mark_submitted(first)
    assert gate.active_count == 1


def test_pre_submit_permit_release_needs_no_model_cleanup():
    gate = pool()
    first = gate.try_acquire()
    assert gate.release(first, cleanup_succeeded=False)
    assert gate.try_acquire() is not None


def test_permit_capacity_and_foreign_handle_ownership():
    gate, other = pool(2), pool()
    first, second = gate.try_acquire(), gate.try_acquire()
    assert first is not second and gate.active_count == 2
    assert gate.try_acquire() is None
    for handle in (first, object(), None):
        with pytest.raises(ValueError, match="invalid_auxiliary_permit"):
            other.release(handle, cleanup_succeeded=True)
        with pytest.raises(ValueError, match="invalid_auxiliary_permit"):
            other.mark_submitted(handle)
    assert gate.active_count == 2 and other.active_count == 0


def test_pool_close_never_releases_a_submitted_resource_before_cleanup():
    gate = pool()
    first = gate.try_acquire()
    gate.mark_submitted(first)
    gate.close()
    gate.close()
    assert gate.try_acquire() is None
    assert gate.active_count == 1
    assert not gate.release(first, cleanup_succeeded=False)
    assert gate.release(first, cleanup_succeeded=True)
    assert gate.active_count == 0 and gate.try_acquire() is None


@pytest.mark.parametrize("capacity", [True, 0, -1, 1.0, "1", 4097])
def test_invalid_permit_capacity_is_rejected(capacity):
    with pytest.raises(ValueError, match="invalid_auxiliary_permit_capacity"):
        pool(capacity)


def test_closed_pool_cannot_turn_an_unsubmitted_permit_into_a_forward():
    gate = pool()
    held = gate.try_acquire()
    gate.close()
    assert not gate.mark_submitted(held)
    assert gate.release(held, cleanup_succeeded=False)


@pytest.mark.parametrize("flag", [0, 1, "true", None])
def test_accounting_flags_are_strict_booleans(flag):
    account, gate = budget(), pool()
    reservation = account.try_reserve(4096, 1024, now=1)
    permit = gate.try_acquire()
    with pytest.raises(ValueError, match="invalid_auxiliary_submission_flag"):
        account.settle(reservation, submitted=flag, now=1)
    with pytest.raises(ValueError, match="invalid_auxiliary_cleanup_flag"):
        gate.release(permit, cleanup_succeeded=flag)
    assert account.usage(now=1) == (4096, 1024) and gate.active_count == 1


def test_two_accounts_share_one_permit_without_refunding_a_submitted_attempt():
    """Exercise the primitives' handoff contract, not production runtime wiring."""
    first, second, gate = budget(), budget(), pool()
    first_reservation = first.try_reserve(2048, 512, now=1)
    first_permit = gate.try_acquire()
    assert gate.mark_submitted(first_permit)
    assert first.settle(first_reservation, submitted=True, now=1)

    second_reservation = second.try_reserve(2048, 512, now=1)
    assert gate.try_acquire() is None
    assert second.settle(second_reservation, submitted=False, now=1)
    assert second.usage(now=1) == (0, 0)

    # Cancellation keeps the first debit and cleanup failure keeps its permit.
    assert not first.settle(first_reservation, submitted=False, now=2)
    assert not gate.release(first_permit, cleanup_succeeded=False)
    assert gate.try_acquire() is None
    assert gate.release(first_permit, cleanup_succeeded=True)

    second_reservation = second.try_reserve(2048, 512, now=2)
    second_permit = gate.try_acquire()
    assert second_permit is not None
    assert gate.mark_submitted(second_permit)
    assert second.settle(second_reservation, submitted=True, now=2)
    assert first.usage(now=2) == second.usage(now=2) == (2048, 512)
    assert gate.release(second_permit, cleanup_succeeded=True)
    assert gate.active_count == 0


@pytest.mark.parametrize("started", [0.4, 1.3, 2.6, 1000.3, 10_000_000.3])
def test_returned_fractional_deadline_expires_debits_without_overbooking(started):
    account = budget(idle_s=0, cooldown_s=0)
    submit(account, 4096, 1024, now=started)
    ready = account.next_ready_at(4096, 1024, now=started)
    assert ready == started + 60
    assert account.try_reserve(4096, 1024, now=math.nextafter(ready, -math.inf)) is None
    reservation = account.try_reserve(4096, 1024, now=ready)
    assert reservation is not None
    assert account.usage(now=ready) == (4096, 1024)
    assert account.settle(reservation, submitted=True, now=ready)
    assert account.usage(now=ready) == (4096, 1024)


def test_fractional_expiry_also_releases_the_metadata_capacity():
    account = budget(max_input_tokens=10000, max_output_tokens=10000, cooldown_s=0, idle_s=0)
    for _ in range(4096):
        submit(account, 1, 1, now=1.3)
    ready = account.next_ready_at(1, 1, now=1.3)
    reservation = account.try_reserve(1, 1, now=ready)
    assert reservation is not None
    assert account.usage(now=ready) == (1, 1)
    assert account.settle(reservation, submitted=True, now=ready)
    assert account.usage(now=ready) == (1, 1)
