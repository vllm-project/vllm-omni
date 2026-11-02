# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Bounded auxiliary accounting primitives, without live scheduling or policy.

A session owner can retain one budget across live re-leases, and an engine
manager can share a permit pool across auxiliary consumers. Callers must wire
dispatch, activity, wakeup and successful cleanup to these primitives before
enabling them. Nothing in this module starts work or allocates prompt space.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

_MAX_COUNT = 2**31 - 1
_MAX_DEBITS = 4096
_MAX_PERMITS = 4096


def _token_count(value: object) -> bool:
    return type(value) is int and 1 <= value <= _MAX_COUNT


def _finite_nonnegative(value: object) -> bool:
    if type(value) is not int and type(value) is not float:
        return False
    try:
        return math.isfinite(value) and value >= 0
    except OverflowError:
        return False


@dataclass(frozen=True)
class DuplexAuxiliaryBudgetLimits:
    """Internal experimental limits, not deployment configuration defaults."""

    max_input_tokens: int = 4096
    max_output_tokens: int = 1024
    window_s: float = 60.0
    cooldown_s: float = 5.0
    idle_s: float = 1.0

    def __post_init__(self) -> None:
        if (
            not _token_count(self.max_input_tokens)
            or not _token_count(self.max_output_tokens)
            or not _finite_nonnegative(self.window_s)
            or self.window_s == 0
            or not _finite_nonnegative(self.cooldown_s)
            or not _finite_nonnegative(self.idle_s)
        ):
            raise ValueError("invalid_auxiliary_budget_limits")


@dataclass(frozen=True, eq=False)
class DuplexAuxiliaryReservation:
    _owner: object = field(repr=False)
    input_tokens: int
    output_tokens: int


class DuplexAuxiliaryBudget:
    """One held reservation and a bounded rolling ledger of submitted costs.

    All calls belong on the same owner loop. A reservation has no expiry:
    only settlement can release it. Submitted costs expire normally after
    the rolling window, regardless of cancellation, failure or session close.
    """

    def __init__(self, limits: DuplexAuxiliaryBudgetLimits, *, now: float) -> None:
        if not isinstance(limits, DuplexAuxiliaryBudgetLimits):
            raise ValueError("invalid_auxiliary_budget_limits")
        if not _finite_nonnegative(now):
            raise ValueError("invalid_auxiliary_clock")
        self._limits = limits
        self._owner = object()
        self._now = float(now)
        self._last_activity = self._now
        self._last_submitted: float | None = None
        self._debits: deque[tuple[float, int, int]] = deque()
        self._inputs = 0
        self._outputs = 0
        self._pending: DuplexAuxiliaryReservation | None = None
        self._closed = False

    def _observe(self, now: float) -> float:
        if not _finite_nonnegative(now) or now < self._now:
            raise ValueError("invalid_auxiliary_clock")
        self._now = float(now)
        # Use the same deadline arithmetic as next_ready_at: subtraction can
        # round differently and leave an old debit alive at its returned time.
        while self._debits and self._debits[0][0] + self._limits.window_s <= self._now:
            _, inputs, outputs = self._debits.popleft()
            self._inputs -= inputs
            self._outputs -= outputs
        return self._now

    def next_ready_at(self, input_tokens: int, output_tokens: int, *, now: float) -> float | None:
        """Return an eligibility time; None means no time-only wakeup can help."""
        if not _token_count(input_tokens) or not _token_count(output_tokens):
            raise ValueError("invalid_auxiliary_token_cost")
        stamp = self._observe(now)
        limits = self._limits
        if (
            self._closed
            or self._pending is not None
            or input_tokens > limits.max_input_tokens
            or output_tokens > limits.max_output_tokens
        ):
            return None
        ready = max(stamp, self._last_activity + limits.idle_s)
        if self._last_submitted is not None:
            ready = max(ready, self._last_submitted + limits.cooldown_s)
        inputs, outputs, count = self._inputs, self._outputs, len(self._debits)

        def fits() -> bool:
            return (
                inputs + input_tokens <= limits.max_input_tokens
                and outputs + output_tokens <= limits.max_output_tokens
                and count < _MAX_DEBITS
            )

        if not fits():
            for submitted_at, charged_input, charged_output in self._debits:
                inputs -= charged_input
                outputs -= charged_output
                count -= 1
                if fits():
                    ready = max(ready, submitted_at + limits.window_s)
                    break
        return ready if math.isfinite(ready) else None

    def try_reserve(self, input_tokens: int, output_tokens: int, *, now: float) -> DuplexAuxiliaryReservation | None:
        ready = self.next_ready_at(input_tokens, output_tokens, now=now)
        if ready is None or ready > self._now:
            return None
        # A projected wakeup is not authority to overbook the current ledger.
        if (
            self._inputs + input_tokens > self._limits.max_input_tokens
            or self._outputs + output_tokens > self._limits.max_output_tokens
            or len(self._debits) >= _MAX_DEBITS
        ):
            return None
        reservation = DuplexAuxiliaryReservation(self._owner, input_tokens, output_tokens)
        self._pending = reservation
        return reservation

    def settle(self, reservation: DuplexAuxiliaryReservation, *, submitted: bool, now: float) -> bool:
        """Commit at dispatch, or refund before dispatch, at most once."""
        if not isinstance(reservation, DuplexAuxiliaryReservation) or reservation._owner is not self._owner:
            raise ValueError("invalid_auxiliary_reservation")
        if type(submitted) is not bool:
            raise ValueError("invalid_auxiliary_submission_flag")
        stamp = self._observe(now)
        if reservation is not self._pending:
            return False
        self._pending = None
        if submitted:
            self._debits.append((stamp, reservation.input_tokens, reservation.output_tokens))
            self._inputs += reservation.input_tokens
            self._outputs += reservation.output_tokens
            self._last_submitted = stamp
        return True

    def note_activity(self, *, now: float) -> None:
        self._last_activity = self._observe(now)

    def usage(self, *, now: float) -> tuple[int, int]:
        """Include held reservations so other callers cannot overbook them."""
        self._observe(now)
        pending = self._pending
        return (
            self._inputs + (pending.input_tokens if pending is not None else 0),
            self._outputs + (pending.output_tokens if pending is not None else 0),
        )

    def close(self, *, now: float) -> None:
        self._observe(now)
        self._closed = True
        self._pending = None


@dataclass(frozen=True, eq=False)
class DuplexAuxiliaryPermit:
    _owner: object = field(repr=False)


class DuplexAuxiliaryPermitPool:
    """Nonblocking manager-local admission, with no waiter queue.

    A submitted permit remains held after cancellation or failed cleanup.
    Only its owner reporting successful cleanup can release that resource.
    Closing the pool forbids new dispatch but does not pretend cleanup ran.
    """

    def __init__(self, *, capacity: int = 1) -> None:
        if type(capacity) is not int or not 1 <= capacity <= _MAX_PERMITS:
            raise ValueError("invalid_auxiliary_permit_capacity")
        self._capacity = capacity
        self._owner = object()
        self._held: dict[DuplexAuxiliaryPermit, bool] = {}
        self._closed = False

    @property
    def active_count(self) -> int:
        return len(self._held)

    def _validate(self, permit: DuplexAuxiliaryPermit) -> None:
        if not isinstance(permit, DuplexAuxiliaryPermit) or permit._owner is not self._owner:
            raise ValueError("invalid_auxiliary_permit")

    def try_acquire(self) -> DuplexAuxiliaryPermit | None:
        if self._closed or len(self._held) >= self._capacity:
            return None
        permit = DuplexAuxiliaryPermit(self._owner)
        self._held[permit] = False
        return permit

    def mark_submitted(self, permit: DuplexAuxiliaryPermit) -> bool:
        self._validate(permit)
        if self._closed or permit not in self._held or self._held[permit]:
            return False
        self._held[permit] = True
        return True

    def release(self, permit: DuplexAuxiliaryPermit, *, cleanup_succeeded: bool) -> bool:
        self._validate(permit)
        if type(cleanup_succeeded) is not bool:
            raise ValueError("invalid_auxiliary_cleanup_flag")
        if permit not in self._held or (self._held[permit] and not cleanup_succeeded):
            return False
        del self._held[permit]
        return True

    def close(self) -> None:
        self._closed = True
