# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Session-owned ledger for the function calls the runtime exposed to the client.

A native-duplex model can emit a function call at any point in its token
stream. The runtime never executes the call: it projects it to the client
(``function_call.done``) and the client later sends the result back as a
``function_call_output`` conversation item. Between those two moments the call
is *pending*, and everything that can happen to a pending call -- the client
answering it, the client answering it twice, the epoch being cancelled under it,
a deadline elapsing -- is a decision about one record.

This module is that record. It holds facts and monotonic decisions only: it
never touches the session, the plugin or the emitter, and it performs no
awaits, so every mutation stays on the mailbox worker and inside one command
(``docs/design/fullduplex.md``: "Cancel is atomic at the session", "single
writer per session").

The state machine is one-way::

    pending -> accepted | timed_out | cancelled

Why a ledger rather than the projection: eligibility used to be derived from
``RealtimeProjectionState.conversation_items`` -- conversation history, which is
history-wide and epoch-blind, so a result that arrived after a barge-in could
still be admitted against a call the session had already moved past. The
ledger also outlives the projection: terminal records are *tombstones*, kept
until the session closes, so a late ``function_call_output`` can never rebind
to a newer call that happens to reuse the same call id.

Bounded bookkeeping: one session may track at most
``DuplexSessionRuntimeConfig.max_tool_calls_per_session`` records (pending and
terminal together, until close). Exceeding the budget is an explicit failure,
never an eviction -- evicting a tombstone is exactly the reuse the tombstones
exist to prevent.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from vllm_omni.engine.duplex.session.timers import (
    LoopTimerScheduler,
    ScheduledTimer,
    TimerScheduler,
)

__all__ = [
    "LoopTimerScheduler",
    "TERMINAL_TOOL_CALL_STATES",
    "TimerScheduler",
    "ToolCall",
    "ToolCallAcceptance",
    "ToolCallLedger",
    "ToolCallOrigin",
    "ToolCallRegistrationError",
    "ToolCallRejection",
    "ToolCallSettlement",
    "ToolCallState",
    "ToolCallTimer",
]


class ToolCallState(str, Enum):
    """Where one runtime-emitted call ended up (single settlement, one-way)."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


TERMINAL_TOOL_CALL_STATES = frozenset({ToolCallState.ACCEPTED, ToolCallState.TIMED_OUT, ToolCallState.CANCELLED})


class ToolCallRejection(str, Enum):
    """Why ``begin_accept`` refused a ``function_call_output``."""

    #: No runtime-emitted call with this call id was ever exposed.
    UNKNOWN = "unknown"
    #: The call already settled (accepted, timed out or cancelled).
    NOT_PENDING = "not_pending"
    #: The call is still pending but its deadline has passed.
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class ToolCallOrigin:
    """The facts a later decision needs about one exposed call.

    ``epoch``/``turn_id`` are captured where the call was emitted, not looked
    up later: the point of recording them is to answer "was this still the same
    session generation when the result came back?".
    """

    call_id: str
    epoch: int
    turn_id: int
    response_id: str | None
    item_id: str
    name: str | None = None
    arguments: str | None = None


@dataclass(frozen=True, slots=True)
class ToolCallSettlement:
    """One new terminal decision, independent of model delivery or consumption.

    The owner's notification callback only queues this immutable fact. It must
    not perform model or wire effects from inside the ledger's mutation.
    """

    origin: ToolCallOrigin
    state: ToolCallState
    settled_epoch: int


@dataclass(slots=True)
class ToolCall:
    """One ledger record: the origin, the settlement and the deadline timer."""

    origin: ToolCallOrigin
    state: ToolCallState = ToolCallState.PENDING
    #: Monotonic instant the result must arrive *before*; ``None`` disables it.
    deadline: float | None = None
    #: Session epoch the record settled in (``None`` while pending).
    settled_epoch: int | None = None
    #: Scheduled expiry, owned by this record until it settles.
    timer: ToolCallTimer | None = None

    @property
    def is_pending(self) -> bool:
        return self.state is ToolCallState.PENDING

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_TOOL_CALL_STATES


@dataclass(frozen=True, slots=True)
class ToolCallAcceptance:
    """A pending call that passed eligibility, before anything was committed.

    Separating "may I accept this?" from "accept it" is what lets the caller do
    the awaiting part (the append tail) in between and re-check afterwards,
    without holding a half-applied admission.
    """

    call_id: str
    epoch: int
    turn_id: int
    deadline: float | None


#: The ledger's own name for the shared handle: an armed expiry it can cancel.
ToolCallTimer: TypeAlias = ScheduledTimer


class ToolCallRegistrationError(RuntimeError):
    """The session cannot track a call it is about to expose.

    Raised rather than swallowed: exposing an untrackable call would leave the
    client holding a call id the runtime will refuse, and rebinding or evicting
    a record to make room is what the tombstones exist to prevent.
    """

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


class ToolCallLedger:
    """Per-session function-call bookkeeping (see the module docstring)."""

    def __init__(
        self,
        *,
        max_calls: int,
        clock: Callable[[], float],
        timeout_s: float | None = None,
        on_expire: Callable[[str], None] | None = None,
        on_settle: Callable[[ToolCallSettlement], None] | None = None,
    ) -> None:
        # A record budget is a count, so only an integer will do: ``True`` would
        # become a limit of one and ``0.5`` a silently truncated zero.
        if isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls <= 0:
            raise ValueError("ToolCallLedger max_calls must be a positive integer")
        if timeout_s is not None and not (timeout_s > 0):
            raise ValueError("ToolCallLedger timeout_s must be positive or None")
        self._calls: dict[str, ToolCall] = {}
        self._max_calls = int(max_calls)
        self._clock = clock
        self._timeout_s = timeout_s
        #: Called by the timer callback with the call id only -- enqueueing, never
        #: deciding. The decision is the mailbox handler's (``expire``).
        self._on_expire = on_expire
        # Enqueue-only notification: the owner must not mutate this ledger or
        # perform plugin/model work synchronously from the callback.
        self._on_settle = on_settle
        self._scheduler: TimerScheduler | None = None
        self._closed = False

    # -- wiring --------------------------------------------------------- #

    def set_scheduler(self, scheduler: TimerScheduler | None) -> None:
        """Attach the loop-timer seam (called once the runner has a running loop)."""
        self._scheduler = scheduler

    # -- facts ---------------------------------------------------------- #

    def record(self, call_id: object) -> ToolCall | None:
        if not isinstance(call_id, str):
            return None
        return self._calls.get(call_id)

    def __len__(self) -> int:
        return len(self._calls)

    def __contains__(self, call_id: object) -> bool:
        return isinstance(call_id, str) and call_id in self._calls

    def origins(self) -> tuple[ToolCallOrigin, ...]:
        """Immutable identities for final plugin disposal when the session closes."""
        return tuple(call.origin for call in self._calls.values())

    # -- exposure ------------------------------------------------------- #

    def register(self, origin: ToolCallOrigin) -> ToolCall:
        """Record a call the runtime is about to project to the client.

        Must happen *before* the call is exposed: a call the session does not
        know about can never be answered, so the only safe failure is to refuse
        to expose it.
        """
        if self._closed:
            raise ToolCallRegistrationError(
                "the session is closing and cannot expose another function call",
                code="tool_call_registration_failed",
            )
        existing = self._calls.get(origin.call_id)
        if existing is not None:
            raise ToolCallRegistrationError(
                f"the runtime re-emitted function call {origin.call_id}; the session refuses to "
                "rebind it, so a late result cannot be matched to the wrong call",
                code="tool_call_registration_failed",
            )
        if len(self._calls) >= self._max_calls:
            raise ToolCallRegistrationError(
                f"this session already tracks {len(self._calls)} function calls "
                "(max_tool_calls_per_session); refusing to expose another one",
                code="resource_exhausted",
            )
        now = self._clock()
        call = ToolCall(origin=origin, deadline=None if self._timeout_s is None else now + self._timeout_s)
        self._calls[origin.call_id] = call
        self._arm(call, now=now)
        return call

    def _arm(self, call: ToolCall, *, now: float) -> None:
        scheduler = self._scheduler
        if call.deadline is None or scheduler is None:
            return
        if call.timer is not None:
            call.timer.cancel()
        call_id = call.origin.call_id
        call.timer = scheduler.call_later(max(0.0, call.deadline - now), lambda: self._fire(call_id))

    def _fire(self, call_id: str) -> None:
        """Timer callback: hand the identity to the owner and decide nothing."""
        if self._closed or self._on_expire is None:
            return
        self._on_expire(call_id)

    # -- settlement ----------------------------------------------------- #

    def begin_accept(
        self,
        call_id: object,
        *,
        now: float | None = None,
    ) -> ToolCallAcceptance | ToolCallRejection:
        """Check eligibility for a ``function_call_output`` without committing it.

        A pending call whose deadline has already passed settles as
        ``timed_out`` here and is reported as :attr:`ToolCallRejection.EXPIRED`:
        the deadline is the arbiter, and "the result was already in this
        mailbox" is not visible to it.
        """
        if not isinstance(call_id, str) or not call_id:
            return ToolCallRejection.UNKNOWN
        call = self._calls.get(call_id)
        if call is None:
            return ToolCallRejection.UNKNOWN
        if not call.is_pending:
            return ToolCallRejection.NOT_PENDING
        instant = self._clock() if now is None else now
        if call.deadline is not None and instant >= call.deadline:
            self._settle(call, ToolCallState.TIMED_OUT, epoch=None)
            return ToolCallRejection.EXPIRED
        return ToolCallAcceptance(
            call_id=call_id,
            epoch=call.origin.epoch,
            turn_id=call.origin.turn_id,
            deadline=call.deadline,
        )

    def revalidate(
        self,
        acceptance: ToolCallAcceptance,
        *,
        epoch: int,
        now: float | None = None,
    ) -> ToolCallRejection | None:
        """Re-check an acceptance after the awaiting part of the command.

        The session may have advanced its epoch (or the deadline may have
        passed) while the append tail was awaited; both make the acceptance
        stale, and the caller must not commit it. ``None`` means "still valid".
        """
        call = self._calls.get(acceptance.call_id)
        if call is None or not call.is_pending:
            return ToolCallRejection.NOT_PENDING
        if call.origin.epoch != epoch or call.origin.epoch != acceptance.epoch:
            # Defensive: ``invalidate_epoch`` already settles these, this only
            # covers an epoch bump that bypassed it.
            self._settle(call, ToolCallState.CANCELLED, epoch=epoch)
            return ToolCallRejection.NOT_PENDING
        instant = self._clock() if now is None else now
        if call.deadline is not None and instant >= call.deadline:
            self._settle(call, ToolCallState.TIMED_OUT, epoch=epoch)
            return ToolCallRejection.EXPIRED
        return None

    def commit_accept(self, acceptance: ToolCallAcceptance, *, epoch: int) -> bool:
        """Recheck and settle atomically; the caller must respect a refusal.

        Conversion can cross a deadline even without yielding to the loop.
        Keep the final eligibility check inside the settlement operation so a
        caller cannot install a candidate using an earlier successful check.
        """
        if self.revalidate(acceptance, epoch=epoch) is not None:
            return False
        call = self._calls.get(acceptance.call_id)
        if call is None or not call.is_pending:
            return False
        self._settle(call, ToolCallState.ACCEPTED, epoch=epoch)
        return True

    def expire(self, call_id: object, *, now: float | None = None) -> bool:
        """Settle a pending call as timed out at its deadline (idempotent).

        Returns whether *this* call settled, so the owner emits the timeout
        feedback exactly once; a late or duplicated timer callback is inert.
        """
        if not isinstance(call_id, str) or not call_id:
            return False
        call = self._calls.get(call_id)
        if call is None or not call.is_pending:
            return False
        # A session without a configured deadline has no expiry at all: this is
        # the only place that would otherwise time a call out unconditionally.
        if call.deadline is None:
            return False
        instant = self._clock() if now is None else now
        if instant < call.deadline:
            # Event loops may deliver a timer slightly early. That message
            # cannot consume the only wakeup for an otherwise pending call.
            self._arm(call, now=instant)
            return False
        self._settle(call, ToolCallState.TIMED_OUT, epoch=None)
        return True

    def invalidate_epoch(self, epoch: int) -> tuple[ToolCallOrigin, ...]:
        """Cancel every call left over from an earlier epoch.

        A cancellation advances the session epoch, which is the one place that
        means "everything the model already committed to is void". Terminal
        records are deliberately *not* touched: they are the tombstones that
        keep a late result from binding to a newer call.
        """
        invalidated: list[ToolCallOrigin] = []
        for call in self._calls.values():
            if call.is_pending and call.origin.epoch < epoch:
                invalidated.append(call.origin)
                self._settle(call, ToolCallState.CANCELLED, epoch=epoch)
        return tuple(invalidated)

    def _settle(self, call: ToolCall, state: ToolCallState, *, epoch: int | None) -> None:
        call.state = state
        call.settled_epoch = call.origin.epoch if epoch is None else epoch
        timer, call.timer = call.timer, None
        if timer is not None:
            timer.cancel()
        if self._on_settle is not None:
            self._on_settle(ToolCallSettlement(call.origin, state, call.settled_epoch))

    def close(self) -> None:
        """Drop every record and timer: the session is gone, nothing can settle.

        Idempotent, because every terminal path closes it: the public
        close/expire/shutdown, an output-backpressure close, and a runtime close
        raised from inside the model channel all funnel through
        ``DuplexSessionRunner._begin_close``, and several of them can run for one
        session.
        """
        if self._closed:
            return
        for call in self._calls.values():
            timer, call.timer = call.timer, None
            if timer is not None:
                timer.cancel()
        self._calls.clear()
        self._closed = True
