# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Session capacity reservation, scheduler side.

The roadmap splits KV capacity into two jobs (2.3): the scheduler admits a
request and reserves the capacity it needs, and the worker only uses what was
reserved. A session's retained history counts against its ongoing holding, so a
follow-up request reserves the *increment* rather than the whole window.

A worker never grows a session past its grant: when the demand does not fit, the
caller gets ``ReservationDeniedError`` and the request goes back to admission instead
of taking capacity from another session.

Admission is only half of the loop. After a batch the worker reports what the
session actually holds (``CapacityUsage``: logical versions plus worker bytes) and
the ledger reconciles it against the grant. A worker that holds more than it was
granted is an overcommit: the session fails closed instead of silently borrowing
capacity from another session.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vllm.logger import init_logger

logger = init_logger(__name__)


class ReservationDeniedError(RuntimeError):
    """The session's incremental demand does not fit the remaining budget."""


@dataclass(frozen=True)
class CapacityDemand:
    """What one request needs on top of what its session already holds."""

    versions: int
    bytes: int

    def __post_init__(self) -> None:
        if self.versions < 0 or self.bytes < 0:
            raise ValueError(f"capacity demand must be non-negative, got {self}")


@dataclass(frozen=True)
class CapacityGrant:
    """What the scheduler reserved for a session."""

    session_id: str
    versions: int
    bytes: int
    incremental: bool


@dataclass(frozen=True)
class CapacityUsage:
    """What a worker reports a session actually holds after a batch.

    ``versions`` is the logical retained-version count and ``bytes`` the
    worker-side footprint. Both are strategy-neutral: the paged layout and the
    Noisy ``VersionPool`` report the same pair even though their storage and
    logical block granularity differ.
    """

    versions: int
    bytes: int

    def __post_init__(self) -> None:
        if self.versions < 0 or self.bytes < 0:
            raise ValueError(f"capacity usage must be non-negative, got {self}")


class ReservationOvercommitError(ReservationDeniedError):
    """The worker holds more than the scheduler granted this session."""


@dataclass
class SessionCapacityLedger:
    """Per-session capacity reservations against one device budget."""

    budget_bytes: int
    _held: dict[str, int] = field(default_factory=dict)
    _grants: dict[str, CapacityGrant] = field(default_factory=dict)
    _usage: dict[str, CapacityUsage] = field(default_factory=dict)
    _overcommit: dict[str, CapacityUsage] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.budget_bytes < 0:
            raise ValueError(f"budget_bytes must be non-negative, got {self.budget_bytes}")

    def held_bytes(self, session_id: str) -> int:
        return self._held.get(session_id, 0)

    def used_bytes(self) -> int:
        return sum(self._held.values())

    def free_bytes(self) -> int:
        return self.budget_bytes - self.used_bytes()

    def reserve(self, session_id: str, demand: CapacityDemand) -> CapacityGrant:
        """Reserve the increment a request needs, or refuse it.

        A session that already holds enough for this request only needs a
        temporary grant: its history is counted once, not per request.
        """
        held = self._held.get(session_id, 0)
        incremental = demand.bytes > held
        needed = demand.bytes - held if incremental else 0
        if needed > self.free_bytes():
            raise ReservationDeniedError(
                f"session {session_id!r} needs {needed} more bytes but only "
                f"{self.free_bytes()} of {self.budget_bytes} are free"
            )
        if incremental:
            self._held[session_id] = demand.bytes
        grant = CapacityGrant(
            session_id=session_id,
            versions=demand.versions,
            bytes=self._held.get(session_id, demand.bytes),
            incremental=incremental,
        )
        self._grants[session_id] = grant
        return grant

    def grant(self, session_id: str) -> CapacityGrant | None:
        return self._grants.get(session_id)

    def reconcile(self, session_id: str, usage: CapacityUsage) -> CapacityUsage:
        """Accept a worker's reported usage, or fail closed on an overcommit.

        The grant is the contract; a worker that reports more than it was granted
        cannot be absorbed by another session's budget, so the ledger records the
        overcommit and raises. The caller decides the session's fate (this module
        keeps accounting only).
        """
        held = self._held.get(session_id, 0)
        if usage.bytes > held:
            self._overcommit[session_id] = usage
            raise ReservationOvercommitError(
                f"session {session_id!r} reports {usage.bytes} bytes but only {held} were granted"
            )
        self._usage[session_id] = usage
        self._overcommit.pop(session_id, None)
        return usage

    def usage(self, session_id: str) -> CapacityUsage | None:
        """Last usage the worker reported for ``session_id``."""
        return self._usage.get(session_id)

    def overcommit(self, session_id: str) -> CapacityUsage | None:
        """The overcommitted usage that failed this session, if any."""
        return self._overcommit.get(session_id)

    def release(self, session_id: str) -> int:
        """Give a session's holding back (reset, close, or a dead replica)."""
        released = self._held.pop(session_id, 0)
        self._grants.pop(session_id, None)
        self._usage.pop(session_id, None)
        self._overcommit.pop(session_id, None)
        return released
