# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Hy-SimulMT contract scaffolding; no transport, queue, or model execution.

See docs/design/feature/hy_simulmt_streaming.md. One connection owner must
serialize admission and state transitions (without awaiting between them).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Protocol

SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class StreamingConfig:
    language: Literal["zh", "en"]
    cadence_ms: int = 1600
    reset_ms: int = 30_000
    max_pending_requests: int = 2

    def __post_init__(self) -> None:
        if self.language not in ("zh", "en"):
            raise ValueError("language must be explicitly set to zh or en")
        for name in ("cadence_ms", "reset_ms", "max_pending_requests"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if not 1 <= self.cadence_ms <= self.reset_ms <= 30_000:
            raise ValueError("require 1 <= cadence_ms <= reset_ms <= 30000")
        if not 1 <= self.max_pending_requests <= 8:
            raise ValueError("max_pending_requests must be between 1 and 8")

    @property
    def cadence_samples(self) -> int:
        return self.cadence_ms * (SAMPLE_RATE // 1000)

    @property
    def reset_samples(self) -> int:
        return self.reset_ms * (SAMPLE_RATE // 1000)


class Admission(Enum):
    ACCEPTED = "accepted"
    COALESCED = "coalesced"
    BACKPRESSURE = "backpressure"


class SessionPhase(Enum):
    OPEN = "open"
    FINALIZING = "finalizing"
    DRAINING = "draining"
    CLOSED = "closed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class Revision:
    """Identity is scoped to one connection; engine IDs also include its ID."""

    epoch: int
    segment_id: int
    revision_id: int
    status: Literal["partial", "final"]


@dataclass(frozen=True)
class Hypothesis:
    revision: Revision
    text: str  # Complete replacement, including when empty; never a delta.


@dataclass(frozen=True)
class InferenceRequest:
    request_id: str  # Globally unique engine request ID, allocated by the owner.
    revision: Revision
    pcm16: bytes  # Immutable complete segment, mono little-endian PCM16, 16 kHz.
    language: Literal["zh", "en"]
    context: str  # Bounded translation history; empty after an epoch change.


class InferenceBackend(Protocol):
    async def infer(self, request: InferenceRequest) -> str:
        """Return one complete hypothesis, or raise; no transport events."""
        ...

    async def cancel(self, request_id: str) -> None:
        """Idempotently abort engine work and await resource release.

        Safe for completed/unknown IDs. Late completion can still race with
        cancellation; the owner must also apply the Revision validity check.
        """
        ...


class PendingWorkPolicy(Protocol):
    def admit(self, status: Literal["partial", "final"]) -> Admission:
        """Atomically reserve capacity before allocating a revision.

        Count queued + running work, including cancellation awaiting cleanup.
        COALESCED replaces a queued partial for this segment only. A final
        may evict that partial but MUST return ACCEPTED or BACKPRESSURE.
        BACKPRESSURE leaves both the reservation and session unchanged.
        The owner enqueues the snapshot without an intervening await.
        """
        ...

    def release(self) -> None:
        """Release one reservation after completion or acknowledged abort.

        Replacing a queued partial reuses its reservation, so does not release
        it. The owner must release each reservation exactly once.
        """
        ...


class StreamingSession:
    """Pure metadata state machine. The future adapter owns audio and tasks.

    Finalization is a barrier: new input waits for the final to complete.
    This deliberately avoids parallel segments and unbounded final queues.
    """

    def __init__(self, config: StreamingConfig) -> None:
        self.config = config
        self.phase = SessionPhase.OPEN
        self.epoch = 0
        self.segment_id = 0
        self.segment_samples = 0
        self.context_samples = 0
        self._revision_id = 0
        self._published_revision_id = 0
        self._partial_samples = 0
        self._latest: Revision | None = None

    def _require_open(self) -> None:
        if self.phase is not SessionPhase.OPEN:
            raise ValueError(f"session is {self.phase.value}")

    @property
    def remaining_samples(self) -> int:
        return self.config.reset_samples - self.context_samples

    @property
    def partial_due(self) -> bool:
        return (
            self.phase is SessionPhase.OPEN
            and self.remaining_samples > 0
            and self.segment_samples - self._partial_samples >= self.config.cadence_samples
        )

    def append_samples(self, count: int) -> None:
        """Account for accepted audio; reject a crossing append atomically."""
        self._require_open()
        if type(count) is not int or not 0 < count <= self.remaining_samples:
            raise ValueError("append must contain positive samples within the reset boundary")
        self.segment_samples += count
        self.context_samples += count

    def _next_revision(self, status: Literal["partial", "final"]) -> Revision:
        self._revision_id += 1
        self._latest = Revision(self.epoch, self.segment_id, self._revision_id, status)
        return self._latest

    def partial(self, admission: Admission) -> Revision | None:
        self._require_open()
        if not self.partial_due:
            raise ValueError("partial is not due")
        if admission is Admission.BACKPRESSURE:
            return None
        self._partial_samples = self.segment_samples
        return self._next_revision("partial")

    def commit(self, admission: Admission, *, end_input: bool = False) -> Revision | None:
        """Seal this segment only after final capacity has been reserved.

        Empty commit is invalid. Empty end-of-input closes without inference.
        A backpressured final consumes neither audio nor a revision ID.
        """
        self._require_open()
        if not self.segment_samples:
            if not end_input:
                raise ValueError("cannot commit an empty segment")
            self.phase = SessionPhase.CLOSED
            return None
        if admission is Admission.COALESCED:
            raise ValueError("finalization requires explicit admission")
        if admission is Admission.BACKPRESSURE:
            return None
        self.phase = SessionPhase.DRAINING if end_input else SessionPhase.FINALIZING
        return self._next_revision("final")

    def is_current(self, revision: Revision) -> bool:
        """Check successes or failures without advancing the lifecycle.

        On a current backend failure, the owner cancels the session directly;
        it must not call complete() to check validity and fabricate a success.
        """
        if (
            self.phase in (SessionPhase.CLOSED, SessionPhase.CANCELLED)
            or revision.epoch != self.epoch
            or revision.segment_id != self.segment_id
            or not self._published_revision_id < revision.revision_id <= self._revision_id
        ):
            return False
        if self.phase is SessionPhase.OPEN:
            # A queued partial must not starve the running partial's progress.
            return revision.status == "partial"
        return revision == self._latest  # Final admission invalidates all partials.

    def complete(self, revision: Revision, text: str) -> Hypothesis | None:
        """Publish admitted revisions in increasing order, at most once."""
        if not self.is_current(revision):
            return None
        self._published_revision_id = revision.revision_id
        event = Hypothesis(revision, text)
        if revision.status == "final":
            self.segment_samples = 0
            self._partial_samples = 0
            self._revision_id = 0
            self._published_revision_id = 0
            self._latest = None
            self.segment_id += 1
            self.phase = SessionPhase.CLOSED if self.phase is SessionPhase.DRAINING else SessionPhase.OPEN
            if self.remaining_samples == 0:
                self.epoch += 1
                self.context_samples = 0
        return event

    def reset_context(self) -> None:
        """Reset at an empty boundary; never silently discard accepted audio."""
        self._require_open()
        if self.segment_samples:
            raise ValueError("finalize accepted audio before resetting context")
        self.epoch += 1
        self.context_samples = 0
        self._latest = None

    def cancel(self) -> None:
        """Invalidate results immediately; owner still MUST await backend aborts."""
        if self.phase in (SessionPhase.CLOSED, SessionPhase.CANCELLED):
            return
        self.phase = SessionPhase.CANCELLED
        self._latest = None
        self.segment_samples = 0
        self.context_samples = 0
