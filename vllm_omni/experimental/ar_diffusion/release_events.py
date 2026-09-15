# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reverse lifecycle path: session releases a worker decided on its own.

LRU eviction and failed forwards originate inside the runner, where no
coordinator RPC reaches. Runners record them here; the coordinator drains and
acknowledges them at each request boundary. Records carry identity only, since
they cross the collective-RPC boundary.
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

# A release the coordinator asked for needs no event; it already knows.
COORDINATED_REASONS = frozenset({"coordinated_reset", "coordinated_close"})

DEFAULT_MAX_PENDING_RELEASE_EVENTS = 256


class SessionGenerationUnsupportedError(RuntimeError):
    """A participant cannot bind the coordinator's generation, so it cannot fence."""


@dataclass(frozen=True)
class ARDiffusionReleaseEvent:
    """One session release a worker performed without being asked.

    ``generation`` is 0 when the runner never saw one; a stale generation lets
    the coordinator drop an event for an already-replaced session.
    """

    event_id: str
    session_id: str
    reason: str
    generation: int = 0
    stage_id: int = -1
    cleanup_failed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "reason": self.reason,
            "generation": self.generation,
            "stage_id": self.stage_id,
            "cleanup_failed": self.cleanup_failed,
        }

    @classmethod
    def from_dict(cls, raw: object) -> ARDiffusionReleaseEvent:
        if isinstance(raw, ARDiffusionReleaseEvent):
            return raw
        if not isinstance(raw, dict):
            raise TypeError(f"AR-Diffusion release event must be a dict, got {type(raw).__name__}.")
        event_id = str(raw.get("event_id") or "")
        session_id = str(raw.get("session_id") or "")
        if not event_id or not session_id:
            raise ValueError("AR-Diffusion release event needs both event_id and session_id.")
        return cls(
            event_id=event_id,
            session_id=session_id,
            reason=str(raw.get("reason") or "unknown"),
            generation=int(raw.get("generation") or 0),
            stage_id=int(raw.get("stage_id", -1)),
            cleanup_failed=bool(raw.get("cleanup_failed", False)),
        )


class ARDiffusionReleaseEventLog:
    """Bounded, ack-based outbox of self-initiated session releases.

    Reads are non-destructive, so a lost RPC reply does not lose the release.
    A backlog means synchronization is broken; ``overflowed`` tells the caller
    to stop admitting work instead of queueing more.
    """

    def __init__(
        self,
        *,
        stage_id: int = -1,
        max_pending: int = DEFAULT_MAX_PENDING_RELEASE_EVENTS,
    ) -> None:
        if max_pending <= 0:
            raise ValueError(f"max_pending must be positive, got {max_pending}")
        self._stage_id = int(stage_id)
        self._max_pending = int(max_pending)
        self._events: dict[str, ARDiffusionReleaseEvent] = {}
        self._ids = itertools.count()
        self._lock = threading.Lock()
        self._ready = False
        self._overflowed = False
        self._suppressed: set[str] = set()
        self._generations: dict[str, int] = {}

    def set_ready(self, ready: bool = True) -> None:
        """Start (or stop) recording; anything queued while not ready is dropped.

        Startup warmup releases real rollouts, but they are not user sessions.
        """
        with self._lock:
            self._ready = bool(ready)
            if ready and self._events:
                logger.debug(
                    "Discarding %d AR-Diffusion release event(s) recorded before readiness",
                    len(self._events),
                )
                self._events.clear()
                self._overflowed = False

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def overflowed(self) -> bool:
        return self._overflowed

    def register_generation(self, session_id: str, generation: int) -> None:
        """Remember the generation a session is currently running under."""
        with self._lock:
            self._generations[str(session_id)] = int(generation)

    def forget_generation(self, session_id: str) -> None:
        with self._lock:
            self._generations.pop(str(session_id), None)

    @contextmanager
    def coordinated(self, session_id: str) -> Iterator[None]:
        """Mark releases for ``session_id`` as coordinator-driven while inside.

        Otherwise the coordinator's own cleanup RPC records an event it would
        then fan out again.
        """
        key = str(session_id)
        with self._lock:
            already = key in self._suppressed
            self._suppressed.add(key)
        try:
            yield
        finally:
            if not already:
                with self._lock:
                    self._suppressed.discard(key)

    def record(
        self,
        session_id: str,
        *,
        reason: str,
        cleanup_failed: bool = False,
    ) -> ARDiffusionReleaseEvent | None:
        """Record one self-initiated release; returns None when suppressed.

        A partly failed cleanup is still recorded: peers must learn this stage
        dropped the session.
        """
        key = str(session_id)
        with self._lock:
            if not self._ready:
                return None
            if key in self._suppressed or reason in COORDINATED_REASONS:
                return None
            if len(self._events) >= self._max_pending:
                # Dropping it would silently desynchronize the stages.
                self._overflowed = True
                logger.error(
                    "AR-Diffusion release event log is full (%d pending); session=%s reason=%s "
                    "could not be queued and stage state is now unsynchronized",
                    self._max_pending,
                    key,
                    reason,
                )
                return None
            event = ARDiffusionReleaseEvent(
                event_id=f"rel-{self._stage_id}-{next(self._ids)}",
                session_id=key,
                reason=str(reason),
                generation=int(self._generations.get(key, 0)),
                stage_id=self._stage_id,
                cleanup_failed=bool(cleanup_failed),
            )
            self._events[event.event_id] = event
            return event

    def pending(self) -> list[dict[str, Any]]:
        """Wire records for every unacknowledged release, oldest first."""
        with self._lock:
            return [event.to_dict() for event in self._events.values()]

    def acknowledge(self, event_ids: Iterable[str]) -> int:
        """Drop acknowledged records; returns how many were still present."""
        removed = 0
        with self._lock:
            for event_id in event_ids or ():
                if self._events.pop(str(event_id), None) is not None:
                    removed += 1
            if not self._events:
                self._overflowed = False
        return removed

    def pending_count(self) -> int:
        with self._lock:
            return len(self._events)
