# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Session state for the Noisy KV cell path.

The contract is recorded in ``docs/design/feature/noisy_kv.md``: a session owns
the history a later request continues from, a request owns the versions it
produces plus its scratch, and a failed request leaves the session unusable
until an explicit reset.

This module owns the lifecycle decision and the identities of the versions a
session retains. The bytes stay in the KV pool under the session owner
(``SESSION_OWNER_PREFIX`` in ``noisy.py``), which is why dropping a session has
to hand those slots back rather than only clearing this dictionary.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum

from vllm.logger import init_logger

logger = init_logger(__name__)


class NoisySessionState(str, Enum):
    """Lifecycle of one Noisy KV session."""

    ACTIVE = "active"
    INVALID = "invalid"
    CLOSING = "closing"
    CLOSED = "closed"


class NoisySessionError(RuntimeError):
    """The session cannot serve the request in its current state."""


@dataclass
class NoisyKVSession:
    """Cross-request state owned by one ``session_id``.

    ``generation`` fences late results: a request binds the generation it saw
    and may only commit if that generation is still current, so a reset or a
    failed request cannot be overwritten by work that started earlier.
    """

    session_id: str
    state: NoisySessionState = NoisySessionState.ACTIVE
    committed_chunks: int = 0
    generation: int = 0
    retained: dict[tuple[int, int], object] = field(default_factory=dict)

    def admit(self) -> int:
        """Return the generation this request must present when committing."""
        if self.state is not NoisySessionState.ACTIVE:
            raise NoisySessionError(f"session {self.session_id!r} is {self.state.value}, not active")
        return self.generation

    def commit_chunks(self, chunks: int, *, generation: int) -> None:
        """Advance the continuation offset after a successful request."""
        if chunks < 0:
            raise ValueError(f"committed chunk count must be non-negative, got {chunks}")
        if generation != self.generation or self.state is not NoisySessionState.ACTIVE:
            raise NoisySessionError(f"stale commit for session {self.session_id!r}")
        self.committed_chunks += chunks

    def invalidate(self, reason: str) -> None:
        """Poison the session; only an explicit reset can revive it."""
        if self.state is NoisySessionState.ACTIVE:
            logger.warning("Noisy KV session %s became invalid: %s", self.session_id, reason)
            self.state = NoisySessionState.INVALID
        # Invalidating must also fence work that started before the failure.
        self.generation += 1

    def reset(self) -> None:
        if self.state in (NoisySessionState.CLOSING, NoisySessionState.CLOSED):
            raise NoisySessionError(f"cannot reset {self.state.value} session {self.session_id!r}")
        self.retained.clear()
        self.committed_chunks = 0
        self.generation += 1
        self.state = NoisySessionState.ACTIVE

    def close(self) -> None:
        if self.state is NoisySessionState.CLOSED:
            return
        self.state = NoisySessionState.CLOSING
        self.retained.clear()
        self.generation += 1
        self.state = NoisySessionState.CLOSED

    @property
    def is_resumable(self) -> bool:
        return self.state is NoisySessionState.ACTIVE and self.committed_chunks > 0


class NoisyKVSessionRegistry:
    """LRU-bounded registry; capacity counts sessions, not bytes."""

    def __init__(
        self,
        capacity: int = 1,
        on_release: Callable[[str], None] | None = None,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"session capacity must be positive, got {capacity}")
        self.capacity = capacity
        self.on_release = on_release
        self._sessions: OrderedDict[str, NoisyKVSession] = OrderedDict()

    def _release(self, session_id: str) -> None:
        """Hand a session's slots back before the registry forgets the session.

        Retention re-keys versions to the session owner, so request cleanup
        never frees them; without this the pool stays short of those slots for
        every reset, close and eviction.
        """
        if self.on_release is not None:
            self.on_release(session_id)

    def __len__(self) -> int:
        return len(self._sessions)

    def __iter__(self) -> Iterator[NoisyKVSession]:
        return iter(self._sessions.values())

    def get(self, session_id: str) -> NoisyKVSession | None:
        session = self._sessions.get(session_id)
        if session is not None:
            self._sessions.move_to_end(session_id)
        return session

    def admit(self, session_id: str) -> tuple[NoisyKVSession, int]:
        """Return the session and the generation to present when committing."""
        session = self.get(session_id)
        if session is None:
            while len(self._sessions) >= self.capacity:
                evicted_id, evicted = next(iter(self._sessions.items()))
                if evicted.state is NoisySessionState.ACTIVE:
                    evicted.close()
                logger.info("Noisy KV session %s evicted from a full registry", evicted_id)
                self._release(evicted_id)
                del self._sessions[evicted_id]
            session = NoisyKVSession(session_id=session_id)
            self._sessions[session_id] = session
        return session, session.admit()

    def reset(self, session_id: str) -> None:
        session = self.get(session_id)
        if session is not None:
            self._release(session_id)
            session.reset()

    def close(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            self._release(session_id)
            session.close()

    def inactivate(self, session_id: str, reason: str) -> None:
        """Mark a session unusable after a failed request; keep it resident."""
        session = self.get(session_id)
        if session is not None:
            session.invalidate(reason)

    def clear(self) -> None:
        for session_id, session in self._sessions.items():
            self._release(session_id)
            session.close()
        self._sessions.clear()
