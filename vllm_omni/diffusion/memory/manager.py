# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session memory manager (RFC #4480).

``SessionMemoryManager`` maps ``session_id -> {name: MemoryObject}`` and is the
single authority for session lifecycle and eviction. It evicts by session count
(LRU), matching the bespoke per-model caches it replaces; the byte budget is
recorded for observability but not yet enforced (enforcing it would diverge from
the budget-less bespoke paths this currently stays equivalent to).
"""

from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from typing import Any

from vllm_omni.diffusion.memory.base import MemoryObject

logger = logging.getLogger(__name__)

# Matches the bespoke DreamZero cap so the new path evicts identically.
DEFAULT_MAX_SESSIONS = 64


class SessionMemory:
    """The named ``MemoryObject`` collection for one session."""

    def __init__(self) -> None:
        self._objects: dict[str, MemoryObject] = {}
        # Session-scoped scalar/tensor metadata that is not itself a typed
        # MemoryObject (e.g. counters, cached conditioning tensors). Persists
        # across the per-call adapters that read/write it.
        self.attrs: dict[str, Any] = {}

    def get(self, name: str) -> MemoryObject | None:
        return self._objects.get(name)

    def put(self, name: str, obj: MemoryObject) -> MemoryObject:
        self._objects[name] = obj
        return obj

    def names(self) -> list[str]:
        return list(self._objects)

    def reset(self) -> None:
        for obj in self._objects.values():
            obj.reset()

    def evict(self) -> int:
        return sum(obj.evict() for obj in self._objects.values())

    @property
    def nbytes(self) -> int:
        return sum(obj.nbytes for obj in self._objects.values())


class SessionMemoryManager:
    """Owns per-session memory and arbitrates the (count-based) LRU.

    The manager lives beside ``DiffusionRequestState`` (owned by the pipeline),
    not inside its ``extra`` dict: the manager is cross-request and long-lived,
    whereas ``DiffusionRequestState`` is per-request and transient.
    """

    def __init__(self, max_sessions: int = DEFAULT_MAX_SESSIONS, byte_budget: int | None = None) -> None:
        if max_sessions <= 0:
            raise ValueError(f"max_sessions must be positive, got {max_sessions}")
        self.max_sessions = max_sessions
        # Recorded but not enforced yet. See RFC #4480.
        self.byte_budget = byte_budget
        self._sessions: OrderedDict[str, SessionMemory] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    @staticmethod
    def _key(session_id: str | None) -> str:
        return str(session_id or "default")

    def get_or_create_session(self, session_id: str | None) -> SessionMemory:
        key = self._key(session_id)
        with self._lock:
            session = self._sessions.get(key)
            if session is None:
                self.misses += 1
                session = SessionMemory()
                self._sessions[key] = session
                while len(self._sessions) > self.max_sessions:
                    _, evicted = self._sessions.popitem(last=False)
                    evicted.evict()
                    self.evictions += 1
            else:
                self.hits += 1
                self._sessions.move_to_end(key)
            return session

    def get_session(self, session_id: str | None) -> SessionMemory | None:
        key = self._key(session_id)
        with self._lock:
            session = self._sessions.get(key)
            if session is not None:
                self._sessions.move_to_end(key)
            return session

    def reset_session(self, session_id: str | None) -> None:
        key = self._key(session_id)
        with self._lock:
            session = self._sessions.get(key)
        if session is not None:
            session.reset()

    def evict_session(self, session_id: str | None) -> int:
        key = self._key(session_id)
        with self._lock:
            session = self._sessions.pop(key, None)
            if session is None:
                return 0
            self.evictions += 1
        return session.evict()

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def __contains__(self, session_id: str | None) -> bool:
        with self._lock:
            return self._key(session_id) in self._sessions

    def stats(self) -> dict[str, int]:
        with self._lock:
            total_nbytes = sum(session.nbytes for session in self._sessions.values())
            return {
                "sessions": len(self._sessions),
                "max_sessions": self.max_sessions,
                "total_nbytes": total_nbytes,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
            }


def resolve_session_memory_config(
    enable: bool | None = None,
    max_sessions: int | None = None,
) -> tuple[bool, int]:
    """Combine explicit args with env-var overrides.

    Environment variables (for quick enablement without touching config files):

        ``OMNI_DIFFUSION_SESSION_MEMORY_MANAGER`` (``1``/``0``/``true``/``false``)
        ``OMNI_DIFFUSION_SESSION_MEMORY_MANAGER_MAX_SESSIONS`` (positive int)
    """
    env_enable = os.environ.get("OMNI_DIFFUSION_SESSION_MEMORY_MANAGER")
    if env_enable is not None:
        parsed = env_enable.strip().lower()
        if parsed in ("1", "true", "yes", "on"):
            enable = True
        elif parsed in ("0", "false", "no", "off"):
            enable = False

    env_size = os.environ.get("OMNI_DIFFUSION_SESSION_MEMORY_MANAGER_MAX_SESSIONS")
    if env_size is not None:
        try:
            env_size_int = int(env_size)
            if env_size_int > 0:
                max_sessions = env_size_int
        except ValueError:
            logger.warning(
                "Ignoring non-integer OMNI_DIFFUSION_SESSION_MEMORY_MANAGER_MAX_SESSIONS=%r.",
                env_size,
            )

    return bool(enable), int(max_sessions or DEFAULT_MAX_SESSIONS)
