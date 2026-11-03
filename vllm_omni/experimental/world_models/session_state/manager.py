# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session memory manager (RFC #4480).

``SessionStateManager`` maps ``session_id -> {name: StateObject}`` and is the
single authority for session lifecycle and eviction. It evicts by session count
(LRU), matching the per-model session caches it adapts (e.g. DreamZero's), so
an adapted model keeps the same eviction behaviour. A byte budget is recorded
for observability; enforcing it is left to a byte-budget eviction planner (see
RFC #4480).
"""

from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import TypeVar

from vllm_omni.experimental.world_models.session_state.accounting import (
    SeenStorages,
    device_bytes,
    merge_bytes,
)
from vllm_omni.experimental.world_models.session_state.base import StateObject

logger = logging.getLogger(__name__)

# Matches DreamZero's MAX_DREAMZERO_SESSIONS cap so an adapted DreamZero
# evicts identically.
DEFAULT_MAX_SESSIONS = 64

M = TypeVar("M", bound=StateObject)


class SessionState:
    """The named ``StateObject`` collection for one session."""

    def __init__(self) -> None:
        self._objects: dict[str, StateObject] = {}
        # Session-scoped scalar/tensor metadata that is not itself a typed
        # StateObject (e.g. counters, cached conditioning tensors). Persists
        # across the per-call adapters that read/write it.
        # ``object`` (not ``Any``) so every read site must narrow explicitly.
        self.attrs: dict[str, object] = {}

    def get(self, name: str) -> StateObject | None:
        return self._objects.get(name)

    def put(self, name: str, obj: StateObject) -> StateObject:
        self._objects[name] = obj
        return obj

    def get_or_create(self, name: str, factory: Callable[[], M], cls: type[M]) -> M:
        """Return the resident object of type ``cls`` under ``name``, building
        it via ``factory`` when absent, wrong-typed, or no longer resident."""
        obj = self._objects.get(name)
        if not isinstance(obj, cls) or not obj.resident:
            obj = factory()
            self._objects[name] = obj
        return obj

    def names(self) -> list[str]:
        return list(self._objects)

    def reset(self) -> None:
        for obj in self._objects.values():
            obj.reset()
        # Clear session-scoped metadata too, so a reset leaves nothing stale
        # (otherwise language/current_start_frame/clip_feas/ys would survive).
        self.attrs.clear()

    def nbytes_by_device(self, seen: SeenStorages | None = None) -> dict[str, int]:
        """Bytes this session holds, keyed by device.

        Both buckets are walked. A ``StateObject`` and a session attribute are
        different in what the *manager* may do with them -- release, accumulate,
        stage and commit versus merely carry -- but not in whether their bytes
        occupy memory, so the accounting covers both. One ``seen`` set spans the
        whole session, so a tensor aliased between the two buckets is counted
        once.
        """
        shared: SeenStorages = set() if seen is None else seen
        totals: dict[str, int] = {}
        for obj in self._objects.values():
            merge_bytes(totals, obj.nbytes_by_device(shared))
        merge_bytes(totals, device_bytes(self.attrs, shared))
        return totals

    @property
    def nbytes(self) -> int:
        """Total bytes held, summed over every device -- see
        ``StateObject.nbytes`` on why a budget wants the per-device split."""
        return sum(self.nbytes_by_device().values())


class SessionStateManager:
    """Owns per-session state and arbitrates the (count-based) LRU.

    The manager lives beside ``DiffusionRequestState`` (owned by the pipeline),
    not inside its ``extra`` dict: the manager is cross-request and long-lived,
    whereas ``DiffusionRequestState`` is per-request and transient.
    """

    def __init__(
        self,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        byte_budget: int | None = None,
        lock_factory: Callable[[], AbstractContextManager[object]] = threading.Lock,
    ) -> None:
        if max_sessions <= 0:
            raise ValueError(f"max_sessions must be positive, got {max_sessions}")
        self.max_sessions = max_sessions
        # Recorded for observability; enforcement is left to an eviction
        # planner (see RFC #4480). Not scoped to a device: which pool a budget
        # applies to is a question for whoever enforces it, so read
        # ``nbytes_by_device()`` and pick the pool at the point of enforcement.
        self.byte_budget = byte_budget
        self._sessions: OrderedDict[str, SessionState] = OrderedDict()
        # The locking strategy is injected rather than hard-coded: callers
        # that run the manager from worker threads keep the default
        # ``threading.Lock``, while an async or single-threaded owner can pass
        # e.g. ``contextlib.nullcontext`` (or a custom strategy) without any
        # call-site change -- the manager only ever does ``with self._lock:``.
        # Heavier operations (paged allocation, eviction planning) keep this
        # seam.
        self._lock = lock_factory()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    @staticmethod
    def _key(session_id: str | None) -> str:
        return str(session_id or "default")

    def get_or_create_session(self, session_id: str | None) -> SessionState:
        key = self._key(session_id)
        with self._lock:
            session = self._sessions.get(key)
            if session is None:
                self.misses += 1
                session = SessionState()
                self._sessions[key] = session
                while len(self._sessions) > self.max_sessions:
                    # Drop the oldest from the table only; do not free its
                    # buffers. An adapter still using the session keeps its own
                    # reference, so its state survives (matching a model's own
                    # per-session state object, which the caller holds even
                    # after it leaves the table). Unreferenced sessions are
                    # GC-reclaimed.
                    self._sessions.popitem(last=False)
                    self.evictions += 1
            else:
                self.hits += 1
                self._sessions.move_to_end(key)
            return session

    def drop_session(self, session_id: str | None) -> bool:
        """Release one session explicitly and free its buffers.

        This is the end-of-session signal a model's engine raises on close,
        eviction, or reset -- distinct from the count-based LRU in
        ``get_or_create_session``, which only drops the *table* entry and lets an
        adapter still holding the session keep it alive. Here the session is
        finished, so its ``StateObject`` buffers and carried attributes are reset
        to release their memory. Returns whether a session was present.
        """
        key = self._key(session_id)
        with self._lock:
            session = self._sessions.pop(key, None)
            if session is None:
                return False
            session.reset()
            return True

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def __contains__(self, session_id: str | None) -> bool:
        with self._lock:
            return self._key(session_id) in self._sessions

    def nbytes_by_device(self) -> dict[str, int]:
        """Bytes held across every live session, keyed by device."""
        with self._lock:
            return self._nbytes_by_device_locked()

    def _nbytes_by_device_locked(self) -> dict[str, int]:
        # One ``seen`` set across all sessions: sessions can legitimately share
        # a storage (a conditioning tensor broadcast to several of them), and
        # the total is what the process holds, not the sum of what each session
        # would hold alone.
        shared: SeenStorages = set()
        totals: dict[str, int] = {}
        for session in self._sessions.values():
            merge_bytes(totals, session.nbytes_by_device(shared))
        return totals

    def stats(self) -> dict[str, int]:
        """Counters plus per-device byte totals.

        Byte keys are prefixed ``nbytes:`` and named by device -- ``nbytes:cpu``,
        ``nbytes:cuda:0`` -- because host and device memory are different
        resources and their sum answers no question. ``total_nbytes`` is kept
        for callers that only want a single number.
        """
        with self._lock:
            by_device = self._nbytes_by_device_locked()
            stats: dict[str, int] = {
                "sessions": len(self._sessions),
                "max_sessions": self.max_sessions,
                "total_nbytes": sum(by_device.values()),
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
            }
            for device, count in sorted(by_device.items()):
                stats[f"nbytes:{device}"] = count
            return stats


def resolve_session_state_config(
    enable: bool | None = None,
    max_sessions: int | None = None,
) -> tuple[bool, int]:
    """Combine explicit args with env-var overrides.

    Environment variables (for quick enablement without touching config files):

        ``OMNI_DIFFUSION_SESSION_STATE_MANAGER`` (``1``/``0``/``true``/``false``)
        ``OMNI_DIFFUSION_SESSION_STATE_MANAGER_MAX_SESSIONS`` (positive int)

    Precedence: a *set* environment variable wins over the value passed in
    (i.e. over ``OmniDiffusionConfig.enable_session_state_manager``), in both
    directions -- ``...SESSION_MEMORY_MANAGER=0`` force-disables the manager
    even when the config enables it. Unset or unparsable values leave the
    passed-in value untouched (an unparsable ``MAX_SESSIONS`` logs a warning
    and is ignored). This mirrors how other omni feature toggles allow quick
    A/B flips on a deployed config.
    """
    env_enable = os.environ.get("OMNI_DIFFUSION_SESSION_STATE_MANAGER")
    if env_enable is not None:
        parsed = env_enable.strip().lower()
        if parsed in ("1", "true", "yes", "on"):
            enable = True
        elif parsed in ("0", "false", "no", "off"):
            enable = False

    env_size = os.environ.get("OMNI_DIFFUSION_SESSION_STATE_MANAGER_MAX_SESSIONS")
    if env_size is not None:
        try:
            env_size_int = int(env_size)
            if env_size_int > 0:
                max_sessions = env_size_int
        except ValueError:
            logger.warning(
                "Ignoring non-integer OMNI_DIFFUSION_SESSION_STATE_MANAGER_MAX_SESSIONS=%r.",
                env_size,
            )

    return bool(enable), int(max_sessions or DEFAULT_MAX_SESSIONS)
