# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ``MemoryObject`` contract for session memory (RFC #4480).

A session is a named collection of ``MemoryObject`` instances. The interface
deliberately speaks only of bytes, growth, read views, commitment, and
evictability -- never tokens, layers, or attention. Writes are two-phase:
``stage()`` holds data for an in-progress window, ``commit()`` promotes it to
persistent context, and ``discard()`` drops it.

The objects are backed by plain (non-paged) buffers. Several methods are
intentionally inert today and documented as such; they exist so that paged-KV
backing and speculative forking can be added later without changing any call
site. See RFC #4480 for the full design and roadmap.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MemoryObject(ABC):
    """One typed unit of session memory with a uniform lifecycle.

    Concrete classes (``PagedKV``, ``EncodeOnceKV``, ``LatentBuffer``) implement
    the abstract methods below. The non-abstract methods provide the default
    behaviour described in their docstrings.
    """

    # The dependency edge: what this object rebuilds from. ``None`` means it is
    # never recomputable (it must be preempted or rejected, not silently
    # dropped). No concrete class sets this yet.
    recompute_source: MemoryObject | None = None
    # Estimated work to rebuild, given ``recompute_source`` is resident.
    recompute_cost: int = 0

    @abstractmethod
    def allocate(self, **spec: Any) -> None:
        """Size and initialise the backing buffer to its empty state."""

    def stage(self, payload: Any) -> None:
        """Hold data for an in-progress window.

        Currently single-window writes go straight through ``commit()``, so
        staging only records the payload and is otherwise inert. Speculative
        forking (not yet implemented) will make this the real staging path.
        """
        self._staged = payload

    @abstractmethod
    def commit(self, payload: Any = None) -> None:
        """Promote data to persistent context.

        Currently this is the synchronous write the bespoke caches already do
        (e.g. ``cache[i] = updated_kv.clone()``).
        """

    def discard(self) -> None:
        """Drop staged data (speculative rejection).

        There is no speculative path yet, so this only clears any recorded
        staged payload and never touches committed bytes.
        """
        self._staged = None

    def append(self, payload: Any) -> None:
        """``stage()`` + ``commit()``; single-shot writers use only this."""
        self.commit(payload)

    @abstractmethod
    def view(self, *, include_staged: bool = True) -> Any:
        """Return what the consumer (attention metadata, pipeline) reads."""

    def evict(self, policy: Any = None) -> int:
        """Release this object's backing storage so it can be reused; return the
        bytes freed.

        This is the backend-agnostic release hook. For a plain buffer it drops
        the buffer and lets the garbage collector reclaim it; for a paged backend
        it returns the blocks to the pool so they can be recycled. Never reclaims
        staged bytes.

        Precondition: only call this when the object is no longer in use.
        Releasing storage that is still referenced is a correctness bug (another
        consumer could reuse it). Count-based session eviction therefore does NOT
        call this — it only drops the lookup-table entry and lets an unreferenced
        session be garbage-collected. A byte-budget eviction planner (a later
        phase) is what drives this hook, with the recompute-DAG safety from the
        RFC; no caller invokes it yet.
        """
        freed = self.nbytes
        self.reset()
        return freed

    @abstractmethod
    def reset(self) -> None:
        """Clear back to the unallocated state."""

    @property
    @abstractmethod
    def nbytes(self) -> int:
        """Bytes currently held (committed + staged)."""

    @property
    @abstractmethod
    def resident(self) -> bool:
        """Whether data is currently in memory (not yet evicted)."""

    @property
    def recomputable(self) -> bool:
        """Droppable-and-rebuildable *right now*.

        Derived, not declared: true only while ``recompute_source`` is still
        resident, so the answer changes as other objects are evicted.
        """
        return self.recompute_source is not None and self.recompute_source.resident
