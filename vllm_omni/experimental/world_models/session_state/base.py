# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ``StateObject`` contract for session state (RFC #4480).

A session is a named collection of ``StateObject`` instances. The interface
deliberately speaks only of bytes, growth, read views, commitment, and
evictability -- never tokens, layers, or attention. Writes are two-phase:
``stage()`` holds data for an in-progress window, ``commit()`` promotes it to
persistent context, and ``discard()`` drops it.

A concrete class may back its storage with plain (non-paged) buffers; the
contract itself does not require that. Several methods are
intentionally inert and documented as such; they exist so that paged-KV
backing and speculative forking can be added without changing any call site.
See RFC #4480 for the full design and roadmap.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from vllm_omni.experimental.world_models.session_state.accounting import SeenStorages

# What writers pass into ``stage()`` / ``commit()`` / ``append()``.
PayloadT = TypeVar("PayloadT")
# What ``view()`` hands to the consumer (not necessarily the payload type:
# an append buffer views as a list of payloads).
ViewT = TypeVar("ViewT")


class StateObject(ABC, Generic[PayloadT, ViewT]):
    """One typed unit of session state with a uniform lifecycle.

    A session's state is a named collection of these objects; the kinds of
    memory they represent include attention K/V (growing self-attention
    caches, or conditioning K/V written once and reread), constant-size
    recurrent state (e.g. a linear-attention carry), frame or latent buffers,
    and retrieval pools. A concrete class binds ``PayloadT`` and ``ViewT`` to
    its real payload and view types and implements the abstract methods; the
    non-abstract methods provide the default behaviour described in their
    docstrings.

    The contract is storage-agnostic. A concrete class may hold its bytes in
    a buffer it owns, or it may be a handle to storage owned by another
    component, such as a paged KV pool. In the handle case it delegates its
    lifecycle operations to that owner instead of copying the data.
    """

    # The dependency edge: what this object rebuilds from. ``None`` means it is
    # never recomputable (it must be preempted or rejected, not silently
    # dropped).
    recompute_source: StateObject | None = None
    # Estimated work to rebuild, given ``recompute_source`` is resident.
    recompute_cost: int = 0
    # Payload held by ``stage()`` until ``discard()`` clears it.
    _staged: PayloadT | None = None

    @abstractmethod
    def allocate(self) -> None:
        """Size and initialise the backing storage to its empty state.

        Concrete classes declare their sizing parameters as typed keyword-only
        arguments (e.g. ``LatentBuffer.allocate(maxlen=...)``); a mistyped
        keyword raises ``TypeError`` instead of being silently ignored.
        """

    def stage(self, payload: PayloadT) -> None:
        """Hold data for an in-progress window.

        Single-window writers go straight through ``commit()``, so the default
        implementation only records the payload and is otherwise inert.
        Speculative forking is the intended consumer of staged state.
        """
        self._staged = payload

    @abstractmethod
    def commit(self, payload: PayloadT | None = None) -> None:
        """Promote data to persistent context.

        For a plain-buffer object this is the synchronous write into its
        committed storage (e.g. ``cache[i] = updated_kv.clone()``).
        """

    def discard(self) -> None:
        """Drop staged data (speculative rejection).

        Clears any recorded staged payload; committed bytes are never touched.
        """
        self._staged = None

    def append(self, payload: PayloadT) -> None:
        """``stage()`` + ``commit()``; single-shot writers use only this."""
        self.commit(payload)

    @abstractmethod
    def view(self, *, include_staged: bool = True) -> ViewT:
        """Return what the consumer (attention metadata, pipeline) reads."""

    # ``policy`` deliberately stays ``Any``: the policy protocol belongs to
    # the byte-budget eviction planner (RFC #4480), and defining it here
    # would invent an interface that design owns.
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
        session be garbage-collected. The intended driver of this hook is a
        byte-budget eviction planner with the recompute-DAG safety described in
        RFC #4480.
        """
        freed = self.nbytes
        self.reset()
        return freed

    @abstractmethod
    def reset(self) -> None:
        """Clear back to the unallocated state."""

    @abstractmethod
    def nbytes_by_device(self, seen: SeenStorages) -> dict[str, int]:
        """Bytes held in this object's committed storage, keyed by device.

        Staged payloads are not counted. ``seen`` deduplicates storages across
        the whole session -- pass it through to ``device_bytes`` rather than
        starting a fresh set, or two views on one buffer are counted twice.
        """

    @property
    def nbytes(self) -> int:
        """Total bytes held, summed over every device.

        Only meaningful when the caller genuinely does not care where the bytes
        live (``evict()`` reporting what it freed, for instance). Anything
        deciding against a budget wants ``nbytes_by_device`` instead: host and
        device memory are different resources and their sum is not a quantity.
        """
        return sum(self.nbytes_by_device(set()).values())

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
