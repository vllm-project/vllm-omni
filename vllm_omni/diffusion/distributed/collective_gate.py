# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cross-module rendezvous that orders background collectives after the
foreground all-to-all of a transformer block.

Distributed layerwise offload (DLO) issues the *next* block's weight AllGather
on a dedicated ``comm_stream`` while the *current* block computes. Both that
AllGather and the block's QKV sequence-parallel AllToAll are collectives over
the same ranks, so whichever reaches the links first wins: the AllGather is
submitted at block entry, well before the block reaches its attention, and the
critical-path AllToAll ends up queued behind it.

A plain event wait cannot fix this by itself. ``wait_event`` snapshots the
records enqueued *before* the call, so "enqueue the wait at block entry, record
the event later in the block" is a no-op -- the wait resolves against the
previous iteration's record. The gather therefore has to be *issued* after the
AllToAll: the hook arms this barrier at block entry and defers the gather, the
attention signals it right after ``strategy.pre_attention``, and the signal
records the event and lets the hook submit the gather gated on it.

The gather still overlaps the rest of the block (attention output, MLP, the
next block's projections); it simply cannot start before the AllToAll finished.
"""

from __future__ import annotations

from collections.abc import Callable

__all__ = ["QKVAllToAllBarrier"]


class QKVAllToAllBarrier:
    """One-block rendezvous between a DLO hook and the block's attention.

    Owned by a single :class:`DistributedLayerwiseOffloadHook` (one per
    streaming block) and shared with the attention layer inside that block.
    Both live on the same block, so the arming and the signalling always come
    from the same module forward.

    Args:
        on_signal: Invoked at most once per :meth:`arm` when the block's
            attention reports its QKV AllToAll. It must submit the deferred
            weight AllGather.
    """

    def __init__(self, on_signal: Callable[[], None]) -> None:
        self._on_signal = on_signal
        self._armed = False

    @property
    def armed(self) -> bool:
        """True between :meth:`arm` and the matching :meth:`signal`."""
        return self._armed

    def arm(self) -> None:
        """Mark the next attention QKV AllToAll as the gather's gate."""
        self._armed = True

    def disarm(self) -> None:
        """Drop a pending arm without firing it (the caller flushes instead)."""
        self._armed = False

    def signal(self) -> None:
        """Report that the QKV AllToAll is now enqueued on the compute stream.

        A no-op unless armed, so it is cheap to call unconditionally from the
        attention hot path.
        """
        if not self._armed:
            return
        self._armed = False
        self._on_signal()
