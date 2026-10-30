# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ordered PCM reservation ownership shared by duplex input buffers.

Models keep their own emit policy. This only transfers a reserved span:
commit drops it, rollback puts its bytes back.
"""

from __future__ import annotations


def commit_ordered_reservation(reservations: list, reservation: object, *, head_only: bool) -> None:
    """Drop one active reservation. ``head_only`` rejects an out-of-order commit."""
    if not getattr(reservation, "_active", False):
        return
    if head_only:
        if not reservations or reservations[0] is not reservation:
            raise RuntimeError("PCM append reservations must commit in wire order")
        reservations.pop(0)
    elif reservation in reservations:
        reservations.remove(reservation)
    reservation._active = False


def rollback_ordered_reservation(
    reservations: list,
    reservation: object,
    buffer: bytearray,
    *,
    active_only: bool,
) -> list:
    """Restore this reservation and every later one. Returns the restored items."""
    if not getattr(reservation, "_active", False):
        return []
    try:
        index = reservations.index(reservation)
    except ValueError:
        reservation._active = False
        return []
    rolled_back = list(reservations[index:])
    restore = bytearray()
    for item in rolled_back:
        if active_only and not getattr(item, "_active", False):
            continue
        restore.extend(getattr(item, "_raw", b""))
        item._active = False
    del reservations[index:]
    buffer[:0] = restore
    reservation._active = False
    return rolled_back


__all__ = ["commit_ordered_reservation", "rollback_ordered_reservation"]
