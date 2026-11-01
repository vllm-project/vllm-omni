# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-device byte accounting for session state (RFC #4480).

A session holds bytes in more than one place -- device tensors, host tensors,
NumPy frame buffers -- and those are different resources: a GPU pool that is
scarce and contended, and host memory that is not. Summing them into a single
number answers no question anyone has, so the accounting here is keyed by the
device the bytes live on and is never reduced to a scalar by this module.

Two details make the numbers honest:

* **Storages, not elements.** ``numel() * element_size()`` measures the logical
  view, but a narrow slice of a large buffer keeps the whole allocation alive.
  Counting ``untyped_storage().nbytes()`` reports what is actually pinned.
* **Views deduplicated.** Two tensors sharing one storage are counted once, via
  the caller-supplied ``seen`` set. Pass the *same* set across every value in a
  session so an alias between two state objects (or between an object and an
  attribute) does not double-count.

Anything that is not a tensor, an array, or a container of those contributes
nothing. That is not a heuristic: in this codebase the only device memory a
session can hold is a ``torch.Tensor``. CUDA graph pools, collective buffers
and the paged KV blocks of PR #4534 are owned by the engine, never by a session.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch

# ``(device string, storage address)`` of a storage already counted.
SeenStorages = set[tuple[str, int]]

_HOST = "cpu"


def device_bytes(value: object, seen: SeenStorages) -> dict[str, int]:
    """Bytes pinned by ``value``, keyed by the device they live on.

    ``seen`` accumulates the storages already counted and is mutated in place;
    share one set across a whole session to deduplicate aliases.
    """
    totals: dict[str, int] = {}
    _accumulate(value, seen, totals)
    return totals


def merge_bytes(into: dict[str, int], other: Mapping[str, int]) -> dict[str, int]:
    """Add ``other`` into ``into`` per device, in place."""
    for device, count in other.items():
        into[device] = into.get(device, 0) + count
    return into


def _accumulate(value: object, seen: SeenStorages, totals: dict[str, int]) -> None:
    if isinstance(value, torch.Tensor):
        storage = value.untyped_storage()
        device = str(value.device)
        key = (device, storage.data_ptr())
        if key in seen:
            return
        seen.add(key)
        totals[device] = totals.get(device, 0) + storage.nbytes()
        return

    if isinstance(value, np.ndarray):
        # ``base is not None`` marks a view; NumPy views are cheap to alias and
        # carry no address we can key on as reliably as a torch storage, so the
        # owning array is what gets counted.
        owner = value if value.base is None else value.base
        if isinstance(owner, np.ndarray):
            key = (_HOST, owner.__array_interface__["data"][0])
            if key in seen:
                return
            seen.add(key)
            totals[_HOST] = totals.get(_HOST, 0) + int(owner.nbytes)
        return

    if isinstance(value, Mapping):
        for item in value.values():
            _accumulate(item, seen, totals)
        return

    # ``str``/``bytes`` are Sequences but hold no device memory; excluding them
    # also stops the recursion from walking a string one character at a time.
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _accumulate(item, seen, totals)
        return
