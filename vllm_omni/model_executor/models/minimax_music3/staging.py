# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pinned host staging for the small per-step index vectors.

Building a device tensor from a Python list (``torch.tensor(rows,
device="cuda")``) stages through pageable host memory, which makes the copy
synchronous: the host blocks until the copy retires, and because the copy is
queued behind the step's kernels that means blocking until the whole forward
has finished. On this model that turned four one-line conveniences into the
single largest entry in a CPU profile of the decode loop.

Writing into a pinned block and pushing it with one non-blocking copy costs
one transfer for the whole step and leaves the host free to keep enqueueing.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

__all__ = ["PinnedIndexStage"]


class PinnedIndexStage:
    """A fixed ``[rows, capacity]`` int64 block staged host-to-device.

    The device view returned by :meth:`push` is only populated once the stream
    reaches the copy, so it may be consumed by kernels on that stream and by
    nothing else. The host side is double-buffered: the next step writes the
    slot the device is not reading, so refilling cannot corrupt a copy that is
    still in flight.
    """

    def __init__(self, rows: int, capacity: int, device: torch.device) -> None:
        self._capacity = capacity
        self._host = [torch.zeros((rows, capacity), dtype=torch.int64, pin_memory=True) for _ in range(2)]
        self._device = torch.zeros((rows, capacity), dtype=torch.int64, device=device)
        self._slot = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    def push(self, values: Sequence[Sequence[int]], count: int) -> torch.Tensor:
        """Stage ``count`` entries of each row and return the device view.

        Args:
            values: One sequence of Python ints per row of the block.
            count: How many entries of each row are meaningful this step.

        Returns:
            ``[rows, count]`` int64 view of the device block.

        Raises:
            ValueError: If ``count`` exceeds the reserved capacity.
        """
        if count > self._capacity:
            raise ValueError(f"staged {count} entries but only {self._capacity} were reserved")
        self._slot ^= 1
        host = self._host[self._slot]
        view = host.numpy()
        for index, row in enumerate(values):
            view[index, :count] = row
        self._device.copy_(host, non_blocking=True)
        return self._device[:, :count]
