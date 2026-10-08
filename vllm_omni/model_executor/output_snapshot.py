# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Owned, dtype-grouped snapshots for asynchronous runner outputs."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten


@dataclass(frozen=True)
class RequestOutputSnapshot:
    """Already partitioned, CPU-owned payloads in the current request order.

    Model finalizers may return this after D2H completes to avoid recursively
    cloning and partitioning data whose request ownership is already known.
    """

    inter_stage: list[dict[str, Any] | None]
    client: list[dict[str, Any] | None] | None = None


class PackedOutputSnapshot(dict):
    """A normal payload mapping with an internal batched-copy plan.

    The runner's snapshot-slot event protects the device slabs until D2H is
    complete. Host slabs are allocated per output, so downstream consumers can
    retain their views without depending on reuse of the device ring slot.
    """

    def __init__(self, leaves: list[Any], spec: Any, groups: list[tuple[list[int], torch.Tensor]]) -> None:
        super().__init__(tree_unflatten(leaves, spec))
        self._leaves = leaves
        self._spec = spec
        self._groups = groups
        self.producer_event: torch.cuda.Event | None = None

    def record_producer_event(self, stream: torch.cuda.Stream) -> None:
        """Publish readiness when a model produces on a separate CUDA stream."""
        self.producer_event = torch.cuda.Event()
        self.producer_event.record(stream)

    def copy_to_cpu(self, copy_tensor: Callable[[torch.Tensor], torch.Tensor]) -> dict[str, Any]:
        leaves = list(self._leaves)
        for indices, slab in self._groups:
            cpu = copy_tensor(slab)
            offset = 0
            for index in indices:
                value = leaves[index]
                size = value.numel()
                leaves[index] = cpu.narrow(0, offset, size).view(value.shape)
                offset += size
        return tree_unflatten(leaves, self._spec)


def pack_output_snapshot(
    payload: dict[str, Any],
    slot: dict[tuple[Any, ...], torch.Tensor],
    *,
    max_buckets: int,
    reuse_existing_storage: bool = False,
) -> PackedOutputSnapshot | None:
    """Copy tensor leaves once per dtype/device, retaining their exact values.

    No CUDA synchronization is introduced here. The caller must select the
    producer stream and wait for the previous consumer before reusing a slot.
    """
    leaves, spec = tree_flatten(payload)
    grouped: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, value in enumerate(leaves):
        if isinstance(value, torch.Tensor):
            # Preserve the existing per-tensor path for unusual layouts.
            if value.layout != torch.strided or value.is_quantized:
                return None
            grouped[(value.dtype, value.device)].append(index)
    groups = []
    for (dtype, device), indices in grouped.items():
        values = [leaves[index].detach().reshape(-1) for index in indices]
        size = sum(value.numel() for value in values)
        # A producer may have already written the leaves into one contiguous
        # per-forward slab.  Reuse that storage directly instead of copying
        # it into a second packed slab.  The strict offset/stride check keeps
        # this optimization limited to views whose order matches the copy
        # plan; arbitrary payloads retain the existing copy path.
        if reuse_existing_storage and values and all(value.is_contiguous() for value in values):
            storage = values[0].untyped_storage()
            offset = values[0].storage_offset()
            cursor = offset
            shares_storage = True
            for value in values:
                if value.untyped_storage().data_ptr() != storage.data_ptr() or value.storage_offset() != cursor:
                    shares_storage = False
                    break
                cursor += value.numel()
            if shares_storage:
                slab = values[0].as_strided((size,), (1,), storage_offset=offset)
                offset = 0
                for index, value in zip(indices, values):
                    shape = leaves[index].shape
                    leaves[index] = slab.narrow(0, offset, value.numel()).view(shape)
                    offset += value.numel()
                groups.append((indices, slab))
                continue
        key = ("packed-output", dtype, device, size)
        slab = slot.get(key)
        if slab is None:
            slab = torch.empty(size, dtype=dtype, device=device)
            if len(slot) < max_buckets:
                slot[key] = slab
        if len(values) == 1:
            slab.copy_(values[0])
        else:
            torch.cat(values, out=slab)
        offset = 0
        for index, value in zip(indices, values):
            shape = leaves[index].shape
            leaves[index] = slab.narrow(0, offset, value.numel()).view(shape)
            offset += value.numel()
        groups.append((indices, slab))
    return PackedOutputSnapshot(leaves, spec, groups)
