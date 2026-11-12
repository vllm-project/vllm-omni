# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cell-granularity compute Batch: member identity, KV views, result mapping.

Mirrors the member-mapping contract of step-level ``InputBatch`` for the Noisy
KV path. Batch organization groups plan-allowed work by compatibility; it does
not reschedule CellPlan slots or own Session history.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_plan import Cell


@dataclass(frozen=True)
class CellMemberRef:
    """One compute member; identity is independent of the Batch row index."""

    request_id: str
    cell: Cell
    task_index: int


@dataclass(frozen=True)
class CellComputeBatch:
    """Compatible members for one model call (Noisy KV Cell granularity).

    Public fields match the design contract:

    * ordered members + compute identity
    * per-member KV read/write views (references, not Session copies)
    * tensor row → member result mapping
    """

    members: tuple[CellMemberRef, ...]
    kv_views: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("CellComputeBatch requires at least one member")
        if len(self.members) != len(self.kv_views):
            raise ValueError(
                f"members ({len(self.members)}) and kv_views ({len(self.kv_views)}) length mismatch"
            )
        cells = {m.cell for m in self.members}
        if len(cells) != 1:
            raise ValueError(f"CellComputeBatch members must share one cell, got {cells}")

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def cell(self) -> Cell:
        return self.members[0].cell

    @property
    def request_ids(self) -> tuple[str, ...]:
        return tuple(m.request_id for m in self.members)

    @property
    def task_indices(self) -> tuple[int, ...]:
        return tuple(m.task_index for m in self.members)

    def row_slice(self, member_index: int) -> slice:
        """Map a packed batch-axis row back to one member."""
        if member_index < 0 or member_index >= self.size:
            raise IndexError(f"member_index {member_index} out of range for size {self.size}")
        return slice(member_index, member_index + 1)

    def split_outputs(self, packed: Mapping[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        """Split packed model outputs into per-member dicts (batch dim 0)."""
        outs: list[dict[str, torch.Tensor]] = []
        for i in range(self.size):
            sl = self.row_slice(i)
            outs.append({k: v[sl] for k, v in packed.items()})
        return outs


def organize_cell_batches(
    tasks: Sequence[tuple[str, Cell]],
    kv_contexts: Sequence[Any],
    *,
    compatibility_of: Callable[[str, Cell], Hashable],
    max_batch_size: int,
) -> list[CellComputeBatch]:
    """Group this slot's ready tasks into compatible model-call Batches.

    Preserves first-seen group order and within-group task order. Does not defer
    Cells to a later plan slot. Oversized compatible groups are split by
    ``max_batch_size`` (same source as KV pool admission).
    """
    if max_batch_size < 1:
        raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")
    if not tasks:
        return []
    if len(kv_contexts) < len(tasks):
        raise ValueError(f"kv_contexts length {len(kv_contexts)} < tasks length {len(tasks)}")

    groups: dict[Hashable, list[int]] = {}
    order: list[Hashable] = []
    for i, (req, cell) in enumerate(tasks):
        key = compatibility_of(req, cell)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(i)

    batches: list[CellComputeBatch] = []
    for key in order:
        indices = groups[key]
        for start in range(0, len(indices), max_batch_size):
            chunk = indices[start : start + max_batch_size]
            members = tuple(
                CellMemberRef(request_id=tasks[i][0], cell=tasks[i][1], task_index=i) for i in chunk
            )
            views = tuple(_single_kv_view(kv_contexts[i]) for i in chunk)
            batches.append(CellComputeBatch(members=members, kv_views=views))
    return batches


def _single_kv_view(ctx_list: Any) -> Any:
    if isinstance(ctx_list, list):
        return ctx_list[0] if ctx_list else None
    return ctx_list
