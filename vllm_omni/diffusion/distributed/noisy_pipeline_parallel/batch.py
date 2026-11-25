# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compute Batch: member identity, KV views, result mapping.

Internal to the Noisy PP executor. Grouping algorithm is generic; which members
may share a model call is declared by each ``CellAdapter.compatibility_key``.
Members may differ in chunk/step; they must share the DiT ``block`` index.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_plan import Cell


@dataclass(frozen=True)
class MemberRef:
    """One compute member; identity is independent of the Batch row index."""

    request_id: str
    chunk: int
    step: int
    block: int
    task_index: int


@dataclass(frozen=True)
class ComputeBatch:
    """Compatible members for one model call (Noisy KV schedule granularity)."""

    members: tuple[MemberRef, ...]
    kv_views: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("ComputeBatch requires at least one member")
        if len(self.members) != len(self.kv_views):
            raise ValueError(f"members ({len(self.members)}) and kv_views ({len(self.kv_views)}) length mismatch")
        blocks = {int(m.block) for m in self.members}
        if len(blocks) != 1:
            raise ValueError(f"ComputeBatch members must share one block, got {blocks}")

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def block(self) -> int:
        return int(self.members[0].block)

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
        for key, tensor in packed.items():
            if tensor.ndim < 1 or tensor.shape[0] != self.size:
                raise ValueError(f"Output {key!r} must have one row per member ({self.size})")
        outs: list[dict[str, torch.Tensor]] = []
        for i in range(self.size):
            sl = self.row_slice(i)
            outs.append({k: v[sl] for k, v in packed.items()})
        return outs


def organize_batches(
    tasks: Sequence[tuple[str, Cell]],
    kv_contexts: Sequence[Any],
    *,
    compatibility_of: Callable[[str, Cell], Hashable],
    max_batch_size: int,
) -> list[ComputeBatch]:
    """Group this slot's ready tasks into compatible model-call Batches.

    Preserves first-seen group order and within-group task order. Does not defer
    work to a later plan slot. Oversized compatible groups are split by
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

    batches: list[ComputeBatch] = []
    for key in order:
        indices = groups[key]
        for start in range(0, len(indices), max_batch_size):
            chunk_idx = indices[start : start + max_batch_size]
            members = tuple(
                MemberRef(
                    request_id=tasks[i][0],
                    chunk=int(tasks[i][1][0]),
                    step=int(tasks[i][1][1]),
                    block=int(tasks[i][1][2]),
                    task_index=i,
                )
                for i in chunk_idx
            )
            views = tuple(_single_kv_view(kv_contexts[i]) for i in chunk_idx)
            batches.append(ComputeBatch(members=members, kv_views=views))
    return batches


def _single_kv_view(ctx_list: Any) -> Any:
    if isinstance(ctx_list, list):
        if len(ctx_list) != 1:
            raise ValueError(f"Cell requires exactly one KV view, got {len(ctx_list)}")
        return ctx_list[0]
    return ctx_list


def slice_activation_for_task(
    src: Mapping[str, torch.Tensor] | None,
    *,
    task_index: int,
    task_count: int,
) -> dict[str, torch.Tensor] | None:
    """Map a slot-packed activation payload to one member's tensors."""
    if src is None:
        return None
    if not 0 <= task_index < task_count:
        raise ValueError("Activation task index is out of range")
    out: dict[str, torch.Tensor] = {}
    for key, tensor in src.items():
        if not isinstance(tensor, torch.Tensor) or tensor.ndim < 1 or tensor.shape[0] != task_count:
            raise ValueError(f"Activation {key!r} must have one row per member ({task_count})")
        out[key] = tensor[task_index : task_index + 1]
    return out
