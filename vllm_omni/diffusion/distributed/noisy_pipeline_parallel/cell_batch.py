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


# Pack one compatible CellComputeBatch into model tensors and split-ready outputs.
CellBatchPackFn = Callable[..., Mapping[str, torch.Tensor]]


class BatchedCellAdapter:
    """Framework multi-member CellAdapter: organize Batches, delegate pack to the model.

    Members are per-request adapters (``run_one`` for B=1). ``pack_batch`` is the
    model backend that packs a ``CellComputeBatch`` (B>1) and returns batch-axis
    tensors for ``split_outputs``.
    """

    def __init__(
        self,
        members: Mapping[str, Any],
        *,
        max_batch_size: int,
        compatibility_of: Callable[[str, Cell], Hashable],
        pack_batch: CellBatchPackFn,
    ) -> None:
        if not members:
            raise ValueError("BatchedCellAdapter requires at least one member")
        if max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")
        self.members = dict(members)
        self.max_batch_size = int(max_batch_size)
        self._compatibility_of = compatibility_of
        self._pack_batch = pack_batch
        first = next(iter(self.members.values()))
        self.device = getattr(first, "device", None)
        self.dtype = getattr(first, "dtype", None)
        self._stash: dict[tuple[str, int, int], dict[str, torch.Tensor]] = {}
        self._last_meta: tuple[str, int, int] | None = None
        self.last_batches: list[CellComputeBatch] = []
        self.block_call_members: list[int] = []

    def stash_activation(
        self,
        payload: dict,
        *,
        handles: list | None = None,
        meta: tuple[str, int, int] | None = None,
    ) -> None:
        del handles
        tensors = {k: v for k, v in payload.items() if isinstance(v, torch.Tensor) and k not in ("chunk", "step")}
        key = meta or self._last_meta
        if key is None:
            raise RuntimeError("stash_activation needs meta")
        self._stash[key] = tensors

    def take_stashed_activation(self, req: str, chunk: int, step: int) -> Any | None:
        return self._stash.pop((req, chunk, step), None) or self._stash.pop(("", chunk, step), None)

    def pack_activation(self, output: Any) -> dict:
        return next(iter(self.members.values())).pack_activation(output)

    def activation_spec(self, *, batch: int, include_hidden: bool):
        return next(iter(self.members.values())).activation_spec(batch=batch, include_hidden=include_hidden)

    def unpack_activation(self, payload: dict) -> Any:
        return payload

    def forward(self, tasks, kv_contexts, *, hidden):
        if not tasks:
            return hidden
        batches = organize_cell_batches(
            tasks,
            kv_contexts,
            compatibility_of=self._compatibility_of,
            max_batch_size=self.max_batch_size,
        )
        self.last_batches = batches
        self.block_call_members = []
        outs: list[dict[str, torch.Tensor] | None] = [None] * len(tasks)
        for batch in batches:
            chunk, step, _block = batch.cell
            if batch.size == 1:
                member_ref = batch.members[0]
                member = self.members[member_ref.request_id]
                self._last_meta = (member_ref.request_id, chunk, step)
                self.block_call_members.append(1)
                outs[member_ref.task_index] = member.run_one(
                    member_ref.request_id,
                    member_ref.cell,
                    batch.kv_views[0],
                    index=member_ref.task_index,
                    n_tasks=len(tasks),
                    hidden=hidden,
                    stash_take=self.take_stashed_activation,
                    bind_text=True,
                )
                continue
            self._last_meta = ("", chunk, step)
            self.block_call_members.append(batch.size)
            packed = dict(
                self._pack_batch(
                    batch,
                    members=[self.members[m.request_id] for m in batch.members],
                    tasks=tasks,
                    hidden=hidden,
                    stash_take=self.take_stashed_activation,
                )
            )
            for local_i, per_member in enumerate(batch.split_outputs(packed)):
                outs[batch.members[local_i].task_index] = per_member
        assert all(item is not None for item in outs)
        typed_outs = [item for item in outs if item is not None]
        if len(typed_outs) == 1:
            return typed_outs[0]
        keys = typed_outs[0].keys()
        return {k: torch.cat([o[k] for o in typed_outs], dim=0) for k in keys}
