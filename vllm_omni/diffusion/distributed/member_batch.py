# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ordered compute members and output ownership, independent of plan granularity.

Step InputBatch and Cell microbatches retain their own packing rules. This module
owns the common invariant: one request unit per member, one KV view per member,
and an exact, stable mapping from packed result rows back to those members.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

import torch

InputT = TypeVar("InputT")
ViewT = TypeVar("ViewT")
ResultT = TypeVar("ResultT")
PositionT_co = TypeVar("PositionT_co", covariant=True)


class ComputePlan(Protocol[PositionT_co]):
    """Immutable ordering/placement; backends supply their own dependency rules."""

    @property
    def num_slots(self) -> int: ...

    def task(self, slot: int, rank: int) -> PositionT_co | None: ...


def planned_task(plan: ComputePlan[PositionT_co], slot: int, rank: int) -> PositionT_co | None:
    """Read a ready unit only inside this plan's execution extent."""
    return plan.task(slot, rank) if 0 <= slot < plan.num_slots else None


@dataclass(frozen=True)
class ChunkStepPlan:
    """One chunk's steps; a new plan is bound after the model advances a chunk."""

    chunk: int
    steps: int

    def __post_init__(self) -> None:
        if self.chunk < 0 or self.steps <= 0:
            raise ValueError("A chunk plan requires a non-negative chunk and positive steps")

    @property
    def num_slots(self) -> int:
        return self.steps

    def task(self, slot: int, rank: int) -> tuple[int, int] | None:
        if rank != 0:
            raise ValueError("Chunk-step plan has one logical execution position")
        return (self.chunk, slot) if 0 <= slot < self.steps else None


@dataclass(frozen=True)
class BatchMember(Generic[InputT, ViewT]):
    request_id: str
    position: tuple[int, ...]
    inputs: InputT
    kv_view: ViewT
    rows: int = 1


@dataclass(frozen=True)
class ExecutionBatch(Generic[InputT, ViewT]):
    members: tuple[BatchMember[InputT, ViewT], ...]

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("ExecutionBatch cannot be empty")
        ids = [member.request_id for member in self.members]
        if any(not identity for identity in ids) or len(ids) != len(set(ids)):
            raise ValueError("An execution batch requires unique non-empty request ids")
        if any(member.rows <= 0 for member in self.members):
            raise ValueError("Each batch member must own a positive number of rows")

    @property
    def request_ids(self) -> tuple[str, ...]:
        return tuple(member.request_id for member in self.members)

    @property
    def kv_views(self) -> tuple[ViewT, ...]:
        return tuple(member.kv_view for member in self.members)

    def map_outputs(self, outputs: Sequence[ResultT]) -> dict[str, ResultT]:
        if len(outputs) != len(self.members):
            raise ValueError(f"Batch produced {len(outputs)} outputs for {len(self.members)} members")
        return dict(zip(self.request_ids, outputs, strict=True))

    def split_rows(self, output: torch.Tensor | None) -> dict[str, torch.Tensor | None]:
        if output is None:
            return dict.fromkeys(self.request_ids)
        expected = sum(member.rows for member in self.members)
        if output.ndim == 0 or output.shape[0] != expected:
            raise ValueError(f"Batch requires {expected} result rows, got shape {tuple(output.shape)}")
        offset = 0
        rows: list[torch.Tensor] = []
        for member in self.members:
            rows.append(output[offset : offset + member.rows])
            offset += member.rows
        return self.map_outputs(rows)


def execute_member_batch(
    batch: ExecutionBatch[InputT, ViewT],
    compute: Callable[[ExecutionBatch[InputT, ViewT]], ResultT],
    publish: Callable[[ExecutionBatch[InputT, ViewT], ResultT], object],
) -> ResultT:
    """Publish only after compute succeeds; failure cleanup belongs to the owner."""
    result = compute(batch)
    publish(batch, result)
    return result
