# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

from vllm_omni.diffusion.runtime_v2.protocol import ExecutionGroupSpec, ParallelSpec, TaskKind


@dataclass(frozen=True)
class WorkerSpec:
    worker_rank: int
    device_id: int | None = None
    label: str = ""


class RuntimeTopology:
    """Static runtime_v2 topology."""

    def __init__(self, workers: tuple[WorkerSpec, ...], groups: tuple[ExecutionGroupSpec, ...]) -> None:
        if not workers:
            raise ValueError("runtime topology requires at least one worker")
        if not groups:
            raise ValueError("runtime topology requires at least one execution group")

        self.workers = workers
        self.groups = groups
        self._workers_by_rank = {worker.worker_rank: worker for worker in workers}
        self._groups_by_id = {group.group_id: group for group in groups}
        self._groups_by_worker_rank: dict[int, list[ExecutionGroupSpec]] = {
            worker.worker_rank: [] for worker in workers
        }

        if len(self._workers_by_rank) != len(workers):
            raise ValueError("worker ranks must be unique")
        if len(self._groups_by_id) != len(groups):
            raise ValueError("group ids must be unique")

        for group in groups:
            for rank in group.ranks:
                if rank not in self._workers_by_rank:
                    raise ValueError(f"group {group.group_id} references unknown worker rank {rank}")
                self._groups_by_worker_rank[rank].append(group)
        missing = [rank for rank, groups_for_rank in self._groups_by_worker_rank.items() if not groups_for_rank]
        if missing:
            raise ValueError(f"workers must belong to at least one execution group, missing ranks: {missing!r}")

    @classmethod
    def single_group(cls, num_gpus: int, parallel_spec: ParallelSpec) -> RuntimeTopology:
        """Build a single-group topology with all TaskKinds supported."""
        workers = tuple(WorkerSpec(worker_rank=r, device_id=r) for r in range(num_gpus))
        group = ExecutionGroupSpec(
            group_id="g0",
            ranks=tuple(range(num_gpus)),
            parallel_spec=parallel_spec,
            supported_task_kinds=tuple(TaskKind),
        )
        return cls(workers=workers, groups=(group,))

    def get_group(self, group_id: str) -> ExecutionGroupSpec:
        return self._groups_by_id[group_id]

    def get_group_leader(self, group_id: str) -> int:
        return self.get_group(group_id).ranks[0]

    def get_groups_for_worker(self, worker_rank: int) -> tuple[ExecutionGroupSpec, ...]:
        return tuple(self._groups_by_worker_rank[worker_rank])

    def get_groups_for_task(self, task_kind: TaskKind) -> tuple[ExecutionGroupSpec, ...]:
        return tuple(group for group in self.groups if task_kind in group.supported_task_kinds)

    def select_group_for_task(self, task_kind: TaskKind) -> str | None:
        for group in self.groups:
            if task_kind in group.supported_task_kinds:
                return group.group_id
        return None
