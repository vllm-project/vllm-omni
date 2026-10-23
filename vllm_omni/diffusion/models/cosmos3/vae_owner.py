# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Spread one request's VAE tasks over the sequence-parallel x CFG-parallel grid.

Each task runs on one owner rank. Encode results are broadcast to every grid
rank; decode results are gathered on grid rank 0, the rank that returns the
request output. Without a usable grid every rank runs every task itself.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.parallel_state import (
    get_cfg_group,
    get_sp_group,
    model_parallel_is_initialized,
)
from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    from vllm_omni.diffusion.distributed.group_coordinator import GroupCoordinator

logger = init_logger(__name__)


@dataclass(frozen=True)
class VaeTask:
    """One VAE call that produces one tensor.

    ``cost`` only decides placement. ``shape`` and ``dtype`` describe the
    result, so ranks that do not own the task can allocate its buffer.
    ``prepare`` is CPU work run one task ahead on a background thread;
    ``run`` receives its result and runs on the owner's main thread.
    """

    key: str
    cost: float
    shape: tuple[int, ...]
    dtype: torch.dtype
    run: Callable[[Any], torch.Tensor]
    prepare: Callable[[], Any] | None = None


class OwnerScheduler:
    """Run VAE tasks on owner ranks of the SP x CFG grid.

    Grid rank = ``cfg_rank * sp_size + sp_rank``. Only the SP and CFG groups
    communicate, never WORLD, so data- and tensor-parallel replicas stay
    independent. Every rank must pass the same tasks in the same order.
    """

    def __init__(
        self, device: torch.device, sp: GroupCoordinator | None = None, cfg: GroupCoordinator | None = None
    ) -> None:
        self.device = device
        self.sp = sp
        self.cfg = cfg
        self.distributed = sp is not None and cfg is not None
        self.rank = cfg.rank_in_group * sp.world_size + sp.rank_in_group if self.distributed else 0
        self.is_output_rank = self.rank == 0
        self.stream = torch.cuda.Stream(device=device) if self.distributed else None

    @classmethod
    def create(cls, device: torch.device, vae: Any) -> OwnerScheduler:
        groups = _grid_groups(device, vae)
        if groups is None:
            logger.info_once("Cosmos3 multiview VAE tasks run on every rank.")
            return cls(device)
        sp, cfg = groups
        logger.info_once(
            "Cosmos3 multiview VAE tasks are spread over the SP=%d x CFG=%d grid.", sp.world_size, cfg.world_size
        )
        return cls(device, sp, cfg)

    def run_on_all(self, tasks: Sequence[VaeTask]) -> dict[str, torch.Tensor]:
        """Run every task once and return all results on every rank.

        With fewer than two tasks each rank runs them itself: a broadcast
        would take as long as the task.
        """
        distributed = self.distributed and len(tasks) > 1
        order, owners = _place(tasks, self.sp.world_size, self.cfg.world_size) if distributed else (tasks, None)
        mine = [task for task in order if owners is None or owners[task.key] == self.rank]
        results = {}
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="cosmos3_vae_prepare") as pool:
            prepared = _prefetched(pool, mine)
            for task in order:
                owner = self.rank if owners is None else owners[task.key]
                if owner == self.rank:
                    value = _checked(task, task.run(next(prepared)))
                else:
                    value = torch.empty(task.shape, dtype=task.dtype, device=self.device)
                if distributed:
                    value = value.contiguous()
                    self._broadcast(value, owner)
                results[task.key] = value
        if distributed:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
        return results

    def run_to_output(self, tasks: Sequence[VaeTask], outputs: Mapping[str, torch.Tensor] | None) -> None:
        """Run every task once and copy each result into ``outputs[key]`` on the output rank.

        ``outputs`` is None on the other ranks. Tasks take no ``prepare``.
        """
        if not self.distributed:
            for task in tasks:
                outputs[task.key].copy_(_checked(task, task.run(None)))
            return
        order, owners = _place(tasks, self.sp.world_size, self.cfg.world_size)
        # Queue every remote result before the output rank runs its own tasks.
        order = sorted(order, key=lambda task: owners[task.key] == 0)
        sp_rank, cfg_rank = self.sp.rank_in_group, self.cfg.rank_in_group
        for task in order:
            owner = owners[task.key]
            owner_cfg, owner_sp = divmod(owner, self.sp.world_size)
            if owner == self.rank:
                value = _checked(task, task.run(None)).contiguous()
            elif sp_rank == 0 and ((cfg_rank == owner_cfg and owner_sp != 0) or (cfg_rank == 0 and owner_cfg != 0)):
                # This rank receives the result, to relay it or to output it.
                value = torch.empty(task.shape, dtype=task.dtype, device=self.device)
            else:
                continue
            self._gather(value, owner, outputs[task.key] if self.is_output_rank else None)
            del value
        self.stream.synchronize()

    def _broadcast(self, tensor: torch.Tensor, owner: int) -> None:
        """Owner to its CFG row along SP, then every SP column along CFG."""
        owner_cfg, owner_sp = divmod(owner, self.sp.world_size)
        with self._on_stream(tensor):
            if self.cfg.rank_in_group == owner_cfg and self.sp.world_size > 1:
                dist.broadcast(tensor, src=self.sp.ranks[owner_sp], group=self.sp.device_group)
            if self.cfg.world_size > 1:
                dist.broadcast(tensor, src=self.cfg.ranks[owner_cfg], group=self.cfg.device_group)

    def _gather(self, tensor: torch.Tensor, owner: int, destination: torch.Tensor | None) -> None:
        """Owner to SP rank 0 of its CFG row, then along CFG to grid rank 0."""
        owner_cfg, owner_sp = divmod(owner, self.sp.world_size)
        sp, cfg = self.sp, self.cfg
        with self._on_stream(tensor):
            if cfg.rank_in_group == owner_cfg and owner_sp != 0:
                if sp.rank_in_group == owner_sp:
                    dist.send(tensor, dst=sp.ranks[0], group=sp.device_group)
                elif sp.rank_in_group == 0:
                    dist.recv(tensor, src=sp.ranks[owner_sp], group=sp.device_group)
            if sp.rank_in_group == 0 and owner_cfg != 0:
                if cfg.rank_in_group == owner_cfg:
                    dist.send(tensor, dst=cfg.ranks[0], group=cfg.device_group)
                elif cfg.rank_in_group == 0:
                    dist.recv(tensor, src=cfg.ranks[owner_cfg], group=cfg.device_group)
            if destination is not None:
                _copy_to_host(destination, tensor)

    @contextmanager
    def _on_stream(self, tensor: torch.Tensor) -> Iterator[None]:
        """Communicate on the side stream after the compute stream's pending work.

        NCCL waits only order the side stream, so the host keeps queuing VAE
        work. ``record_stream`` keeps the allocator from reusing ``tensor``
        while the side stream may still access it.
        """
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.stream):
            yield
        tensor.record_stream(self.stream)


def _grid_groups(device: torch.device, vae: Any) -> tuple[GroupCoordinator, GroupCoordinator] | None:
    """Return the SP and CFG groups to spread VAE tasks over, or None."""
    if device.type != "cuda" or not current_omni_platform.is_cuda() or not dist.is_initialized():
        return None
    # A tile-parallel VAE already splits each call over the ranks.
    distributed_vae = getattr(vae, "is_distributed_enabled", None)
    if (callable(distributed_vae) and distributed_vae()) or not model_parallel_is_initialized():
        return None
    sp, cfg = get_sp_group(), get_cfg_group()
    if sp.world_size * cfg.world_size == 1:
        return None
    if any(dist.get_backend(group.device_group) != "nccl" for group in (sp, cfg) if group.world_size > 1):
        return None
    return sp, cfg


def _place(tasks: Sequence[VaeTask], sp_size: int, cfg_size: int) -> tuple[list[VaeTask], dict[str, int]]:
    """Longest processing time first; returns the shared task order and owner grid ranks.

    The costliest task goes first, ties keep the given order, onto the least
    loaded rank; load ties prefer fewer hops to grid rank 0, then the lower rank.
    """
    order = sorted(tasks, key=lambda task: -task.cost)
    loads = [0.0] * (sp_size * cfg_size)
    owners = {}
    for task in order:
        owner = min(range(len(loads)), key=lambda r: (loads[r], (r % sp_size != 0) + (r >= sp_size), r))
        owners[task.key] = owner
        loads[owner] += task.cost
    return order, owners


def _prefetched(pool: ThreadPoolExecutor, tasks: Sequence[VaeTask]) -> Iterator[Any]:
    """Yield each task's ``prepare()`` result while ``pool`` prepares the following task."""

    def submit(task: VaeTask) -> Future | None:
        return pool.submit(task.prepare) if task.prepare is not None else None

    following = submit(tasks[0]) if tasks else None
    for index in range(len(tasks)):
        current, following = following, submit(tasks[index + 1]) if index + 1 < len(tasks) else None
        yield None if current is None else current.result()


def _checked(task: VaeTask, value: torch.Tensor) -> torch.Tensor:
    if tuple(value.shape) != task.shape or value.dtype != task.dtype:
        raise ValueError(
            f"VAE task {task.key!r} returned {tuple(value.shape)} {value.dtype}, expected {task.shape} {task.dtype}."
        )
    return value


def _copy_to_host(destination: torch.Tensor, tensor: torch.Tensor) -> None:
    if destination.is_contiguous():
        destination.copy_(tensor, non_blocking=True)
        return
    # A camera slice of [B, C, V*F, H, W] is not contiguous. Copy contiguous
    # [F, H, W] spans directly into pinned host memory: a strided D2H copy may
    # stage through a temporary on the host and is then not asynchronous.
    for batch in range(destination.shape[0]):
        for channel in range(destination.shape[1]):
            destination[batch, channel].copy_(tensor[batch, channel], non_blocking=True)
