# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.parallel_state import get_dit_group

logger = init_logger(__name__)


@dataclass
class GridSpec:
    """The Grid shape split"""

    split_dims: tuple[int, ...]  # For example: (3,4) for (B, C, T, H, W), (2,3) for (B, C, H, W)
    grid_shape: tuple[int, ...]  # For example: (nh, nw) for (B, C, T, H, W), (nh, nw) for (B, C, H, W)
    tile_spec: dict = field(default_factory=dict)
    output_dtype: torch.dtype | None = None


@dataclass
class TileTask:
    tile_id: int
    grid_coord: tuple[int, ...]  # The coordinate of the tile in GridSpec.grid_shape
    tensor: torch.Tensor | list[torch.Tensor]  # The tile tensor
    workload: int | float = 1


@dataclass
class DistributedOperator:
    split: callable
    exec: callable
    merge: callable


class DistributedVaeExecutor:
    """
    Abstract util class for distributed patch/tile parallel VAE decoding.
    """

    def __init__(self):
        self.group = get_dit_group()
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        self.parallel_size = 1

    def set_parallel_size(self, parallel_size: int):
        self.parallel_size = parallel_size

    def gather_tensors(self, tensor: torch.Tensor):
        gather_list = [torch.empty_like(tensor) for _ in range(self.world_size)] if self.rank == 0 else None
        dist.gather(tensor, gather_list=gather_list, dst=0, group=self.group)
        return gather_list

    def broadcast_tensor(self, tensor: torch.Tensor):
        dist.broadcast(tensor, src=0, group=self.group)
        return tensor

    def _compute_global_padding_shape(self, local_results, output_ndim: int, device):
        local_tile_max_dims = [0] * output_ndim
        for _, tile_tensor in local_results:
            for dim_idx, dim_size in enumerate(tile_tensor.shape):
                local_tile_max_dims[dim_idx] = max(local_tile_max_dims[dim_idx], dim_size)
        local_shape_stat = torch.tensor([len(local_results), *local_tile_max_dims], device=device)
        dist.all_reduce(
            local_shape_stat,
            op=dist.ReduceOp.MAX,
            group=self.group,
        )
        return local_shape_stat.tolist()

    def _pack_local_tiles(self, local_results, global_padding_shape, grid_spec, device, dtype):
        tile_tensor = torch.zeros(global_padding_shape, device=device, dtype=dtype)
        meta_tensor = torch.full(
            (global_padding_shape[0], len(grid_spec.split_dims) + 1), -1, device=device, dtype=torch.int64
        )
        for idx, (tid, t_tensor) in enumerate(local_results):
            meta_tensor[idx, 0] = tid
            for i, dim in enumerate(grid_spec.split_dims):
                meta_tensor[idx, i + 1] = t_tensor.shape[dim]
            slices = tuple(slice(0, s) for s in t_tensor.shape)
            tile_tensor[idx][slices] = t_tensor
        return tile_tensor, meta_tensor

    def _unpack_tiles(self, meta_gather, tile_gather, grid_spec, tid_coord_map):
        coord_tensor_map = {}
        for r in range(self.world_size):
            meta_src = meta_gather[r]
            tiles_src = tile_gather[r]
            for idx in range(meta_src.shape[0]):
                tid = int(meta_src[idx, 0])
                if tid < 0:
                    continue
                slices = [slice(None)] * tiles_src[idx].ndim
                for i, dim in enumerate(grid_spec.split_dims):
                    slices[dim] = slice(0, int(meta_src[idx, i + 1]))
                slices = tuple(slices)
                coord_tensor_map[tid_coord_map[tid]] = tiles_src[idx][slices]

        return coord_tensor_map

    def _balance_tasks(self, task_list, num_rank):
        workloads = [0] * num_rank
        assigned = [[] for _ in range(num_rank)]

        for task in sorted(task_list, key=lambda t: t.workload, reverse=True):
            r = workloads.index(min(workloads))
            assigned[r].append(task)
            workloads[r] += task.workload

        return assigned

    def execute(self, z: torch.Tensor, operator: DistributedOperator, broadcast_result: bool = True):
        pp_size = min(self.parallel_size, self.world_size)

        # 1. Split into tiles
        tiletask_list, grid_spec = operator.split(z)
        tid_coord_map = {task.tile_id: task.grid_coord for task in tiletask_list}

        # 2. local decode
        assigned = self._balance_tasks(tiletask_list, pp_size)
        local_tasks = assigned[self.rank] if pp_size <= self.world_size else []
        local_results = [(t.tile_id, operator.exec(t)) for t in local_tasks]

        # 3. compute shape per rank
        global_padding_shape = self._compute_global_padding_shape(local_results, z.ndim, z.device)

        # 5. prepare tile tensors
        output_dtype = grid_spec.output_dtype if grid_spec.output_dtype is not None else z.dtype
        local_tile_tensor, local_meta_tensor = self._pack_local_tiles(
            local_results, global_padding_shape, grid_spec, z.device, output_dtype
        )

        # 6. gather tiles & meta
        meta_gather = self.gather_tensors(local_meta_tensor)
        tile_gather = self.gather_tensors(local_tile_tensor)

        if self.rank != 0:
            result = torch.empty(0, device=z.device)  # Dummy return for non-zero ranks
        else:
            # 7. reconstruct full tensor (rank 0)
            coord_tensor_map = self._unpack_tiles(meta_gather, tile_gather, grid_spec, tid_coord_map)
            result = operator.merge(coord_tensor_map, grid_spec)

        if broadcast_result:
            result = self._sync_final_result(result, z.ndim, z.device, output_dtype)
        return result

    def _sync_final_result(self, rank0_result, output_ndim, output_device, output_dtype):
        shape_tensor = torch.empty((output_ndim,), device=output_device, dtype=torch.int64)
        if self.rank == 0:
            shape_tensor.copy_(torch.tensor(tuple(rank0_result.shape), device=output_device, dtype=torch.int64))
        dist.broadcast(shape_tensor, src=0, group=self.group)

        if self.rank != 0:
            sync_result = torch.empty(tuple(shape_tensor.tolist()), device=output_device, dtype=output_dtype)
        else:
            sync_result = rank0_result
        dist.broadcast(sync_result, src=0, group=self.group)
        return sync_result


class DistributedVaeMixin:
    def init_distributed(self):
        self.distributed_executor = DistributedVaeExecutor()

    def set_parallel_size(self, parallel_size: int) -> None:
        self.distributed_executor.set_parallel_size(parallel_size)

    def is_distributed_enabled(self) -> bool:
        if (
            self.distributed_executor.parallel_size <= 1
            or not dist.is_initialized()
            or not getattr(self, "use_tiling", False)
        ):
            return False
        world_size = dist.get_world_size(group=self.distributed_executor.group)
        pp_size = min(int(self.distributed_executor.parallel_size), int(world_size))
        if pp_size <= 1:
            return False
        if self.distributed_executor.parallel_size > pp_size:
            logger.warning(
                f"vae_patch_parallel_size={self.distributed_executor.parallel_size} "
                f"is greater than dit_group={world_size};"
                f" using dit_group size={world_size}"
            )
        return True
