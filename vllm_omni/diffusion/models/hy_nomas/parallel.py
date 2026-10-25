# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from vllm_omni.diffusion.distributed.parallel_state import get_sp_group


def initialize_reference_parallelism(sp_size: int, device: torch.device) -> DeviceMesh | None:
    from hymm.distributed.parallel_states import (
        ParallelDims,
        initialize_sequence_parallel_state_v2,
        nccl_info,
        set_sequence_parallel_state,
        set_tensor_parallel_state,
    )

    dims = ParallelDims(dp_replicate=1, dp_shard=1, sp=sp_size, tp=1, pp=1, world_size=sp_size)
    mesh = None
    if sp_size > 1:
        group = get_sp_group()
        if group.ulysses_world_size != sp_size or group.world_size != sp_size:
            raise ValueError("Hy-Nomas requires the initialized SP group to match ulysses_degree.")
        # Reuse the worker's group; vLLM-Omni owns its creation and destruction.
        mesh = DeviceMesh.from_group(group.ulysses_group, device.type, mesh_dim_names=("sp",))
        dims.sp_mesh = mesh
        dims.sp_group = group.ulysses_group
    else:
        dims.sp_group = None
    dims.device_mesh = mesh
    # The upstream single-device path leaves stale SP state when a model is reloaded.
    set_sequence_parallel_state(False)
    set_tensor_parallel_state(False)
    nccl_info.sp_group = None
    nccl_info.tp_group = None
    nccl_info.use_dynamic_ring_attention = False
    nccl_info.sp_stream = None
    nccl_info.sp_rank_list = None
    initialize_sequence_parallel_state_v2(dims, mesh)
    # Use the initialized worker rank instead of the torchrun RANK environment variable.
    nccl_info.global_rank = dist.get_rank() if dist.is_initialized() else 0
    return mesh
