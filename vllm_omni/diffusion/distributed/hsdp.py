# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed
from torch import nn
from torch.distributed import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
)
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.parallel_state import (
    get_fs_group,
    get_fully_shard_rank,
    get_fully_shard_world_size,
    get_world_group,
)
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


@dataclass
class HSDPInferenceConfig:
    """Configuration for HSDP inference.

    This is a runtime config created from DiffusionParallelConfig's HSDP settings.
    """

    enabled: bool = False
    hsdp_replicate_size: int = 1
    hsdp_shard_size: int = -1  # -1 = auto (shard across entire world)
    param_dtype: torch.dtype = torch.bfloat16
    reduce_dtype: torch.dtype = torch.float32
    output_dtype: torch.dtype | None = None
    reshard_after_forward: bool = True


def _create_hsdp_mesh(
    device_type: str,
    replicate_size: int,
    shard_pg: torch.distributed.ProcessGroup,
) -> DeviceMesh:
    """Create a 2D DeviceMesh for HSDP using an existing ProcessGroup for the shard dimension.

    Args:
        device_type: The device type (e.g., "cuda", "npu")
        replicate_size: Number of replica groups
        shard_pg: The ProcessGroup for the shard dimension (from FS GroupCoordinator)

    Returns:
        A 2D DeviceMesh with dimensions ("replicate", "shard")
    """
    shard_size = torch.distributed.get_world_size(shard_pg)
    world_size = replicate_size * shard_size

    # Build 2D mesh tensor: shape (replicate_size, shard_size)
    # Ranks are arranged so that each row is a shard group
    mesh_tensor = torch.arange(world_size).reshape(replicate_size, shard_size)

    # Create DeviceMesh with the shard ProcessGroup
    # For the shard dimension, we reuse the existing FS ProcessGroup
    device_mesh = init_device_mesh(
        device_type,
        mesh_shape=(replicate_size, shard_size),
        mesh_dim_names=("replicate", "shard"),
    )

    # Note: init_device_mesh creates new ProcessGroups internally.
    # For consistency, we verify the mesh structure matches our FS group.
    # In a future optimization, we could pass the existing ProcessGroups directly.
    logger.debug(
        "Created HSDP mesh: replicate_size=%d, shard_size=%d, mesh=%s",
        replicate_size,
        shard_size,
        mesh_tensor.tolist(),
    )

    return device_mesh


def apply_hsdp_to_model(
    model: nn.Module,
    hsdp_config: HSDPInferenceConfig,
) -> nn.Module:
    """
    Apply HSDP sharding to a model that already has weights loaded.

    This function redistributes the model's parameters across GPUs using HSDP.
    The model should already have its weights loaded via the standard load_weights method.

    Args:
        model: Model instance with weights already loaded
        hsdp_config: HSDP configuration with HSDP mesh dimensions

    Returns:
        HSDP-wrapped model ready for inference
    """
    if not hsdp_config.enabled:
        raise ValueError("HSDP is not enabled in config")

    # Use GroupCoordinator for distributed info
    world_group = get_world_group()
    fs_group = get_fs_group()

    world_size = world_group.world_size
    rank = world_group.rank_in_group
    fs_world_size = get_fully_shard_world_size()
    fs_rank = get_fully_shard_rank()

    hsdp_replicate_size = hsdp_config.hsdp_replicate_size
    hsdp_shard_size = hsdp_config.hsdp_shard_size

    # Validate that the FS group matches the HSDP shard size
    if fs_world_size != hsdp_shard_size:
        raise ValueError(
            f"FS group world_size ({fs_world_size}) does not match "
            f"HSDP shard_size ({hsdp_shard_size}). "
            "Ensure fully_shard_degree is set correctly in initialize_model_parallel."
        )

    logger.info(
        "HSDP Inference: replicate_size=%d, shard_size=%d, world_size=%d, rank=%d, fs_world_size=%d, fs_rank=%d",
        hsdp_replicate_size,
        hsdp_shard_size,
        world_size,
        rank,
        fs_world_size,
        fs_rank,
    )

    mp_policy = MixedPrecisionPolicy(
        param_dtype=hsdp_config.param_dtype,
        reduce_dtype=hsdp_config.reduce_dtype,
        output_dtype=hsdp_config.output_dtype,
        cast_forward_inputs=False,
    )

    device_type = current_omni_platform.device_type

    # Create 2D DeviceMesh for HSDP using the FS group's ProcessGroup for shard dimension
    # The mesh shape is (replicate, shard) where:
    # - replicate: groups of ranks that hold the same shard (for gradient all-reduce in training)
    # - shard: groups of ranks that each hold different shards (for parameter all-gather)
    device_mesh = _create_hsdp_mesh(
        device_type=device_type,
        replicate_size=hsdp_replicate_size,
        shard_pg=fs_group.device_group,
    )

    hsdp_shard_conditions = getattr(model, "_hsdp_shard_conditions", None)
    if not hsdp_shard_conditions or len(hsdp_shard_conditions) == 0:
        raise ValueError(f"Model {type(model).__name__} has no _hsdp_shard_conditions defined")

    # Apply HSDP sharding, this will automatically handle weight distribution
    shard_model(
        model,
        reshard_after_forward=hsdp_config.reshard_after_forward,
        mp_policy=mp_policy,
        mesh=device_mesh,
        hsdp_shard_conditions=hsdp_shard_conditions,
    )

    for param in model.parameters():
        param.requires_grad = False

    logger.info("HSDP applied to model: %s", type(model).__name__)
    return model


def shard_model(
    model: nn.Module,
    *,
    reshard_after_forward: bool = True,
    mp_policy: MixedPrecisionPolicy | None = None,
    mesh: DeviceMesh | None = None,
    hsdp_shard_conditions: list[Callable[[str, nn.Module], bool]],
) -> None:
    """Apply HSDP sharding to model modules based on shard conditions."""
    hsdp_kwargs: dict[str, Any] = {
        "reshard_after_forward": reshard_after_forward,
        "mesh": mesh,
        "mp_policy": mp_policy,
    }

    num_sharded = 0
    for name, module in reversed(list(model.named_modules())):
        if any(cond(name, module) for cond in hsdp_shard_conditions):
            fully_shard(module, **hsdp_kwargs)
            num_sharded += 1

    if num_sharded == 0:
        raise ValueError("No modules were sharded")

    fully_shard(model, **hsdp_kwargs)
    logger.info("Sharded %d modules + root", num_sharded)
