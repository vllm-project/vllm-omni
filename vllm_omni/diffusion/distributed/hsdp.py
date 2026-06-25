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
    target_device: torch.device | None = None,
) -> nn.Module:
    """
    Apply HSDP sharding to a model that already has weights loaded.

    This function redistributes the model's parameters across GPUs using HSDP.
    The model should already have its weights loaded via the standard load_weights method.

    Args:
        model: Model instance with weights already loaded
        hsdp_config: HSDP configuration with HSDP mesh dimensions
        target_device: Worker's execution device. When the model declares
            _hsdp_ignored_modules, those modules are excluded from FSDP's
            mesh-driven device placement, so the caller must specify where to
            put them. Optional only when there are no ignored modules.

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

    # When the model contains FP8 parameters (online quantization), let FSDP
    # keep the original storage dtype on all-gather instead of casting to
    # hsdp_config.param_dtype (typically bfloat16). FP8 GEMM kernels expect
    # FP8 inputs; an implicit FP8 -> bf16 cast would silently break them.
    has_fp8_params = any(p.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) for p in model.parameters())
    mp_policy = MixedPrecisionPolicy(
        param_dtype=None if has_fp8_params else hsdp_config.param_dtype,
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

    # Collect parameters of any modules the model wants excluded from FSDP sharding.
    # See _hsdp_ignored_modules on each model class. These params keep their
    # original dtype/storage and are not subject to MixedPrecisionPolicy on the root
    # FSDP wrap. Useful for small auxiliary modules (e.g., timestep embedders) that
    # need to stay in a higher-precision dtype than the bulk of the model.
    ignored_module_names = getattr(model, "_hsdp_ignored_modules", []) or []
    ignored_params: set[nn.Parameter] = set()
    # FSDP wraps move sharded params to mesh devices automatically, but
    # ignored modules stay wherever load_weights left them (CPU under HSDP).
    # The caller (diffusers_loader / similar) is responsible for telling us
    # which device the worker should run on, matching the convention used
    # for VAEs / encoders / resident_modules.
    if ignored_module_names and target_device is None:
        raise ValueError(
            f"Model {type(model).__name__} declares _hsdp_ignored_modules="
            f"{ignored_module_names} but apply_hsdp_to_model was called "
            "without target_device. The caller must pass target_device so "
            "the ignored modules can be placed on the worker's execution device."
        )
    for mod_name in ignored_module_names:
        sub_mod = getattr(model, mod_name, None)
        if sub_mod is None:
            logger.warning("_hsdp_ignored_modules entry %r not found on model", mod_name)
            continue
        sub_mod.to(target_device)
        ignored_params.update(sub_mod.parameters())
    if ignored_params:
        logger.info(
            "HSDP excluding %d parameter tensors from sharding (modules: %s, moved to %s)",
            len(ignored_params),
            ignored_module_names,
            target_device,
        )

    # Apply HSDP sharding, this will automatically handle weight distribution
    shard_model(
        model,
        reshard_after_forward=hsdp_config.reshard_after_forward,
        mp_policy=mp_policy,
        mesh=device_mesh,
        hsdp_shard_conditions=hsdp_shard_conditions,
        ignored_params=ignored_params if ignored_params else None,
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
    ignored_params: set[nn.Parameter] | None = None,
) -> None:
    """Apply HSDP sharding to model modules based on shard conditions.

    ignored_params (if provided) are excluded from the root fully_shard
    wrap, so they are not collected into the root flat-parameter, are not
    subject to MixedPrecisionPolicy, and retain their original dtype.
    Per-submodule shard wraps do not receive ignored_params because the
    ignored modules are expected to live at the root level, not inside any
    block matched by hsdp_shard_conditions.
    """
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

    root_kwargs = dict(hsdp_kwargs)
    if ignored_params:
        root_kwargs["ignored_params"] = ignored_params
    fully_shard(model, **root_kwargs)
    logger.info(
        "Sharded %d modules + root (ignored_params=%d)",
        num_sharded,
        len(ignored_params) if ignored_params else 0,
    )
