# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Process-group state for independently sized VAE parallelism."""

from __future__ import annotations

import torch.distributed as dist
from vllm.logger import init_logger

logger = init_logger(__name__)

_VAE_GROUP: dist.ProcessGroup | None = None
_VAE_GROUP_RANKS: list[int] | None = None
_INDEPENDENT_VAE_GROUP_MODELS = frozenset({"MiniMaxH3Pipeline", "MiniMaxH3ModularPipeline"})


def supports_independent_vae_process_group(model_class_name: str) -> bool:
    """Return whether a pipeline can bind its VAE after construction."""

    return model_class_name in _INDEPENDENT_VAE_GROUP_MODELS


def requires_independent_vae_process_group(model_class_name: str, world_size: int, group_size: int) -> bool:
    """Return whether a smaller dedicated group is required."""

    return supports_independent_vae_process_group(model_class_name) and 1 < group_size < world_size


def validate_vae_parallel_group_size(world_size: int, group_size: int) -> None:
    """Validate a VAE subgroup size against the actual worker WORLD."""
    if world_size <= 0:
        raise ValueError(f"world_size must be greater than 0, got {world_size}")
    if group_size <= 0:
        raise ValueError(f"vae_patch_parallel_size must be greater than 0, got {group_size}")
    if group_size > world_size:
        raise ValueError(f"vae_patch_parallel_size ({group_size}) cannot exceed diffusion world_size ({world_size})")
    if world_size % group_size != 0:
        raise ValueError(
            f"vae_patch_parallel_size ({group_size}) must evenly divide diffusion world_size ({world_size})"
        )


def validate_independent_vae_parallel_config(world_size: int, group_size: int, mode: str) -> None:
    """Validate the MiniMax-H3 subgroup contract before collectives start."""

    validate_vae_parallel_group_size(world_size, group_size)
    if mode != "tile":
        raise ValueError(f"independent MiniMax-H3 VAE process groups support tile mode only, got {mode!r}")


def generate_contiguous_rank_groups(world_size: int, group_size: int) -> list[list[int]]:
    """Partition WORLD into deterministic contiguous groups of equal size."""
    validate_vae_parallel_group_size(world_size, group_size)
    return [list(range(start, start + group_size)) for start in range(0, world_size, group_size)]


def initialize_vae_parallel_group(group_size: int, backend: str | None = None) -> None:
    """Create this rank's VAE subgroup using one global creation schedule."""
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before VAE parallel initialization")

    global _VAE_GROUP, _VAE_GROUP_RANKS
    if _VAE_GROUP_RANKS is not None:
        raise RuntimeError("VAE parallel group is already initialized")

    group_size = int(group_size)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    rank_groups = generate_contiguous_rank_groups(world_size, group_size)
    if group_size == 1:
        _VAE_GROUP_RANKS = [rank]
        return

    try:
        for ranks in rank_groups:
            group = dist.new_group(ranks=ranks, backend=backend)
            if rank in ranks:
                _VAE_GROUP = group
                _VAE_GROUP_RANKS = ranks
    except Exception:
        local_group = _VAE_GROUP
        _VAE_GROUP = None
        _VAE_GROUP_RANKS = None
        if local_group is not None:
            try:
                dist.destroy_process_group(local_group)
            except Exception:
                logger.exception("Failed to destroy a partially initialized VAE process group")
        raise

    if _VAE_GROUP is None or _VAE_GROUP_RANKS is None:
        raise RuntimeError(f"rank {rank} was not assigned to a VAE parallel group")
    logger.info(
        "Initialized independent VAE process group: global_rank=%d, ranks=%s, size=%d",
        rank,
        _VAE_GROUP_RANKS,
        group_size,
    )


def get_vae_group() -> dist.ProcessGroup:
    """Return this rank's VAE process group."""
    if _VAE_GROUP is None:
        raise RuntimeError("VAE group is not initialized (vae_patch_parallel_size must be greater than 1)")
    return _VAE_GROUP


def get_vae_parallel_world_size() -> int:
    return len(get_vae_group_ranks())


def get_vae_parallel_rank() -> int:
    ranks = get_vae_group_ranks()
    return ranks.index(dist.get_rank())


def get_vae_group_ranks() -> list[int]:
    if _VAE_GROUP_RANKS is None:
        raise RuntimeError("VAE parallel state is not initialized")
    return list(_VAE_GROUP_RANKS)


def destroy_vae_parallel_group() -> None:
    global _VAE_GROUP, _VAE_GROUP_RANKS
    group = _VAE_GROUP
    _VAE_GROUP = None
    _VAE_GROUP_RANKS = None
    if group is not None:
        dist.destroy_process_group(group)
