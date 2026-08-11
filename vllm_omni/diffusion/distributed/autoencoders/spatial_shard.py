# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch
import torch.distributed as dist
from vllm.logger import init_logger

logger = init_logger(__name__)


class SpatialShardExecutor(Protocol):
    group: dist.ProcessGroup
    parallel_mode: str
    parallel_size: int

    def _sync_final_result(
        self,
        rank0_result: torch.Tensor,
        output_ndim: int,
        output_device: torch.device,
        output_dtype: torch.dtype,
    ) -> torch.Tensor: ...


class SpatialShardVAE(Protocol):
    distributed_executor: SpatialShardExecutor
    decoder: torch.nn.Module
    post_quant_conv: torch.nn.Module
    dtype: torch.dtype
    _conv_idx: list[int]
    _feat_map: list[object]

    def is_distributed_enabled(self) -> bool: ...

    def clear_cache(self) -> None: ...


def select_auto_spatial_shard_split_dim(
    z_shape: tuple[int, ...],
    world_size: int,
) -> str | None:
    """Select the longer spatial axis that can be split across every rank."""
    if len(z_shape) != 5 or world_size <= 1:
        return None

    _, _, _, height, width = z_shape
    candidates = [
        (extent, split_dim) for extent, split_dim in ((height, "height"), (width, "width")) if extent >= world_size
    ]
    if not candidates:
        return None
    # Prefer width for square latents. It was modestly faster in the original
    # four-GPU Wan benchmark and keeps the choice deterministic across models.
    return max(candidates, key=lambda candidate: (candidate[0], candidate[1] == "width"))[1]


def resolve_spatial_shard_split_dim(
    mode: str,
    z: torch.Tensor | None = None,
    world_size: int | None = None,
) -> str | None:
    if mode == "spatial_shard_width":
        return "width"
    if mode == "spatial_shard_height":
        return "height"
    if mode == "auto" and z is not None and world_size is not None:
        return select_auto_spatial_shard_split_dim(tuple(z.shape), world_size)
    return None


def spatial_shard_decode_enabled(vae: SpatialShardVAE, z: torch.Tensor, *, model_name: str) -> bool:
    """Apply the shared request- and topology-level spatial decode policy."""
    executor = vae.distributed_executor
    if executor.parallel_mode == "tile":
        return False
    if z.ndim != 5:
        logger.warning(
            "%s VAE spatial sharded decode expects 5D latent input; falling back to tiled decode.", model_name
        )
        return False
    if not vae.is_distributed_enabled():
        return False

    group = executor.group
    world_size = dist.get_world_size(group=group)
    requested_size = int(executor.parallel_size)
    requested_split_dim = resolve_spatial_shard_split_dim(executor.parallel_mode)
    if requested_size != world_size:
        if executor.parallel_mode == "auto":
            logger.debug(
                "%s VAE auto decode selected tile mode for a partial DiT group (requested=%s dit_group=%s)",
                model_name,
                requested_size,
                world_size,
            )
        elif requested_split_dim is not None:
            logger.warning(
                "%s VAE spatial sharded decode currently requires vae_patch_parallel_size "
                "to match the DIT group size; falling back to tiled decode. "
                "requested=%s dit_group=%s split_dim=%s",
                model_name,
                requested_size,
                world_size,
                requested_split_dim,
            )
        return False
    return resolve_spatial_shard_split_dim(executor.parallel_mode, z, world_size) is not None


def prepare_spatial_shard_decode(
    vae: SpatialShardVAE,
    *,
    install: Callable[[SpatialShardVAE, dist.ProcessGroup, str], None],
) -> None:
    """Install one VAE's dynamic wrappers while allocations are still tagged."""
    executor = vae.distributed_executor
    mode = executor.parallel_mode
    if mode not in {"auto", "spatial_shard_height", "spatial_shard_width"}:
        return
    requested_size = int(executor.parallel_size)
    world_size = dist.get_world_size(group=executor.group)
    if requested_size <= 1 or requested_size != world_size:
        return
    split_dim = "width" if mode == "spatial_shard_width" else "height"
    install(vae, executor.group, split_dim)


def prepare_pipeline_spatial_shard_decode(pipeline: torch.nn.Module) -> None:
    """Prepare every VAE that advertises spatial-shard decode capability."""
    from vllm_omni.diffusion.offloader.module_collector import ModuleDiscovery

    if not dist.is_initialized():
        return
    for vae in ModuleDiscovery.discover(pipeline).vaes:
        prepare = getattr(vae, "_prepare_spatial_shard_decode", None)
        if callable(prepare):
            prepare()
