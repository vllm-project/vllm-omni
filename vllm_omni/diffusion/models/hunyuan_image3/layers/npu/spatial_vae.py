# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Opt-in height sharding using the framework-owned VAE executor group."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from typing import TYPE_CHECKING, Literal

import torch
import torch.distributed as dist
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.autoencoders.wan_spatial_shard import all_gather_along_dim
from vllm_omni.diffusion.models.hunyuan_image3.layers.native.vae_context import VAE_SPATIAL_CONTEXT, VaeSpatialContext
from vllm_omni.diffusion.models.hunyuan_image3.layers.npu.autoencoder_blocks import VaeConv3d

if TYPE_CHECKING:
    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_hunyuan import DistributedAutoencoderKLHunyuan
    from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import DistributedVaeExecutor

logger = init_logger(__name__)


def spatial_vae_forward(
    vae: DistributedAutoencoderKLHunyuan,
    x: torch.Tensor,
    stage: Literal["encode", "decode"],
) -> torch.Tensor | None:
    if stage not in ("encode", "decode"):
        raise ValueError(f"Unknown VAE stage: {stage}")
    # Keep conditional-image latents on the established tile encoder.
    # Full-image normalization changes the conditioning seen by DiT.
    if stage == "encode":
        return None
    if vae.distributed_executor.parallel_mode != "spatial_shard_height":
        return None
    if x.device.type not in ("npu", "privateuseone"):
        logger.warning_once("HunyuanImage3 spatial VAE decode requires NPU input; using tile execution")
        return None
    if torch.is_grad_enabled() or torch.compiler.is_compiling():
        logger.warning_once(
            "HunyuanImage3 spatial VAE decode requires eager inference without gradients; using tile execution"
        )
        return None
    executor = vae.distributed_executor
    group = executor.group
    world = dist.get_world_size(group)
    network = vae.decoder
    if world <= 1 or executor.parallel_size != world:
        logger.warning_once("HunyuanImage3 spatial VAE requires a matching VAE device group; using tile execution")
        return None
    if x.ndim != 5 or x.shape[3] % world or not x.numel():
        logger.warning_once(
            "HunyuanImage3 spatial %s requires aligned nonempty NCTHW input; got %s", stage, tuple(x.shape)
        )
        return None
    if network.training or any(
        not isinstance(m, VaeConv3d)
        or m.stride != (1, 1, 1)
        or m.dilation != (1, 1, 1)
        or m.padding_mode != "zeros"
        or not isinstance(m.padding, tuple)
        or any(k != 2 * p + 1 for k, p in zip(m.kernel_size, m.padding))
        for m in network.modules()
        if isinstance(m, nn.Conv3d)
    ):
        logger.warning_once("HunyuanImage3 spatial %s found an unsupported convolution; using tile execution", stage)
        return None
    return run_spatial_vae(network, x, group, executor=executor)


def run_spatial_vae(
    network: nn.Module,
    x: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    executor: DistributedVaeExecutor | None = None,
) -> torch.Tensor:
    """Execute equal-height, prevalidated shards; also used by the component benchmark."""
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    with ExitStack() as local_stage:
        communication_guard: Callable[[str], AbstractContextManager] = nullcontext
        if executor is not None:
            local_stage.enter_context(executor._sync_errors("spatial decode before first communication"))

            @contextmanager
            def checked_communication(stage: str):
                # Agree on all local work since the previous collective,
                # including input/receive allocations for this collective.
                local_stage.close()
                yield
                local_stage.enter_context(executor._sync_errors(f"spatial decode after {stage}"))

            communication_guard = checked_communication

        token = VAE_SPATIAL_CONTEXT.set(VaeSpatialContext(group, communication_guard))
        try:
            with torch.autocast("npu", dtype=torch.bfloat16):
                local = x.chunk(world, dim=3)[rank].to(torch.bfloat16).contiguous()
                y = network(local)
                return all_gather_along_dim(y, group=group, dim=3, communication_guard=communication_guard)
        finally:
            VAE_SPATIAL_CONTEXT.reset(token)
