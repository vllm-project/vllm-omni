# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact resolution and layer-count profiles for Ming-Image compilation."""

import time

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.forward_context import (
    set_forward_context_direct_condition,
    set_forward_context_ref_latent,
)

logger = init_logger(__name__)


@torch.inference_mode()
def warmup_compile_buckets(pipeline) -> None:
    """Compile and capture repeated DiT blocks directly for each profile shape."""
    start = time.perf_counter()
    dtype, device = pipeline.od_config.dtype, pipeline.device
    query = torch.zeros((1, 256, 2048), device=device, dtype=dtype)
    direct = torch.zeros((1, 1, 6144), device=device, dtype=dtype)
    cap_feats, direct_condition = pipeline.conditioning(query, direct)
    layered = pipeline.is_layer_decomposition
    batch = 2 if layered else 1
    if layered:
        direct_condition = torch.cat([direct_condition, torch.zeros_like(direct_condition)])
        cap_feats = torch.cat([cap_feats, torch.zeros_like(cap_feats)])
    timestep = torch.ones(batch, device=device, dtype=torch.float32)

    set_forward_context_direct_condition(direct_condition)
    for height, width, num_layers in pipeline.compile_buckets:
        latent_height = height // pipeline.vae_scale_factor
        latent_width = width // pipeline.vae_scale_factor
        frames = num_layers + 1 if layered else 1
        latents = torch.zeros(
            (batch, pipeline.transformer.in_channels, frames, latent_height, latent_width),
            device=device,
            dtype=dtype,
        )
        reference = torch.zeros_like(latents[:, :, :1]) if layered else None
        set_forward_context_ref_latent(reference)
        for _ in range(3):
            torch.compiler.cudagraph_mark_step_begin()
            pipeline.transformer(list(latents.unbind()), timestep, list(cap_feats.unbind()))
        torch.accelerator.synchronize()
        logger.info("Ming-Image compile bucket warmed: %dx%d, layers=%d", height, width, num_layers)
    logger.info("Ming-Image compile warmup complete in %.3f s", time.perf_counter() - start)
