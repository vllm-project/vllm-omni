# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact resolution and layer-count profiles for Ming-Image compilation."""

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.forward_context import (
    set_forward_context_direct_condition,
    set_forward_context_ref_latent,
)

if TYPE_CHECKING:
    from .pipeline import MingImageDiffusionPipeline

logger = init_logger(__name__)


@dataclass(frozen=True)
class MingImageCompileBucket:
    height: int
    width: int
    num_layers: int
    num_condition_tokens: int
    guidance_scale: float | None = None
    has_reference: bool = False

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0 or self.height % 16 or self.width % 16:
            raise ValueError("Ming-Image compile bucket height and width must be positive multiples of 16.")
        if self.num_layers < 1 or self.num_condition_tokens < 1:
            raise ValueError("Ming-Image compile bucket layer and conditioning token counts must be positive.")

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.height, self.width, self.num_layers


@torch.inference_mode()
def warmup_compile_buckets(pipeline: "MingImageDiffusionPipeline") -> None:
    """Compile and capture repeated DiT blocks directly for each profile shape."""
    start = time.perf_counter()
    dtype, device = pipeline.od_config.dtype, pipeline.device
    conditioning = pipeline.conditioning
    query = torch.zeros((1, 256, conditioning.query_encoder.thinker_hidden_size), device=device, dtype=dtype)
    layered = pipeline.is_layer_decomposition
    for bucket in pipeline.compile_buckets:
        bucket_start = time.perf_counter()
        direct = torch.zeros(
            (1, bucket.num_condition_tokens, conditioning.direct_projector[1].in_features), device=device, dtype=dtype
        )
        cap_feats, direct_condition = conditioning(query, direct)
        guidance = pipeline.default_guidance_scale if bucket.guidance_scale is None else bucket.guidance_scale
        batch = 2 if guidance > 1 else 1
        if batch == 2:
            direct_condition = torch.cat([direct_condition, torch.zeros_like(direct_condition)])
            cap_feats = torch.cat([cap_feats, torch.zeros_like(cap_feats)])
        timestep = torch.ones(batch, device=device, dtype=torch.float32)
        latent_height = bucket.height // pipeline.vae_scale_factor
        latent_width = bucket.width // pipeline.vae_scale_factor
        frames = bucket.num_layers + 1 if layered else 1
        latents = torch.zeros(
            (batch, pipeline.transformer.in_channels, frames, latent_height, latent_width),
            device=device,
            dtype=dtype,
        )
        has_reference = layered or bucket.has_reference
        reference = torch.zeros_like(latents[:, :, :1]) if has_reference else None
        set_forward_context_direct_condition(direct_condition)
        set_forward_context_ref_latent(reference)
        for _ in range(3):
            torch.compiler.cudagraph_mark_step_begin()
            pipeline.transformer(list(latents.unbind()), timestep, list(cap_feats.unbind()))
        torch.accelerator.synchronize()
        logger.info(
            "Ming-Image compile bucket warmed: %dx%d, layers=%d, num_condition_tokens=%d, "
            "cfg=%g, reference=%s, time=%.3f ms",
            bucket.height,
            bucket.width,
            bucket.num_layers,
            bucket.num_condition_tokens,
            guidance,
            has_reference,
            (time.perf_counter() - bucket_start) * 1000,
        )
    logger.info("Ming-Image compile warmup complete in %.3f s", time.perf_counter() - start)
