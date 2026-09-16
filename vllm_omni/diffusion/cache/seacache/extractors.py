# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Image adapters using first-block modulation with SeaCache's shared hook."""

from __future__ import annotations

from typing import Any

import torch

from vllm_omni.diffusion.cache.teacache.extractors import (
    CacheContext,
    get_extractor,
)


def _target_features(modulated_input: torch.Tensor, height: int, width: int) -> list[torch.Tensor]:
    """View modulated target tokens as B,D,T,H,W for the shared Sea filter."""
    count = height * width
    if height <= 0 or width <= 0 or count > modulated_input.shape[1]:
        return []
    target = modulated_input[:, :count]
    batch, _, channels = target.shape
    return [target.reshape(batch, height, width, channels).permute(0, 3, 1, 2).unsqueeze(2)]


def _target_features_from_ids(modulated_input: torch.Tensor, img_ids: torch.Tensor) -> list[torch.Tensor]:
    ids = img_ids[0] if img_ids.ndim == 3 else img_ids
    # The noisy target is first at T=0; FLUX.2 references use T=10,20,...
    target_ids = ids[ids[:, 0] == 0]
    if not len(target_ids):
        return []
    height = int(target_ids[:, 1].max().item()) + 1
    width = int(target_ids[:, 2].max().item()) + 1
    if height * width != len(target_ids):
        return []
    return _target_features(modulated_input, height, width)


def _with_latents(ctx: CacheContext, latents: list[torch.Tensor]) -> CacheContext:
    # Keep condition tokens in the execution context, outside the indicator.
    ctx.extra_states = {**(ctx.extra_states or {}), "sea_cache_latents": latents}
    return ctx


def extract_flux_seacache_context(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    pooled_projections: torch.Tensor,
    timestep: torch.Tensor,
    img_ids: torch.Tensor,
    *args: Any,
    **kwargs: Any,
) -> CacheContext:
    extract = get_extractor("FluxTransformer2DModel")
    ctx = extract(module, hidden_states, encoder_hidden_states, pooled_projections, timestep, img_ids, *args, **kwargs)
    return _with_latents(ctx, _target_features_from_ids(ctx.modulated_input, img_ids))


def extract_flux2_seacache_context(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    img_ids: torch.Tensor,
    *args: Any,
    **kwargs: Any,
) -> CacheContext:
    """Use the same complete dual/single-stream boundary for FLUX.2 and Klein."""
    extract = get_extractor("Flux2Transformer2DModel")
    ctx = extract(module, hidden_states, encoder_hidden_states, timestep, img_ids, *args, **kwargs)
    return _with_latents(ctx, _target_features_from_ids(ctx.modulated_input, img_ids))


def extract_qwen_seacache_context(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_mask: torch.Tensor,
    timestep: torch.Tensor | float | int,
    img_shapes: list[list[tuple[int, int, int]]],
    *args: Any,
    **kwargs: Any,
) -> CacheContext | None:
    batch_size = hidden_states.shape[0]
    if getattr(module, "zero_cond_t", False) and batch_size > 1:
        # The shared extractor's 2B modulations only broadcast to B rows for B=1.
        return None
    extract = get_extractor("QwenImageTransformer2DModel")
    ctx = extract(
        module,
        hidden_states,
        encoder_hidden_states,
        encoder_hidden_states_mask,
        timestep,
        img_shapes,
        *args,
        **kwargs,
    )
    # The first B rows use the real timestep; zero_cond_t appends zero-time rows.
    ctx.modulated_input = ctx.modulated_input[:batch_size]
    frames, height, width = img_shapes[0][0]
    features = _target_features(ctx.modulated_input, height, width) if frames == 1 else []
    return _with_latents(ctx, features)
