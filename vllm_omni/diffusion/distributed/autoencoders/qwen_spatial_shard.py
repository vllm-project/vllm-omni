# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen decoder adaptation of the shared Wan spatial halo primitives."""

from types import MethodType
from typing import Any

import torch.distributed as dist
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import QwenImageAttentionBlock, QwenImageCausalConv3d
from torch import nn

from vllm_omni.diffusion.distributed.autoencoders.wan_spatial_shard import (
    _SPATIAL_SHARD_CONTEXT,
    SpatialShardContext,
    WanDistCausalConv3d,
    WanDistConv2d,
    _decoder_upsample_count,
    _patch_attention_block,
    _rank_world,
    _spatial_dim,
    gather_and_trim_extent,
    split_for_parallel_decode,
)


def _patch_modules(module: nn.Module, group: dist.ProcessGroup, split_dim: str) -> None:
    if isinstance(module, QwenImageAttentionBlock):
        _patch_attention_block(module, group, split_dim)
        # Attention runs on the full feature map, including its projection convs.
        return
    for name, child in list(module.named_children()):
        if isinstance(child, QwenImageCausalConv3d):
            replacement = WanDistCausalConv3d(child, group, split_dim)
        elif isinstance(child, nn.Conv2d):
            replacement = WanDistConv2d(child, group, split_dim)
        else:
            _patch_modules(child, group, split_dim)
            continue
        # Preserve the loaded Parameters themselves, including requires_grad.
        replacement.weight = child.weight
        replacement.bias = child.bias
        replacement.train(child.training)
        setattr(module, name, replacement)


def install_qwen_spatial_shard_decode(vae: Any, group: dist.ProcessGroup, split_dim: str) -> None:
    """Bind the already-loaded decoder permanently to a group and direction."""
    _spatial_dim(split_dim)
    _, world_size = _rank_world(group)
    config = (group, split_dim, world_size)
    installed = getattr(vae, "_qwen_spatial_shard_config", None)
    if installed is not None:
        if installed != config:
            raise ValueError("Qwen spatial-shard configuration changed; create a fresh VAE instance.")
        return
    decoder = vae.decoder
    _patch_modules(decoder, group, split_dim)
    upsample_count = _decoder_upsample_count(decoder)
    original_forward = decoder.forward

    def forward(self, x, feat_cache=None, feat_idx=None):
        dim = _spatial_dim(split_dim)
        input_extent = x.shape[dim]
        local, expected_extent = split_for_parallel_decode(
            x, upsample_count=upsample_count, split_dim=split_dim, group=group
        )
        rank, size = _rank_world(group)
        token = _SPATIAL_SHARD_CONTEXT.set(SpatialShardContext(input_extent, local.shape[dim], split_dim, rank, size))
        try:
            out = original_forward(local, feat_cache=feat_cache, feat_idx=[0] if feat_idx is None else feat_idx)
        finally:
            _SPATIAL_SHARD_CONTEXT.reset(token)
        return gather_and_trim_extent(out, expected_extent=expected_extent, split_dim=split_dim, group=group, dst=None)

    decoder.forward = MethodType(forward, decoder)
    vae._qwen_spatial_shard_config = config
