# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""QwenImage adapter for the shared spatial-shard VAE decoder kernels."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import QwenImageCausalConv3d
from diffusers.models.autoencoders.vae import DecoderOutput

from vllm_omni.diffusion.distributed.autoencoders.spatial_shard import SpatialShardVAE
from vllm_omni.diffusion.distributed.autoencoders.wan_spatial_shard import (
    SpatialShardCausalConv3dMixin,
    install_spatial_shard_decode,
    spatial_shard_decode_impl,
)


class QwenImageDistCausalConv3d(SpatialShardCausalConv3dMixin, QwenImageCausalConv3d):
    """Context-aware QwenImage causal conv that remains cache-discoverable."""

    def __init__(self, source: nn.Conv3d, group: dist.ProcessGroup) -> None:
        # Initialize Conv3d directly so the replacement preserves source device,
        # dtype, groups, and bias while remaining an isinstance() match for
        # QwenImageCausalConv3d. Diffusers clear_cache() relies on that match.
        nn.Conv3d.__init__(
            self,
            source.in_channels,
            source.out_channels,
            source.kernel_size,
            stride=source.stride,
            padding=0,
            dilation=source.dilation,
            groups=source.groups,
            bias=source.bias is not None,
            padding_mode=source.padding_mode,
            device=source.weight.device,
            dtype=source.weight.dtype,
        )
        self._init_spatial_shard_causal_conv(source, group)

    def forward(self, x: torch.Tensor, cache_x: torch.Tensor | None = None) -> torch.Tensor:
        return self._spatial_shard_causal_conv_forward(x, cache_x)


def install_qwenimage_spatial_shard_decode(
    vae: SpatialShardVAE,
    group: dist.ProcessGroup,
    split_dim: str = "height",
) -> None:
    install_spatial_shard_decode(
        vae,
        group,
        split_dim,
        causal_conv_class_name="QwenImageCausalConv3d",
        causal_conv_factory=QwenImageDistCausalConv3d,
        attention_block_class_name="QwenImageAttentionBlock",
        installed_attr="_vllm_omni_qwenimage_spatial_shard_installed",
        model_name="QwenImage",
    )


def spatial_shard_decode(
    vae: SpatialShardVAE,
    z: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    return_dict: bool = True,
    split_dim: str = "height",
) -> DecoderOutput | tuple[torch.Tensor]:
    """Decode QwenImage latents and preserve its all-rank output contract."""
    return spatial_shard_decode_impl(
        vae,
        z,
        group=group,
        install=install_qwenimage_spatial_shard_decode,
        model_name="QwenImage",
        pass_first_chunk=False,
        broadcast_result=True,
        unpatchify_patch_size=None,
        return_dict=return_dict,
        split_dim=split_dim,
    )
