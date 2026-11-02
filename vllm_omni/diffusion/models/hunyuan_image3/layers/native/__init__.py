# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Native (plain PyTorch) layer implementations."""

from vllm_omni.diffusion.models.hunyuan_image3.layers.native.autoencoder_blocks import AttnBlock, ResnetBlock
from vllm_omni.diffusion.models.hunyuan_image3.layers.native.transformer_blocks import ResBlock
from vllm_omni.diffusion.models.hunyuan_image3.layers.native.vae import (
    VaeConv3d,
    spatial_vae_forward,
    vae_group_norm_for_conv,
)

__all__ = ["AttnBlock", "ResBlock", "ResnetBlock", "VaeConv3d", "spatial_vae_forward", "vae_group_norm_for_conv"]
