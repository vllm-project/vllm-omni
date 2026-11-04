# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Native (plain PyTorch) layer implementations."""

from vllm_omni.diffusion.models.hunyuan_image3.layers.native.autoencoder_blocks import AttnBlock, ResnetBlock, VaeConv3d
from vllm_omni.diffusion.models.hunyuan_image3.layers.native.normalization import vae_group_norm_for_conv
from vllm_omni.diffusion.models.hunyuan_image3.layers.native.spatial_vae import spatial_vae_forward
from vllm_omni.diffusion.models.hunyuan_image3.layers.native.transformer_blocks import ResBlock

__all__ = ["AttnBlock", "ResBlock", "ResnetBlock", "VaeConv3d", "spatial_vae_forward", "vae_group_norm_for_conv"]
