# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Ascend VAE implementations; imported only by NPU platform dispatch."""

from vllm_omni.diffusion.models.hunyuan_image3.layers.npu.autoencoder_blocks import AttnBlock, ResnetBlock, VaeConv3d
from vllm_omni.diffusion.models.hunyuan_image3.layers.npu.normalization import vae_group_norm_for_conv
from vllm_omni.diffusion.models.hunyuan_image3.layers.npu.spatial_vae import spatial_vae_forward

__all__ = ["AttnBlock", "ResnetBlock", "VaeConv3d", "spatial_vae_forward", "vae_group_norm_for_conv"]
