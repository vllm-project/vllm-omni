# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Platform-dispatched layer implementations for HunyuanImage3.

Blocks and VAE helpers are selected by platform. Native implementations provide
fallbacks; provider-specific imports stay behind platform selection.

Layout::
    native/autoencoder_blocks.py          default ResnetBlock and AttnBlock
    native/transformer_blocks.py          default ResBlock
    native/vae.py                         default VAE convolution and helpers
    nvidia/autoencoder_blocks.py          CUDA ResnetBlock (fused)
    nvidia/transformer_blocks.py          CUDA ResBlock (fused)
    npu/autoencoder_blocks.py             NPU ResnetBlock, AttnBlock and VaeConv3d
    npu/normalization.py                  NPU fused and spatial GroupNorm helpers
    npu/spatial_vae.py                    NPU spatial VAE encode/decode
    npu/vae_context.py                    spatial VAE execution context
"""

from vllm_omni.diffusion.models.hunyuan_image3.layers.native.autoencoder_blocks import AttnBlock
from vllm_omni.diffusion.models.hunyuan_image3.layers.native.vae import (
    VaeConv3d,
    spatial_vae_forward,
    vae_group_norm_for_conv,
)
from vllm_omni.platforms import current_omni_platform

if current_omni_platform.is_cuda():
    from vllm_omni.diffusion.models.hunyuan_image3.layers.nvidia import ResBlock, ResnetBlock
elif current_omni_platform.is_npu():
    from vllm_omni.diffusion.models.hunyuan_image3.layers.npu import (
        AttnBlock,
        ResBlock,
        ResnetBlock,
        VaeConv3d,
        spatial_vae_forward,
        vae_group_norm_for_conv,
    )
else:
    from vllm_omni.diffusion.models.hunyuan_image3.layers.native import ResBlock, ResnetBlock

__all__ = ["AttnBlock", "ResBlock", "ResnetBlock", "VaeConv3d", "spatial_vae_forward", "vae_group_norm_for_conv"]
