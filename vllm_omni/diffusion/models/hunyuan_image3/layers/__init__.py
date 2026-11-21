# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Platform-dispatched layer implementations for HunyuanImage3.

Blocks and VAE helpers are selected by platform. Native implementations provide
fallbacks; provider-specific imports stay behind platform selection.

Layout::
    native/autoencoder_blocks.py          default ResnetBlock, AttnBlock and VaeConv3d
    native/transformer_blocks.py          default ResBlock
    native/normalization.py               default GroupNorm helper
    native/spatial_vae.py                 unsupported-backend spatial fallback
    native/vae_context.py                 shared spatial VAE execution context
    nvidia/autoencoder_blocks.py          CUDA ResnetBlock (fused)
    nvidia/transformer_blocks.py          CUDA ResBlock (fused)
    npu/autoencoder_blocks.py             NPU ResnetBlock, AttnBlock and VaeConv3d
    npu/normalization.py                  NPU fused and spatial GroupNorm helpers
    npu/spatial_vae.py                    NPU spatial VAE decode
"""

from vllm_omni.diffusion.models.hunyuan_image3.layers.native import (
    AttnBlock,
    ResBlock,
    ResnetBlock,
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
        ResnetBlock,
        VaeConv3d,
        spatial_vae_forward,
        vae_group_norm_for_conv,
    )

__all__ = ["AttnBlock", "ResBlock", "ResnetBlock", "VaeConv3d", "spatial_vae_forward", "vae_group_norm_for_conv"]
