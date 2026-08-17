# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hunyuan Image 3 diffusion model components.

The two residual blocks are selected by platform:

* ``ResnetBlock`` -- the VAE block, ``GroupNorm -> SiLU`` fusion.
* ``ResBlock``    -- the DiT block, ``GroupNorm -> SiLU`` plus AdaGN fusion.

Layout::

    autoencoder_blocks.py         default ResnetBlock
    transformer_blocks.py         default ResBlock
    nvidia/autoencoder_blocks.py  CUDA ResnetBlock
    nvidia/transformer_blocks.py  CUDA ResBlock

Note there is no ``try/except ImportError`` around the CUDA branch. A previous
version wrapped it, and when the import raised for an unrelated reason the
warning was easy to miss and every CUDA machine silently ran the default
blocks. If the CUDA blocks fail to import on a CUDA machine that is a bug, and
it should be loud. Falling back on a *non*-CUDA platform is the job of the
``else`` branch, which is a plain, always-valid import.
"""

# The dispatch must stay ahead of the imports below. ``autoencoder.py`` and
# ``hunyuan_image3_transformer.py`` import these two names back out of this
# package, so they have to be bound before anything that (directly or not)
# pulls those modules in. Do not sort this block down.
from vllm_omni.platforms import current_omni_platform

if current_omni_platform.is_cuda():
    from vllm_omni.diffusion.models.hunyuan_image3.nvidia.autoencoder_blocks import ResnetBlock
    from vllm_omni.diffusion.models.hunyuan_image3.nvidia.transformer_blocks import ResBlock
else:
    # NPU, ROCm, XPU, CPU, out-of-tree: plain PyTorch blocks.
    from vllm_omni.diffusion.models.hunyuan_image3.autoencoder_blocks import ResnetBlock
    from vllm_omni.diffusion.models.hunyuan_image3.transformer_blocks import ResBlock

from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
    HunyuanImage3Model,
    HunyuanImage3Text2ImagePipeline,
)
from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
    HunyuanImage3Pipeline,
)

__all__ = [
    "HunyuanImage3Pipeline",
    "HunyuanImage3Model",
    "HunyuanImage3Text2ImagePipeline",
    "ResnetBlock",
    "ResBlock",
]
