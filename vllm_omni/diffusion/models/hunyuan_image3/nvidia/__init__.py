# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NVIDIA (CUDA) block implementations for HunyuanImage3.

Only the blocks that actually have a CUDA-specific implementation live here.
Everything else is served by the default modules one level up.

Keep this list in sync with what the sibling modules really define -- an
earlier version of this file re-exported ``AttnBlock``/``Encoder``/``Decoder``
and friends that were never implemented here, so the import raised and the
package silently fell back to the default blocks on every CUDA machine.
"""

from vllm_omni.diffusion.models.hunyuan_image3.nvidia.autoencoder_blocks import (
    Conv3d,
    ResnetBlock,
)
from vllm_omni.diffusion.models.hunyuan_image3.nvidia.transformer_blocks import ResBlock

__all__ = ["Conv3d", "ResnetBlock", "ResBlock"]
