# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Native (plain PyTorch) layer implementations."""

from vllm_omni.diffusion.models.hunyuan_image3.layers.native.autoencoder_blocks import ResnetBlock
from vllm_omni.diffusion.models.hunyuan_image3.layers.native.transformer_blocks import ResBlock

__all__ = ["ResBlock", "ResnetBlock"]
