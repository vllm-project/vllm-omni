# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-local CUDA MXFP8 scope; checkpoint fusion stays in the normal loader."""

from vllm_omni.platforms import current_omni_platform
from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config


def projection_quant_config(config, prefix):
    """Keep conditioning, AdaLN, VSA gates and token refinement in BF16."""
    if not isinstance(config, DiffusionMXFP8Config) or not current_omni_platform.is_cuda():
        return config
    if prefix.startswith("blocks.") and prefix.endswith((".attn.qkv_proj", ".attn.out_proj", ".mlp.fc1", ".mlp.fc2")):
        return config
    return None
