# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified quantization framework for vLLM-OMNI.

Delegates to vLLM's quantization registry (35+ methods, all platforms).
Adds per-component quantization for multi-stage models.

    from vllm_omni.quantization import build_quant_config

    config = build_quant_config("fp8")
    config = build_quant_config({"transformer": {"method": "fp8"}, "vae": None})
"""

from .component_config import ComponentQuantizationConfig
from .factory import SUPPORTED_QUANTIZATION_METHODS, build_quant_config
from .inc_config import OmniINCConfig

# DiffusionGGUFConfig is NOT imported here to avoid pulling in
# GGUF -> fused_moe -> pynvml at module load time.

__all__ = [
    "build_quant_config",
    "ComponentQuantizationConfig",
    "OmniINCConfig",
    "SUPPORTED_QUANTIZATION_METHODS",
]
