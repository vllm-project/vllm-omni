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

# Heavy configs (GGUF, MXFP8) are NOT imported here to avoid pulling in
# optional dependencies (pynvml, torch_npu) at module load time.
# Import them directly when needed:
#   from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

__all__ = [
    "build_quant_config",
    "ComponentQuantizationConfig",
    "OmniINCConfig",
    "SUPPORTED_QUANTIZATION_METHODS",
]
