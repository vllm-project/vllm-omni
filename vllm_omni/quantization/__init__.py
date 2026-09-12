# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unified quantization framework for vLLM-OMNI.

Delegates to vLLM's quantization registry (35+ methods, all platforms).
Adds per-component quantization for multi-stage models.

    from vllm_omni.quantization.factory import build_quantization_config

    config = build_quantization_config("fp8")
    config = build_quantization_config({"transformer": {"method": "fp8"}, "vae": None})
"""

from .component_config import ComponentQuantizationConfig, resolve_component_quant_config
from .factory import (
    SUPPORTED_QUANTIZATION_METHODS,
    build_quantization_config,
    register_omni_quantization_configs,
)
from .inc_config import OmniINCConfig

# Heavy configs are NOT imported here to avoid pulling in
# optional dependencies (pynvml, torch_npu) at module load time.
# Import them directly when needed:
#   from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

__all__ = [
    "build_quantization_config",
    "ComponentQuantizationConfig",
    "resolve_component_quant_config",
    "OmniINCConfig",
    "SUPPORTED_QUANTIZATION_METHODS",
    "register_omni_quantization_configs",
]
