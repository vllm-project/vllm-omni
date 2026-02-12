# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantization support for diffusion models.

This module provides a unified interface for quantizing diffusion transformers
using various methods (FP8, etc.). It wraps vLLM's quantization infrastructure
while allowing diffusion-model-specific defaults and optimizations.

Example usage:
    from vllm_omni.diffusion.quantization import (
        get_diffusion_quant_config,
        get_vllm_quant_config_for_layers,
    )

    # Create FP8 config for diffusion model
    diff_config = get_diffusion_quant_config("fp8")

    # Get vLLM config to pass to linear layers
    vllm_config = get_vllm_quant_config_for_layers(diff_config)

    # Use in model initialization
    linear_layer = QKVParallelLinear(..., quant_config=vllm_config)
"""

from typing import TYPE_CHECKING

from vllm.logger import init_logger

from .base import DiffusionQuantizationConfig
from .fp8 import DiffusionFp8Config

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )

logger = init_logger(__name__)

# Registry of supported quantization methods
# To add a new method, create a new config class and register it here
_QUANT_CONFIG_REGISTRY: dict[str, type[DiffusionQuantizationConfig]] = {
    "fp8": DiffusionFp8Config,
}

SUPPORTED_QUANTIZATION_METHODS = list(_QUANT_CONFIG_REGISTRY.keys())


def get_diffusion_quant_config(
    quantization: str | None,
    **kwargs,
) -> DiffusionQuantizationConfig | None:
    """Factory function to create quantization config for diffusion models.

    Args:
        quantization: Quantization method name ("fp8", etc.) or None to disable
        **kwargs: Method-specific parameters passed to the config constructor

    Returns:
        DiffusionQuantizationConfig instance or None if quantization is disabled

    Raises:
        ValueError: If the quantization method is not supported

    Example:
        # Default FP8 with dynamic activation scaling
        config = get_diffusion_quant_config("fp8")

        # FP8 with custom parameters
        config = get_diffusion_quant_config(
            "fp8",
            activation_scheme="static",
            ignored_layers=["proj_out"],
        )
    """
    if quantization is None or quantization.lower() == "none":
        return None

    quantization = quantization.lower()
    if quantization not in _QUANT_CONFIG_REGISTRY:
        raise ValueError(
            f"Unknown quantization method: {quantization!r}. Supported methods: {SUPPORTED_QUANTIZATION_METHODS}"
        )

    config_cls = _QUANT_CONFIG_REGISTRY[quantization]
    logger.info("Creating diffusion quantization config: %s", quantization)
    return config_cls(**kwargs)


def get_vllm_quant_config_for_layers(
    diffusion_quant_config: DiffusionQuantizationConfig | None,
) -> "QuantizationConfig | None":
    """Get the vLLM QuantizationConfig to pass to linear layers.

    This extracts the underlying vLLM config from a DiffusionQuantizationConfig,
    which can then be passed to vLLM linear layers (QKVParallelLinear, etc.).

    Args:
        diffusion_quant_config: The diffusion quantization config, or None

    Returns:
        vLLM QuantizationConfig instance, or None if input is None
    """
    if diffusion_quant_config is None:
        return None
    return diffusion_quant_config.get_vllm_quant_config()


__all__ = [
    "DiffusionQuantizationConfig",
    "DiffusionFp8Config",
    "get_diffusion_quant_config",
    "get_vllm_quant_config_for_layers",
    "SUPPORTED_QUANTIZATION_METHODS",
]
