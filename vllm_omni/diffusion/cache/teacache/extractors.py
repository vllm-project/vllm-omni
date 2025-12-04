# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Model-specific extractors for TeaCache.

This module provides a registry of extractor functions that know how to extract
modulated inputs from different transformer architectures. Adding support for
a new model requires only adding a new extractor function to the registry.
"""

from typing import Callable, Union

import torch
import torch.nn as nn


def extract_qwen_modulated_input(
    module: nn.Module, hidden_states: torch.Tensor, temb: torch.Tensor
) -> torch.Tensor:
    """
    Extract modulated input for QwenImageTransformer2DModel.

    The modulated input is extracted from the first transformer block:
    1. Apply norm1: img_normed = block.img_norm1(hidden_states)
    2. Get modulation params: img_mod1 = block.img_mod(temb)[:3*dim]
    3. Apply modulation: img_modulated, _ = block._modulate(img_normed, img_mod1)

    Args:
        module: QwenImageTransformer2DModel instance
        hidden_states: Input hidden states tensor
        temb: Timestep embedding tensor

    Returns:
        Modulated input tensor from first transformer block
    """
    if not hasattr(module, "transformer_blocks") or len(module.transformer_blocks) == 0:
        raise ValueError("Module must have transformer_blocks")

    block = module.transformer_blocks[0]

    # Get modulation parameters
    img_mod_params = block.img_mod(temb)  # [B, 6*dim]
    img_mod1, _ = img_mod_params.chunk(2, dim=-1)  # [B, 3*dim]

    # Apply norm1
    img_normed = block.img_norm1(hidden_states)

    # Apply modulation
    img_modulated, _ = block._modulate(img_normed, img_mod1)

    return img_modulated


# Registry for model-specific extractors
# Key: pipeline/model architecture name
EXTRACTOR_REGISTRY: dict[str, Callable] = {
    "QwenImagePipeline": extract_qwen_modulated_input,
    "qwen": extract_qwen_modulated_input,
    "Qwen": extract_qwen_modulated_input,
}


def register_extractor(model_identifier: str, extractor_fn: Callable) -> None:
    """
    Register a new extractor function for a model type.

    This allows extending TeaCache support to new models without modifying
    the core TeaCache code.

    Args:
        model_identifier: Model type identifier (class name or type string)
        extractor_fn: Function with signature (module, hidden_states, temb) -> modulated_input

    Example:
        >>> def extract_flux_modulated_input(module, hidden_states, temb):
        ...     return module.transformer_blocks[0].norm1(hidden_states, emb=temb)[0]
        >>> register_extractor("FluxTransformer2DModel", extract_flux_modulated_input)
    """
    EXTRACTOR_REGISTRY[model_identifier] = extractor_fn


def get_extractor(model_or_type: Union[nn.Module, str]) -> Callable:
    """
    Get extractor function for given model or model type.

    This function auto-detects the appropriate extractor based on:
    1. Explicit model type string
    2. Module class name (exact match)
    3. Module class name (partial match)

    Args:
        model_or_type: Either a torch.nn.Module instance or a model type string

    Returns:
        Extractor function with signature (module, hidden_states, temb) -> modulated_input

    Raises:
        ValueError: If model type not found in registry

    Example:
        >>> # Auto-detect from module
        >>> extractor = get_extractor(transformer)
        >>> modulated = extractor(transformer, hidden_states, temb)
        >>>
        >>> # Explicit type
        >>> extractor = get_extractor("Qwen")
    """
    # Resolve model type string
    if isinstance(model_or_type, nn.Module):
        model_type = model_or_type.__class__.__name__
    else:
        model_type = model_or_type

    # Exact, case-sensitive match only
    if model_type in EXTRACTOR_REGISTRY:
        return EXTRACTOR_REGISTRY[model_type]

    # No exact match found
    available_types = list(EXTRACTOR_REGISTRY.keys())
    raise ValueError(
        f"Unknown model type: {model_type}. "
        f"Available types: {available_types}\n"
        f"To add support for a new model, use register_extractor() or add to EXTRACTOR_REGISTRY."
    )
