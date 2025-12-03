# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Callable

import torch


def extract_qwen_modulated_input(
    module, hidden_states: torch.Tensor, temb: torch.Tensor
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
EXTRACTOR_REGISTRY: dict[str, Callable] = {
    "qwen": extract_qwen_modulated_input,
    "Qwen": extract_qwen_modulated_input,
    "QwenImageTransformer2DModel": extract_qwen_modulated_input,
}


def get_extractor(model_type: str) -> Callable:
    """
    Get extractor function for given model type.

    Args:
        model_type: Model type identifier

    Returns:
        Extractor function

    Raises:
        ValueError: If model type not found in registry
    """
    if model_type in EXTRACTOR_REGISTRY:
        return EXTRACTOR_REGISTRY[model_type]

    # Try case-insensitive match
    for key, extractor in EXTRACTOR_REGISTRY.items():
        if key.lower() == model_type.lower():
            return extractor

    raise ValueError(
        f"Unknown model type: {model_type}. "
        f"Available types: {list(EXTRACTOR_REGISTRY.keys())}"
    )
