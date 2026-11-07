# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared LeRobot-compatible inference dtype policy for Pi models."""

import torch
from torch import nn

SUPPORTED_INFERENCE_DTYPES = (torch.float32, torch.bfloat16)

# LeRobot keeps the full vision path and numerically sensitive normalization
# parameters in float32. All other PaliGemma/action-expert parameters use
# bfloat16, while projections owned by the outer Pi model remain float32.
FLOAT32_IN_BFLOAT16_SELECTORS = (
    "vision_tower",
    "multi_modal_projector",
    "input_layernorm",
    "post_attention_layernorm",
    "model.norm",
)


def match_module_input_dtype(tensor: torch.Tensor, module: nn.Module) -> torch.Tensor:
    """Cast an activation to the dtype expected by ``module``'s weight."""
    weight = getattr(module, "weight", None)
    if weight is None or tensor.dtype == weight.dtype:
        return tensor
    return tensor.to(dtype=weight.dtype)


def apply_pi_inference_dtype(model: nn.Module, dtype: torch.dtype) -> None:
    """Apply the shared FP32 or LeRobot-style mixed-BF16 layout.

    Pi0 and Pi0.5 both expose their PaliGemma/action-expert pair through
    ``paligemma_with_expert``. In BF16 mode that inner module follows
    LeRobot's mixed policy; the variant-specific outer projections stay FP32.
    """
    if dtype == torch.float32:
        model.to(dtype=torch.float32)
        return
    if dtype != torch.bfloat16:
        raise ValueError(f"Unsupported Pi-family inference dtype: {dtype!r}.")

    inner_model = model.paligemma_with_expert
    for name, parameter in inner_model.named_parameters():
        target_dtype = (
            torch.float32 if any(selector in name for selector in FLOAT32_IN_BFLOAT16_SELECTORS) else torch.bfloat16
        )
        parameter.data = parameter.data.to(dtype=target_dtype)
    for buffer in inner_model.buffers():
        if buffer.is_floating_point():
            buffer.data = buffer.data.to(dtype=torch.bfloat16)

    for name, parameter in model.named_parameters():
        if not name.startswith("paligemma_with_expert."):
            parameter.data = parameter.data.to(dtype=torch.float32)
    for name, buffer in model.named_buffers():
        if not name.startswith("paligemma_with_expert.") and buffer.is_floating_point():
            buffer.data = buffer.data.to(dtype=torch.float32)
