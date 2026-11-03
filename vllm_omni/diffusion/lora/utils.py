# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Backward-compatible import path for shared LoRA binding utilities."""

from vllm_omni.lora.binding import (
    _expand_expected_modules_for_packed_layers,
    _match_target_modules,
    _move_moe_lora_buffers,
    _select_moe_lora_wrapper_cls,
    _validate_moe_lora_token_capacity,
    detach_moe_lora_context,
    from_layer_diffusion,
    moe_lora_kernel_requires_context,
)

__all__ = ["from_layer_diffusion"]
