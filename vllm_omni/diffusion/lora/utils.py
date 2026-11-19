# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Backward-compatible import path for shared LoRA binding utilities."""

from vllm_omni.lora.binding import (
    MOE_LORA_UNAVAILABLE_ERROR,
    from_layer_diffusion,
)

__all__ = ["MOE_LORA_UNAVAILABLE_ERROR", "from_layer_diffusion"]
