# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Backward-compatible import path for shared LoRA binding utilities."""

from vllm_omni.lora.binding import (
    from_layer_diffusion,
)

__all__ = ["from_layer_diffusion"]
