# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Backward-compatible import path for the shared LoRA manager."""

from vllm_omni.lora.manager import (
    DiffusionLoRAManager,
    LoRABackend,
    LoRAManager,
)

__all__ = ["DiffusionLoRAManager", "LoRABackend", "LoRAManager"]
