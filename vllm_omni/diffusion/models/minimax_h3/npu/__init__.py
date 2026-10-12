# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MiniMax-H3 components for artifacts produced on Ascend NPU.

The package name records where these artifacts come from, not where they can
run. Everything here is plain ``torch`` plus ``safetensors`` and carries no
``torch_npu`` dependency, so the same checkpoint loads and serves on CUDA and
CPU as well. See ``tests/diffusion/models/minimax_h3/test_minimax_h3_native_lora.py``
for the platform-agnostic and CUDA coverage that pins this.
"""

from .lora import MINIMAX_H3_NATIVE_INFERENCE_STEPS, load_minimax_h3_native_lora

__all__ = [
    "MINIMAX_H3_NATIVE_INFERENCE_STEPS",
    "load_minimax_h3_native_lora",
]
