# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Compatibility exports for the shared Q/K RMSNorm + RoPE operation.

The canonical implementation lives under :mod:`vllm_omni.diffusion.layers.ops`.
Keep this module as a forwarding shim for existing consumers of the original
import path.
"""

from vllm_omni.diffusion.layers.ops import (
    fused_qk_norm_rope,
    fused_qk_norm_rope_min_tokens,
    fused_qk_norm_rope_supported,
)

__all__ = [
    "fused_qk_norm_rope",
    "fused_qk_norm_rope_min_tokens",
    "fused_qk_norm_rope_supported",
]
