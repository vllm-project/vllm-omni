# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compatibility imports for Q/K RMSNorm and RoPE operations."""

from vllm_omni.diffusion.layers.ops.rope.qk_norm_rope import (
    QK_NORM_ROPE_TABLE_KEY,
    fused_joint_qkv_norm_rope,
    fused_qk_norm_rope,
    fused_qk_norm_rope_available,
    fused_qk_norm_rope_min_tokens,
    pack_qk_norm_rope_table,
)
from vllm_omni.diffusion.layers.ops.rope.qk_norm_rope import (
    _eager_qk_norm_rope as _eager_qk_norm_rope,
)
from vllm_omni.diffusion.layers.ops.rope.qk_norm_rope import (
    _fused_cuda_supported as _fused_cuda_supported,
)
from vllm_omni.diffusion.layers.ops.rope.qk_norm_rope import (
    _launch_fused_qk_norm_rope as _launch_fused_qk_norm_rope,
)

__all__ = [
    "QK_NORM_ROPE_TABLE_KEY",
    "fused_joint_qkv_norm_rope",
    "fused_qk_norm_rope",
    "fused_qk_norm_rope_available",
    "fused_qk_norm_rope_min_tokens",
    "pack_qk_norm_rope_table",
]
