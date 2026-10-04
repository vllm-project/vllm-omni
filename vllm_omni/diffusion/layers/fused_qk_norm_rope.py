# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compatibility import for the shared Q/K RMSNorm + RoPE operation.

New callers should import from ``vllm_omni.diffusion.layers.ops``.
Implementation and torch registration live in ``ops.rope.qk_norm_rope``.
"""

from vllm_omni.diffusion.layers.ops.rope.qk_norm_rope import (
    _fused_cuda_supported as _fused_cuda_supported,
)
from vllm_omni.diffusion.layers.ops.rope.qk_norm_rope import (
    fused_qk_norm_rope,
    fused_qk_norm_rope_min_tokens,
)

# Boogu still imports the private predicate here until its public support-query
# adoption in #7422. Keep it an alias to the single canonical implementation.
__all__ = ["fused_qk_norm_rope", "fused_qk_norm_rope_min_tokens"]
