# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compatibility import for the shared Q/K RMSNorm + RoPE operation.

New callers should import from ``vllm_omni.diffusion.layers.ops``.
Implementation and torch registration live in ``ops.rope.qk_norm_rope``.
"""

from vllm_omni.diffusion.layers.ops.rope.qk_norm_rope import fused_qk_norm_rope

__all__ = ["fused_qk_norm_rope"]
