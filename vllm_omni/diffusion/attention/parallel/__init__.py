# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Parallel attention strategies.

This package provides **communication / resharding strategies** for attention,
orthogonal to the **attention kernel backend** (SDPA/Flash/Sage).

The goal is to keep `vllm_omni.diffusion.attention.layer.Attention` small and
extensible: adding a new parallelism method should not require editing the core
Attention module, only adding a new strategy and selecting it in the factory.
"""

from .base import NoParallelAttention, ParallelAttentionContext, ParallelAttentionStrategy
from .factory import build_parallel_attention_strategy

__all__ = [
    "ParallelAttentionStrategy",
    "ParallelAttentionContext",
    "NoParallelAttention",
    "build_parallel_attention_strategy",
]
