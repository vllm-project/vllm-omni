# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared access to the production ``KVCacheTensor`` layout derivation.

Tests deliberately go through the same code production uses
(``vllm_omni.diffusion.diffusion_kv.layout``) rather than recomputing strides,
so a regression in that derivation fails the suite instead of being masked by a
parallel test-only copy.
"""

from types import SimpleNamespace

from vllm.config import CacheConfig
from vllm.v1.kv_cache_layout import KVCacheLayout

from vllm_omni.diffusion.diffusion_kv.layout import (
    build_kv_cache_tensor,
    resolve_diffusion_kv_cache_layout,
)

__all__ = ["build_kv_cache_tensor", "layout_for_backend"]


def layout_for_backend(attn_backend) -> KVCacheLayout:
    """The physical layout ``attn_backend``'s block-stride declaration selects.

    Before 0.29 a backend's ``indexes_kv_by_block_stride()`` was copied onto
    ``AttentionSpec``; asserting on the resolved layout is the surviving way to
    check that declaration still reaches the cache.
    """

    vllm_config = SimpleNamespace(cache_config=CacheConfig())
    return resolve_diffusion_kv_cache_layout(
        vllm_config,
        indexes_kv_by_block_stride=attn_backend.indexes_kv_by_block_stride(),
    )
