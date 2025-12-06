# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Cache module for diffusion model inference acceleration.

This module provides a unified cache adapter system for different caching strategies:
- TeaCache: Timestep Embedding Aware Cache for adaptive transformer caching
- (Future: DeepCache, DistriFusion, etc.)

The cache adapter system uses a registry pattern where cache types can be
dynamically registered and instantiated. Cache is configured via OmniDiffusionConfig
or DIFFUSION_CACHE_ADAPTER environment variable.
"""

from vllm_omni.diffusion.cache.apply import setup_cache
from vllm_omni.diffusion.cache.base import CacheAdapter
from vllm_omni.diffusion.cache.registry import (
    CacheType,
    get_cache_adapter,
    register_cache_adapter,
)
from vllm_omni.diffusion.cache.teacache import (
    CacheContext,
    TeaCacheConfig,
    apply_teacache_hook,
)

# Import teacache adapter to trigger registration
from vllm_omni.diffusion.cache.teacache.adapter import TeaCacheAdapter  # noqa: F401

__all__ = [
    "CacheAdapter",
    "CacheType",
    "TeaCacheConfig",
    "CacheContext",
    "setup_cache",
    "get_cache_adapter",
    "register_cache_adapter",
    "apply_teacache_hook",
]
