# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Cache module for diffusion model inference acceleration.

This module provides a unified cache backend system for different caching strategies:
- TeaCache: Timestep Embedding Aware Cache for adaptive transformer caching
- cache-dit: DBCache, SCM, and TaylorSeer caching strategies

Cache backends are instantiated directly via their constructors and configured via OmniDiffusionConfig.
"""

from vllm_omni.diffusion.cache.base import CacheBackend
from vllm_omni.diffusion.cache.teacache import (
    CacheContext,
    TeaCacheConfig,
    apply_teacache_hook,
)
from vllm_omni.diffusion.cache.teacache.backend import TeaCacheBackend

__all__ = [
    "CacheBackend",
    "TeaCacheConfig",
    "CacheContext",
    "TeaCacheBackend",
    "apply_teacache_hook",
]
