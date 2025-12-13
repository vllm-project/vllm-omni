# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
TeaCache: Timestep Embedding Aware Cache for diffusion model acceleration.

TeaCache speeds up diffusion inference by reusing transformer block computations
when consecutive timestep embeddings are similar.

This implementation uses a hooks-based approach that requires zero changes to
model code. Model developers only need to add an extractor function to support
new models.

Usage:
    from vllm_omni import Omni

    omni = Omni(
        model="Qwen/Qwen-Image",
        cache_backend="tea_cache",
        cache_config={"rel_l1_thresh": 0.2}
    )
    images = omni.generate("a cat")

    # Alternative: Using environment variable
    # export DIFFUSION_CACHE_BACKEND=tea_cache
"""

from vllm_omni.diffusion.cache.teacache.backend import TeaCacheBackend
from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.extractors import (
    CacheContext,
    register_extractor,
)
from vllm_omni.diffusion.cache.teacache.hook import TeaCacheHook, apply_teacache_hook
from vllm_omni.diffusion.cache.teacache.state import TeaCacheState

__all__ = [
    "TeaCacheBackend",
    "TeaCacheConfig",
    "TeaCacheState",
    "TeaCacheHook",
    "apply_teacache_hook",
    "register_extractor",
    "CacheContext",
]
