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
    # Recommended: Via OmniDiffusionConfig
    config = OmniDiffusionConfig(
        model="Qwen/Qwen-Image",
        cache_adapter="tea_cache",
        cache_config={"rel_l1_thresh": 0.2}
    )
    omni = OmniDiffusion(od_config=config)
    images = omni.generate("a cat")

    # Alternative: Environment variable
    export DIFFUSION_CACHE_ADAPTER=tea_cache
"""

from vllm_omni.diffusion.cache.teacache.adapter import TeaCacheAdapter
from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.extractors import (
    CacheContext,
    register_extractor,
)
from vllm_omni.diffusion.cache.teacache.hook import TeaCacheHook, apply_teacache_hook
from vllm_omni.diffusion.cache.teacache.state import TeaCacheState

__all__ = [
    "TeaCacheAdapter",
    "TeaCacheConfig",
    "TeaCacheState",
    "TeaCacheHook",
    "apply_teacache_hook",
    "register_extractor",
    "CacheContext",
]
