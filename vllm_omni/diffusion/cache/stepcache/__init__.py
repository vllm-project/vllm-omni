# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
StepCache: velocity-based step skipping for DreamZero-style DiT denoising.

Unlike block-level Cache-DiT, this backend skips entire DiT forwards during the
pipeline denoise loop when successive velocity predictions are highly aligned.

Usage:
    from vllm_omni import Omni

    omni = Omni(
        model="...",
        cache_backend="step_cache",
    )

    # Alternative: environment variable
    # export DIFFUSION_CACHE_BACKEND=step_cache
"""

from vllm_omni.diffusion.cache.stepcache.backend import (
    CUSTOM_STEP_CACHE_DIT_ENABLERS,
    CUSTOM_STEPCACHE_ENABLERS,
    STEP_CACHE_CONFIG_ATTR,
    STEP_CACHE_DIT_CONFIG_ATTR,
    STEP_CACHE_DIT_STATE_ATTR,
    STEP_CACHE_STATE_ATTR,
    StepCacheBackend,
    StepCacheDiTBackend,
    StepCacheDiTConfig,
    StepCacheDiTState,
    enable_dreamzero_step_cache_dit,
    enable_dreamzero_stepcache,
    get_step_cache_dit_config,
    get_step_cache_dit_state,
    get_stepcache_config,
    get_stepcache_state,
    is_step_cache_dit_active,
    is_stepcache_active,
)
from vllm_omni.diffusion.cache.stepcache.config import StepCacheConfig
from vllm_omni.diffusion.cache.stepcache.state import StepCacheState

__all__ = [
    "CUSTOM_STEP_CACHE_DIT_ENABLERS",
    "CUSTOM_STEPCACHE_ENABLERS",
    "STEP_CACHE_CONFIG_ATTR",
    "STEP_CACHE_DIT_CONFIG_ATTR",
    "STEP_CACHE_DIT_STATE_ATTR",
    "STEP_CACHE_STATE_ATTR",
    "StepCacheBackend",
    "StepCacheConfig",
    "StepCacheDiTBackend",
    "StepCacheDiTConfig",
    "StepCacheDiTState",
    "StepCacheState",
    "enable_dreamzero_step_cache_dit",
    "enable_dreamzero_stepcache",
    "get_step_cache_dit_config",
    "get_step_cache_dit_state",
    "get_stepcache_config",
    "get_stepcache_state",
    "is_step_cache_dit_active",
    "is_stepcache_active",
]
