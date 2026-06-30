# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_omni.diffusion.cache.magcache.config import MagCacheConfig
from vllm_omni.diffusion.cache.magcache.hook import (
    MagCacheBlockHook,
    MagCacheHeadHook,
    MagCacheState,
    apply_mag_cache_hook,
)
from vllm_omni.diffusion.cache.magcache.strategy import (
    Flux2MagCacheStrategy,
    FluxMagCacheStrategy,
    MagCacheStrategy,
    MagCacheStrategyRegistry,
    get_strategy,
    register_strategy,
)

__all__ = [
    "Flux2MagCacheStrategy",
    "FluxMagCacheStrategy",
    "MagCacheBlockHook",
    "MagCacheConfig",
    "MagCacheHeadHook",
    "MagCacheState",
    "MagCacheStrategy",
    "MagCacheStrategyRegistry",
    "apply_mag_cache_hook",
    "get_strategy",
    "register_strategy",
]
