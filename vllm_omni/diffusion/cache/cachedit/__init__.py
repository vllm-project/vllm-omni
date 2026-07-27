# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Public API for the Cache-DiT diffusion cache backend."""

from vllm_omni.diffusion.cache.cachedit.backend import (
    CUSTOM_DIT_ENABLERS,
    CacheDiTBackend,
    cache_summary,
    enable_cache_for_dit,
)
from vllm_omni.diffusion.cache.cachedit.config import (
    CacheDiTAdapterConfig,
    CacheDiTConfig,
)
from vllm_omni.diffusion.cache.cachedit.model_specific import (
    BagelCachedAdapter,
    SensenovaCachedAdapter,
)
from vllm_omni.diffusion.cache.cachedit.model_specific import (
    register_custom_dit_enablers as _register_custom_dit_enablers,
)

_register_custom_dit_enablers()

__all__ = [
    "BagelCachedAdapter",
    "CUSTOM_DIT_ENABLERS",
    "CacheDiTAdapterConfig",
    "CacheDiTBackend",
    "CacheDiTConfig",
    "SensenovaCachedAdapter",
    "cache_summary",
    "enable_cache_for_dit",
]
