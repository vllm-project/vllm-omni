from typing import Any

from vllm_omni.diffusion.cache.base import CacheBackend
from vllm_omni.diffusion.cache.cache_dit_backend import CacheDiTBackend
from vllm_omni.diffusion.cache.teacache.backend import TeaCacheBackend
from vllm_omni.diffusion.data import DiffusionCacheConfig


def get_cache_backend(cache_backend: str | None, cache_config: Any) -> CacheBackend | None:
    """Get cache backend instance based on cache_backend string.

    This is a selector function that routes to the appropriate backend implementation.
    - cache_dit: Uses CacheDiTBackend with enable()/refresh() interface
    - tea_cache: Uses TeaCacheBackend with enable()/refresh() interface

    Args:
        cache_backend: Cache backend name ("cache_dit", "tea_cache", or None).
        cache_config: Cache configuration (dict or DiffusionCacheConfig instance).

    Returns:
        Cache backend instance (CacheDiTBackend or TeaCacheBackend) if cache_backend is set,
        None otherwise.

    Raises:
        ValueError: If cache_backend is unsupported.
    """
    if cache_backend is None or cache_backend == "none":
        return None

    if isinstance(cache_config, dict):
        cache_config = DiffusionCacheConfig.from_dict(cache_config)

    if cache_backend == "cache_dit":
        return CacheDiTBackend(cache_config)
    elif cache_backend == "tea_cache":
        return TeaCacheBackend(cache_config)
    else:
        raise ValueError(f"Unsupported cache backend: {cache_backend}. Supported: 'cache_dit', 'tea_cache'")
