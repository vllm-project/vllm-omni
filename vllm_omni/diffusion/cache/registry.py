# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Cache adapter registry for diffusion models.

This module provides a registry pattern for cache adapters, allowing dynamic
registration and instantiation of different cache types (TeaCache, DeepCache, etc.).
"""

from enum import Enum
from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.cache.base import CacheAdapter

logger = init_logger(__name__)


class CacheType(Enum):
    """Supported cache adapter types."""

    NONE = "none"
    TEA_CACHE = "tea_cache"
    # Future cache types can be added here:
    # DEEP_CACHE = "deep_cache"
    # DISTRI_FUSION = "distri_fusion"


# Global registry mapping cache types to adapter classes
CACHE_ADAPTER_REGISTRY: dict[CacheType, type[CacheAdapter]] = {}


def register_cache_adapter(cache_type: CacheType, adapter_class: type[CacheAdapter]) -> None:
    """
    Register a cache adapter class for a given cache type.

    This allows extending the system with new cache types without modifying
    the core cache infrastructure.

    Args:
        cache_type: CacheType enum value
        adapter_class: CacheAdapter subclass to register

    Example:
        >>> register_cache_adapter(CacheType.TEA_CACHE, TeaCacheAdapter)
    """
    if not issubclass(adapter_class, CacheAdapter):
        raise TypeError(f"{adapter_class} must be a subclass of CacheAdapter")

    CACHE_ADAPTER_REGISTRY[cache_type] = adapter_class
    logger.debug(f"Registered cache adapter: {cache_type.value} -> {adapter_class.__name__}")


def get_cache_adapter(cache_type: str, config: dict[str, Any]) -> CacheAdapter:
    """
    Factory function to get cache adapter instance.

    Converts string cache type to enum, looks up in registry, and instantiates
    the appropriate adapter class with the provided configuration.

    Args:
        cache_type: String name of cache type ("tea_cache", "deep_cache", etc.)
        config: Configuration dictionary to pass to adapter constructor

    Returns:
        Instantiated CacheAdapter subclass

    Raises:
        ValueError: If cache_type is unknown or not registered

    Example:
        >>> adapter = get_cache_adapter("tea_cache", {"rel_l1_thresh": 0.2})
        >>> adapter.apply(transformer)
    """
    # Normalize cache type string
    cache_type_str = cache_type.lower().strip()

    # Convert string to enum
    try:
        cache_enum = CacheType(cache_type_str)
    except ValueError:
        available = [ct.value for ct in CacheType if ct != CacheType.NONE]
        raise ValueError(f"Unknown cache type: '{cache_type}'. Available types: {available}")

    # Check if it's the special "none" case
    if cache_enum == CacheType.NONE:
        raise ValueError("Cannot instantiate adapter for cache_type='none'. Use setup_cache() which handles this case.")

    # Lookup in registry
    if cache_enum not in CACHE_ADAPTER_REGISTRY:
        raise ValueError(
            f"Cache type '{cache_type}' is not registered. Registered types: {list(CACHE_ADAPTER_REGISTRY.keys())}"
        )

    adapter_class = CACHE_ADAPTER_REGISTRY[cache_enum]

    # Instantiate and return
    return adapter_class(config)
