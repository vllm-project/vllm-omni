# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unified cache setup function for diffusion models.

This module provides the main entry point for setting up cache adapters
on transformer models. It handles cache type selection, adapter instantiation,
and application to the model.
"""

from typing import Any, Optional

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.cache.base import CacheAdapter
from vllm_omni.diffusion.cache.registry import get_cache_adapter

logger = init_logger(__name__)


def setup_cache(
    transformer: torch.nn.Module,
    cache_type: str = "none",
    cache_config: Optional[dict[str, Any]] = None,
) -> Optional[CacheAdapter]:
    """
    Setup cache adapter for transformer (one-time initialization).

    This is the main entry point for applying caching to diffusion transformers.
    It's called once during pipeline initialization and applies the appropriate
    cache adapter based on configuration.

    Args:
        transformer: Transformer module to apply cache to
        cache_type: Type of cache to apply. Options:
            - "none": No caching (default)
            - "tea_cache": TeaCache adaptive caching
            - Additional types can be registered via register_cache_adapter()
        cache_config: Cache-specific configuration dictionary. Contents depend
            on cache type:
            - For "tea_cache": {"rel_l1_thresh": 0.2, "model_type": "QwenImagePipeline"}
            - For other types: see their respective adapter documentation

    Returns:
        CacheAdapter instance for state management, or None if cache_type="none"

    Example:
        >>> # Setup TeaCache during pipeline initialization
        >>> adapter = setup_cache(
        ...     transformer,
        ...     cache_type="tea_cache",
        ...     cache_config={"rel_l1_thresh": 0.2}
        ... )
        >>>
        >>> # Reset cache state before each generation
        >>> if adapter is not None:
        ...     adapter.reset(transformer)
    """
    # No caching requested
    if cache_type == "none":
        logger.debug("No cache adapter requested (cache_type='none')")
        return None

    # Get adapter instance from registry
    try:
        adapter = get_cache_adapter(cache_type, cache_config or {})
    except ValueError as e:
        logger.error(f"Failed to get cache adapter: {e}")
        raise

    # Apply cache to transformer
    logger.info(f"Applying {cache_type} cache to transformer")
    try:
        adapter.apply(transformer)
    except Exception as e:
        logger.error(f"Failed to apply cache adapter: {e}")
        raise

    logger.info(f"Cache adapter '{cache_type}' successfully applied")
    return adapter
