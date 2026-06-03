# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
MagCache backend implementation.

This module provides the MagCache backend that implements the CacheBackend
interface using the hooks-based MagCache system.
"""

from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.cache.base import CacheBackend
from vllm_omni.diffusion.cache.magcache.config import MagCacheConfig
from vllm_omni.diffusion.cache.magcache.hook import (
    apply_mag_cache_hook,
)
from vllm_omni.diffusion.cache.magcache.strategy import (
    get_strategy,
)
from vllm_omni.diffusion.data import DiffusionCacheConfig

logger = init_logger(__name__)


class MagCacheBackend(CacheBackend):
    """
    MagCache implementation using hooks.

    MagCache (Magnitude-based Cache) is an adaptive caching technique that
    speeds up diffusion inference by reusing transformer block computations
    based on accumulated magnitude error between timesteps.

    The backend applies MagCache hooks to the transformer which intercept the
    forward pass and implement the caching logic transparently.

    Example:
        >>> from vllm_omni.diffusion.data import DiffusionCacheConfig
        >>> from vllm_omni.diffusion.cache.magcache.config import MagCacheConfig
        >>> from vllm_omni.diffusion.cache.magcache.strategy import FluxMagCacheStrategy
        >>> cache_config = DiffusionCacheConfig(
        ...     mag_ratios=FluxMagCacheStrategy.FLUX_MAG_RATIOS,
        ...     num_inference_steps=28,
        ...     mag_threshold=0.24,
        ...     mag_max_skip_steps=5,
        ...     mag_retention_ratio=0.1,
        ... )
        >>> backend = MagCacheBackend(cache_config)
        >>> backend.enable(pipeline)
        >>> backend.refresh(pipeline, num_inference_steps=50)
    """

    def __init__(self, config: DiffusionCacheConfig):
        super().__init__(config)
        self._registered = False
        self._magcache_config: MagCacheConfig | None = None
        self._transformer_id: int | None = None

    def enable(self, pipeline: Any) -> None:
        """Enable MagCache on transformer using hooks.

        This creates a MagCacheConfig from the backend's DiffusionCacheConfig
        and applies the MagCache hook to the transformer.

        Args:
            pipeline: Diffusion pipeline instance. Extracts transformer and transformer_type:
                     - transformer: pipeline.transformer
                     - transformer_type: pipeline.transformer.__class__.__name__
        """
        transformer = pipeline.transformer
        transformer_type = transformer.__class__.__name__

        num_inference_steps = self.config.num_inference_steps or 28

        mag_ratios = self.config.mag_ratios
        strategy = None

        if mag_ratios is None and not self.config.mag_calibrate:
            strategy = get_strategy(transformer_type)
            original_ratios = strategy.mag_ratios

            if len(original_ratios) != num_inference_steps and hasattr(strategy, "nearest_interp"):
                mag_ratios = strategy.nearest_interp(original_ratios, num_inference_steps)
                logger.info(
                    f"MagCache: Interpolated mag_ratios from {len(original_ratios)} to {num_inference_steps} steps"
                )
            else:
                mag_ratios = original_ratios
                if len(original_ratios) != num_inference_steps:
                    logger.warning(
                        f"MagCache: mag_ratios length ({len(original_ratios)}) != "
                        f"num_inference_steps ({num_inference_steps}), "
                        f"this may cause unexpected behavior"
                    )

            logger.info(f"MagCache: Using mag_ratios from {type(strategy).__name__}")

        if mag_ratios is None and not self.config.mag_calibrate:
            raise ValueError(
                f"mag_ratios must be provided for MagCache. "
                f"For {transformer_type}, you need to provide mag_ratios or run in calibrate mode."
            )

        self._magcache_config = MagCacheConfig(
            transformer_type=transformer_type,
            threshold=self.config.mag_threshold,
            max_skip_steps=self.config.mag_max_skip_steps,
            retention_ratio=self.config.mag_retention_ratio,
            num_inference_steps=num_inference_steps,
            mag_calibrate=self.config.mag_calibrate,
            mag_ratios=mag_ratios if not self.config.mag_calibrate else None,
        )
        self._transformer_id = id(transformer)

        apply_mag_cache_hook(transformer, self._magcache_config, strategy=strategy)

        self._registered = True
        self.enabled = True

    def refresh(self, pipeline: Any, num_inference_steps: int) -> None:
        """Refresh MagCache state for new generation.

        Clears all cached residuals and resets counters/accumulators.
        Should be called before each generation to ensure clean state.

        Args:
            pipeline: Diffusion pipeline instance. Extracts transformer via pipeline.transformer.
            num_inference_steps: Number of inference steps for the current generation.
                                May be used for cache context updates.
        """
        transformer = pipeline.transformer
        current_transformer_id = id(transformer)

        needs_re_register = False

        if self._registered and hasattr(self, "_transformer_id"):
            if current_transformer_id != self._transformer_id:
                logger.warning(
                    f"Transformer was replaced (id changed from {self._transformer_id} "
                    f"to {current_transformer_id}), re-registering hooks"
                )
                needs_re_register = True

        if not self._registered or needs_re_register:
            self.enable(pipeline)
            return

        blocks_with_hooks = []

        for name, submodule in transformer.named_children():
            if not isinstance(submodule, torch.nn.ModuleList):
                continue
            for index, block in enumerate(submodule):
                registry = getattr(block, "_hook_registry", None)
                if registry is not None and len(registry._hooks) > 0:
                    blocks_with_hooks.append((f"{name}.{index}", block, registry))

        if not blocks_with_hooks:
            logger.warning("No hooks found on transformer blocks, re-registering")
            apply_mag_cache_hook(transformer, self._magcache_config)
            self._transformer_id = current_transformer_id
        else:
            for name, block, registry in blocks_with_hooks:
                for hook in registry._hooks.values():
                    if hasattr(hook, "reset_state"):
                        hook.reset_state(block)

    def is_enabled(self) -> bool:
        """Check if MagCache is enabled.

        Returns:
            True if enabled, False otherwise.
        """
        return self.enabled
