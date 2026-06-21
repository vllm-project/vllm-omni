# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
StepCache backend implementation.

This module provides the stepcache backend that implements the CacheBackend
interface for DreamZero-style velocity-based step skipping in the denoise loop.
"""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.cache.base import CacheBackend
from vllm_omni.diffusion.cache.stepcache.config import (
    STEP_CACHE_CONFIG_ATTR,
    STEP_CACHE_STATE_ATTR,
    StepCacheConfig,
)
from vllm_omni.diffusion.cache.stepcache.state import StepCacheState
from vllm_omni.diffusion.data import DiffusionCacheConfig

logger = init_logger(__name__)

# Backward-compatible aliases for older imports and pipeline attribute names.
STEP_CACHE_DIT_CONFIG_ATTR = STEP_CACHE_CONFIG_ATTR
STEP_CACHE_DIT_STATE_ATTR = STEP_CACHE_STATE_ATTR
StepCacheDiTConfig = StepCacheConfig
StepCacheDiTState = StepCacheState


def is_stepcache_active(pipeline: Any) -> bool:
    """Return True when stepcache is enabled on *pipeline*."""
    cfg = getattr(pipeline, STEP_CACHE_CONFIG_ATTR, None)
    return isinstance(cfg, StepCacheConfig) and cfg.enabled


def get_stepcache_config(pipeline: Any) -> StepCacheConfig | None:
    cfg = getattr(pipeline, STEP_CACHE_CONFIG_ATTR, None)
    return cfg if isinstance(cfg, StepCacheConfig) else None


def get_stepcache_state(pipeline: Any) -> StepCacheState | None:
    state = getattr(pipeline, STEP_CACHE_STATE_ATTR, None)
    return state if isinstance(state, StepCacheState) else None


def is_step_cache_dit_active(pipeline: Any) -> bool:
    return is_stepcache_active(pipeline)


def get_step_cache_dit_config(pipeline: Any) -> StepCacheConfig | None:
    return get_stepcache_config(pipeline)


def get_step_cache_dit_state(pipeline: Any) -> StepCacheState | None:
    return get_stepcache_state(pipeline)


def enable_dreamzero_stepcache(pipeline: Any, config: DiffusionCacheConfig) -> None:
    """Enable stepcache for DreamZeroPipeline."""
    cache_config = StepCacheConfig.from_diffusion_cache_config(config)
    setattr(pipeline, STEP_CACHE_CONFIG_ATTR, cache_config)
    setattr(pipeline, STEP_CACHE_STATE_ATTR, StepCacheState(cache_config))

    logger.info(
        "stepcache enabled on DreamZeroPipeline (enabled=%s, thresholds=%s, countdowns=%s)",
        cache_config.enabled,
        cache_config.sim_thresholds,
        cache_config.skip_countdowns,
    )


enable_dreamzero_step_cache_dit = enable_dreamzero_stepcache

CUSTOM_STEPCACHE_ENABLERS = {
    "DreamZeroPipeline": enable_dreamzero_stepcache,
}

CUSTOM_STEP_CACHE_DIT_ENABLERS = CUSTOM_STEPCACHE_ENABLERS


class StepCacheBackend(CacheBackend):
    """
    Velocity cosine step-skipping cache backend for DreamZero.

    Attaches :class:`StepCacheConfig` and :class:`StepCacheState` to supported
    pipelines. The denoise loop calls :meth:`StepCacheState.should_run_step` to
    decide whether to run ``predict_noise``.

    Example:
        >>> from vllm_omni.diffusion.data import DiffusionCacheConfig
        >>> backend = StepCacheBackend(DiffusionCacheConfig())
        >>> backend.enable(pipeline)
        >>> backend.refresh(pipeline, num_inference_steps=16)
    """

    def enable(self, pipeline: Any) -> None:
        """Enable stepcache on a supported pipeline."""
        pipeline_type = pipeline.__class__.__name__

        if pipeline_type in CUSTOM_STEPCACHE_ENABLERS:
            logger.info("Using custom stepcache enabler for model: %s", pipeline_type)
            CUSTOM_STEPCACHE_ENABLERS[pipeline_type](pipeline, self.config)
        else:
            raise ValueError(
                f"step_cache backend does not support {pipeline_type}. Supported: {sorted(CUSTOM_STEPCACHE_ENABLERS)}"
            )

        self.enabled = True

    def refresh(self, pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        """Refresh stepcache state for a new generation."""
        state = get_stepcache_state(pipeline)
        if state is not None:
            state.reset()
            if verbose:
                logger.debug(
                    "stepcache state refreshed (num_inference_steps=%d)",
                    num_inference_steps,
                )
        elif verbose and is_stepcache_active(pipeline):
            logger.warning("stepcache config active but state not found on pipeline")


StepCacheDiTBackend = StepCacheBackend
