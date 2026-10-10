# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.cache.base import CacheBackend
from vllm_omni.diffusion.cache.seacache.config import SeaCacheConfig
from vllm_omni.diffusion.cache.seacache.extractors import (
    extract_flux2_seacache_context,
    extract_flux_seacache_context,
    extract_qwen_seacache_context,
)
from vllm_omni.diffusion.cache.seacache.hook import (
    SeaCacheRootHook,
    apply_sea_cache_hook,
)
from vllm_omni.diffusion.cache.teacache.extractors import CacheContext
from vllm_omni.diffusion.data import DiffusionCacheConfig

logger = init_logger(__name__)


def _enable_seacache(
    pipeline: Any,
    config: DiffusionCacheConfig,
    *,
    extractor_fn: Callable[..., CacheContext | None] | None = None,
    can_cache_callback: Callable[[], bool] | None = None,
) -> SeaCacheRootHook:
    """Install the shared indicator and residual extrapolation hook."""
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        raise ValueError("SeaCache requires a pipeline with a transformer")

    image_pipeline = type(pipeline).__name__ in _IMAGE_EXTRACTORS
    max_consecutive_cached = config.sea_max_consecutive_cached
    power_exp = config.sea_power_exp
    if max_consecutive_cached is None:
        max_consecutive_cached = 0 if image_pipeline else 2
    if power_exp is None:
        power_exp = 2.0 if image_pipeline else 3.0

    sea_config = SeaCacheConfig(
        threshold=config.sea_threshold,
        residual_order=config.sea_residual_order,
        max_consecutive_cached=max_consecutive_cached,
        power_exp=power_exp,
    )
    hook = apply_sea_cache_hook(
        transformer,
        sea_config,
        current_step_callback=lambda: getattr(pipeline, "current_step_index", None),
        current_sigma_callback=lambda: getattr(pipeline, "current_sigma", None),
        num_inference_steps_callback=lambda: getattr(pipeline, "num_timesteps", None),
        extractor_fn=extractor_fn,
        can_cache_callback=can_cache_callback,
    )
    logger.info(
        "SeaCache enabled for %s (threshold=%s, residual_order=%d, max_consecutive_cached=%d, power_exp=%s)",
        pipeline.__class__.__name__,
        sea_config.threshold,
        sea_config.residual_order,
        sea_config.max_consecutive_cached,
        sea_config.power_exp,
    )
    return hook


enable_cosmos3_seacache = _enable_seacache


_IMAGE_EXTRACTORS: dict[str, Callable[..., CacheContext | None]] = {
    "FluxPipeline": extract_flux_seacache_context,
    "Flux2Pipeline": extract_flux2_seacache_context,
    "Flux2KleinPipeline": extract_flux2_seacache_context,
    "QwenImagePipeline": extract_qwen_seacache_context,
    "QwenImageEditPipeline": extract_qwen_seacache_context,
    "QwenImageEditPlusPipeline": extract_qwen_seacache_context,
}


def enable_image_seacache(pipeline: Any, config: DiffusionCacheConfig) -> SeaCacheRootHook:
    """Adapt image modulation features."""

    def can_cache() -> bool:
        # Check after SP/offload hooks are installed: their collectives cannot be skipped.
        od_config = getattr(pipeline, "od_config", None)
        parallel = getattr(pipeline.transformer, "parallel_config", None)
        if parallel is None:
            parallel = getattr(od_config, "parallel_config", None)
        return (getattr(parallel, "sequence_parallel_size", 1) or 1) == 1 and not getattr(
            od_config, "enable_distributed_layerwise_offload", False
        )

    return _enable_seacache(
        pipeline,
        config,
        extractor_fn=_IMAGE_EXTRACTORS[type(pipeline).__name__],
        can_cache_callback=can_cache,
    )


CUSTOM_SEACACHE_ENABLERS = {
    "Cosmos3OmniDiffusersPipeline": enable_cosmos3_seacache,
    "Cosmos3OmniPipeline": enable_cosmos3_seacache,
    **dict.fromkeys(_IMAGE_EXTRACTORS, enable_image_seacache),
}


class SeaCacheBackend(CacheBackend):
    """Backend for spectral-evolution-aware diffusion caching."""

    def __init__(self, config: DiffusionCacheConfig):
        super().__init__(config)
        self._transformer_id: int | None = None

    def enable(self, pipeline: Any) -> None:
        pipeline_type = pipeline.__class__.__name__
        enabler = CUSTOM_SEACACHE_ENABLERS.get(pipeline_type)
        if enabler is None:
            raise ValueError(f"SeaCache does not support pipeline type {pipeline_type}")
        hook = enabler(pipeline, self.config)
        self._transformer_id = id(pipeline.transformer)
        self.enabled = True
        pipeline._cache_context_factory = hook.cache_context

    def refresh(
        self,
        pipeline: Any,
        num_inference_steps: int,
        verbose: bool = True,
    ) -> None:
        del num_inference_steps
        transformer = getattr(pipeline, "transformer", None)
        if transformer is None:
            raise ValueError("SeaCache requires a pipeline with a transformer")
        if not self.enabled or self._transformer_id != id(transformer):
            self.enable(pipeline)

        registry = getattr(transformer, "_hook_registry", None)
        hook = registry.get_hook(SeaCacheRootHook._HOOK_NAME) if registry is not None else None
        if not isinstance(hook, SeaCacheRootHook):
            raise RuntimeError("SeaCache hook is not installed on the pipeline transformer")
        hook.refresh(transformer)
        pipeline._cache_context_factory = hook.cache_context
        if verbose:
            logger.debug("SeaCache state refreshed")
