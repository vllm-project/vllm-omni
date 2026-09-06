# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.cache.base import CacheBackend
from vllm_omni.diffusion.cache.seacache.config import SeaCacheConfig
from vllm_omni.diffusion.cache.seacache.hook import (
    SeaCacheRootHook,
    apply_sea_cache_hook,
)
from vllm_omni.diffusion.data import DiffusionCacheConfig

logger = init_logger(__name__)


def enable_cosmos3_seacache(
    pipeline: Any,
    config: DiffusionCacheConfig,
) -> SeaCacheRootHook:
    transformer = getattr(pipeline, "transformer", None)
    if transformer is None:
        raise ValueError("SeaCache requires a pipeline with a transformer")
    if not callable(getattr(transformer, "_run_gen_layers", None)):
        raise ValueError("Pipeline transformer does not expose the block boundary required by SeaCache")

    sea_config = SeaCacheConfig(
        threshold=config.sea_threshold,
        residual_order=config.sea_residual_order,
        max_consecutive_cached=config.sea_max_consecutive_cached,
        power_exp=config.sea_power_exp,
    )
    hook = apply_sea_cache_hook(
        transformer,
        sea_config,
        current_step_callback=lambda: getattr(pipeline, "current_step_index", None),
        current_sigma_callback=lambda: getattr(pipeline, "current_sigma", None),
        num_inference_steps_callback=lambda: getattr(
            pipeline,
            "num_timesteps",
            None,
        ),
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


CUSTOM_SEACACHE_ENABLERS = {
    "Cosmos3OmniDiffusersPipeline": enable_cosmos3_seacache,
    "Cosmos3OmniPipeline": enable_cosmos3_seacache,
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
        pipeline._sea_cache_hook = hook

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
        pipeline._sea_cache_hook = hook
        if verbose:
            logger.debug("SeaCache state refreshed")
