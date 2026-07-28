# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Generic Cache-DiT backend lifecycle and integration."""

from collections.abc import Callable
from typing import Any, TypeAlias

import cache_dit
from cache_dit import BlockAdapter, DBCacheConfig
from cache_dit.caching.cache_adapters.cache_adapter import CachedAdapter
from vllm.logger import init_logger

from vllm_omni.diffusion.cache.base import CacheBackend
from vllm_omni.diffusion.cache.cachedit.config import (
    CacheDiTAdapterConfig,
    CacheDiTConfig,
)
from vllm_omni.diffusion.data import DiffusionCacheConfig

logger = init_logger(__name__)

RefreshCacheContextFunc: TypeAlias = Callable[[Any, int, bool], None]
CacheDiTEnabler: TypeAlias = Callable[[Any, DiffusionCacheConfig], RefreshCacheContextFunc]

# Model-specific implementations register themselves when the package loads.
CUSTOM_DIT_ENABLERS: dict[str, CacheDiTEnabler] = {}


def cache_summary(pipeline: Any, details: bool = True) -> None:
    """Log Cache-DiT statistics for every transformer on the pipeline."""

    transformers = [
        transformer
        for attribute in ("transformer", "transformer_2")
        if (transformer := getattr(pipeline, attribute, None)) is not None
    ]
    for transformer in transformers:
        cache_dit.summary(transformer, details=details)

    if not transformers:
        logger.warning("CacheDiT summary failed; this pipeline has no defined transformer attribute")


def _default_get_pipeline_transformer(pipeline: Any) -> Any:
    return pipeline.transformer


def _build_cache_context_refresh(
    cache_config: DiffusionCacheConfig,
    get_pipeline_transformer: Callable[[Any], Any] = _default_get_pipeline_transformer,
) -> RefreshCacheContextFunc:
    """Build the cache context refresh callback for one transformer."""

    projected_config = CacheDiTConfig.from_diffusion_config(cache_config)

    def refresh_cache_context(pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        transformer = get_pipeline_transformer(pipeline)

        # Cache-DiT has no predefined SCM mask for these small step counts.
        scm_supported_steps = num_inference_steps >= 8 or num_inference_steps in (4, 6)
        if projected_config.scm_steps_mask_policy is None or not scm_supported_steps:
            cache_dit.refresh_context(transformer, num_inference_steps=num_inference_steps, verbose=verbose)
            return

        cache_dit.refresh_context(
            transformer,
            cache_config=DBCacheConfig().reset(
                num_inference_steps=num_inference_steps,
                steps_computation_mask=cache_dit.steps_mask(
                    mask_policy=projected_config.scm_steps_mask_policy,
                    total_steps=num_inference_steps,
                ),
                steps_computation_policy=projected_config.scm_steps_policy,
            ),
            verbose=verbose,
        )

    return refresh_cache_context


def enable_cache_for_dit(
    pipeline: Any,
    cache_config: DiffusionCacheConfig,
    block_adapter: BlockAdapter | None = None,
    adapter_cls: type[CachedAdapter] | None = None,
) -> RefreshCacheContextFunc:
    """Enable Cache-DiT for a standard single-transformer DiT pipeline."""

    projected_config = CacheDiTConfig.from_diffusion_config(cache_config)
    db_cache_config = projected_config.to_db_cache_config()
    calibrator_config = projected_config.to_calibrator_config()

    logger.info(
        "Enabling cache-dit on transformer: Fn=%s, Bn=%s, W=%s",
        db_cache_config.Fn_compute_blocks,
        db_cache_config.Bn_compute_blocks,
        db_cache_config.max_warmup_steps,
    )

    transformer = _default_get_pipeline_transformer(pipeline)
    cache_target = transformer if block_adapter is None else block_adapter
    if adapter_cls is not None:
        adapter_cls.apply(
            cache_target,
            cache_config=db_cache_config,
            calibrator_config=calibrator_config,
        )
    elif block_adapter is None:
        try:
            cache_dit.enable_cache(
                cache_target,
                cache_config=db_cache_config,
                calibrator_config=calibrator_config,
            )
        except ValueError as exc:
            raise ValueError(
                "Failed to enable Cache-DiT for pipeline "
                f"{type(pipeline).__name__} with transformer "
                f"{type(transformer).__name__}: no model-declared "
                "_cache_dit_adapter_config or compatible Cache-DiT built-in "
                "adapter was found."
            ) from exc
    else:
        cache_dit.enable_cache(
            cache_target,
            cache_config=db_cache_config,
            calibrator_config=calibrator_config,
        )

    return _build_cache_context_refresh(cache_config)


def _maybe_build_block_adapter(pipeline: Any) -> BlockAdapter | None:
    """Build the model-declared block adapter, when one is configured."""

    transformer = _default_get_pipeline_transformer(pipeline)
    adapter_config: CacheDiTAdapterConfig | None = getattr(transformer, "_cache_dit_adapter_config", None)
    if adapter_config is None:
        logger.info(
            "Transformer %s does not declare _cache_dit_adapter_config; "
            "falling back to Cache-DiT's built-in adapter registry.",
            type(transformer).__name__,
        )
        return None

    block_attributes, forward_patterns = zip(*adapter_config.block_forward_patterns.items())
    missing_attributes = [
        block_attribute for block_attribute in block_attributes if not hasattr(transformer, block_attribute)
    ]
    if missing_attributes:
        raise AttributeError(f"Missing Cache-DiT block attributes: {missing_attributes}")

    return BlockAdapter(
        transformer=transformer,
        blocks=[getattr(transformer, block_attribute) for block_attribute in block_attributes],
        forward_pattern=list(forward_patterns),
        has_separate_cfg=adapter_config.has_separate_cfg,
        check_forward_pattern=adapter_config.check_forward_pattern,
    )


def _maybe_get_cached_adapter_cls(pipeline: Any) -> type[CachedAdapter] | None:
    """Return the custom cached adapter declared by the transformer."""

    transformer = _default_get_pipeline_transformer(pipeline)
    adapter_config: CacheDiTAdapterConfig | None = getattr(transformer, "_cache_dit_adapter_config", None)
    return None if adapter_config is None else adapter_config.cached_adapter_cls


class CacheDiTBackend(CacheBackend):
    """Manage Cache-DiT through the common diffusion cache lifecycle."""

    def __init__(self, cache_config: Any = None):
        if cache_config is None:
            config = DiffusionCacheConfig()
        elif isinstance(cache_config, dict):
            config = DiffusionCacheConfig.from_dict(cache_config)
        else:
            config = cache_config

        super().__init__(config)
        self._refresh_func: RefreshCacheContextFunc | None = None

    def enable(self, pipeline: Any) -> None:
        pipeline_name = type(pipeline).__name__
        custom_enabler = CUSTOM_DIT_ENABLERS.get(pipeline_name)
        if custom_enabler is not None:
            logger.info("Using custom cache-dit enabler for model: %s", pipeline_name)
            self._refresh_func = custom_enabler(pipeline, self.config)
        else:
            block_adapter = _maybe_build_block_adapter(pipeline)
            adapter_cls = _maybe_get_cached_adapter_cls(pipeline)
            self._refresh_func = enable_cache_for_dit(
                pipeline,
                self.config,
                block_adapter,
                adapter_cls,
            )

        self.enabled = True
        logger.info("Cache-dit enabled successfully on %s", pipeline_name)

    def refresh(self, pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        if not self.enabled or self._refresh_func is None:
            logger.warning("Cache-dit is not enabled. Cannot refresh cache context.")
            return

        if verbose:
            logger.info(
                "Refreshing cache context for transformer with num_inference_steps: %s",
                num_inference_steps,
            )
        self._refresh_func(pipeline, num_inference_steps, verbose)


__all__ = [
    "CUSTOM_DIT_ENABLERS",
    "CacheDiTBackend",
    "cache_summary",
    "enable_cache_for_dit",
]
