# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Generic Cache-DiT backend lifecycle and integration."""

from __future__ import annotations

from collections.abc import Callable
from operator import attrgetter
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import cache_dit
import torch.nn as nn
from cache_dit import BlockAdapter, DBCacheConfig
from cache_dit.caching.cache_adapters.cache_adapter import CachedAdapter
from vllm.logger import init_logger

from vllm_omni.diffusion.cache.base import CacheBackend
from vllm_omni.diffusion.cache.cachedit.config import (
    CacheDiTAdapterConfig,
    CacheDiTConfig,
)
from vllm_omni.diffusion.data import DiffusionCacheConfig

if TYPE_CHECKING:
    from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery

logger = init_logger(__name__)

RefreshCacheContextFunc: TypeAlias = Callable[[Any, int, bool], None]
CacheDiTEnabler: TypeAlias = Callable[[Any, DiffusionCacheConfig], RefreshCacheContextFunc]

# Model-specific implementations register themselves when the package loads.
CUSTOM_DIT_ENABLERS: dict[str, CacheDiTEnabler] = {}


def _dit_module_names(pipeline: SupportsComponentDiscovery) -> tuple[str, ...]:
    """Return the pipeline DiT attributes that are present at runtime."""
    names = getattr(pipeline, "_dit_modules", None)
    if not isinstance(names, (list, tuple)):
        names = ("transformer",)

    resolved_names = []
    for name in names:
        if not isinstance(name, str):
            continue
        try:
            module = attrgetter(name)(pipeline)
        except AttributeError:
            continue
        if module is not None:
            resolved_names.append(name)
    return tuple(resolved_names)


def cache_summary(pipeline: SupportsComponentDiscovery, details: bool = True) -> None:
    """Log Cache-DiT statistics for every transformer on the pipeline."""

    transformers = [attrgetter(name)(pipeline) for name in _dit_module_names(pipeline)]
    for transformer in transformers:
        if not BlockAdapter.is_cached(transformer):
            continue
        cache_dit.summary(transformer, details=details)

    if not transformers:
        logger.warning("CacheDiT summary failed; this pipeline has no defined transformer attribute")


def _default_get_pipeline_transformer(
    pipeline: SupportsComponentDiscovery,
) -> nn.Module:
    return cast(nn.Module, getattr(pipeline, "transformer"))


def _make_pipeline_transformer_getter(
    name: str,
) -> Callable[[SupportsComponentDiscovery], nn.Module]:
    def get_pipeline_transformer(
        pipeline: SupportsComponentDiscovery,
    ) -> nn.Module:
        return cast(nn.Module, attrgetter(name)(pipeline))

    return get_pipeline_transformer


def _build_cache_context_refresh(
    cache_config: DiffusionCacheConfig,
    get_pipeline_transformer: Callable[[SupportsComponentDiscovery], nn.Module] = _default_get_pipeline_transformer,
) -> RefreshCacheContextFunc:
    """Build the cache context refresh callback for one transformer."""

    projected_config = CacheDiTConfig.from_diffusion_config(cache_config)

    def refresh_cache_context(
        pipeline: SupportsComponentDiscovery, num_inference_steps: int, verbose: bool = True
    ) -> None:
        transformer = get_pipeline_transformer(pipeline)

        # Cache-DiT has no predefined SCM mask for these small step counts.
        scm_supported_steps = num_inference_steps >= 8 or num_inference_steps in (4, 6)
        if projected_config.scm_steps_mask_policy is None or not scm_supported_steps:
            if projected_config.force_refresh_step_hint is not None:
                # Cache-DiT's ``once`` policy clears the hint after it fires.
                # Reapply it at every request boundary so a repeated request
                # with the same configured policy receives the same behavior.
                cache_dit.refresh_context(
                    transformer,
                    cache_config=DBCacheConfig().reset(
                        num_inference_steps=num_inference_steps,
                        force_refresh_step_hint=projected_config.force_refresh_step_hint,
                        force_refresh_step_policy=projected_config.force_refresh_step_policy,
                    ),
                    verbose=verbose,
                )
                return
            cache_dit.refresh_context(transformer, num_inference_steps=num_inference_steps, verbose=verbose)
            return

        refresh_config = DBCacheConfig().reset(
            num_inference_steps=num_inference_steps,
            steps_computation_mask=cache_dit.steps_mask(
                mask_policy=projected_config.scm_steps_mask_policy,
                total_steps=num_inference_steps,
            ),
            steps_computation_policy=projected_config.scm_steps_policy,
        )
        if projected_config.force_refresh_step_hint is not None:
            refresh_config.force_refresh_step_hint = projected_config.force_refresh_step_hint
            refresh_config.force_refresh_step_policy = projected_config.force_refresh_step_policy
        cache_dit.refresh_context(
            transformer,
            cache_config=refresh_config,
            verbose=verbose,
        )

    return refresh_cache_context


def enable_cache_for_dit(
    pipeline: SupportsComponentDiscovery,
    cache_config: DiffusionCacheConfig,
    block_adapter: BlockAdapter | None = None,
    adapter_cls: type[CachedAdapter] | None = None,
    get_pipeline_transformer: Callable[[SupportsComponentDiscovery], nn.Module] = _default_get_pipeline_transformer,
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

    transformer = get_pipeline_transformer(pipeline)
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

    return _build_cache_context_refresh(cache_config, get_pipeline_transformer)


def _maybe_build_block_adapter(
    pipeline: SupportsComponentDiscovery,
    get_pipeline_transformer: Callable[[SupportsComponentDiscovery], nn.Module] = _default_get_pipeline_transformer,
) -> BlockAdapter | None:
    """Build the model-declared block adapter, when one is configured."""

    transformer = get_pipeline_transformer(pipeline)
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

    block_adapter = BlockAdapter(
        transformer=transformer,
        blocks=[getattr(transformer, block_attribute) for block_attribute in block_attributes],
        forward_pattern=list(forward_patterns),
        has_separate_cfg=adapter_config.has_separate_cfg,
        check_forward_pattern=adapter_config.check_forward_pattern,
    )
    # cache_dit marks the wrapper pipe's class as cached, and for bare-transformer
    # adapters that class is shared process-wide; a marker left by a previous
    # engine would make this enable skip cache-context installation.
    pipe_cls = type(block_adapter.pipe)
    if "_is_cached" in vars(pipe_cls):
        del pipe_cls._is_cached
    return block_adapter


def _maybe_get_cached_adapter_cls(
    pipeline: SupportsComponentDiscovery,
    get_pipeline_transformer: Callable[[SupportsComponentDiscovery], nn.Module] = _default_get_pipeline_transformer,
) -> type[CachedAdapter] | None:
    """Return the custom cached adapter declared by the transformer."""

    transformer = get_pipeline_transformer(pipeline)
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
        self._refresh_funcs: list[RefreshCacheContextFunc] = []
        self._cache_targets: list[Any] = []

    def enable(self, pipeline: SupportsComponentDiscovery) -> None:
        pipeline_name = type(pipeline).__name__
        custom_enabler = CUSTOM_DIT_ENABLERS.get(pipeline_name)
        self._refresh_funcs = []
        self._cache_targets = []
        if custom_enabler is not None:
            logger.info("Using custom cache-dit enabler for model: %s", pipeline_name)
            self._refresh_funcs = [custom_enabler(pipeline, self.config)]
            self._cache_targets = [_default_get_pipeline_transformer(pipeline)]
        else:
            for name in _dit_module_names(pipeline):
                get_transformer = _make_pipeline_transformer_getter(name)
                block_adapter = _maybe_build_block_adapter(pipeline, get_transformer)
                adapter_cls = _maybe_get_cached_adapter_cls(pipeline, get_transformer)
                cache_target = get_transformer(pipeline) if block_adapter is None else block_adapter
                self._refresh_funcs.append(
                    enable_cache_for_dit(
                        pipeline,
                        self.config,
                        block_adapter,
                        adapter_cls,
                        get_transformer,
                    )
                )
                self._cache_targets.append(cache_target)
            if not self._refresh_funcs:
                raise ValueError(f"Pipeline {pipeline_name} has no declared DiT modules for Cache-DiT")

        self.enabled = True
        logger.info("Cache-dit enabled successfully on %s", pipeline_name)

    def disable(self, pipeline: SupportsComponentDiscovery) -> None:
        """Remove Cache-DiT hooks so later requests use native forwards."""

        if not self.enabled:
            return

        logger.info(
            "Disabling cache-dit on %d DiT target(s) for %s",
            len(self._cache_targets),
            type(pipeline).__name__,
        )
        try:
            for target in self._cache_targets:
                cache_dit.disable_cache(target)
        finally:
            self._refresh_funcs = []
            self._cache_targets = []
            self.enabled = False

    def refresh(self, pipeline: SupportsComponentDiscovery, num_inference_steps: int, verbose: bool = True) -> None:
        if not self.enabled or not self._refresh_funcs:
            logger.warning("Cache-dit is not enabled. Cannot refresh cache context.")
            return

        if verbose:
            logger.info(
                "Refreshing cache context for transformer with num_inference_steps: %s",
                num_inference_steps,
            )
        for refresh_func in self._refresh_funcs:
            refresh_func(pipeline, num_inference_steps, verbose)


__all__ = [
    "CUSTOM_DIT_ENABLERS",
    "CacheDiTBackend",
    "cache_summary",
    "enable_cache_for_dit",
]
