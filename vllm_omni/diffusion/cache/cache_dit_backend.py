# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
cache-dit integration backend for vllm-omni.

This module provides a CacheDiTBackend class to enable cache-dit acceleration on diffusion
pipelines in vllm-omni, supporting both single and dual-transformer architectures.
"""

from collections.abc import Callable
from typing import Any, Optional

import cache_dit
from cache_dit import BlockAdapter, DBCacheConfig, ForwardPattern, ParamsModifier, TaylorSeerCalibratorConfig
from vllm.logger import init_logger

from vllm_omni.diffusion.cache.base import CacheBackend
from vllm_omni.diffusion.data import DiffusionCacheConfig, OmniDiffusionConfig

logger = init_logger(__name__)


# Registry of custom cache-dit enablers for specific models
# Maps pipeline names to their cache-dit enablement functions
# Models in this registry require custom handling (e.g., dual-transformer architectures)
# Will be populated after function definitions
CUSTOM_DIT_ENABLERS: dict[str, Callable] = {}


def _build_db_cache_config(cache_config: Any) -> DBCacheConfig:
    """Build DBCacheConfig with optional SCM (Step Computation Masking) support.

    Args:
        cache_config: DiffusionCacheConfig instance.

    Returns:
        DBCacheConfig instance with SCM support if configured.
    """

    return DBCacheConfig(
        # we will refresh the context when gets num_inference_steps in the first inference request
        num_inference_steps=None,
        Fn_compute_blocks=cache_config.Fn_compute_blocks,
        Bn_compute_blocks=cache_config.Bn_compute_blocks,
        max_warmup_steps=cache_config.max_warmup_steps,
        max_cached_steps=cache_config.max_cached_steps,
        max_continuous_cached_steps=cache_config.max_continuous_cached_steps,
        residual_diff_threshold=cache_config.residual_diff_threshold,
    )


def enable_cache_for_wan22(pipeline: Any, cache_config: Any) -> Callable[[int], None]:
    """Enable cache-dit for Wan2.2 dual-transformer architecture.

    Wan2.2 uses two transformers (transformer and transformer_2) that need
    to be enabled together using BlockAdapter.

    Args:
        pipeline: The Wan2.2 pipeline instance.
        cache_config: DiffusionCacheConfig instance with cache configuration.

    Returns:
        A refresh function that can be called to update cache context with new num_inference_steps.
    """

    # Build DBCacheConfig for primary transformer
    primary_cache_config = _build_db_cache_config(cache_config)

    # FIXME: secondary cache shares the same config with primary cache for now,
    # but we should support different config for secondary transformer in the future
    # Build DBCacheConfig for secondary transformer (can use same or different config)
    secondary_cache_config = _build_db_cache_config(cache_config)

    # Build calibrator configs if TaylorSeer is enabled
    primary_calibrator = None
    secondary_calibrator = None
    if cache_config.enable_taylorseer:
        taylorseer_order = cache_config.taylorseer_order
        primary_calibrator = TaylorSeerCalibratorConfig(taylorseer_order=taylorseer_order)
        secondary_calibrator = TaylorSeerCalibratorConfig(taylorseer_order=taylorseer_order)
        logger.info(f"TaylorSeer enabled with order={taylorseer_order}")

    # Build ParamsModifier for each transformer
    primary_modifier = ParamsModifier(
        cache_config=primary_cache_config,
        calibrator_config=primary_calibrator,
    )
    secondary_modifier = ParamsModifier(
        cache_config=secondary_cache_config,
        calibrator_config=secondary_calibrator,
    )

    logger.info(
        "Enabling cache-dit on Wan2.2 dual transformers with BlockAdapter: "
        f"Fn={primary_cache_config.Fn_compute_blocks}, "
        f"Bn={primary_cache_config.Bn_compute_blocks}, "
        f"W={primary_cache_config.max_warmup_steps}, "
    )

    transformer = pipeline.transformer
    transformer_2 = pipeline.transformer_2
    transformer_blocks = transformer.blocks
    transformer_2_blocks = transformer_2.blocks
    # Enable cache-dit using BlockAdapter for both transformers simultaneously
    cache_dit.enable_cache(
        BlockAdapter(
            transformer=[transformer, transformer_2],
            blocks=[transformer_blocks, transformer_2_blocks],
            forward_pattern=[ForwardPattern.Pattern_2, ForwardPattern.Pattern_2],
            params_modifiers=[primary_modifier, secondary_modifier],
            has_separate_cfg=True,
        ),
    )

    # from https://github.com/vipshop/cache-dit/pull/542
    def _split_inference_steps(num_inference_steps: int) -> tuple[int, int]:
        """Split inference steps into high-noise and low-noise steps for Wan2.2.

        This is an internal helper function specific to Wan2.2's dual-transformer
        architecture that uses boundary_ratio to determine the split point.

        Args:
            num_inference_steps: Total number of inference steps.

        Returns:
            A tuple of (num_high_noise_steps, num_low_noise_steps).
        """
        if pipeline.config.boundary_ratio is not None:
            boundary_timestep = pipeline.config.boundary_ratio * pipeline.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None

        # Set timesteps to calculate the split
        device = next(pipeline.transformer.parameters()).device
        pipeline.scheduler.set_timesteps(num_inference_steps, device=device)

        timesteps = pipeline.scheduler.timesteps
        num_high_noise_steps = 0  # high-noise steps for transformer
        for t in timesteps:
            if boundary_timestep is None or t >= boundary_timestep:
                num_high_noise_steps += 1
        # low-noise steps for transformer_2
        num_low_noise_steps = num_inference_steps - num_high_noise_steps
        return num_high_noise_steps, num_low_noise_steps

    def refresh_cache_context(pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        """Refresh cache context for both transformers with new num_inference_steps.

        Args:
            pipeline: The Wan2.2 pipeline instance.
            num_inference_steps: New number of inference steps.
        """
        num_high_noise_steps, num_low_noise_steps = _split_inference_steps(num_inference_steps)
        # Refresh context for high-noise transformer
        if cache_config.scm_steps_mask_policy is None:
            cache_dit.refresh_context(pipeline.transformer, num_inference_steps=num_high_noise_steps, verbose=verbose)
        else:
            cache_dit.refresh_context(
                pipeline.transformer,
                cache_config=DBCacheConfig().reset(
                    num_inference_steps=num_high_noise_steps,
                    steps_computation_mask=cache_dit.steps_mask(
                        mask_policy=cache_config.scm_steps_mask_policy, total_steps=num_high_noise_steps
                    ),
                    steps_computation_policy=cache_config.scm_steps_policy,
                ),
                verbose=verbose,
            )

        # Refresh context for low-noise transformer
        if cache_config.scm_steps_mask_policy is None:
            cache_dit.refresh_context(pipeline.transformer_2, num_inference_steps=num_low_noise_steps, verbose=verbose)
        else:
            cache_dit.refresh_context(
                pipeline.transformer_2,
                cache_config=DBCacheConfig().reset(
                    num_inference_steps=num_low_noise_steps,
                    steps_computation_mask=cache_dit.steps_mask(
                        mask_policy=cache_config.scm_steps_mask_policy, total_steps=num_low_noise_steps
                    ),
                    steps_computation_policy=cache_config.scm_steps_policy,
                ),
                verbose=verbose,
            )

    return refresh_cache_context


def enable_cache_for_longcat_image(pipeline: Any, cache_config: Any) -> Callable[[int], None]:
    """Enable cache-dit for LongCatImage pipeline.

    Args:
        pipeline: The LongCatImage pipeline instance.
        cache_config: DiffusionCacheConfig instance with cache configuration.
    """
    # Build DBCacheConfig for transformer
    db_cache_config = _build_db_cache_config(cache_config)

    calibrator = None
    if cache_config.enable_taylorseer:
        taylorseer_order = cache_config.taylorseer_order
        calibrator = TaylorSeerCalibratorConfig(taylorseer_order=taylorseer_order)
        logger.info(f"TaylorSeer enabled with order={taylorseer_order}")

    # Build ParamsModifier for transformer
    modifier = ParamsModifier(
        cache_config=db_cache_config,
        calibrator_config=calibrator,
    )

    logger.info(
        f"Enabling cache-dit on LongCatImage transformer with BlockAdapter: "
        f"Fn={db_cache_config.Fn_compute_blocks}, "
        f"Bn={db_cache_config.Bn_compute_blocks}, "
        f"W={db_cache_config.max_warmup_steps}, "
    )

    # Enable cache-dit using BlockAdapter for transformer
    cache_dit.enable_cache(
        (
            BlockAdapter(
                transformer=pipeline.transformer,
                blocks=[
                    pipeline.transformer.transformer_blocks,
                    pipeline.transformer.single_transformer_blocks,
                ],
                forward_pattern=[ForwardPattern.Pattern_1, ForwardPattern.Pattern_1],
                params_modifiers=[modifier],
            )
        ),
        cache_config=db_cache_config,
    )

    def refresh_cache_context(pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        """Refresh cache context for the transformer with new num_inference_steps.

        Args:
            pipeline: The LongCatImage pipeline instance.
            num_inference_steps: New number of inference steps.
        """
        if cache_config.scm_steps_mask_policy is None:
            cache_dit.refresh_context(pipeline.transformer, num_inference_steps=num_inference_steps, verbose=verbose)
        else:
            cache_dit.refresh_context(
                pipeline.transformer,
                cache_config=DBCacheConfig().reset(
                    num_inference_steps=num_inference_steps,
                    steps_computation_mask=cache_dit.steps_mask(
                        mask_policy=cache_config.scm_steps_mask_policy,
                        total_steps=num_inference_steps,
                    ),
                    steps_computation_policy=cache_config.scm_steps_policy,
                ),
                verbose=verbose,
            )

    return refresh_cache_context


def enable_cache_for_flux(pipeline: Any, cache_config: Any) -> Callable[[int], None]:
    """Enable cache-dit for Flux dual-transformer architecture.

    Flux uses two transformers (transformer and transformer_2) that need
    to be enabled together using BlockAdapter.

    Args:
        pipeline: The Flux pipeline instance.
        cache_config: DiffusionCacheConfig instance with cache configuration.

    Returns:
        A refresh function that can be called to update cache context with new num_inference_steps.
    """
    raise NotImplementedError("cache-dit is not implemented for Flux pipeline.")


def enable_cache_for_sd3(pipeline: Any, cache_config: Any) -> Callable[[int], None]:
    """Enable cache-dit for StableDiffusion3Pipeline.

    Args:
        pipeline: The StableDiffusion3 pipeline instance.
        cache_config: DiffusionCacheConfig instance with cache configuration.
    """
    # Build DBCacheConfig for transformer
    db_cache_config = _build_db_cache_config(cache_config)

    calibrator = None
    if cache_config.enable_taylorseer:
        taylorseer_order = cache_config.taylorseer_order
        calibrator = TaylorSeerCalibratorConfig(taylorseer_order=taylorseer_order)
        logger.info(f"TaylorSeer enabled with order={taylorseer_order}")

    # Build ParamsModifier for transformer
    modifier = ParamsModifier(
        cache_config=db_cache_config,
        calibrator_config=calibrator,
    )

    logger.info(
        f"Enabling cache-dit on StableDiffusion3 transformer with BlockAdapter: "
        f"Fn={db_cache_config.Fn_compute_blocks}, "
        f"Bn={db_cache_config.Bn_compute_blocks}, "
        f"W={db_cache_config.max_warmup_steps}, "
    )

    # Enable cache-dit using BlockAdapter for transformer
    cache_dit.enable_cache(
        (
            BlockAdapter(
                transformer=pipeline.transformer,
                blocks=pipeline.transformer.transformer_blocks,
                forward_pattern=ForwardPattern.Pattern_1,
                params_modifiers=[modifier],
            )
        ),
        cache_config=db_cache_config,
    )

    def refresh_cache_context(pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        """Refresh cache context for the transformer with new num_inference_steps.

        Args:
            pipeline: The LongCatImage pipeline instance.
            num_inference_steps: New number of inference steps.
        """
        if cache_config.scm_steps_mask_policy is None:
            cache_dit.refresh_context(pipeline.transformer, num_inference_steps=num_inference_steps, verbose=verbose)
        else:
            cache_dit.refresh_context(
                pipeline.transformer,
                cache_config=DBCacheConfig().reset(
                    num_inference_steps=num_inference_steps,
                    steps_computation_mask=cache_dit.steps_mask(
                        mask_policy=cache_config.scm_steps_mask_policy,
                        total_steps=num_inference_steps,
                    ),
                    steps_computation_policy=cache_config.scm_steps_policy,
                ),
                verbose=verbose,
            )

    return refresh_cache_context


def enable_cache_for_dit(pipeline: Any, cache_config: Any) -> Callable[[int], None]:
    """Enable cache-dit for regular single-transformer DiT models.

    Args:
        pipeline: The diffusion pipeline instance.
        cache_config: DiffusionCacheConfig instance with cache configuration.

    Returns:
        A refresh function that can be called to update cache context with new num_inference_steps.
    """
    # Build DBCacheConfig with optional SCM support
    db_cache_config = _build_db_cache_config(cache_config)

    # Build calibrator config if TaylorSeer is enabled
    calibrator_config = None
    if cache_config.enable_taylorseer:
        taylorseer_order = cache_config.taylorseer_order
        calibrator_config = TaylorSeerCalibratorConfig(taylorseer_order=taylorseer_order)
        logger.info(f"TaylorSeer enabled with order={taylorseer_order}")

    logger.info(
        f"Enabling cache-dit on transformer: "
        f"Fn={db_cache_config.Fn_compute_blocks}, "
        f"Bn={db_cache_config.Bn_compute_blocks}, "
        f"W={db_cache_config.max_warmup_steps}, "
    )

    # Enable cache-dit on the transformer
    cache_dit.enable_cache(
        pipeline.transformer,
        cache_config=db_cache_config,
        calibrator_config=calibrator_config,
    )

    def refresh_cache_context(pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        """Refresh cache context for the transformer with new num_inference_steps.

        Args:
            pipeline: The diffusion pipeline instance.
            num_inference_steps: New number of inference steps.
        """
        if cache_config.scm_steps_mask_policy is None:
            cache_dit.refresh_context(pipeline.transformer, num_inference_steps=num_inference_steps, verbose=verbose)
        else:
            cache_dit.refresh_context(
                pipeline.transformer,
                cache_config=DBCacheConfig().reset(
                    num_inference_steps=num_inference_steps,
                    steps_computation_mask=cache_dit.steps_mask(
                        mask_policy=cache_config.scm_steps_mask_policy,
                        total_steps=num_inference_steps,
                    ),
                    steps_computation_policy=cache_config.scm_steps_policy,
                ),
                verbose=verbose,
            )

    return refresh_cache_context


# Register custom cache-dit enablers after function definitions
CUSTOM_DIT_ENABLERS.update(
    {
        "WanPipeline": enable_cache_for_wan22,
        "FluxPipeline": enable_cache_for_flux,
        "LongCatImagePipeline": enable_cache_for_longcat_image,
        "LongCatImageEditPipeline": enable_cache_for_longcat_image,
        "StableDiffusion3Pipeline": enable_cache_for_sd3,
    }
)


class CacheDiTBackend(CacheBackend):
    """Backend class for cache-dit acceleration on diffusion pipelines.

    This class implements cache-dit acceleration (DBCache, SCM, TaylorSeer) using
    the cache-dit library. It inherits from CacheBackend and provides a unified
    interface for managing cache-dit acceleration on diffusion models.

    Attributes:
        config: Cache configuration (DiffusionCacheConfig instance), inherited from CacheBackend.
        enabled: Whether cache-dit is enabled on this pipeline, inherited from CacheBackend.
        _refresh_func: Internal refresh function for updating cache context.
        _last_num_inference_steps: Last num_inference_steps used for refresh optimization.
    """

    def __init__(self, cache_config: Any = None):
        """Initialize the cache-dit backend.

        Args:
            cache_config: Cache configuration (DiffusionCacheConfig instance, dict, or None).
                         If None or empty, uses default DiffusionCacheConfig().
        """
        # Use default config if cache_config is not provided or is empty
        if cache_config is None:
            config = DiffusionCacheConfig()
        elif isinstance(cache_config, dict):
            # Convert dict to DiffusionCacheConfig, using defaults for missing keys
            config = DiffusionCacheConfig.from_dict(cache_config)
        else:
            config = cache_config

        # Initialize base class with normalized config
        super().__init__(config)

        # Cache-dit specific attributes
        self._refresh_func: Callable[[Any, int, bool], None] | None = None
        self._last_num_inference_steps: int | None = None

    def enable(self, pipeline: Any) -> None:
        """Enable cache-dit on the pipeline if configured.

        This method applies cache-dit acceleration to the appropriate transformer(s)
        in the pipeline. It handles both single-transformer and dual-transformer
        architectures (e.g., Wan2.2).

        Args:
            pipeline: The diffusion pipeline instance.
        """

        # Extract pipeline name from pipeline
        pipeline_name = pipeline.__class__.__name__
        # Check if this model has a custom cache-dit enabler
        if pipeline_name in CUSTOM_DIT_ENABLERS:
            logger.info(f"Using custom cache-dit enabler for model: {pipeline_name}")
            self._refresh_func = CUSTOM_DIT_ENABLERS[pipeline_name](pipeline, self.config)
        else:
            # For regular single-transformer models
            self._refresh_func = enable_cache_for_dit(pipeline, self.config)

        self.enabled = True
        logger.info(f"Cache-dit enabled successfully on {pipeline_name}")

    def refresh(self, pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        """Refresh cache context with new num_inference_steps.

        This method updates the cache context when num_inference_steps changes
        during inference. For dual-transformer models (e.g., Wan2.2), it automatically
        splits the steps based on boundary_ratio.

        Args:
            pipeline: The diffusion pipeline instance.
            num_inference_steps: New number of inference steps.
            verbose: Whether to log refresh operations.
        """
        if not self.enabled or self._refresh_func is None:
            logger.warning("Cache-dit is not enabled. Cannot refresh cache context.")
            return

        # Only refresh if num_inference_steps has changed
        if self._last_num_inference_steps is None or num_inference_steps != self._last_num_inference_steps:
            if verbose:
                logger.info(f"Refreshing cache context for transformer with num_inference_steps: {num_inference_steps}")
            self._refresh_func(pipeline, num_inference_steps, verbose)
            self._last_num_inference_steps = num_inference_steps

    def is_enabled(self) -> bool:
        """Check if cache-dit is enabled on this pipeline.

        Returns:
            True if cache-dit is enabled, False otherwise.
        """
        return self.enabled


def may_enable_cache_dit(pipeline: Any, od_config: OmniDiffusionConfig) -> Optional["CacheDiTBackend"]:
    """Enable cache-dit on the pipeline if configured (convenience function).

    This is a convenience function that creates and enables a CacheDiTBackend.
    For new code, consider using CacheDiTBackend directly.

    Args:
        pipeline: The diffusion pipeline instance.
        od_config: OmniDiffusionConfig with cache configuration.

    Returns:
        A CacheDiTBackend instance if cache-dit is enabled, None otherwise.
    """
    if od_config.cache_backend != "cache-dit" or not od_config.cache_config:
        return None

    backend = CacheDiTBackend(od_config.cache_config)
    backend.enable(pipeline)
    return backend if backend.is_enabled() else None
