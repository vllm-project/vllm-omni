# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
TeaCache backend implementation.

This module provides the TeaCache backend that implements the CacheBackend
interface using the hooks-based TeaCache system.
"""

from operator import attrgetter
from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.cache.base import CacheBackend
from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.hook import TeaCacheHook, apply_teacache_hook
from vllm_omni.diffusion.data import DiffusionCacheConfig

logger = init_logger(__name__)


def enable_hunyuan_image3_teacache(pipeline: Any, config: DiffusionCacheConfig) -> None:
    """
    Enable TeaCache for HunyuanImage3 model.

    HunyuanImage3 uses a GPT-based architecture with KV cache, which is incompatible
    with the standard hook-based TeaCache approach. Instead, we store the TeaCacheConfig
    on the pipeline so the denoising loop can implement caching directly.
    """
    teacache_config = TeaCacheConfig(
        transformer_type="HunyuanImage3Pipeline",
        rel_l1_thresh=config.rel_l1_thresh,
        coefficients=config.coefficients,
    )
    pipeline._tea_cache_config = teacache_config

    logger.info(f"TeaCache enabled for HunyuanImage3 with rel_l1_thresh={teacache_config.rel_l1_thresh}")


def enable_bagel_teacache(pipeline: Any, config: DiffusionCacheConfig) -> None:
    """
    Enable TeaCache for Bagel model.
    """
    teacache_config = TeaCacheConfig(
        transformer_type="Bagel",
        rel_l1_thresh=config.rel_l1_thresh,
        coefficients=config.coefficients,
    )
    transformer = pipeline.bagel
    apply_teacache_hook(transformer, teacache_config)
    pipeline.transformer = transformer

    logger.info(
        f"TeaCache applied with rel_l1_thresh={teacache_config.rel_l1_thresh}, "
        f"transformer_class={teacache_config.transformer_type}"
    )


def enable_sensenova_u1_teacache(pipeline: Any, config: DiffusionCacheConfig) -> None:
    """Enable TeaCache for SenseNova-U1 denoising forwards."""
    teacache_config = TeaCacheConfig(
        transformer_type="SenseNovaU1ForCausalLM",
        rel_l1_thresh=config.rel_l1_thresh,
        coefficients=config.coefficients,
    )
    transformer = pipeline.denoising_transformer
    apply_teacache_hook(transformer, teacache_config)

    logger.info(
        f"TeaCache applied with rel_l1_thresh={teacache_config.rel_l1_thresh}, "
        f"transformer_class={teacache_config.transformer_type}"
    )


def enable_flux2_klein_teacache(pipeline: Any, config: DiffusionCacheConfig) -> None:
    """
    Enable TeaCache for Flux2 Klein model.
    """
    teacache_config = TeaCacheConfig(
        transformer_type="Flux2Klein",
        rel_l1_thresh=config.rel_l1_thresh,
        coefficients=config.coefficients,
    )
    transformer = pipeline.transformer

    apply_teacache_hook(transformer, teacache_config)

    logger.info(
        f"TeaCache applied with rel_l1_thresh={teacache_config.rel_l1_thresh}, "
        f"transformer_class={teacache_config.transformer_type}"
    )


CUSTOM_TEACACHE_ENABLERS = {
    "BagelPipeline": enable_bagel_teacache,
    "Flux2KleinPipeline": enable_flux2_klein_teacache,
    "HunyuanImage3Pipeline": enable_hunyuan_image3_teacache,
    "SenseNovaU1Pipeline": enable_sensenova_u1_teacache,
}


class TeaCacheBackend(CacheBackend):
    """
    TeaCache implementation using hooks.

    TeaCache (Timestep Embedding Aware Cache) is an adaptive caching technique
    that speeds up diffusion inference by reusing transformer block computations
    when consecutive timestep embeddings are similar.

    The backend applies TeaCache hooks to the transformer which intercept the
    forward pass and implement the caching logic transparently.

    Example:
        >>> from vllm_omni.diffusion.data import DiffusionCacheConfig
        >>> backend = TeaCacheBackend(DiffusionCacheConfig(rel_l1_thresh=0.2))
        >>> backend.enable(pipeline)
        >>> # Generate with cache enabled
        >>> backend.refresh(pipeline, num_inference_steps=50)  # Refresh before each generation
        >>> # Access config attributes: backend.config.rel_l1_thresh
    """

    def enable(self, pipeline: Any) -> None:
        """
        Enable TeaCache on transformer using hooks.

        This creates a TeaCacheConfig from the backend's DiffusionCacheConfig
        and applies the TeaCache hook to the transformer.

        Args:
            pipeline: Diffusion pipeline instance. Extracts transformer and transformer_type:
                     - transformer: pipeline.transformer
                     - transformer_type: pipeline.transformer.__class__.__name__
        """
        pipeline_type = pipeline.__class__.__name__
        config_factory = getattr(type(pipeline), "_teacache_hook_configs", None)

        if callable(config_factory):
            hook_configs = config_factory(pipeline, self.config)
            if not hook_configs:
                raise ValueError(f"{pipeline_type} declared no TeaCache hook targets")
            for target_path, teacache_config in hook_configs.items():
                if not isinstance(teacache_config, TeaCacheConfig):
                    raise TypeError(
                        f"{pipeline_type} TeaCache config for {target_path!r} "
                        f"must be TeaCacheConfig, got {type(teacache_config).__name__}"
                    )
                transformer = attrgetter(target_path)(pipeline)
                apply_teacache_hook(transformer, teacache_config)
                logger.info(
                    "TeaCache applied with rel_l1_thresh=%s, transformer_class=%s, component=%s",
                    teacache_config.rel_l1_thresh,
                    teacache_config.transformer_type,
                    target_path,
                )
        elif pipeline_type in CUSTOM_TEACACHE_ENABLERS:
            logger.info(f"Using custom TeaCache enabler for model: {pipeline_type}")
            CUSTOM_TEACACHE_ENABLERS[pipeline_type](pipeline, self.config)
        else:
            transformer = pipeline.transformer
            transformer_type = transformer.__class__.__name__

            # Create TeaCacheConfig from DiffusionCacheConfig with transformer_type
            # Access parameters via attribute access: config.rel_l1_thresh
            try:
                teacache_config = TeaCacheConfig(
                    transformer_type=transformer_type,
                    rel_l1_thresh=self.config.rel_l1_thresh,
                    coefficients=self.config.coefficients,
                )
            except Exception as e:
                logger.error(f"Failed to create TeaCacheConfig: {e}")
                raise ValueError(
                    f"Invalid TeaCache configuration: {e}. "
                    f"Expected keys: rel_l1_thresh, coefficients (optional). "
                    f"transformer_type is automatically extracted from pipeline.transformer.__class__.__name__."
                )

            # Apply hook to transformer
            apply_teacache_hook(transformer, teacache_config)

            logger.info(
                f"TeaCache applied with rel_l1_thresh={teacache_config.rel_l1_thresh}, "
                f"transformer_class={teacache_config.transformer_type}"
            )

        # Mark as enabled
        self.enabled = True

    def refresh(self, pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        """
        Refresh TeaCache state for new generation.

        Clears all cached residuals and resets counters/accumulators.
        Should be called before each generation to ensure clean state.

        Args:
            pipeline: Diffusion pipeline instance. Extracts transformer via pipeline.transformer.
            num_inference_steps: Number of inference steps for the current generation.
                                Currently not used by TeaCache but accepted for interface consistency.
            verbose: Whether to log refresh operations (default: True)
        """
        # HunyuanImage3: tea cache state is managed inside the denoising loop,
        # so refresh is a no-op (state is re-initialized every __call__).
        if (
            hasattr(pipeline, "_tea_cache_config")
            and isinstance(pipeline._tea_cache_config, TeaCacheConfig)
            and pipeline.__class__.__name__ == "HunyuanImage3Pipeline"
        ):
            if verbose:
                logger.debug(f"TeaCache state refreshed for HunyuanImage3 (num_inference_steps={num_inference_steps})")
            return

        target_names = vars(pipeline).get("_dit_modules")
        if target_names is None:
            target_names = getattr(type(pipeline), "_dit_modules", None)
        if target_names:
            targets = [(name, attrgetter(name)(pipeline)) for name in target_names]
        else:
            transformer = pipeline.transformer
            target_name = "transformer"
            if not hasattr(transformer, "_hook_registry") and hasattr(pipeline, "denoising_transformer"):
                transformer = pipeline.denoising_transformer
                target_name = "denoising_transformer"
            targets = [(target_name, transformer)]

        for target_name, transformer in targets:
            if not hasattr(transformer, "_hook_registry"):
                if verbose:
                    logger.warning(f"Transformer {target_name} has no hook registry, TeaCache may not be applied")
                continue

            hook = transformer._hook_registry.get_hook(TeaCacheHook._HOOK_NAME)
            if hook is None:
                if verbose:
                    logger.warning(f"TeaCache hook not found on {target_name}, nothing to refresh")
            else:
                calibration_matches = hook.prepare_for_request(num_inference_steps)
                if not calibration_matches and verbose:
                    logger.warning(
                        "TeaCache on %s is calibrated for %d inference steps; "
                        "got %d, so this request will run uncached",
                        target_name,
                        hook.config.calibrated_num_inference_steps,
                        num_inference_steps,
                    )
                transformer._hook_registry.reset_hook(TeaCacheHook._HOOK_NAME)
                if verbose:
                    logger.debug(
                        f"TeaCache state refreshed for {target_name} (num_inference_steps={num_inference_steps})"
                    )
