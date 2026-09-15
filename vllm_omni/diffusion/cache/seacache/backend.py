# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SeaCache backend implementation, mirroring TeaCacheBackend."""

from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.cache.base import CacheBackend
from vllm_omni.diffusion.cache.seacache.config import SeaCacheConfig
from vllm_omni.diffusion.cache.seacache.hook import SeaCacheHook, apply_seacache_hook
from vllm_omni.diffusion.data import DiffusionCacheConfig

logger = init_logger(__name__)


def enable_flux2_klein_seacache(pipeline: Any, config: DiffusionCacheConfig) -> None:
    """Enable SeaCache for Flux2-Klein by its extractor-registry alias (both
    Flux2 variants share the ``Flux2Transformer2DModel`` class name).
    """
    seacache_config = SeaCacheConfig(
        transformer_type="Flux2Klein",
        sea_thresh=config.sea_thresh,
        sea_norm_mode=config.sea_norm_mode,
    )
    try:
        seacache_config.validate()
    except ValueError as e:
        raise ValueError(f"Invalid SeaCache configuration: {e}") from e

    apply_seacache_hook(pipeline.transformer, seacache_config)
    logger.info(
        "SeaCache applied with sea_thresh=%.3f, sea_norm_mode=%s, transformer_class=%s",
        seacache_config.sea_thresh,
        seacache_config.sea_norm_mode,
        seacache_config.transformer_type,
    )


_CUSTOM_SEACACHE_ENABLERS = {
    "Flux2KleinPipeline": enable_flux2_klein_seacache,
}


class SeaCacheBackend(CacheBackend):
    """SeaCache backend using hooks.

    Example:
        >>> backend = SeaCacheBackend(DiffusionCacheConfig(sea_thresh=0.3))
        >>> backend.enable(pipeline)
        >>> backend.refresh(pipeline, num_inference_steps=50)  # before each generation
    """

    def enable(self, pipeline: Any) -> None:
        # Pipeline-level enablers handle ambiguous transformer class names:
        # Flux2-Klein shares Flux2Transformer2DModel with unsupported full Flux2.
        if pipeline.__class__.__name__ in _CUSTOM_SEACACHE_ENABLERS:
            _CUSTOM_SEACACHE_ENABLERS[pipeline.__class__.__name__](pipeline, self.config)
            self.enabled = True
            return

        transformer = pipeline.transformer
        transformer_type = transformer.__class__.__name__

        seacache_config = SeaCacheConfig(
            transformer_type=transformer_type,
            sea_thresh=self.config.sea_thresh,
            sea_norm_mode=self.config.sea_norm_mode,
        )
        try:
            seacache_config.validate()
        except ValueError as e:
            raise ValueError(f"Invalid SeaCache configuration: {e}") from e

        apply_seacache_hook(transformer, seacache_config)
        self.enabled = True

        logger.info(
            "SeaCache applied with sea_thresh=%.3f, sea_norm_mode=%s, transformer_class=%s",
            seacache_config.sea_thresh,
            seacache_config.sea_norm_mode,
            seacache_config.transformer_type,
        )

    def refresh(self, pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        transformer = pipeline.transformer
        if hasattr(transformer, "_hook_registry"):
            hook = transformer._hook_registry.get_hook(SeaCacheHook._HOOK_NAME)
            if hook is not None:
                # num_inference_steps drives last-step forcing; state itself is
                # reset by reset_hook.
                hook.num_inference_steps = num_inference_steps
                transformer._hook_registry.reset_hook(SeaCacheHook._HOOK_NAME)
                if verbose:
                    logger.debug("SeaCache state refreshed (num_inference_steps=%d)", num_inference_steps)
            elif verbose:
                logger.warning("SeaCache hook not found, nothing to refresh")
        elif verbose:
            logger.warning("Transformer has no hook registry, SeaCache may not be applied")
