# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline-level installation for optional VAE optimizations."""

from __future__ import annotations

from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.offloader.module_collector import ModuleDiscovery
from vllm_omni.diffusion.vae_optimizations.image import maybe_optimize_image_vae
from vllm_omni.diffusion.vae_optimizations.precision import configure_wan_decode_precision
from vllm_omni.diffusion.vae_optimizations.wan import maybe_optimize_wan_vae
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

_FOLDED_CACHE_NAMES = (
    "_fused_weight",
    "_vllm_folded_value_weight",
    "_vllm_folded_value_bias",
)


def optimize_pipeline_vaes(pipeline: nn.Module, od_config: OmniDiffusionConfig) -> None:
    """Configure declared VAE components once after checkpoint loading."""

    precision = getattr(od_config, "vae_decode_precision", None)
    vaes = ModuleDiscovery.discover(pipeline).vaes
    for vae in vaes:
        # An explicit precision override is a serving contract, so failures on
        # a recognized Wan VAE must abort startup instead of being hidden.
        configure_wan_decode_precision(vae, precision)

    if not current_omni_platform.is_cuda():
        return

    for vae in vaes:
        for optimizer in (maybe_optimize_image_vae, maybe_optimize_wan_vae):
            try:
                optimizer(vae)
            except Exception:
                # A wrapper may already have installed its exact fallback
                # before a later optional rewrite failed. With no registered
                # enabled gate, those wrappers remain on the reference path.
                logger.warning(
                    "Failed to install optional VAE fast paths on %s; using the reference path",
                    type(vae).__name__,
                    exc_info=True,
                )


def clear_pipeline_vae_fast_path_caches(pipeline: nn.Module) -> None:
    """Release lazily folded buffers before the worker enters sleep mode."""

    cleared = 0
    for vae in ModuleDiscovery.discover(pipeline).vaes:
        if not getattr(vae, "_vllm_vae_fast_path_installed", False):
            continue
        for module in vae.modules():
            for name in _FOLDED_CACHE_NAMES:
                if name in module._buffers and module._buffers[name] is not None:
                    setattr(module, name, None)
                    cleared += 1
    if cleared:
        logger.debug("Released %d lazy VAE fast-path buffers before sleep", cleared)


__all__ = ["clear_pipeline_vae_fast_path_caches", "optimize_pipeline_vaes"]
