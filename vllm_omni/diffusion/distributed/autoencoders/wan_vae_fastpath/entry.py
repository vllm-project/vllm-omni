# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The Wan-family fast path as a :class:`vllm_omni.diffusion.vae_fast_path.VaeFastPath`."""

from __future__ import annotations

import importlib

import torch.nn as nn

from vllm_omni.diffusion.vae_fast_path import VaeFastPathContext

from .families import find_family
from .install import SPATIAL_LEVEL_ATTR


class WanFamilyVaeFastPath:
    """Every VAE with a registered :class:`~.families.WanVaeFamily`, on CUDA and XPU."""

    name = "wan_family"

    def matches(self, vae: nn.Module) -> bool:
        return find_family(vae) is not None

    def supported(self, ctx: VaeFastPathContext) -> bool:
        platform = ctx.platform
        return platform is not None and bool(platform.is_cuda() or platform.is_xpu())

    def install(self, vae: nn.Module, component: str, level: str, ctx: VaeFastPathContext):
        # Resolved through the package at call time, so the installers stay patchable there.
        package = importlib.import_module(__package__)
        if component == "decoder":
            # LingBot-World installs its own spatial decoder before registry discovery.
            if (
                getattr(vae, "_vllm_omni_wan_spatial_shard_installed", False)
                and getattr(vae, SPATIAL_LEVEL_ATTR, None) is None
                and not package.is_installed(vae)
            ):
                return package.WanVaeFastPathReport(
                    level=level, installed=False, reason="pipeline owns spatial decoder"
                )
            return package.install_wan_vae_fastpath(vae, level=level)
        if component == "encoder":
            return package.install_wan_vae_encoder_fastpath(vae, level=level)
        raise ValueError(f"unknown VAE component {component!r}")


WAN_FAMILY_FAST_PATH = WanFamilyVaeFastPath()

__all__ = ["WAN_FAMILY_FAST_PATH", "WanFamilyVaeFastPath"]
