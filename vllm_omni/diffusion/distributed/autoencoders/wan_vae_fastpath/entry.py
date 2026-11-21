# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The Wan-family fast path as a :class:`vllm_omni.diffusion.vae_fast_path.VaeFastPath`."""

from __future__ import annotations

import importlib

import torch.nn as nn

from vllm_omni.diffusion.vae_fast_path import VaeFastPathContext

from .families import find_family


class WanFamilyVaeFastPath:
    """Every VAE with a registered :class:`~.families.WanVaeFamily`, on CUDA and XPU.

    On Ascend only the decoder's lossless causal-Conv3D spatial-padding optimization is installed.
    """

    name = "wan_family"

    def matches(self, vae: nn.Module) -> bool:
        return find_family(vae) is not None

    def supported(self, ctx: VaeFastPathContext) -> bool:
        platform = ctx.platform
        return platform is not None and bool(platform.is_cuda() or platform.is_xpu() or platform.is_npu())

    def install(self, vae: nn.Module, component: str, level: str, ctx: VaeFastPathContext):
        # Resolved through the package at call time, so the installers stay patchable there.
        package = importlib.import_module(__package__)
        # The platform comes from the context: at load time the VAE can still be on CPU.
        platform = ctx.platform
        is_npu = platform is not None and not (platform.is_cuda() or platform.is_xpu()) and bool(platform.is_npu())
        if component == "decoder":
            if is_npu:
                return package.install_wan_vae_fastpath(vae, level=level, npu_spatial_padding_only=True)
            return package.install_wan_vae_fastpath(vae, level=level)
        if component == "encoder":
            if is_npu:
                return package.WanVaeFastPathReport(
                    level=level, installed=False, reason="the encoder fast path is not supported on Ascend"
                )
            return package.install_wan_vae_encoder_fastpath(vae, level=level)
        raise ValueError(f"unknown VAE component {component!r}")


WAN_FAMILY_FAST_PATH = WanFamilyVaeFastPath()

__all__ = ["WAN_FAMILY_FAST_PATH", "WanFamilyVaeFastPath"]
