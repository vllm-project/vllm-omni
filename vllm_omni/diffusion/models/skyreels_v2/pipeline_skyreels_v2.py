# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SkyReels V2 text-to-video pipeline for vLLM-Omni.

Adapted from Diffusers `SkyReelsV2Pipeline` using the same Omni integration
pattern as Wan2.2 T2V: Diffusers checkpoint layout + Omni Wan DiT/VAE/UMT5 stack.

SkyReels V2 T2V checkpoints are single-transformer (no MoE `transformer_2`).
Recommended Diffusers models:
  - Skywork/SkyReels-V2-T2V-14B-540P-Diffusers
  - Skywork/SkyReels-V2-T2V-14B-720P-Diffusers
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar

import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import (
    Wan22Pipeline,
    get_wan22_post_process_func,
    get_wan22_pre_process_func,
    load_wan_weights_with_optional_gate,
)

# SkyReels Diffusers docs: flow_shift=8.0 for T2V.
# Recommended sampling guidance_scale is 6.0 (set via OmniDiffusionSamplingParams).
SKYREELS_V2_DEFAULT_FLOW_SHIFT = 8.0


def get_skyreels_v2_post_process_func(od_config: OmniDiffusionConfig):
    return get_wan22_post_process_func(od_config)


def get_skyreels_v2_pre_process_func(od_config: OmniDiffusionConfig):
    return get_wan22_pre_process_func(od_config)


class SkyReelsV2Pipeline(Wan22Pipeline):
    """Text-to-video pipeline for SkyReels V2 Diffusers checkpoints.

    Reuses Wan2.2 Omni plumbing because Diffusers' SkyReels V2 DiT is Wan-based.
    Forces single-transformer loading and SkyReels T2V scheduler defaults.
    """

    supports_request_batch = True
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        # Apply SkyReels T2V defaults before Wan init builds the scheduler.
        if od_config.flow_shift is None:
            od_config.flow_shift = SKYREELS_V2_DEFAULT_FLOW_SHIFT
        # Single-DiT T2V: do not enable MoE boundary routing.
        if od_config.boundary_ratio is None:
            od_config.boundary_ratio = 0.0

        super().__init__(od_config=od_config, prefix=prefix)

        # SkyReels T2V Diffusers packs are single-transformer. Drop any MoE stage
        # that Wan auto-detection might have enabled for dual-DiT Wan2.2 packs.
        self.has_transformer_2 = False
        if self.transformer_2 is not None:
            self.transformer_2 = None
        self.weights_sources = [
            source for source in self.weights_sources if not source.prefix.startswith("transformer_2.")
        ]
        if self.transformer is None:
            raise RuntimeError("SkyReelsV2Pipeline requires a `transformer` subfolder in the checkpoint.")
        self.transformer_config = self.transformer.config

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return load_wan_weights_with_optional_gate(self, weights)


__all__ = [
    "SkyReelsV2Pipeline",
    "get_skyreels_v2_post_process_func",
    "get_skyreels_v2_pre_process_func",
    "SKYREELS_V2_DEFAULT_FLOW_SHIFT",
]
