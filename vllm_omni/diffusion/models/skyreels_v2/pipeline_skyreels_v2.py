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
from vllm_omni.diffusion.request import OmniDiffusionRequest, resolve_video_num_frames

# SkyReels Diffusers docs: flow_shift=8.0 for T2V.
SKYREELS_V2_DEFAULT_FLOW_SHIFT = 8.0
SKYREELS_V2_DEFAULT_GUIDANCE_SCALE = 6.0
SKYREELS_V2_DEFAULT_NUM_INFERENCE_STEPS = 50
# Per-checkpoint geometry from the SkyReels V2 model cards.
SKYREELS_V2_540P_DEFAULTS = {"height": 544, "width": 960, "num_frames": 97}
SKYREELS_V2_720P_DEFAULTS = {"height": 720, "width": 1280, "num_frames": 121}


def get_skyreels_v2_resolution_defaults(model: str | None) -> dict[str, int]:
    """540P and 720P checkpoints share one architecture; pick geometry by name."""
    return SKYREELS_V2_720P_DEFAULTS if "720p" in str(model or "").lower() else SKYREELS_V2_540P_DEFAULTS


def get_skyreels_v2_post_process_func(od_config: OmniDiffusionConfig):
    return get_wan22_post_process_func(od_config)


def get_skyreels_v2_pre_process_func(od_config: OmniDiffusionConfig):
    """Fill omitted sampling fields with SkyReels defaults instead of Wan's.

    Wan resolves omitted fields to 480x832, 81 frames, 40 steps and CFG 4.0.
    Applying SkyReels values here covers every entrypoint (offline, online,
    disaggregated stages) before the inherited Wan pipeline reads them.
    """
    wan_pre_process_func = get_wan22_pre_process_func(od_config)
    resolution_defaults = get_skyreels_v2_resolution_defaults(getattr(od_config, "model", None))

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        # Startup warmup sets its own lightweight shape; keep it as Wan does.
        if request.is_dummy_run():
            return wan_pre_process_func(request)
        sampling_params = request.sampling_params
        for field in ("height", "width"):
            if getattr(sampling_params, field) is None:
                setattr(sampling_params, field, resolution_defaults[field])
        sampling_params.num_frames = resolve_video_num_frames(
            sampling_params.num_frames,
            default_num_frames=resolution_defaults["num_frames"],
            is_dummy_run=False,
        )
        if sampling_params.num_inference_steps is None:
            sampling_params.num_inference_steps = SKYREELS_V2_DEFAULT_NUM_INFERENCE_STEPS
        if not sampling_params.guidance_scale_provided:
            sampling_params.guidance_scale = SKYREELS_V2_DEFAULT_GUIDANCE_SCALE
            sampling_params.guidance_scale_provided = True
        return wan_pre_process_func(request)

    return pre_process_func


class SkyReelsV2Pipeline(Wan22Pipeline):
    """Text-to-video pipeline for SkyReels V2 Diffusers checkpoints.

    Reuses Wan2.2 Omni plumbing because Diffusers' SkyReels V2 DiT is Wan-based.
    Forces single-transformer loading and SkyReels T2V scheduler defaults.
    """

    supports_request_batch = True
    default_num_inference_steps: int = SKYREELS_V2_DEFAULT_NUM_INFERENCE_STEPS
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
    "SKYREELS_V2_DEFAULT_GUIDANCE_SCALE",
    "SKYREELS_V2_DEFAULT_NUM_INFERENCE_STEPS",
    "get_skyreels_v2_resolution_defaults",
]
