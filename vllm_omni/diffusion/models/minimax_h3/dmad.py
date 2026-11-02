# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""DMAD H3 release contract and its request-owned CPU noise stream.

Reference: https://github.com/Yzmblog/DMAD, dmad_h3/sampling.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

from .packed_tokens import minimax_h3_patchify_video_latent

if TYPE_CHECKING:
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams


@dataclass(frozen=True)
class DmadSpec:
    denoise_steps: int

    name: ClassVar[str] = "DMAD H3"

    @property
    def base_schedule(self) -> tuple[float, ...]:
        return tuple(torch.linspace(1.0, 0.0, self.denoise_steps + 1, dtype=torch.float32).tolist())

    def check_request(self, sampling: OmniDiffusionSamplingParams, task: str, *, step_execution: bool = False) -> None:
        if task != "t2va":
            raise ValueError("DMAD H3 released students support t2va only")
        if float(sampling.lora_scale) != 1.0:
            raise ValueError("DMAD H3 requires lora_scale=1 (or 0 to disable the adapter)")
        steps = sampling.num_inference_steps
        if steps is not None and steps != self.denoise_steps:
            raise ValueError(f"This DMAD H3 adapter requires num_inference_steps={self.denoise_steps}")
        if step_execution and steps is None:
            raise ValueError(f"DMAD H3 step execution requires explicit num_inference_steps={self.denoise_steps}")
        extra = sampling.extra_args or {}
        for key, expected in (("flow_shift", 12.0), ("audio_flow_shift", 2.0)):
            if float(extra.get(key, expected)) != expected:
                raise ValueError(f"DMAD H3 requires {key}={expected:g}")
        if extra.get("sampler", "euler") != "euler":
            raise ValueError("DMAD H3 selects its own re-noising sampler; remove the sampler override")
        if extra.get("audio_mode", "native") != "native":
            raise ValueError("DMAD H3 requires audio_mode=native")
        if (
            extra.get("long_video_mode", "full") != "full"
            or extra.get("continuation_prompts") is not None
            or extra.get("continue_from") is not None
        ):
            raise ValueError("DMAD H3 does not support continuation")


def dmad_sigmas(steps: int, shift: float) -> list[float]:
    """The released grid computes the shift in FP32 before device transfer."""
    base = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32)
    return (shift * base / (1.0 + (shift - 1.0) * base)).tolist()


class DmadNoise:
    """One CPU generator for initial video, audio, and every re-noising draw.

    Initial video is drawn in channel-first VAE layout; subsequent video
    noise is drawn directly in packed row order. Audio always uses packed
    channel-major rows. Preserve the final zero-sigma draws as well.
    """

    def __init__(
        self, *, seed: int, device: torch.device, latent_t: int, latent_h: int, latent_w: int, audio_t: int
    ) -> None:
        self.device = device
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.video_shape = (1, 24, latent_t, latent_h, latent_w)
        self.audio_shape = (audio_t * 2, 32)

    def initial_rows(self) -> tuple[torch.Tensor, torch.Tensor]:
        video = torch.randn(self.video_shape, generator=self.generator, dtype=torch.float32)
        audio = torch.randn(self.audio_shape, generator=self.generator, dtype=torch.float32)
        return minimax_h3_patchify_video_latent(video.to(self.device), patch_size=(1, 2, 2)), audio.to(self.device)

    def step(self, denoised: torch.Tensor, sigma_next: float, *, modality: str) -> torch.Tensor:
        if modality not in {"video", "audio"}:
            raise ValueError(f"Unknown DMAD modality: {modality}")
        if not math.isfinite(sigma_next) or not 0 <= sigma_next <= 1:
            raise ValueError("DMAD sigma must be finite and in [0, 1]")
        noise = torch.randn(denoised.shape, generator=self.generator, dtype=torch.float32).to(self.device)
        sigma = denoised.new_tensor(sigma_next, dtype=torch.float32)
        return (1.0 - sigma) * denoised.float() + sigma * noise
