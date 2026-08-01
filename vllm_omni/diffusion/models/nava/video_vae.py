# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os

import torch
from torch import nn

from vllm_omni.diffusion.models.nava.config import NAVAConfig
from vllm_omni.diffusion.models.wan2_2.wan_vae import Wan22VAE


class NAVAVideoVAE(nn.Module):
    def __init__(self, model_root: str, config: NAVAConfig, device: torch.device) -> None:
        super().__init__()
        self.config = config
        self.latent_channels = config.video_latent_ch
        vae_path = os.path.join(model_root, config.wan_dir, "Wan2.2_VAE.pth")
        if not os.path.exists(vae_path):
            raise FileNotFoundError(f"NAVA video VAE checkpoint not found: {vae_path}")
        # Upstream NAVA moves the Wan2.2 VAE model to BF16, while its
        # LocalVideoVAEAdapter returns float32 tensors from wrapped encode/decode.
        self.vae = Wan22VAE(
            vae_path=vae_path,
            dtype=torch.bfloat16,
            device=device,
            lazy=True,
            scale_dtype=torch.float32,
            autocast_dtype=torch.float32,
        )
        self.vae.requires_grad_(False).eval()

    def encode_first_frame(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"Expected first-frame tensor [B,C,H,W], got {tuple(image.shape)}.")
        # Upstream NAVA's LocalVideoVAEAdapter uses Wan2_2_VAE.wrapped_encode,
        # which returns the encoder output directly instead of sampling from a
        # posterior. Keep I2AV first-frame conditioning deterministic here.
        latent = self.vae.encode_video(image.unsqueeze(2), use_sample=False).float().squeeze(2)
        return latent.flatten(2).transpose(1, 2).contiguous()

    def decode(self, video_latents: torch.Tensor, *, height: int, width: int, frames: int) -> torch.Tensor:
        batch = video_latents.shape[0]
        latent_h, latent_w = self.config.video_latent_hw(height, width)
        latent_frames = self.config.video_latent_frames(frames)
        expected_tokens = latent_frames * latent_h * latent_w
        if video_latents.shape[1] != expected_tokens:
            raise ValueError(
                f"NAVA video latent token count mismatch: expected {expected_tokens}, got {video_latents.shape[1]}."
            )
        latent = video_latents.reshape(batch, latent_frames, latent_h, latent_w, self.latent_channels)
        latent = latent.permute(0, 4, 1, 2, 3).contiguous()
        video = self.vae.decode_video(latent).float()
        if video.ndim == 5 and video.shape[2] >= frames:
            return video[:, :, :frames].contiguous()
        return video
