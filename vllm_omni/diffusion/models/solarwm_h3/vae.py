# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Conditioning arithmetic adapted from SolarWM, Copyright (c) SolarWM contributors.
# Source: Junchao-cs/SolarWM@ce1da4e7705391eda8eeda6016c0fd3f614b975e
"""Solar's published Diffusers H3 visual VAE and preencode rounding contract."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from vllm_omni.diffusion.models.minimax_h3.packed_tokens import minimax_h3_patchify_video_latent

if TYPE_CHECKING:
    from diffusers import AutoencoderKLMiniMaxH3


class SolarH3VideoVAE(nn.Module):
    """Keep the published VAE's spatial tiles and Solar's pixel/latent math."""

    def __init__(self, model: AutoencoderKLMiniMaxH3) -> None:
        super().__init__()
        self.model = model
        self.config_dict = dict(model.config)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _decode_tiling_context(self, _latent: torch.Tensor) -> AbstractContextManager:
        # This is the upstream Solar VAE precision, independent of DiT BF16.
        return torch.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda")

    def _denormalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        shape = (1, int(self.config_dict["latent_channels"]), 1, 1, 1)
        mean = torch.tensor(self.config_dict["latents_mean"], device=latent.device, dtype=torch.float32).view(shape)
        std = torch.tensor(self.config_dict["latents_std"], device=latent.device, dtype=torch.float32).view(shape)
        return latent.float() * std + mean

    @staticmethod
    def _normalize_decoded_frames(decoded: torch.Tensor) -> torch.Tensor:
        if decoded.ndim != 5 or decoded.shape[1] != 3:
            raise ValueError("Solar H3 VAE expects decoded [B,3,T,H,W] pixels")
        shape = (1, 3, 1, 1, 1)
        mean = decoded.new_tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(shape)
        std = decoded.new_tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(shape)
        return (decoded.float() * std + mean).clamp(0, 1)

    @torch.inference_mode()
    def encode_image(self, image: Image.Image, *, height: int, width: int) -> torch.Tensor:
        """Encode the anchor into BF16 patch rows, matching Solar preencode.

        Rounding the posterior sample to FP16 before CPU FP32 mean/std
        normalization is intentional; normalizing on GPU before rounding
        changes the conditioning payload the published student was trained on.
        """
        if not isinstance(image, Image.Image):
            raise TypeError("Solar H3 anchor must be a PIL image")
        if image.size != (width, height):
            raise ValueError("Solar H3 anchor must already match the requested output size")
        pixels = torch.from_numpy(np.asarray(image.convert("RGB")).copy()).permute(2, 0, 1)
        pixels = pixels.to(device=self.device, dtype=torch.float32).view(1, 3, 1, height, width)
        mean = pixels.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
        std = pixels.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
        normalized = (pixels.div(255.0) - mean) / std
        posterior = self.model.encode(normalized, return_dict=False)[0]
        # The upstream generator is CPU even when the posterior is on CUDA;
        # Diffusers' randn_tensor preserves this CPU RNG sampling contract.
        latent = posterior.sample(generator=torch.Generator().manual_seed(42))
        latent = latent.to(torch.float16).float().cpu()
        channels = int(self.config_dict["latent_channels"])
        latent_mean = torch.tensor(self.config_dict["latents_mean"], device="cpu", dtype=torch.float32).view(
            1, channels, 1, 1, 1
        )
        latent_std = torch.tensor(self.config_dict["latents_std"], device="cpu", dtype=torch.float32).view(
            1, channels, 1, 1, 1
        )
        latent = ((latent - latent_mean) / latent_std).to(torch.bfloat16)
        if latent.shape[2] != 1 or latent.shape[-2] % 2 or latent.shape[-1] % 2:
            raise ValueError("Solar H3 anchor VAE returned incompatible latent geometry")
        return minimax_h3_patchify_video_latent(latent, patch_size=(1, 2, 2))

    @torch.inference_mode()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        with self._decode_tiling_context(latent):
            decoded = self.model.decode(self._denormalize_latent(latent), return_dict=False)[0]
        return self._normalize_decoded_frames(decoded)
