# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from contextlib import nullcontext

import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan, patchify

from vllm_omni.platforms import current_omni_platform


class StreamAutoencoderKLWan:
    """Chunked-encode wrapper over ``AutoencoderKLWan``.

    Args:
        vae: Loaded ``AutoencoderKLWan`` (or subclass). The wrapper holds a
            reference; weights/dtype/device are the caller's responsibility.
        latents_scale: ``[mean, inv_std]`` tensors, each shape
            ``(1, z_dim, 1, 1, 1)``. Used to normalize encoder output and
            de-normalize decoder input.
    """

    def __init__(self, vae: AutoencoderKLWan, latents_scale: list[torch.Tensor]) -> None:
        self._vae = vae
        self._mean, self._inv_std = latents_scale

    def reset(self) -> None:
        """Clear encoder + decoder feat_maps. Call at the start of a session."""
        self._vae.clear_cache()

    @property
    def dtype(self) -> torch.dtype:
        return self._vae.dtype

    @torch.no_grad()
    def init(self, frame: torch.Tensor) -> torch.Tensor:
        """Encode a single pixel frame and prime the encoder feat_map.

        Args:
            frame: ``[C, 1, H, W]`` or ``[B, C, 1, H, W]``.
        Returns:
            ``[z_dim, 1, H, W]`` normalized latent.
        """
        pixels = frame.unsqueeze(0) if frame.dim() == 4 else frame
        with self._autocast(pixels.device):
            out = self._encode_step(pixels)
            mu = self._quant_and_normalize(out)
        return mu.float().squeeze(0)

    @torch.no_grad()
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode ``4*N`` pixel frames using the preserved feat_map.

        Args:
            pixels: ``[C, 4N, H, W]`` or ``[B, C, 4N, H, W]``.
        Returns:
            ``[z_dim, N, H, W]`` normalized latents.
        """
        pixels = pixels.unsqueeze(0) if pixels.dim() == 4 else pixels
        T = pixels.shape[2]
        if T % 4 != 0:
            raise ValueError(f"StreamAutoencoderKLWan.encode expects a multiple of 4 pixel frames, got {T}")
        N = T // 4
        with self._autocast(pixels.device):
            outs = [self._encode_step(pixels[:, :, i * 4 : (i + 1) * 4]) for i in range(N)]
            out = torch.cat(outs, dim=2)
            mu = self._quant_and_normalize(out)
        return mu.float().squeeze(0)

    @torch.no_grad()
    def decode(self, zs: list[torch.Tensor]) -> list[torch.Tensor]:
        """De-normalize each ``[z_dim, T, H, W]`` latent and run VAE decode."""
        dtype = self._vae.dtype
        device = zs[0].device
        mean = self._mean.to(device, dtype)
        inv_std = self._inv_std.to(device, dtype)
        out = []
        for z in zs:
            z_in = z.unsqueeze(0).to(dtype) / inv_std + mean
            sample = self._vae.decode(z_in, return_dict=False)[0]
            out.append(sample.float().clamp_(-1, 1).squeeze(0))
        return out

    def _autocast(self, device: torch.device):
        dtype = self._vae.dtype
        if dtype not in (torch.float16, torch.bfloat16):
            return nullcontext()
        return current_omni_platform.create_autocast_context(
            device_type=device.type,
            dtype=dtype,
            enabled=True,
        )

    def _encode_step(self, chunk_pixels: torch.Tensor) -> torch.Tensor:
        # Reset only the conv-index walker, not the feat_map
        self._vae._enc_conv_idx = [0]
        if self._vae.config.patch_size is not None:
            chunk_pixels = patchify(chunk_pixels, patch_size=self._vae.config.patch_size)
        return self._vae.encoder(
            chunk_pixels,
            feat_cache=self._vae._enc_feat_map,
            feat_idx=self._vae._enc_conv_idx,
        )

    def _quant_and_normalize(self, out: torch.Tensor) -> torch.Tensor:
        out = self._vae.quant_conv(out)
        mu, _ = out.chunk(2, dim=1)
        mean = self._mean.to(mu.device, mu.dtype)
        inv_std = self._inv_std.to(mu.device, mu.dtype)
        return (mu - mean) * inv_std