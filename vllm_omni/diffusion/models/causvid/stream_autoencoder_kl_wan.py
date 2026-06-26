# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import types

import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan, WanRMS_norm, patchify, unpatchify
from torch import nn

def _bf16_wan_rms_norm_forward(self: WanRMS_norm, x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias

class StreamAutoencoderKLWan(nn.Module):

    def __init__(self, vae: AutoencoderKLWan, latents_scale: list[torch.Tensor]) -> None:
        super().__init__()
        self._vae = vae
        self._mean, self._inv_std = latents_scale
        self._first_encode = True
        self._first_decode = True
        for m in self._vae.modules():
            if isinstance(m, WanRMS_norm):
                m.forward = types.MethodType(_bf16_wan_rms_norm_forward, m)

    def reset(self) -> None:
        """Clear encoder + decoder feat-maps. Call at the start of a session."""
        self._vae.clear_cache()
        self._first_encode = True
        self._first_decode = True

    @property
    def dtype(self) -> torch.dtype:
        return self._vae.dtype

    @torch.no_grad()
    def stream_encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """``pixels`` is ``[C, T, H, W]`` (or batched).

        First call of a session: ``T = 1 + 4N`` -> frame 0 (priming) + the
        remaining ``4N`` frames in a single encoder call -> ``1 + N`` latent
        frames. Later calls: ``T = 4N`` grouped per 4 frames -> ``N`` latent
        frames. Returns ``[z_dim, F, h, w]``.
        """
        x = pixels.unsqueeze(0) if pixels.dim() == 4 else pixels
        t = x.shape[2]
        if self._first_encode:
            self._first_encode = False
            out = torch.cat([self._encode_step(x[:, :, :1]), self._encode_step(x[:, :, 1:])], dim=2)
        else:
            out = torch.cat([self._encode_step(x[:, :, i * 4 : (i + 1) * 4]) for i in range(t // 4)], dim=2)
        mu = self._quant(out)
        return mu.float().squeeze(0)

    @torch.no_grad()
    def stream_decode(self, latents: torch.Tensor) -> torch.Tensor:
        """``latents`` is ``[z_dim, T, h, w]`` (or
        batched) normalized DiT output. De-normalizes (``z / inv_std + mean``),
        then drives the decoder with a persistent feat-map. First call: frame 0 +
        the rest in a single call; later calls: frame-by-frame."""
        z = latents.unsqueeze(0) if latents.dim() == 4 else latents
        t = z.shape[2]
        dtype = self._vae.dtype
        mean = self._mean.to(z.device, dtype)
        inv_std = self._inv_std.to(z.device, dtype)
        x = self._vae.post_quant_conv(z.to(dtype) / inv_std + mean)
        if self._first_decode:
            self._first_decode = False
            out = torch.cat([self._decode_step(x[:, :, :1]), self._decode_step(x[:, :, 1:])], dim=2)
        else:
            out = torch.cat([self._decode_step(x[:, :, i : i + 1]) for i in range(t)], dim=2)
        if self._vae.config.patch_size is not None:
            out = unpatchify(out, patch_size=self._vae.config.patch_size)
        return out.float().clamp_(-1, 1).squeeze(0)

    def _encode_step(self, chunk_pixels: torch.Tensor) -> torch.Tensor:
        self._vae._enc_conv_idx = [0]
        if self._vae.config.patch_size is not None:
            chunk_pixels = patchify(chunk_pixels, patch_size=self._vae.config.patch_size)
        return self._vae.encoder(chunk_pixels, feat_cache=self._vae._enc_feat_map, feat_idx=self._vae._enc_conv_idx)

    def _decode_step(self, x_frames: torch.Tensor) -> torch.Tensor:
        self._vae._conv_idx = [0]
        return self._vae.decoder(x_frames, feat_cache=self._vae._feat_map, feat_idx=self._vae._conv_idx)

    def _quant(self, out: torch.Tensor) -> torch.Tensor:
        out = self._vae.quant_conv(out)
        mu, _ = out.chunk(2, dim=1)
        return mu
