# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 The FastWAM Authors
# SPDX-FileCopyrightText: Copyright 2024-2025 The Wan Team and Alibaba Group
# SPDX-FileCopyrightText: Copyright (c) ModelScope DiffSynth-Studio contributors
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Single-frame Wan2.2 VAE encoder, as FastWAM uses it to embed the current camera frame.

FastWAM encodes exactly one frame per call (no temporal context), so the causal
feature cache of the full Wan VAE is never populated and is left out here. The
module names follow the Diffusers ``AutoencoderKLWan`` layout so the encoder loads
unchanged from ``Wan-AI/Wan2.2-TI2V-5B-Diffusers/vae``; the arithmetic follows the
reference (DiffSynth) implementation op for op. On CPU in bfloat16 this module
reproduces the reference first-frame latents bitwise, while the Diffusers
``AutoencoderKLWan.encode`` differs from them by up to 2 bf16 ULPs, which is why
the encoder is vendored rather than reused.
"""

# Wan2.2 provenance: the Wan2.2 modules in FastWAM's wan22/ package come from
# Wan-Video/Wan2.2 (https://github.com/Wan-Video/Wan2.2, Apache-2.0, Copyright 2024-2025
# The Wan Team / Alibaba) via DiffSynth-Studio (https://github.com/modelscope/DiffSynth-Studio,
# Apache-2.0). Those portions remain under the Apache License 2.0.
#
# Adapted from FastWAM (https://github.com/yuantianyuan01/FastWAM @7faa711,
# src/fastwam/models/wan22/wan_video_vae.py). The original portions carry the following MIT license:
# Copyright (c) 2026 The FastWAM Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

# Wan2.2 VAE latent statistics (identical to ``latents_mean``/``latents_std`` in the
# Diffusers vae/config.json).
WAN22_LATENTS_MEAN = (
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
)  # fmt: skip
WAN22_LATENTS_STD = (
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
)  # fmt: skip


class CausalConv3d(nn.Conv3d):
    """Conv3d with causal temporal padding (all of it in front)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pad_t, pad_h, pad_w = (int(p) for p in self.padding)
        self._causal_padding = (pad_w, pad_w, pad_h, pad_h, 2 * pad_t, 0)
        self.padding = (0, 0, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return super().forward(F.pad(x, self._causal_padding))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, images: bool = True) -> None:
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones((dim, 1, 1) if images else (dim, 1, 1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ``+ 0.0`` mirrors the reference's bias-less ``+ self.bias`` (a python 0.).
        return F.normalize(x, dim=1) * self.scale * self.gamma + 0.0


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm1 = RMSNorm(in_dim, images=False)
        self.conv1 = CausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = RMSNorm(out_dim, images=False)
        self.conv2 = CausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_shortcut(x) if self.conv_shortcut is not None else x
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return x + h


class AttentionBlock(nn.Module):
    """Single-head spatial self-attention (per frame)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.norm(x)
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)
        x = self.proj(x)
        x = rearrange(x, "(b t) c h w-> b c t h w", t=t)
        return x + identity


class AvgDown3D(nn.Module):
    """Parameter-free space/time-to-channel shortcut of the Wan2.2 down blocks."""

    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        if in_channels * self.factor % out_channels:
            raise ValueError("AvgDown3D needs in_channels * factor divisible by out_channels.")
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
        b, c, t, h, w = x.shape
        fs, ft = self.factor_s, self.factor_t
        x = x.view(b, c, t // ft, ft, h // fs, fs, w // fs, fs)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(b, c * self.factor, t // ft, h // fs, w // fs)
        x = x.view(b, self.out_channels, self.group_size, t // ft, h // fs, w // fs)
        return x.mean(dim=2)


class Downsample(nn.Module):
    """Spatial 2x downsample. The temporal ``time_conv`` of a ``downsample3d`` block only
    acts from the second frame on, so it is loaded but never run here."""

    def __init__(self, dim: int, temporal: bool) -> None:
        super().__init__()
        self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)) if temporal else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        return rearrange(x, "(b t) c h w -> b c t h w", t=t)


class DownBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_res_blocks: int, temporal: bool, down: bool) -> None:
        super().__init__()
        self.avg_shortcut = AvgDown3D(in_dim, out_dim, factor_t=2 if temporal else 1, factor_s=2 if down else 1)
        resnets = []
        for _ in range(num_res_blocks):
            resnets.append(ResidualBlock(in_dim, out_dim))
            in_dim = out_dim
        self.resnets = nn.ModuleList(resnets)
        self.downsampler = Downsample(out_dim, temporal) if down else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_copy = x.clone()
        for resnet in self.resnets:
            x = resnet(x)
        if self.downsampler is not None:
            x = self.downsampler(x)
        return x + self.avg_shortcut(x_copy)


class MidBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([ResidualBlock(dim, dim), ResidualBlock(dim, dim)])
        self.attentions = nn.ModuleList([AttentionBlock(dim)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        return self.resnets[1](x)


class Encoder(nn.Module):
    def __init__(
        self,
        base_dim: int,
        z_dim: int,
        dim_mult: Sequence[int],
        num_res_blocks: int,
        temporal_downsample: Sequence[bool],
        in_channels: int,
    ) -> None:
        super().__init__()
        dims = [base_dim * u for u in [1, *dim_mult]]
        self.conv_in = CausalConv3d(in_channels, dims[0], 3, padding=1)
        self.down_blocks = nn.ModuleList(
            [
                DownBlock(
                    in_dim,
                    out_dim,
                    num_res_blocks,
                    temporal=temporal_downsample[i] if i < len(temporal_downsample) else False,
                    down=i != len(dim_mult) - 1,
                )
                for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:]))
            ]
        )
        self.mid_block = MidBlock(dims[-1])
        self.norm_out = RMSNorm(dims[-1], images=False)
        self.conv_out = CausalConv3d(dims[-1], z_dim * 2, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        return self.conv_out(F.silu(self.norm_out(x)))


class WanVAEFirstFrameEncoder(nn.Module):
    """Encode one RGB frame [B, 3, 1, H, W] in [-1, 1] into normalised Wan2.2 latents
    [B, z_dim, 1, H/16, W/16] (posterior mean, ``(mu - mean) * (1 / std)``)."""

    mean: torch.Tensor
    inv_std: torch.Tensor

    def __init__(
        self,
        base_dim: int = 160,
        z_dim: int = 48,
        dim_mult: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        temporal_downsample: Sequence[bool] = (False, True, True),
        patch_size: int = 2,
        latents_mean: Sequence[float] = WAN22_LATENTS_MEAN,
        latents_std: Sequence[float] = WAN22_LATENTS_STD,
    ) -> None:
        super().__init__()
        if len(latents_mean) != z_dim or len(latents_std) != z_dim:
            raise ValueError(f"latents_mean/std must have {z_dim} entries.")
        self.z_dim = z_dim
        self.patch_size = patch_size
        self.encoder = Encoder(
            base_dim, z_dim, dim_mult, num_res_blocks, temporal_downsample, in_channels=3 * patch_size * patch_size
        )
        self.quant_conv = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        # Built in float32 whatever the default dtype is, then cast with the module as
        # in the reference: 1/std is rounded to the run dtype after the reciprocal.
        self.register_buffer("mean", torch.tensor(latents_mean, dtype=torch.float32), persistent=False)
        self.register_buffer("inv_std", torch.tensor(latents_std, dtype=torch.float32).reciprocal(), persistent=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() != 5 or image.shape[2] != 1:
            raise ValueError(f"Expected one frame [B, 3, 1, H, W], got {tuple(image.shape)}.")
        p = self.patch_size
        x = rearrange(image, "b c f (h q) (w r) -> b (c r q) f h w", q=p, r=p)
        mu, _log_var = self.quant_conv(self.encoder(x)).chunk(2, dim=1)
        mean = self.mean.to(dtype=mu.dtype, device=mu.device).view(1, self.z_dim, 1, 1, 1)
        inv_std = self.inv_std.to(dtype=mu.dtype, device=mu.device).view(1, self.z_dim, 1, 1, 1)
        return (mu - mean) * inv_std

    @classmethod
    def from_diffusers_config(cls, cfg: Mapping[str, Any]) -> WanVAEFirstFrameEncoder:
        """Build from a Diffusers ``AutoencoderKLWan`` ``config.json`` (Wan2.2 layout)."""
        patch_size = int(cfg.get("patch_size") or 1)
        if not cfg.get("is_residual") or patch_size != 2 or list(cfg.get("attn_scales") or []):
            raise ValueError("FastWAM needs the Wan2.2 (residual, patch_size=2) VAE, e.g. Wan2.2-TI2V-5B-Diffusers.")
        if int(cfg.get("in_channels", 12)) != 3 * patch_size * patch_size:
            raise ValueError(f"Unexpected VAE in_channels={cfg.get('in_channels')} for patch_size={patch_size}.")
        return cls(
            base_dim=int(cfg["base_dim"]),
            z_dim=int(cfg["z_dim"]),
            dim_mult=tuple(int(m) for m in cfg["dim_mult"]),
            num_res_blocks=int(cfg["num_res_blocks"]),
            temporal_downsample=tuple(bool(t) for t in cfg["temperal_downsample"]),
            patch_size=patch_size,
            latents_mean=tuple(float(m) for m in cfg["latents_mean"]),
            latents_std=tuple(float(s) for s in cfg["latents_std"]),
        )

    @staticmethod
    def is_encoder_key(name: str) -> bool:
        """Keys of a Diffusers ``AutoencoderKLWan`` state dict this module consumes."""
        return name.startswith(("encoder.", "quant_conv."))
