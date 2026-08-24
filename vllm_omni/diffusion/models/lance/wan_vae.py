# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Ported from ByteDance Lance upstream (https://github.com/bytedance/Lance,
# modeling/vae/wan/{vae2_2.py,model.py}). Upstream copyright:
#
#   Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
#   Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
#   Licensed under the Apache License, Version 2.0.
#
# Lance ships its VAE as a raw ``Wan2.2_VAE.pth`` whose state dict is keyed for
# the upstream ``WanVAE_`` nn.Module here (not for diffusers' ``AutoencoderKLWan``).
# We port that module verbatim so we can load the .pth directly, and wrap it with
# :class:`LanceWanVAE` which exposes BAGEL's 4-D ``encode(BCHW)/decode(BCHW)``
# surface (and a 5-D path for the video checkpoint).
"""Wan2.2 VAE used by Lance, ported from upstream so ``Wan2.2_VAE.pth`` loads
natively without state-dict surgery."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from vllm.logger import init_logger

logger = init_logger(__name__)

CACHE_T = 2


class CausalConv3d(nn.Conv3d):
    """Causal 3D conv with feature-map caching across temporal chunks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)


class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


class Upsample(nn.Upsample):
    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):
    def __init__(self, dim, mode):
        assert mode in ("none", "upsample2d", "upsample3d", "downsample2d", "downsample3d")
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                        cache_x = torch.cat(
                            [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                            dim=2,
                        )
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == "Rep":
                        cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """Single-head causal self-attention over spatial tokens, per frame."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
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


def _patchify(x, patch_size):
    if patch_size == 1:
        return x
    if x.dim() == 4:
        return rearrange(x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size, r=patch_size)
    if x.dim() == 5:
        return rearrange(x, "b c f (h q) (w r) -> b (c r q) f h w", q=patch_size, r=patch_size)
    raise ValueError(f"Invalid input shape: {x.shape}")


def _unpatchify(x, patch_size):
    if patch_size == 1:
        return x
    if x.dim() == 4:
        return rearrange(x, "b (c r q) h w -> b c (h q) (w r)", q=patch_size, r=patch_size)
    if x.dim() == 5:
        return rearrange(x, "b (c r q) f h w -> b c f (h q) (w r)", q=patch_size, r=patch_size)
    return x


class AvgDown3D(nn.Module):
    def __init__(self, in_channels, out_channels, factor_t, factor_s=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s
        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
        B, C, T, H, W = x.shape
        x = x.view(
            B,
            C,
            T // self.factor_t,
            self.factor_t,
            H // self.factor_s,
            self.factor_s,
            W // self.factor_s,
            self.factor_s,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(B, C * self.factor, T // self.factor_t, H // self.factor_s, W // self.factor_s)
        x = x.view(B, self.out_channels, self.group_size, T // self.factor_t, H // self.factor_s, W // self.factor_s)
        return x.mean(dim=2)


class DupUp3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor_t, factor_s=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s
        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x: torch.Tensor, first_chunk=False) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.view(
            x.size(0),
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
            x.size(2),
            x.size(3),
            x.size(4),
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(
            x.size(0),
            self.out_channels,
            x.size(2) * self.factor_t,
            x.size(4) * self.factor_s,
            x.size(6) * self.factor_s,
        )
        if first_chunk:
            x = x[:, :, self.factor_t - 1 :, :, :]
        return x


class Down_ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, mult, temperal_downsample=False, down_flag=False):
        super().__init__()
        self.avg_shortcut = AvgDown3D(
            in_dim, out_dim, factor_t=2 if temperal_downsample else 1, factor_s=2 if down_flag else 1
        )
        downsamples = []
        for _ in range(mult):
            downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            downsamples.append(Resample(out_dim, mode=mode))
        self.downsamples = nn.Sequential(*downsamples)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        x_copy = x.clone()
        for module in self.downsamples:
            x = module(x, feat_cache, feat_idx)
        return x + self.avg_shortcut(x_copy)


class Up_ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, mult, temporal_upsample=False, up_flag=False):
        super().__init__()
        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim, out_dim, factor_t=2 if temporal_upsample else 1, factor_s=2 if up_flag else 1
            )
        else:
            self.avg_shortcut = None
        upsamples = []
        for _ in range(mult):
            upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        if up_flag:
            mode = "upsample3d" if temporal_upsample else "upsample2d"
            upsamples.append(Resample(out_dim, mode=mode))
        self.upsamples = nn.Sequential(*upsamples)

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        x_main = x.clone()
        for module in self.upsamples:
            x_main = module(x_main, feat_cache, feat_idx)
        if self.avg_shortcut is not None:
            return x_main + self.avg_shortcut(x, first_chunk)
        return x_main


class Encoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        attn_scales=(),
        temperal_downsample=(True, True, False),
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = list(dim_mult)
        self.num_res_blocks = num_res_blocks
        self.attn_scales = list(attn_scales)
        self.temperal_downsample = list(temperal_downsample)

        dims = [dim * u for u in [1] + self.dim_mult]
        self.conv1 = CausalConv3d(12, dims[0], 3, padding=1)

        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_down_flag = self.temperal_downsample[i] if i < len(self.temperal_downsample) else False
            downsamples.append(
                Down_ResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks,
                    temperal_downsample=t_down_flag,
                    down_flag=i != len(self.dim_mult) - 1,
                )
            )
        self.downsamples = nn.Sequential(*downsamples)

        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout),
        )
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1),
        )

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class Decoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        attn_scales=(),
        temporal_upsample=(False, True, True),
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = list(dim_mult)
        self.num_res_blocks = num_res_blocks
        self.attn_scales = list(attn_scales)
        self.temporal_upsample = list(temporal_upsample)

        dims = [dim * u for u in [self.dim_mult[-1]] + self.dim_mult[::-1]]
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout),
        )

        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_up_flag = self.temporal_upsample[i] if i < len(self.temporal_upsample) else False
            upsamples.append(
                Up_ResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks + 1,
                    temporal_upsample=t_up_flag,
                    up_flag=i != len(self.dim_mult) - 1,
                )
            )
        self.upsamples = nn.Sequential(*upsamples)
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, 12, 3, padding=1),
        )

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx, first_chunk)
            else:
                x = layer(x)

        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


def _count_conv3d(model):
    return sum(1 for m in model.modules() if isinstance(m, CausalConv3d))


class WanVAE_(nn.Module):
    """Upstream Wan2.2 VAE module — encoder3d/decoder3d sandwich with 2x patchify
    on input. State-dict-compatible with ``Wan2.2_VAE.pth``."""

    def __init__(
        self,
        dim=160,
        dec_dim=256,
        z_dim=16,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        attn_scales=(),
        temperal_downsample=(True, True, False),
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = list(dim_mult)
        self.num_res_blocks = num_res_blocks
        self.attn_scales = list(attn_scales)
        self.temperal_downsample = list(temperal_downsample)
        self.temporal_upsample = list(temperal_downsample)[::-1]

        self.encoder = Encoder3d(
            dim, z_dim * 2, self.dim_mult, num_res_blocks, self.attn_scales, self.temperal_downsample, dropout
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(
            dec_dim, z_dim, self.dim_mult, num_res_blocks, self.attn_scales, self.temporal_upsample, dropout
        )

    def encode(self, x, scale):
        self.clear_cache()
        x = _patchify(x, patch_size=2)
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        out = None
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(x[:, :, :1, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
                out = torch.cat([out, out_], 2)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(1, self.z_dim, 1, 1, 1)
        else:
            mu = (mu - scale[0]) * scale[1]
        self.clear_cache()
        return mu, log_var

    def decode(self, z, scale):
        self.clear_cache()
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        out = None
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx, first_chunk=True
                )
            else:
                out_ = self.decoder(x[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
                out = torch.cat([out, out_], 2)
        out = _unpatchify(out, patch_size=2)
        self.clear_cache()
        return out

    def clear_cache(self):
        self._conv_num = _count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        self._enc_conv_num = _count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


# ----------------------------------------------------------------------------
# Wan2.2 VAE wrapper + BAGEL-surface adapter
# ----------------------------------------------------------------------------


# Per-channel mean/std for the 48-channel Wan2.2 latent space (upstream constants).
_WAN22_LATENT_MEAN: tuple[float, ...] = (
    -0.2289,
    -0.0052,
    -0.1323,
    -0.2339,
    -0.2799,
    0.0174,
    0.1838,
    0.1557,
    -0.1382,
    0.0542,
    0.2813,
    0.0891,
    0.1570,
    -0.0098,
    0.0375,
    -0.1825,
    -0.2246,
    -0.1207,
    -0.0698,
    0.5109,
    0.2665,
    -0.2108,
    -0.2158,
    0.2502,
    -0.2055,
    -0.0322,
    0.1109,
    0.1567,
    -0.0729,
    0.0899,
    -0.2799,
    -0.1230,
    -0.0313,
    -0.1649,
    0.0117,
    0.0723,
    -0.2839,
    -0.2083,
    -0.0520,
    0.3748,
    0.0152,
    0.1957,
    0.1433,
    -0.2944,
    0.3573,
    -0.0548,
    -0.1681,
    -0.0667,
)
_WAN22_LATENT_STD: tuple[float, ...] = (
    0.4765,
    1.0364,
    0.4514,
    1.1677,
    0.5313,
    0.4990,
    0.4818,
    0.5013,
    0.8158,
    1.0344,
    0.5894,
    1.0901,
    0.6885,
    0.6165,
    0.8454,
    0.4978,
    0.5759,
    0.3523,
    0.7135,
    0.6804,
    0.5833,
    1.4146,
    0.8986,
    0.5659,
    0.7069,
    0.5338,
    0.4889,
    0.4917,
    0.4069,
    0.4999,
    0.6866,
    0.4093,
    0.5709,
    0.6065,
    0.6415,
    0.4944,
    0.5726,
    1.2042,
    0.5458,
    1.6887,
    0.3971,
    1.0600,
    0.3943,
    0.5537,
    0.5444,
    0.4089,
    0.7468,
    0.7744,
)


class LanceWanVAE(nn.Module):
    """Wan2.2 VAE wrapped for BAGEL's pipeline.

    Exposes BAGEL's image-VAE surface — ``encode(BCHW) -> BC_zHW`` and
    ``decode(BC_zHW) -> BCHW`` — by treating each image as a 1-frame video clip.
    A 5-D ``encode_video``/``decode_video`` path is also provided for the
    ``Lance_3B_Video`` checkpoint.

    Construction is lazy: the heavy ``WanVAE_`` and ``Wan2.2_VAE.pth`` are not
    materialized until first use. Once built, the inner module is registered as
    a submodule so ``self.parameters()``, ``self.to(device)`` and ``vae_dtype =
    next(vae.parameters()).dtype`` (used by BAGEL's decode path) all behave.
    """

    z_channels: int = 48
    downsample_spatial: int = 16
    downsample_temporal: int = 4

    def __init__(
        self,
        vae_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
        *,
        lazy: bool = False,
    ):
        super().__init__()
        self._vae_path = vae_path
        self._dtype = dtype
        self._device = torch.device(device) if device is not None else None
        self._built = False
        # Registered lazily once the .pth is loaded:
        # self.model: WanVAE_   (set in _ensure_built)
        if not lazy:
            # BAGEL's decode path inspects ``next(vae.parameters()).dtype`` before
            # the first encode/decode call, so we have to register the submodule
            # eagerly when a real device is provided.  ``lazy=True`` is reserved
            # for config-only / no-GPU paths.
            self._ensure_built()

    def _ensure_built(self) -> None:
        if self._built:
            return
        device = self._device or (
            next(self.parameters(), torch.empty(0)).device
            if any(True for _ in self.parameters())
            else torch.device("cpu")
        )
        # Wan2.2 VAE config from upstream Wan2_2_VAE defaults:
        #   z_dim=48 (channels), c_dim=160 (encoder width), dec_dim=256 (decoder width),
        #   dim_mult=[1,2,4,4], temperal_downsample=[False,True,True].
        model = WanVAE_(
            dim=160,
            dec_dim=256,
            z_dim=self.z_channels,
            dim_mult=(1, 2, 4, 4),
            num_res_blocks=2,
            attn_scales=(),
            temperal_downsample=(False, True, True),
            dropout=0.0,
        )
        logger.info("Loading Wan2.2 VAE state dict from %s", self._vae_path)
        state = torch.load(self._vae_path, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.warning("Wan2.2 VAE missing keys (%d): %s", len(missing), missing[:5])
        if unexpected:
            logger.warning("Wan2.2 VAE unexpected keys (%d): %s", len(unexpected), unexpected[:5])
        model = model.to(device=device, dtype=self._dtype).eval()
        model.requires_grad_(False)
        # Register as submodule so .to() / .parameters() pick it up.
        self.model = model
        # Per-channel normalization tensors (matches upstream Wan2_2_VAE.scale = [mean, 1/std]).
        mean = torch.tensor(_WAN22_LATENT_MEAN, dtype=self._dtype, device=device)
        inv_std = torch.tensor([1.0 / s for s in _WAN22_LATENT_STD], dtype=self._dtype, device=device)
        # Buffers, so .to(device) follows along.
        self.register_buffer("_latent_mean", mean, persistent=False)
        self.register_buffer("_latent_inv_std", inv_std, persistent=False)
        self._built = True

    @torch.inference_mode()
    def encode_video(self, video: torch.Tensor, *, use_sample: bool = True) -> torch.Tensor:
        """Encode a 5-D clip ``[B, 3, T, H, W]`` -> latent ``[B, 48, t, h, w]``."""
        self._ensure_built()
        video = video.to(self.model.encoder.conv1.weight.dtype)
        mu, log_var = self.model.encode(video, [self._latent_mean, self._latent_inv_std])
        if use_sample:
            std = torch.exp(0.5 * log_var)
            mu = mu + std * torch.randn_like(std)
        return mu

    @torch.inference_mode()
    def decode_video(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a 5-D latent ``[B, 48, t, h, w]`` -> video ``[B, 3, T, H, W]``."""
        self._ensure_built()
        latent = latent.to(self.model.decoder.conv1.weight.dtype)
        out = self.model.decode(latent, [self._latent_mean, self._latent_inv_std])
        return out.clamp_(-1.0, 1.0)

    # ----- BAGEL image-VAE surface (4-D) -------------------------------- #

    @torch.inference_mode()
    def encode(self, padded_images: torch.Tensor) -> torch.Tensor:
        """``[B, 3, H, W]`` -> ``[B, 48, H/16, W/16]`` (single-frame image path).

        Each image is wrapped as a 1-frame clip, encoded, and the temporal axis
        is squeezed back out so the result matches BAGEL's iteration pattern.
        """
        if padded_images.dim() == 4:
            video = padded_images.unsqueeze(2)  # B,C,1,H,W
            squeeze_time = True  # image path: caller iterates 3-D latents
        elif padded_images.dim() == 5:
            video = padded_images
            # video path: caller expects 4-D latents (C,T,H,W), even when T==1
            # (e.g. video2video with a 1-frame reference clip).  Squeezing here
            # would drop the temporal axis and break the downstream indexing
            # ``latent[:, :t_lat, ...]`` in ``forward_cache_update_vae``.
            squeeze_time = False
        else:
            raise ValueError(f"LanceWanVAE.encode expects 4-D BCHW or 5-D BCTHW, got {tuple(padded_images.shape)}")
        latent = self.encode_video(video, use_sample=True)
        # B,48,t,h,w -> B,48,h,w when t == 1 (image path only)
        if squeeze_time and latent.shape[2] == 1:
            return latent.squeeze(2)
        return latent

    @torch.inference_mode()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """``[B, 48, h, w]`` -> ``[B, 3, H, W]`` (single-frame image path)."""
        if latent.dim() == 4:
            latent = latent.unsqueeze(2)  # B,48,1,h,w
        elif latent.dim() != 5:
            raise ValueError(f"LanceWanVAE.decode expects 4-D BCHW or 5-D BCTHW, got {tuple(latent.shape)}")
        video = self.decode_video(latent)  # B,3,T,H,W
        if video.shape[2] == 1:
            return video.squeeze(2)
        return video


def build_wan22_vae(vae_path: str, dtype: torch.dtype = torch.bfloat16, device=None) -> LanceWanVAE:
    """Convenience factory: lazy-construct a :class:`LanceWanVAE` adapter."""
    return LanceWanVAE(vae_path=vae_path, dtype=dtype, device=device)


__all__ = ["LanceWanVAE", "WanVAE_", "build_wan22_vae"]
