# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This file vendors the minimal Continuous Image (CI) tokenizer implementation
# needed by InternVLA-A1 from NVIDIA/Cosmos-Tokenizer. The code is intentionally
# reduced to the CI image path only so the runtime does not depend on the
# external `cosmos_tokenizer` package.

from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

_WAVELETS = {
    "haar": torch.tensor([0.7071067811865476, 0.7071067811865476]),
    "rearrange": torch.tensor([1.0, 1.0]),
}
_PERSISTENT = False


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def normalize(in_channels: int, num_groups: int = 32) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class IdentityDistribution(nn.Module):
    def forward(self, parameters: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        zeros = torch.tensor([0.0], device=parameters.device, dtype=parameters.dtype)
        return parameters, (zeros, zeros)


class Patcher(nn.Module):
    def __init__(self, patch_size: int = 1, patch_method: str = "haar") -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_method = patch_method
        self.register_buffer("wavelets", _WAVELETS[patch_method], persistent=_PERSISTENT)
        self.range = range(int(torch.log2(torch.tensor(self.patch_size)).item()))
        self.register_buffer(
            "_arange",
            torch.arange(_WAVELETS[patch_method].shape[0]),
            persistent=_PERSISTENT,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch_method == "haar":
            return self._haar(x)
        if self.patch_method == "rearrange":
            return self._arrange(x)
        raise ValueError(f"Unknown patch method: {self.patch_method}")

    def _dwt(self, x: torch.Tensor, mode: str = "reflect", *, rescale: bool = False) -> torch.Tensor:
        dtype = x.dtype
        h = self.wavelets
        n = h.shape[0]
        g = x.shape[1]
        hl = h.flip(0).reshape(1, 1, -1).repeat(g, 1, 1).to(dtype=dtype)
        hh = (h * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(g, 1, 1).to(dtype=dtype)

        x = F.pad(x, pad=(n - 2, n - 1, n - 2, n - 1), mode=mode).to(dtype)
        xl = F.conv2d(x, hl.unsqueeze(2), groups=g, stride=(1, 2))
        xh = F.conv2d(x, hh.unsqueeze(2), groups=g, stride=(1, 2))
        xll = F.conv2d(xl, hl.unsqueeze(3), groups=g, stride=(2, 1))
        xlh = F.conv2d(xl, hh.unsqueeze(3), groups=g, stride=(2, 1))
        xhl = F.conv2d(xh, hl.unsqueeze(3), groups=g, stride=(2, 1))
        xhh = F.conv2d(xh, hh.unsqueeze(3), groups=g, stride=(2, 1))

        out = torch.cat([xll, xlh, xhl, xhh], dim=1)
        if rescale:
            out = out / 2
        return out

    def _haar(self, x: torch.Tensor) -> torch.Tensor:
        for _ in self.range:
            x = self._dwt(x, rescale=True)
        return x

    def _arrange(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(
            x,
            "b c (h p1) (w p2) -> b (c p1 p2) h w",
            p1=self.patch_size,
            p2=self.patch_size,
        ).contiguous()


class UnPatcher(nn.Module):
    def __init__(self, patch_size: int = 1, patch_method: str = "haar") -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_method = patch_method
        self.register_buffer("wavelets", _WAVELETS[patch_method], persistent=_PERSISTENT)
        self.range = range(int(torch.log2(torch.tensor(self.patch_size)).item()))
        self.register_buffer(
            "_arange",
            torch.arange(_WAVELETS[patch_method].shape[0]),
            persistent=_PERSISTENT,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch_method == "haar":
            return self._ihaar(x)
        if self.patch_method == "rearrange":
            return self._iarrange(x)
        raise ValueError(f"Unknown patch method: {self.patch_method}")

    def _idwt(self, x: torch.Tensor, *, rescale: bool = False) -> torch.Tensor:
        dtype = x.dtype
        h = self.wavelets
        n = h.shape[0]
        g = x.shape[1] // 4
        hl = h.flip([0]).reshape(1, 1, -1).repeat([g, 1, 1]).to(dtype=dtype)
        hh = (h * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(g, 1, 1).to(dtype=dtype)

        xll, xlh, xhl, xhh = torch.chunk(x.to(dtype), 4, dim=1)
        yl = F.conv_transpose2d(xll, hl.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0))
        yl += F.conv_transpose2d(xlh, hh.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0))
        yh = F.conv_transpose2d(xhl, hl.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0))
        yh += F.conv_transpose2d(xhh, hh.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0))
        y = F.conv_transpose2d(yl, hl.unsqueeze(2), groups=g, stride=(1, 2), padding=(0, n - 2))
        y += F.conv_transpose2d(yh, hh.unsqueeze(2), groups=g, stride=(1, 2), padding=(0, n - 2))

        if rescale:
            y = y * 2
        return y

    def _ihaar(self, x: torch.Tensor) -> torch.Tensor:
        for _ in self.range:
            x = self._idwt(x, rescale=True)
        return x

    def _iarrange(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(
            x,
            "b (c p1 p2) h w -> b c (h p1) (w p2)",
            p1=self.patch_size,
            p2=self.patch_size,
        )


class Upsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (0, 1, 0, 1), mode="constant", value=0))


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels: int, out_channels: int | None = None, dropout: float, **_: object) -> None:
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.nin_shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(nonlinearity(self.norm1(x)))
        h = self.conv2(self.dropout(nonlinearity(self.norm2(h))))
        return self.nin_shortcut(x) + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.norm = normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        b, c, height, width = q.shape
        q = q.reshape(b, c, height * width).permute(0, 2, 1)
        k = k.reshape(b, c, height * width)
        attn = torch.bmm(q, k) * (int(c) ** -0.5)
        attn = F.softmax(attn, dim=2)
        v = v.reshape(b, c, height * width)
        attn = attn.permute(0, 2, 1)
        h = torch.bmm(v, attn).reshape(b, c, height, width)
        return x + self.proj_out(h)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        channels_mult: list[int],
        num_res_blocks: int,
        attn_resolutions: list[int],
        dropout: float,
        resolution: int,
        z_channels: int,
        spatial_compression: int,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.num_resolutions = len(channels_mult)
        self.num_res_blocks = num_res_blocks

        patch_size = int(kwargs.get("patch_size", 1))
        self.patcher = Patcher(patch_size, str(kwargs.get("patch_method", "rearrange")))
        in_channels = in_channels * patch_size * patch_size
        self.num_downsamples = int(math.log2(spatial_compression)) - int(math.log2(patch_size))
        assert self.num_downsamples <= self.num_resolutions

        self.conv_in = nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1)
        curr_res = resolution // patch_size
        in_ch_mult = (1,) + tuple(channels_mult)
        self.down = nn.ModuleList()
        block_in = channels
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = channels * in_ch_mult[i_level]
            block_out = channels * channels_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level < self.num_downsamples:
                down.downsample = Downsample(block_in)
                curr_res //= 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.norm_out = normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patcher(x)
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level < self.num_downsamples:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = nonlinearity(self.norm_out(h))
        return self.conv_out(h)


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        channels: int,
        channels_mult: list[int],
        num_res_blocks: int,
        attn_resolutions: list[int],
        dropout: float,
        resolution: int,
        z_channels: int,
        spatial_compression: int,
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.num_resolutions = len(channels_mult)
        self.num_res_blocks = num_res_blocks

        patch_size = int(kwargs.get("patch_size", 1))
        self.unpatcher = UnPatcher(patch_size, str(kwargs.get("patch_method", "rearrange")))
        out_ch = out_channels * patch_size * patch_size
        self.num_upsamples = int(math.log2(spatial_compression)) - int(math.log2(patch_size))
        assert self.num_upsamples <= self.num_resolutions

        block_in = channels * channels_mult[self.num_resolutions - 1]
        curr_res = (resolution // patch_size) // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = channels * channels_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level >= (self.num_resolutions - self.num_upsamples):
                up.upsample = Upsample(block_in)
                curr_res *= 2
            self.up.insert(0, up)

        self.norm_out = normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level >= (self.num_resolutions - self.num_upsamples):
                h = self.up[i_level].upsample(h)
        h = self.conv_out(nonlinearity(self.norm_out(h)))
        return self.unpatcher(h)


class ContinuousImageTokenizer(nn.Module):
    def __init__(self, z_channels: int, z_factor: int, latent_channels: int, **kwargs: object) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.encoder = Encoder(z_channels=z_factor * z_channels, **kwargs)
        self.decoder = Decoder(z_channels=z_channels, **kwargs)
        self.quant_conv = nn.Conv2d(z_factor * z_channels, z_factor * latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, z_channels, 1)
        self.distribution = IdentityDistribution()

    def encoder_module(self) -> nn.Sequential:
        return nn.Sequential(
            OrderedDict(
                [
                    ("encoder", self.encoder),
                    ("quant_conv", self.quant_conv),
                    ("distribution", self.distribution),
                ]
            )
        )

    def decoder_module(self) -> nn.Sequential:
        return nn.Sequential(
            OrderedDict(
                [
                    ("post_quant_conv", self.post_quant_conv),
                    ("decoder", self.decoder),
                ]
            )
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h = self.encoder(x)
        moments = self.quant_conv(h)
        return self.distribution(moments)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.post_quant_conv(z))


def build_cosmos_ci_torch_model(spatial_compression: int) -> ContinuousImageTokenizer:
    return ContinuousImageTokenizer(
        attn_resolutions=[32],
        channels=128,
        channels_mult=[2, 4, 4],
        dropout=0.0,
        in_channels=3,
        spatial_compression=spatial_compression,
        num_res_blocks=2,
        out_channels=3,
        resolution=1024,
        patch_size=4,
        patch_method="haar",
        latent_channels=16,
        z_channels=16,
        z_factor=1,
        name="CI",
    )
