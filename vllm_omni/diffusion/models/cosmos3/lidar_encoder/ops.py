# -----------------------------------------------------------------------------
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact Cosmos Lab at cosmoslab@exchange.nvidia.com.
# -----------------------------------------------------------------------------

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn.modules.utils import _pair, _quadruple


def zero_out(m: nn.Module) -> None:
    for p in m.parameters():
        p.data.zero_()


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, channels: int, max_period: int = 10_000) -> None:
        super().__init__()
        self.channels = channels
        self.max_period = max_period

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B], returns [B,C]
        assert len(x.shape) == 1
        h = -np.log(self.max_period) / (self.channels // 2 - 1)
        h = torch.exp(h * torch.arange(self.channels // 2, device=x.device))  # [C/2]
        h = x[:, None] * h[None, :]  # [B,C/2]
        h = torch.cat([h.sin(), h.cos()], dim=-1)  # [B,C]
        return h.to(x)  # [B,C]

    def extra_repr(self) -> str:
        return f"dim={self.channels} max_period={self.max_period}"


class Pad(nn.Module):
    def __init__(
        self,
        padding: int | tuple[int, int] | tuple[int, int, int, int],
        ring: bool = False,
        mode: str = "constant",
    ) -> None:
        super().__init__()
        self.padding = _quadruple(padding)
        self.horizontal = "circular" if ring else mode
        self.vertical = mode

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # h: [B,C,H,W], returns [B,C,H',W']
        left, right, top, bottom = self.padding
        h = F.pad(h, (left, right, 0, 0), mode=self.horizontal)  # [B,C,H,W']
        h = F.pad(h, (0, 0, top, bottom), mode=self.vertical)  # [B,C,H',W']
        return h

    def extra_repr(self) -> str:
        return f"padding={self.padding}, " + f"horizontal={self.horizontal}, vertical={self.vertical}"


class Resample(nn.Module):
    def __init__(
        self,
        up: int | tuple[int, int] = 1,
        down: int | tuple[int, int] = 1,
        window: Sequence[float] = (1, 3, 3, 1),  # bilinear
        ring: bool = True,
        normalize: bool = True,
        direction: str = "hw",
        mode: str = "constant",
    ) -> None:
        super().__init__()
        self.up = np.asarray(_pair(up))
        self.down = np.asarray(_pair(down))
        self.window = window
        self.n_taps = len(window)
        self.ring = ring
        self.pad_mode_w = "circular" if ring else mode
        self.pad_mode_h = mode
        self.normalize = normalize
        self.direction = direction
        assert self.direction in ("h", "w", "hw")

        # setup sizes
        if "h" in self.direction:
            self.k_h = self.n_taps
            self.up_h = self.up[0]
            self.down_h = self.down[0]
        else:
            self.k_h = self.up_h = self.down_h = 1

        if "w" in self.direction:
            self.k_w = self.n_taps
            self.up_w = self.up[1]
            self.down_w = self.down[1]
        else:
            self.k_w = self.up_w = self.down_w = 1

        # setup filter
        kernel = torch.tensor(self.window, dtype=torch.float32)  # [K]
        if self.normalize:
            kernel /= kernel.sum()
        kernel *= (self.up_h * self.up_w) ** (kernel.ndim / 2)
        self.register_buffer("kernel", kernel)

        # setup padding
        if self.up[0] > 1:
            self.ph0 = (self.k_h - self.up_h + 1) // 2 + self.up_h - 1
            self.ph1 = (self.k_h - self.up_h) // 2
        elif self.down[0] >= 1:
            self.ph0 = (self.k_h - self.down_h + 1) // 2
            self.ph1 = (self.k_h - self.down_h) // 2
        if self.up[1] > 1:
            self.pw0 = (self.k_w - self.up_w + 1) // 2 + self.up_w - 1
            self.pw1 = (self.k_w - self.up_w) // 2
        elif self.down[1] >= 1:
            self.pw0 = (self.k_w - self.down_w + 1) // 2
            self.pw1 = (self.k_w - self.down_w) // 2

        self.margin = int(max(self.ph0, self.ph1, self.pw0, self.pw1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # h: [B,C,H,W], returns [B,C,H',W']
        # margin
        h = F.pad(h, (self.margin, self.margin, 0, 0), mode=self.pad_mode_w)  # [B,C,H,Wm]
        h = F.pad(h, (0, 0, self.margin, self.margin), mode=self.pad_mode_h)  # [B,C,Hm,Wm]
        # up by zero-insertion
        B, C, H, W = h.shape
        h = h.view(B, C, H, 1, W, 1)  # [B,C,H,1,W,1]
        h = F.pad(h, [0, self.up_w - 1, 0, 0, 0, self.up_h - 1])  # [B,C,H,UH,W,UW]
        h = h.view(B, C, H * self.up_h, W * self.up_w)  # [B,C,Hu,Wu]
        # crop
        h = h[
            ...,
            self.margin * self.up_h - self.ph0 : (H - self.margin) * self.up_h + self.ph1,
            self.margin * self.up_w - self.pw0 : (W - self.margin) * self.up_w + self.pw1,
        ]
        # fir
        kernel = self.kernel[None, None].repeat(C, 1, 1).to(dtype=h.dtype)  # [C,1,K]
        if self.direction == "hw":
            h = F.conv2d(h, kernel[..., None, :], groups=C)
            h = F.conv2d(h, kernel[..., :, None], groups=C)
        elif self.direction == "h":
            h = F.conv2d(h, kernel[..., :, None], groups=C)
        elif self.direction == "w":
            h = F.conv2d(h, kernel[..., None, :], groups=C)
        # down
        h = h[:, :, :: self.down_h, :: self.down_w]
        return h

    def extra_repr(self) -> str:
        return f"up={tuple(self.up)}, down={tuple(self.down)}, ring={self.ring}"


class Conv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int],
        padding: int | tuple[int, int],
        bias: bool = True,
        ring: bool = False,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )
        self.pad = Pad(padding=padding, ring=ring) if padding != 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,Cin,H,W], returns [B,Cout,H',W']
        if self.pad is not None:
            x = self.pad(x)
        return super().forward(x)


class AdaGN(nn.GroupNorm):
    def __init__(
        self,
        emb_channels: int,
        out_channels: int,
        num_groups: int,
        eps: float = 1e-5,
    ) -> None:
        super().__init__(
            num_groups=num_groups,
            num_channels=out_channels,
            eps=eps,
            affine=False,
        )
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, out_channels * 2),
            Rearrange("B C -> B C 1 1"),
        )

    def forward(
        self, x: torch.Tensor, emb: torch.Tensor
    ) -> torch.Tensor:  # x: [B,C,H,W], emb: [B,E], returns [B,C,H,W]
        h = super().forward(x)  # [B,C,H,W]
        scale, shift = self.proj(emb).chunk(2, dim=1)  # [B,C,1,1], [B,C,1,1]
        h = h * (1 + scale) + shift  # [B,C,H,W]
        return h


class ConditionalSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        for module in self:
            x = module(x, condition)
        return x
