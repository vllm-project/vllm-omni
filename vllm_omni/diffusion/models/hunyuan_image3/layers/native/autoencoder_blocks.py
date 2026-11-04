# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""ResnetBlock for the HunyuanImage3 autoencoder — default fallback implementation.

Uses native PyTorch ``nn.GroupNorm`` + ``F.silu``. Hardware-specific optimized
versions live in subdirectories (e.g., nvidia/) and are selected via __init__.py
dispatch.
"""

import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

VaeConv3d = nn.Conv3d


class ResnetBlock(nn.Module):
    conv_cls = nn.Conv3d

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = self.conv_cls(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = self.conv_cls(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = self.conv_cls(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    conv_cls = nn.Conv3d

    """Attention with torch sdpa implementation."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        # The memory concern is already handled via VAE tiling,
        # so there's no need to chunk here — calling nn.Conv3d directly is fine.
        self.q = self.conv_cls(in_channels, in_channels, kernel_size=1)
        self.k = self.conv_cls(in_channels, in_channels, kernel_size=1)
        self.v = self.conv_cls(in_channels, in_channels, kernel_size=1)
        self.proj_out = self.conv_cls(in_channels, in_channels, kernel_size=1)

    def _normalize(self, x: Tensor) -> Tensor:
        return self.norm(x)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self._normalize(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, f, h, w = q.shape
        q = rearrange(q, "b c f h w -> b 1 (f h w) c").contiguous()
        k = rearrange(k, "b c f h w -> b 1 (f h w) c").contiguous()
        v = rearrange(v, "b c f h w -> b 1 (f h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "b 1 (f h w) c -> b c f h w", f=f, h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


__all__ = ["AttnBlock", "ResnetBlock", "VaeConv3d"]
