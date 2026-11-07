# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Full-frame VAE semantics with height-sharded activations and tiled convolutions."""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from vllm_omni.diffusion.distributed.autoencoders.wan_spatial_shard import halo_exchange


@dataclass(frozen=True)
class SpatialContext:
    group: dist.ProcessGroup
    rank: int
    units: tuple[int, ...]

    def split(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.shape[-2] // sum(self.units)
        start = sum(self.units[: self.rank]) * scale
        return x.narrow(-2, start, self.units[self.rank] * scale).contiguous()

    def gather(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.shape[-2] // self.units[self.rank]
        heights = [n * scale for n in self.units]
        padded = F.pad(x, (0, 0, 0, max(heights) - x.shape[-2])).contiguous()
        parts = [torch.empty_like(padded) for _ in heights]
        dist.all_gather(parts, padded, group=self.group)
        return torch.cat([part.narrow(-2, 0, h) for part, h in zip(parts, heights)], dim=-2)

    def normalize(self, norm: nn.GroupNorm, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        grouped = x.to(torch.float32, copy=True).reshape(b, norm.num_groups, c // norm.num_groups, t, h, w)
        variance, mean = torch.var_mean(grouped, dim=(2, 4, 5), correction=0, keepdim=True)
        count = c // norm.num_groups * h * w
        moments = torch.stack((mean * count, torch.full_like(mean, count)))
        dist.all_reduce(moments, group=self.group)
        global_mean = moments[0] / moments[1]
        centered = (variance + (mean - global_mean).square()) * count
        dist.all_reduce(centered, group=self.group)
        normalized = grouped.sub_(global_mean).mul_(torch.rsqrt(centered / moments[1] + norm.eps)).reshape_as(x)
        normalized.mul_(norm.weight.float().view(1, -1, 1, 1, 1))
        normalized.add_(norm.bias.float().view(1, -1, 1, 1, 1))
        return normalized.to(x.dtype)

    def convolution(self, conv: nn.Conv3d, x: torch.Tensor) -> torch.Tensor:
        halo = conv.kernel_size[1] // 2
        if halo:
            x, _, _ = halo_exchange(x, group=self.group, halo_size=halo)
            if conv.padding[1] == 0:
                # Downsampling uses right/bottom padding, not symmetric padding.
                x = x[:, :, :, halo:]
        return F.conv3d(x, conv.weight, conv.bias, conv.stride, (0, 0, conv.padding[2]), conv.dilation, conv.groups)


def tiled_convolution(conv: nn.Conv3d, x: torch.Tensor, rows: int = 128) -> torch.Tensor:
    """Tile output rows; only each tile's spatial padding/workspace is materialized."""
    shape = [(x.shape[i + 2] + 2 * conv.padding[i] - conv.kernel_size[i]) // conv.stride[i] + 1 for i in range(3)]
    output = x.new_empty((x.shape[0], conv.out_channels, *shape))
    stride, padding, kernel = conv.stride[1], conv.padding[1], conv.kernel_size[1]
    for start in range(0, shape[1], rows):
        end = min(start + rows, shape[1])
        lo, hi = start * stride - padding, (end - 1) * stride - padding + kernel
        tile = x[:, :, :, max(0, lo) : min(x.shape[-2], hi)]
        tile = F.pad(tile, (0, 0, max(0, -lo), max(0, hi - x.shape[-2])))
        output[:, :, :, start:end] = F.conv3d(
            tile, conv.weight, conv.bias, conv.stride, (0, 0, conv.padding[2]), conv.dilation, conv.groups
        )
    return output
