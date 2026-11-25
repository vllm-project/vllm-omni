# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialParallelConv3d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, padding_mode="reflect")
        self.pad_mode = "reflect"
        self.pad_mode_t = "constant"
        self.causal = True
        self.spatial_parallel = False

    def _apply_padding(self, x):
        if sum(self.padding) == 0:
            return x
        x = F.pad(x, (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 0, 0), mode=self.pad_mode)
        if x.shape[2] > 1:
            return F.pad(x, (0, 0, 0, 0, 2 * self.padding[0], 0), mode=self.pad_mode_t)
        zeros = torch.zeros_like(x[:, :, :1]).expand(-1, -1, self.kernel_size[0] - 1, -1, -1)
        return torch.cat([zeros, x], dim=2)

    def forward(self, x):
        if sum(self.padding) == 0:
            return super().forward(x)
        x = self._apply_padding(x)
        return F.conv3d(x, self.weight, self.bias, stride=self.stride, padding=0, dilation=self.dilation)


class SpatialParallelGroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial_parallel = False


class TemporalIsolatedSpatialParallelGroupNorm(SpatialParallelGroupNorm):
    def forward(self, input):
        if input.dim() != 5:
            return super().forward(input)
        b, c, t, h, w = input.shape
        out = super().forward(input.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, 1, h, w))
        return out.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


def _group_norm(channels, time_isolated):
    norm_cls = TemporalIsolatedSpatialParallelGroupNorm if time_isolated else SpatialParallelGroupNorm
    norm = norm_cls(num_groups=32, num_channels=channels, eps=1e-6, affine=True)
    with torch.no_grad():
        norm.weight.normal_(1.0, 0.2)
        norm.bias.normal_(0.0, 0.2)
    return norm


class Downsample3D(nn.Module):
    def __init__(self, channels, time_stride, space_stride):
        super().__init__()
        self.space_stride = space_stride
        self.conv = SpatialParallelConv3d(
            channels, channels, kernel_size=3, padding=(1, 0, 0), stride=(time_stride, space_stride, space_stride)
        )
        self.pad_mode = self.conv.pad_mode

    def forward(self, x):
        if self.space_stride == 2:
            x = F.pad(x, (0, 1, 0, 1, 0, 0), mode=self.pad_mode)
        return self.conv(x)


class ResnetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, time_isolated):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_fused_norm = False
        self.norm1 = _group_norm(in_channels, time_isolated)
        self.norm2 = _group_norm(out_channels, time_isolated)
        self.conv1 = SpatialParallelConv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = SpatialParallelConv3d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = SpatialParallelConv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, zq=None):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class EncoderFCN3D(nn.Module):
    def __init__(self, time_isolated):
        super().__init__()
        levels = ((32, 32, 1, 2), (32, 64, 2, 2), (64, 64, 1, 1))
        self.num_levels = len(levels)
        self.num_res_blocks = [2] * self.num_levels
        self.use_fused_norm = False
        self.conv_in = SpatialParallelConv3d(3, 32, kernel_size=3, padding=1)
        self.down = nn.ModuleList()
        for in_channels, channels, time_stride, space_stride in levels:
            down = nn.Module()
            down.block = nn.ModuleList(
                [ResnetBlock3D(in_channels, channels, time_isolated), ResnetBlock3D(channels, channels, time_isolated)]
            )
            if time_stride * space_stride > 1:
                down.downsample = Downsample3D(channels, time_stride, space_stride)
            self.down.append(down)
        self.norm_out = _group_norm(64, time_isolated)
        self.conv_out = SpatialParallelConv3d(64, 8, kernel_size=3, padding=1)

    def forward(self, x, zq=None):
        h = self.conv_in(x)
        for i_level in range(self.num_levels):
            for i_block in range(self.num_res_blocks[i_level]):
                h = self.down[i_level].block[i_block](h, zq)
            if hasattr(self.down[i_level], "downsample"):
                h = self.down[i_level].downsample(h)
        h = F.silu(self.norm_out(h))
        return self.conv_out(h)


def make_encoder(time_isolated=True):
    torch.manual_seed(0)
    return EncoderFCN3D(time_isolated).eval()


def checkpoint_pad_chain(source, *, front, top, bottom, left, right, apply_silu):
    source = source.contiguous()
    if apply_silu:
        source = F.silu(source)
    padded = F.pad(source, (left, right, top, bottom, 0, 0), mode="reflect")
    if padded.shape[2] > 1:
        return F.pad(padded, (0, 0, 0, 0, front, 0))
    zeros = torch.zeros_like(padded[:, :, :1]).expand(-1, -1, front, -1, -1)
    return torch.cat([zeros, padded], dim=2)
