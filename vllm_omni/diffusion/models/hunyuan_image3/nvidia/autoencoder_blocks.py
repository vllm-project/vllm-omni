# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""ResnetBlock for the HunyuanImage3 autoencoder — NVIDIA CUDA + Triton implementation.

Split out from autoencoder.py because its ``GroupNorm -> SiLU`` pairs are served
by :func:`fused_group_norm_silu`, a single Triton kernel that falls back to
native ``F.silu(F.group_norm(...))`` when Triton is unavailable. The submodule
layout is untouched, so state_dict keys are identical to the unfused version.

``Conv3d`` is duplicated here rather than imported from autoencoder.py, which
would make the two modules import each other.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from vllm_omni.model_executor.models.common.ops import fused_group_norm_silu
from ._cudnn_settings import cudnn_settings


class Conv3d(nn.Conv3d):
    """
    Perform Conv3d on patches with numerical differences from nn.Conv3d within 1e-5.
    Only symmetric padding is supported.
    """

    def forward(self, input):
        B, C, T, H, W = input.shape
        memory_count = (C * T * H * W) * 2 / 1024**3
        if memory_count > 2:
            n_split = math.ceil(memory_count / 2)
            assert n_split >= 2
            chunks = torch.chunk(input, chunks=n_split, dim=-3)
            padded_chunks = []
            for i in range(len(chunks)):
                if self.padding[0] > 0:
                    padded_chunk = F.pad(
                        chunks[i],
                        (0, 0, 0, 0, self.padding[0], self.padding[0]),
                        mode="constant" if self.padding_mode == "zeros" else self.padding_mode,
                        value=0,
                    )
                    if i > 0:
                        padded_chunk[:, :, : self.padding[0]] = chunks[i - 1][:, :, -self.padding[0] :]
                    if i < len(chunks) - 1:
                        padded_chunk[:, :, -self.padding[0] :] = chunks[i + 1][:, :, : self.padding[0]]
                else:
                    padded_chunk = chunks[i]
                padded_chunks.append(padded_chunk)
            padding_bak = self.padding
            self.padding = (0, self.padding[1], self.padding[2])
            outputs = []
            for i in range(len(padded_chunks)):
                outputs.append(super().forward(padded_chunks[i]))
            self.padding = padding_bak
            return torch.cat(outputs, dim=-3)
        else:
            return super().forward(input)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        with cudnn_settings(benchmark = True, deterministic = True):
            h = x
            h = fused_group_norm_silu(
                h, self.norm1.weight, self.norm1.bias, num_groups=self.norm1.num_groups, eps=self.norm1.eps
            )
            h = self.conv1(h)

            h = fused_group_norm_silu(
                h, self.norm2.weight, self.norm2.bias, num_groups=self.norm2.num_groups, eps=self.norm2.eps
            )
            h = self.conv2(h)

            if self.in_channels != self.out_channels:
                x = self.nin_shortcut(x)
            return x + h


__all__ = ["ResnetBlock"]
