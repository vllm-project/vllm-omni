# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright contributors to the Amphion project
"""ConvNeXt Vocos backbone used by the IndexTTS 2.5 codec decoder."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=7,
            padding=3,
            groups=dim,
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones(dim),
            requires_grad=True,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        hidden = self.dwconv(inputs).transpose(1, 2)
        hidden = self.norm(hidden)
        hidden = self.pwconv1(hidden)
        hidden = self.act(hidden)
        hidden = self.pwconv2(hidden)
        hidden = self.gamma * hidden
        return residual + hidden.transpose(1, 2)


class VocosBackbone(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
    ):
        super().__init__()
        self.embed = nn.Conv1d(
            input_channels,
            dim,
            kernel_size=7,
            padding=3,
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        layer_scale = 1 / num_layers
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    layer_scale_init_value=layer_scale,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(inputs)
        hidden = self.norm(hidden.transpose(1, 2)).transpose(1, 2)
        for block in self.convnext:
            hidden = block(hidden)
        return self.final_layer_norm(hidden.transpose(1, 2))
