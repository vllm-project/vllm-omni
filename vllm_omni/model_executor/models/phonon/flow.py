# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Phonon LSD flow head, matching xn-ptts's SimpleMLPAdaLN."""

import torch
from torch import nn
from torch.nn import functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, width: int, frequency_size: int = 256):
        super().__init__()
        self.register_buffer("freqs", torch.zeros(frequency_size // 2))
        self.mlp = nn.Sequential(
            nn.Linear(frequency_size, width),
            nn.SiLU(),
            nn.Linear(width, width),
            TimeRMSNorm(width),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        angles = time * self.freqs
        return self.mlp(torch.cat((angles.cos(), angles.sin()), dim=-1))


class TimeRMSNorm(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(width))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = inputs.float() * torch.rsqrt(inputs.float().square().mean(-1, keepdim=True) + 1e-5)
        return normalized.to(inputs.dtype) * self.alpha


class FlowBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.in_ln = nn.LayerNorm(width, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 3 * width))

    def forward(self, inputs: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.adaLN_modulation(conditioning).chunk(3, dim=-1)
        return inputs + gate * self.mlp(self.in_ln(inputs) * (1 + scale) + shift)


class FlowFinalLayer(nn.Module):
    def __init__(self, width: int, latent_dim: int):
        super().__init__()
        self.linear = nn.Linear(width, latent_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))

    def forward(self, inputs: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(conditioning).chunk(2, dim=-1)
        normalized = F.layer_norm(inputs, (inputs.shape[-1],), eps=1e-6)
        return self.linear(normalized * (1 + scale) + shift)


class PhononFlowHead(nn.Module):
    def __init__(self, latent_dim: int, width: int, condition_dim: int, depth: int):
        super().__init__()
        self.time_embed = nn.ModuleList([TimeEmbedding(width), TimeEmbedding(width)])
        self.cond_embed = nn.Linear(condition_dim, width)
        self.input_proj = nn.Linear(latent_dim, width)
        self.res_blocks = nn.ModuleList([FlowBlock(width) for _ in range(depth)])
        self.final_layer = FlowFinalLayer(width, latent_dim)

    def forward(self, condition: torch.Tensor, noise: torch.Tensor, *, steps: int = 1) -> torch.Tensor:
        if steps < 1:
            raise ValueError("LSD steps must be positive")
        projected_condition = self.cond_embed(condition)
        current = noise
        for step in range(steps):
            start = condition.new_full((1, 1), step / steps)
            end = condition.new_full((1, 1), (step + 1) / steps)
            conditioning = projected_condition + (self.time_embed[0](start) + self.time_embed[1](end)) * 0.5
            hidden = self.input_proj(current)
            for block in self.res_blocks:
                hidden = block(hidden, conditioning)
            current = current + self.final_layer(hidden, conditioning) / steps
        return current
