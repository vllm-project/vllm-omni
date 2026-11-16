# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from https://github.com/higurashi-ai/Haru (Apache-2.0), haru/haru/models/meanflow_head.py @ d3612b0.
"""MeanFlow head: one audio latent from one LM hidden state.

The head predicts the mean velocity ``u(z, t, h | c, w)`` of the flow from
noise (``t = 0``) to data (``t = 1``) over an interval ``h``; sampling is
``z += h * u``. The guidance scale ``w`` is an input of the network, so a
single forward pass of a small MLP produces a guided latent.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from vllm_omni.model_executor.models.haru.configuration_haru import HaruHeadConfig

_FREQUENCY_DIM = 256


def _rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    return F.rms_norm(x.float(), (x.shape[-1],), eps=eps).to(x.dtype)


class ScalarEmbed(nn.Module):
    """Sinusoidal features of a scalar, then a two-layer MLP."""

    def __init__(self, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(_FREQUENCY_DIM, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, hidden, bias=False)
        half = _FREQUENCY_DIM // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """``value [B]`` in the head dtype -> ``[B, hidden]``."""
        args = value.float()[:, None] * self.freqs.float()
        features = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(value.dtype)
        return self.fc2(F.silu(self.fc1(features)))


class FeedForward(nn.Module):
    def __init__(self, hidden: int, ffn: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, ffn, bias=False)
        self.up_proj = nn.Linear(hidden, ffn, bias=False)
        self.down_proj = nn.Linear(ffn, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class HeadLayer(nn.Module):
    def __init__(self, hidden: int, ffn: int, eps: float):
        super().__init__()
        self.norm = nn.Module()
        self.norm.weight = nn.Parameter(torch.ones(hidden))
        self.ada = nn.Linear(hidden, 3 * hidden, bias=False)
        self.ffn = FeedForward(hidden, ffn)
        self.eps = eps

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.ada(c).chunk(3, dim=-1)
        h = _rms_norm(x, self.eps) * self.norm.weight
        return x + gate * self.ffn(h * (1 + scale) + shift)


class FinalLayer(nn.Module):
    def __init__(self, hidden: int, out: int, eps: float):
        super().__init__()
        self.ada = nn.Linear(hidden, 2 * hidden, bias=False)
        self.proj = nn.Linear(hidden, out, bias=False)
        self.eps = eps

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada(c).chunk(2, dim=-1)
        return self.proj(_rms_norm(x, self.eps) * (1 + scale) + shift)


class MeanFlowHead(nn.Module):
    def __init__(self, config: HaruHeadConfig):
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        self.latent_proj = nn.Linear(config.latent_dim, hidden, bias=False)
        self.cond_proj = nn.Linear(hidden, hidden, bias=False)
        self.t_embed = ScalarEmbed(hidden)
        self.dt_embed = ScalarEmbed(hidden)
        self.h_embed = ScalarEmbed(hidden)
        self.omega_embed = ScalarEmbed(hidden)
        ffn = int(hidden * config.ffn_ratio)
        self.layers = nn.ModuleList(HeadLayer(hidden, ffn, config.norm_eps) for _ in range(config.num_layers))
        self.final = FinalLayer(hidden, config.latent_dim, config.norm_eps)

    def velocity(self, z: torch.Tensor, cond: torch.Tensor, t: float, h: float, guidance: torch.Tensor) -> torch.Tensor:
        """``z [B, D]``, projected condition ``cond [B, H]``, ``guidance [B]`` float32."""
        batch, dtype = z.shape[0], z.dtype

        def scalar(value: float) -> torch.Tensor:
            return torch.full((batch,), value, device=z.device, dtype=dtype)

        c = cond + self.t_embed(scalar(t))
        c = c + self.dt_embed(scalar(self.config.interval_embed))
        c = c + self.h_embed(scalar(h))
        c = c + self.omega_embed((1.0 - 1.0 / guidance.float()).to(dtype))
        c = F.silu(c)
        x = self.latent_proj(z)
        for layer in self.layers:
            x = layer(x, c)
        return self.final(x, c)

    @torch.no_grad()
    def sample(
        self,
        hidden: torch.Tensor,
        noise: torch.Tensor,
        guidance: torch.Tensor | float | None = None,
        steps: int | None = None,
    ) -> torch.Tensor:
        """LM hidden states ``[B, H]`` and noise ``[B, D]`` -> normalized latents ``[B, D]``."""
        steps = steps or self.config.num_steps
        if guidance is None:
            guidance = self.config.guidance_scale
        if not torch.is_tensor(guidance):
            guidance = torch.full((noise.shape[0],), float(guidance), device=noise.device)
        cond = self.cond_proj(hidden.to(noise.dtype))
        z, h = noise, 1.0 / steps
        for i in range(steps):
            z = z + h * self.velocity(z, cond, i * h, h, guidance)
        return z
