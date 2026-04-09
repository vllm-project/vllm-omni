# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from NextStep-1.1 (https://huggingface.co/stepfun-ai/NextStep-1.1)
# Original: models/heads.py — FlowMatchingHead and components.
# No TP needed: the FM head is tiny (dim=1536, 12 layers).

from __future__ import annotations

import math

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Utilities (inlined from remote utils/model_utils.py)
# ---------------------------------------------------------------------------


def expand_t(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    dims = [1] * (len(x.size()) - 1)
    return t.view(t.size(0), *dims)


def randn_tensor(
    shape: tuple[int, ...],
    noise_repeat: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    bsz = shape[0]
    if bsz % noise_repeat != 0:
        raise ValueError(f"Batch size ({bsz}) must be divisible by noise repeat ({noise_repeat})")
    _shape = (noise_repeat,) + shape[1:]
    _tensor = torch.randn(_shape, device=device, dtype=dtype).repeat(bsz // noise_repeat, 1)
    return _tensor


# ---------------------------------------------------------------------------
# Adaptive LayerNorm modulation
# ---------------------------------------------------------------------------


def modulate(
    x: torch.Tensor,
    shift: torch.Tensor | None,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    if shift is None:
        return x * (1 + scale)
    return x * (1 + scale) + shift


# ---------------------------------------------------------------------------
# ResBlock
# ---------------------------------------------------------------------------


class ResBlock(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 1.0):
        super().__init__()
        self.channels = channels
        self.intermediate_size = int(channels * mlp_ratio)
        self.in_ln = nn.LayerNorm(self.channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.channels, self.intermediate_size),
            nn.SiLU(),
            nn.Linear(self.intermediate_size, self.channels),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


# ---------------------------------------------------------------------------
# FinalLayer
# ---------------------------------------------------------------------------


class FinalLayer(nn.Module):
    def __init__(self, model_channels: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 2 * model_channels, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# ---------------------------------------------------------------------------
# TimestepEmbedder
# ---------------------------------------------------------------------------


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))


# ---------------------------------------------------------------------------
# SimpleMLPAdaLN
# ---------------------------------------------------------------------------


class SimpleMLPAdaLN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        dim: int = 1536,
        layers: int = 12,
        mlp_ratio: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.dim = dim

        self.time_embed = TimestepEmbedder(dim)
        self.cond_embed = nn.Linear(cond_dim, dim)
        self.input_proj = nn.Linear(input_dim, dim)

        self.res_blocks = nn.ModuleList([ResBlock(dim, mlp_ratio) for _ in range(layers)])
        self.final_layer = FinalLayer(dim, input_dim)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)
        y = t + c

        for block in self.res_blocks:
            x = block(x, y)

        return self.final_layer(x, y)


# ---------------------------------------------------------------------------
# FlowMatchingHead
# ---------------------------------------------------------------------------


class FlowMatchingHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        dim: int = 1536,
        layers: int = 12,
        mlp_ratio: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.net = SimpleMLPAdaLN(
            input_dim=input_dim,
            cond_dim=cond_dim,
            dim=dim,
            layers=layers,
            mlp_ratio=mlp_ratio,
        )

    @property
    def dtype(self):
        return self.net.input_proj.weight.dtype

    @property
    def device(self):
        return self.net.input_proj.weight.device

    def get_score_from_velocity(
        self,
        velocity: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        t = expand_t(t, x)
        alpha_t, d_alpha_t = t, 1
        sigma_t, d_sigma_t = 1 - t, -1
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        score = (reverse_alpha_ratio * velocity - mean) / var
        return score

    def get_velocity_from_cfg(
        self,
        velocity: torch.Tensor,
        cfg: float,
        cfg_img: float,
        cfg_mult: int,
    ) -> torch.Tensor:
        if cfg_mult == 2:
            cond_v, uncond_v = torch.chunk(velocity, 2, dim=0)
            velocity = uncond_v + cfg * (cond_v - uncond_v)
        elif cfg_mult == 3:
            cond_v, uncond_v1, uncond_v2 = torch.chunk(velocity, 3, dim=0)
            velocity = uncond_v2 + cfg_img * (uncond_v1 - uncond_v2) + cfg * (cond_v - uncond_v1)
        return velocity

    @torch.no_grad()
    def sample(
        self,
        c: torch.Tensor,
        cfg: float = 1.0,
        cfg_img: float = 1.0,
        cfg_mult: int | None = None,
        timesteps_shift: float = 1.0,
        num_sampling_steps: int = 20,
        last_step_size: float = 0.0,
        noise_repeat: int = 1,
    ) -> torch.Tensor:
        if cfg_mult is None:
            cfg_mult = 1
            if cfg > 1.0:
                cfg_mult += 1
            if cfg_img > 1.0:
                cfg_mult += 1

        if cfg_mult <= 0:
            raise ValueError(f"Invalid cfg_mult={cfg_mult}; expected a positive value.")
        if c.shape[0] % cfg_mult != 0:
            raise ValueError(
                f"Invalid CFG layout: condition batch size {c.shape[0]} is not divisible by cfg_mult={cfg_mult}."
            )

        noise = randn_tensor((c.shape[0] // cfg_mult, self.input_dim), noise_repeat, self.device)

        x = noise
        xs = []

        t0, t1 = 0, 1
        timesteps = torch.linspace(t0, t1, num_sampling_steps + 1, device=c.device)[:-1]
        timesteps = timesteps / (timesteps_shift - (timesteps_shift - 1) * timesteps)
        timesteps = torch.cat([timesteps, torch.ones(1, device=c.device)])

        for ti, tj in zip(timesteps[:-1], timesteps[1:]):
            dt = tj - ti

            combined = torch.cat([x] * cfg_mult, dim=0)
            velocity = self.net(combined.to(c.dtype), ti.expand(c.shape[0]).to(c), c)
            velocity = velocity.to(torch.float32)

            velocity = self.get_velocity_from_cfg(velocity, cfg, cfg_img, cfg_mult)
            score = self.get_score_from_velocity(velocity, x, ti.expand(x.shape[0]).to(x))
            drift = velocity + (1 - expand_t(ti.expand(x.shape[0]).to(x), x)) * score

            w_cur = randn_tensor((c.shape[0] // cfg_mult, self.input_dim), noise_repeat, self.device)
            dw = w_cur * torch.sqrt(dt)

            mean_x = x + drift * dt
            x = mean_x + torch.sqrt(2 * (1 - expand_t(ti.expand(x.shape[0]).to(x), x))) * dw
            xs.append(x)

        if len(xs) != num_sampling_steps:
            raise ValueError(f"Samples ({len(xs)}) does not match the number of steps ({num_sampling_steps})")

        return xs[-1].to(c.dtype)
