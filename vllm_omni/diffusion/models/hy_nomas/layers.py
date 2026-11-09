# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from Tencent-Hunyuan/Hy-Nomas@911cea6458 (Copyright 2026 Tencent).

import math

import torch
from torch import nn

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, **kwargs):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, **kwargs))

    def forward(self, x):
        output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return output.type_as(x) * self.weight


class MLP(nn.Module):
    def __init__(self, hidden, intermediate, act=nn.SiLU, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(hidden, intermediate, **kwargs)
        self.act = act()
        self.fc2 = nn.Linear(intermediate, hidden, **kwargs)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden, **kwargs):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(256, hidden, **kwargs), nn.SiLU(), nn.Linear(hidden, hidden, **kwargs))

    def forward(self, t):
        freqs = torch.exp(-math.log(10000) * torch.arange(128, dtype=torch.float32) / 128).to(t.device)
        angles = t.flatten()[:, None].float() * freqs[None]
        values = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1).view(*t.shape, 256)
        return self.mlp(values.to(self.mlp[0].weight.dtype))


class TextProjection(nn.Module):
    def __init__(self, channels, hidden, **kwargs):
        super().__init__()
        self.linear_1 = nn.Linear(channels, hidden, **kwargs)
        self.act_1 = nn.SiLU()
        self.linear_2 = nn.Linear(hidden, hidden, **kwargs)

    def forward(self, x):
        return self.linear_2(self.act_1(self.linear_1(x)))


class PatchEmbed(nn.Module):
    def __init__(self, channels, hidden, **kwargs):
        super().__init__()
        self.proj = nn.Conv3d(channels, hidden, 1, stride=1, **kwargs)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class ModulateDiT(nn.Module):
    def __init__(self, hidden, **kwargs):
        super().__init__()
        self.act = nn.SiLU()
        self.linear = nn.Linear(hidden, 6 * hidden, **kwargs)

    def forward(self, x):
        return self.linear(self.act(x))


class FinalLayer(nn.Module):
    def __init__(self, hidden, channels, **kwargs):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6, **kwargs)
        self.linear = nn.Linear(hidden, channels, **kwargs)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden, **kwargs))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class TokenRefinerBlock(nn.Module):
    def __init__(self, hidden, heads, **kwargs):
        super().__init__()
        self.heads_num = heads
        self.norm1 = nn.LayerNorm(hidden, eps=1e-6, **kwargs)
        self.self_attn_qkv = nn.Linear(hidden, 3 * hidden, **kwargs)
        self.self_attn_proj = nn.Linear(hidden, hidden, **kwargs)
        self.norm2 = nn.LayerNorm(hidden, eps=1e-6, **kwargs)
        self.mlp = MLP(hidden, 4 * hidden, **kwargs)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden, **kwargs))
        attention_impl = FlashAttentionImpl if torch.device(kwargs.get("device") or "cpu").type == "cuda" else SDPAImpl
        self.attn = attention_impl(
            num_heads=heads,
            head_size=hidden // heads,
            causal=False,
            softmax_scale=(hidden // heads) ** -0.5,
            qkv_layout="BSND",
        )

    def forward(self, x, c, mask):
        gate_attention, gate_mlp = self.adaLN_modulation(c).chunk(2, dim=1)
        batch, length, hidden = x.shape
        q, k, v = self.self_attn_qkv(self.norm1(x)).view(batch, length, 3, self.heads_num, -1).unbind(2)
        output = self.attn.forward(q, k, v, AttentionMetadata(attn_mask=mask)).reshape(batch, length, hidden)
        output = output.masked_fill(~mask[:, :, None], 0)
        x = x + self.self_attn_proj(output) * gate_attention.unsqueeze(1)
        return x + self.mlp(self.norm2(x)) * gate_mlp.unsqueeze(1)


class IndividualTokenRefiner(nn.Module):
    def __init__(self, hidden, heads, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList(TokenRefinerBlock(hidden, heads, **kwargs) for _ in range(2))

    def forward(self, x, c, mask):
        mask = mask.clone().bool()
        mask[:, 0] = True
        for block in self.blocks:
            x = block(x, c, mask)
        return x


class SingleTokenRefiner(nn.Module):
    def __init__(self, channels, hidden, heads, **kwargs):
        super().__init__()
        self.input_embedder = nn.Linear(channels, hidden, **kwargs)
        self.t_embedder = TimestepEmbedder(hidden, **kwargs)
        self.c_embedder = TextProjection(channels, hidden, **kwargs)
        self.individual_token_refiner = IndividualTokenRefiner(hidden, heads, **kwargs)

    def forward(self, x, t, mask):
        weights = mask.to(x.dtype).unsqueeze(-1)
        context = (x * weights).sum(1) / weights.sum(1)
        condition = self.t_embedder(t) + self.c_embedder(context)
        return self.individual_token_refiner(self.input_embedder(x), condition, mask)


def rotary_tables(sizes, dimensions, theta, *, time_offset=0, device=None):
    # Preserve device-local first-segment tables and CPU offset tables thereafter.
    start = (time_offset, 0, 0)
    axes = [torch.linspace(a, a + n, n + 1, dtype=torch.float32, device=device)[:n] for a, n in zip(start, sizes)]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0)
    cosines, sines = [], []
    for axis, dim in zip(grid, dimensions):
        frequencies = 1.0 / (theta ** (torch.arange(0, dim, 2, device=axis.device).float() / dim))
        angles = torch.outer(axis.reshape(-1), frequencies)
        cosines.append(angles.cos().repeat_interleave(2, dim=1))
        sines.append(angles.sin().repeat_interleave(2, dim=1))
    return torch.cat(cosines, dim=1), torch.cat(sines, dim=1)


def apply_rotary(x, tables):
    cos, sin = (part.to(x.device).view(1, x.shape[1], 1, x.shape[-1]) for part in tables)
    real, imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)
    rotated = torch.stack((-imag, real), dim=-1).flatten(3)
    return (x.float() * cos + rotated * sin).type_as(x)
