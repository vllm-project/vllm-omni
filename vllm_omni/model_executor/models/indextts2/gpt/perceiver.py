# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from lucidrains/naturalspeech2-pytorch  (PerceiverResampler)
# https://github.com/lucidrains/naturalspeech2-pytorch/blob/659bec7f/
#   naturalspeech2_pytorch/naturalspeech2_pytorch.py#L532

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class Attend(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = dropout

    def forward(self, q, k, v, mask=None):
        attn_mask = None
        if exists(mask):
            attn_mask = rearrange(mask, "b j -> b 1 1 j")

        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )


class RMSNorm(nn.Module):
    def __init__(self, dim, scale=True):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if scale else None

    def forward(self, x):
        gamma = default(self.gamma, 1)
        return F.normalize(x, dim=-1) * self.scale * gamma


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def FeedForward(dim, mult=4):
    dim_inner = int(dim * mult * 2 / 3)
    return nn.Sequential(nn.Linear(dim, dim_inner * 2), GEGLU(), nn.Linear(dim_inner, dim))


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        dim,
        depth=2,
        dim_context=None,
        num_latents=32,
        dim_head=64,
        heads=8,
        ff_mult=4,
    ):
        super().__init__()
        dim_context = default(dim_context, dim)

        self.proj_context = nn.Linear(dim_context, dim) if dim_context != dim else nn.Identity()

        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        nn.init.normal_(self.latents, std=0.02)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim=dim, dim_head=dim_head, heads=heads, cross_attn_include_queries=True),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = RMSNorm(dim)

    def forward(self, x, mask=None):
        batch = x.shape[0]

        x = self.proj_context(x)

        latents = repeat(self.latents, "n d -> b n d", b=batch)

        for attn, ff in self.layers:
            latents = attn(latents, x, mask=mask) + latents
            latents = ff(latents) + latents

        return self.norm(latents)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_context=None,
        dim_head=64,
        heads=8,
        dropout=0.0,
        cross_attn_include_queries=False,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.cross_attn_include_queries = cross_attn_include_queries

        dim_inner = dim_head * heads
        dim_context = default(dim_context, dim)

        self.attend = Attend(dropout=dropout)
        self.to_q = nn.Linear(dim, dim_inner, bias=False)
        self.to_kv = nn.Linear(dim_context, dim_inner * 2, bias=False)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)

    def forward(self, x, context=None, mask=None):
        h, has_context = self.heads, exists(context)

        context = default(context, x)

        if has_context and self.cross_attn_include_queries:
            context = torch.cat((x, context), dim=-2)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        out = self.attend(q, k, v, mask=mask)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)
