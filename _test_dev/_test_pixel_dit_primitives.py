# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-identical parity for the 3 pixel-DiT primitives vs inline upstream reference.

Usage:
    python _test_dev/_test_pixel_dit_primitives.py
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from vllm_omni.diffusion.models.hidream_o1_image.hidream_o1_image_transformer import (
    BottleneckPatchEmbed,
    FinalLayer,
    TimestepEmbedder,
)

# from https://github.com/HiDream-ai/HiDream-O1-Image/blob/main/models/qwen3_vl_transformers.py
class Upstream_BottleneckPatchEmbed(nn.Module):
    def __init__(self, config, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True):
        super().__init__()
        self.proj1 = nn.Linear(patch_size*patch_size*in_chans, pca_dim, bias=False)
        self.proj2 = nn.Linear(pca_dim, embed_dim, bias=bias)
        self.initialize_weights()

    def initialize_weights(self):
        w1 = self.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.proj2.bias, 0)

    def forward(self, x):
        x = self.proj2(self.proj1(x))
        return x


class Upstream_FinalLayer(nn.Module):
    def __init__(self, config, hidden_size, patch_size, out_channels):
        super().__init__()
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, adaln_input=None):
        x = self.linear(x)
        return x


class Upstream_TimestepEmbedder(nn.Module):
    def __init__(self, config, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t * 1000, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb
# --- end upstream ref ---

torch.manual_seed(0)
x_emb_impl = BottleneckPatchEmbed(config=None, patch_size=16, in_chans=3, pca_dim=768, embed_dim=4096, bias=True)
x_emb_upstream = Upstream_BottleneckPatchEmbed(config=None, patch_size=16, in_chans=3, pca_dim=768, embed_dim=4096, bias=True)
x_emb_upstream.load_state_dict(x_emb_impl.state_dict())

torch.manual_seed(1)
final_impl = FinalLayer(config=None, hidden_size=4096, patch_size=16, out_channels=3)
final_upstream = Upstream_FinalLayer(config=None, hidden_size=4096, patch_size=16, out_channels=3)
final_upstream.load_state_dict(final_impl.state_dict())

torch.manual_seed(2)
t_emb_impl = TimestepEmbedder(config=None, hidden_size=4096, frequency_embedding_size=256)
t_emb_upstream = Upstream_TimestepEmbedder(config=None, hidden_size=4096, frequency_embedding_size=256)
t_emb_upstream.load_state_dict(t_emb_impl.state_dict())

n_x = sum(p.numel() for p in x_emb_impl.parameters())
n_f = sum(p.numel() for p in final_impl.parameters())
n_t = sum(p.numel() for p in t_emb_impl.parameters())
print(f'BottleneckPatchEmbed params: {n_x:,}')
print(f'FinalLayer          params: {n_f:,}')
print(f'TimestepEmbedder    params: {n_t:,}')

torch.manual_seed(42)
x_in = torch.randn(4, 16 * 16 * 3)
h_in = torch.randn(4, 4096)
t_in = torch.tensor([0.0, 0.5, 1.0, 0.25])

with torch.no_grad():
    x_out_impl = x_emb_impl(x_in)
    x_out_upstream = x_emb_upstream(x_in)
    p_out_impl = final_impl(h_in)
    p_out_upstream = final_upstream(h_in)
    t_out_impl = t_emb_impl(t_in)
    t_out_upstream = t_emb_upstream(t_in)

max_abs_x = (x_out_impl - x_out_upstream).abs().max().item()
max_abs_p = (p_out_impl - p_out_upstream).abs().max().item()
max_abs_t = (t_out_impl - t_out_upstream).abs().max().item()
print(f'x_embedder    forward parity: shape={tuple(x_out_impl.shape)} max|impl - upstream| = {max_abs_x}')
print(f'final_layer2  forward parity: shape={tuple(p_out_impl.shape)} max|impl - upstream| = {max_abs_p}')
print(f't_embedder1   forward parity: shape={tuple(t_out_impl.shape)} max|impl - upstream| = {max_abs_t}')

TOL = 1e-5

assert n_x == 3_739_648, f'BottleneckPatchEmbed params: {n_x:,}'
assert n_f == 3_146_496, f'FinalLayer params: {n_f:,}'
assert n_t == 17_833_984, f'TimestepEmbedder params: {n_t:,}'
assert max_abs_x < TOL, f'BottleneckPatchEmbed forward divergence: {max_abs_x} (tol = {TOL})'
assert max_abs_p < TOL, f'FinalLayer forward divergence: {max_abs_p} (tol = {TOL})'
assert max_abs_t < TOL, f'TimestepEmbedder forward divergence: {max_abs_t} (tol = {TOL})'

print(f'pass (tol = {TOL})')


# output:
# BottleneckPatchEmbed params: 3,739,648
# FinalLayer          params: 3,146,496
# TimestepEmbedder    params: 17,833,984
# x_embedder    forward parity: shape=(4, 4096) max|impl - upstream| = 0.0
# final_layer2  forward parity: shape=(4, 768) max|impl - upstream| = 0.0
# t_embedder1   forward parity: shape=(4, 4096) max|impl - upstream| = 0.0
# pass (tol = 1e-05)
