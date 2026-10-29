# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 The FastWAM Authors
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""umT5-XXL encoder with the Wan/FastWAM arithmetic, loading Diffusers-layout weights.

FastWAM conditions on Wan's own T5 encoder implementation. The weights are the
same tensors as ``transformers.UMT5EncoderModel`` in
``Wan-AI/Wan2.2-TI2V-5B-Diffusers/text_encoder`` (checked tensor by tensor), but
the HF forward pass rounds differently in bfloat16 (max |diff| 0.25 on the
FastWAM context of a LIBERO prompt), so the Wan forward is kept and the HF
weights are renamed on load (:func:`hf_umt5_key_to_wan`). With this module the
context matches the reference bitwise on CPU.

The vLLM-Omni ``t5_encoder`` package implements T5 (one shared relative-position
bias), not umT5 (one bias per layer), so it is not a drop-in replacement either.
"""

# Adapted from FastWAM (https://github.com/yuantianyuan01/FastWAM @7faa711,
# src/fastwam/models/wan22/wan_video_text_encoder.py). The original portions carry the following MIT license:
# Copyright (c) 2026 The FastWAM Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import html
import math
from collections.abc import Mapping
from typing import Any

import regex as re
import torch
import torch.nn.functional as F
from torch import nn


class GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class T5LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            x = x.type_as(self.weight)
        return self.weight * x


class T5Attention(nn.Module):
    def __init__(self, dim: int, dim_attn: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads
        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, pos_bias: torch.Tensor) -> torch.Tensor:
        b, n, c = x.size(0), self.num_heads, self.head_dim
        q = self.q(x).view(b, -1, n, c)
        k = self.k(x).view(b, -1, n, c)
        v = self.v(x).view(b, -1, n, c)
        attn_bias = x.new_zeros(b, n, q.size(1), k.size(1))
        attn_bias += pos_bias
        attn_bias.masked_fill_(mask.view(b, 1, 1, -1) == 0, torch.finfo(x.dtype).min)
        # T5 does not scale the logits.
        attn = torch.einsum("binc,bjnc->bnij", q, k) + attn_bias
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        x = torch.einsum("bnij,bjnc->binc", attn, v)
        return self.o(x.reshape(b, -1, n * c))


class T5FeedForward(nn.Module):
    def __init__(self, dim: int, dim_ffn: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim, dim_ffn, bias=False), GELU())
        self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
        self.fc2 = nn.Linear(dim_ffn, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x) * self.gate(x))


class T5RelativeEmbedding(nn.Module):
    def __init__(self, num_buckets: int, num_heads: int, max_dist: int = 128) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.max_dist = max_dist
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def forward(self, lq: int, lk: int) -> torch.Tensor:
        device = self.embedding.weight.device
        rel_pos = torch.arange(lk, device=device).unsqueeze(0) - torch.arange(lq, device=device).unsqueeze(1)
        rel_pos_embeds = self.embedding(self._relative_position_bucket(rel_pos))
        return rel_pos_embeds.permute(2, 0, 1).unsqueeze(0).contiguous()  # [1, N, Lq, Lk]

    def _relative_position_bucket(self, rel_pos: torch.Tensor) -> torch.Tensor:
        # Bidirectional buckets.
        num_buckets = self.num_buckets // 2
        rel_buckets = (rel_pos > 0).long() * num_buckets
        rel_pos = torch.abs(rel_pos)
        max_exact = num_buckets // 2
        rel_pos_large = (
            max_exact
            + (
                torch.log(rel_pos.float() / max_exact) / math.log(self.max_dist / max_exact) * (num_buckets - max_exact)
            ).long()
        )
        rel_pos_large = torch.min(rel_pos_large, torch.full_like(rel_pos_large, num_buckets - 1))
        return rel_buckets + torch.where(rel_pos < max_exact, rel_pos, rel_pos_large)


class T5Block(nn.Module):
    def __init__(
        self, dim: int, dim_attn: int, dim_ffn: int, num_heads: int, num_buckets: int, max_dist: int, eps: float
    ) -> None:
        super().__init__()
        self.norm1 = T5LayerNorm(dim, eps)
        self.attn = T5Attention(dim, dim_attn, num_heads)
        self.norm2 = T5LayerNorm(dim, eps)
        self.ffn = T5FeedForward(dim, dim_ffn)
        self.pos_embedding = T5RelativeEmbedding(num_buckets, num_heads, max_dist)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        e = self.pos_embedding(x.size(1), x.size(1))
        x = x + self.attn(self.norm1(x), mask=mask, pos_bias=e)
        return x + self.ffn(self.norm2(x))


class WanUMT5Encoder(nn.Module):
    """umT5 encoder (per-layer relative position bias), Wan implementation."""

    def __init__(
        self,
        vocab: int = 256384,
        dim: int = 4096,
        dim_attn: int = 4096,
        dim_ffn: int = 10240,
        num_heads: int = 64,
        num_layers: int = 24,
        num_buckets: int = 32,
        max_dist: int = 128,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.token_embedding = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList(
            [T5Block(dim, dim_attn, dim_ffn, num_heads, num_buckets, max_dist, eps) for _ in range(num_layers)]
        )
        self.norm = T5LayerNorm(dim, eps)

    @classmethod
    def from_hf_config(cls, cfg: Mapping[str, Any]) -> WanUMT5Encoder:
        """Build from a ``UMT5EncoderModel`` ``config.json`` (e.g. Wan2.2-Diffusers ``text_encoder/``)."""
        if cfg.get("feed_forward_proj") != "gated-gelu":
            raise ValueError(
                f"Expected a gated-gelu umT5 encoder, got feed_forward_proj={cfg.get('feed_forward_proj')!r}."
            )
        return cls(
            vocab=int(cfg["vocab_size"]),
            dim=int(cfg["d_model"]),
            dim_attn=int(cfg["num_heads"]) * int(cfg["d_kv"]),
            dim_ffn=int(cfg["d_ff"]),
            num_heads=int(cfg["num_heads"]),
            num_layers=int(cfg["num_layers"]),
            num_buckets=int(cfg["relative_attention_num_buckets"]),
            max_dist=int(cfg.get("relative_attention_max_distance", 128)),
            eps=float(cfg.get("layer_norm_epsilon", 1e-6)),
        )

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(ids)
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)


_HF_LAYER_KEYS = {
    "0": {
        "SelfAttention.q.weight": "attn.q.weight",
        "SelfAttention.k.weight": "attn.k.weight",
        "SelfAttention.v.weight": "attn.v.weight",
        "SelfAttention.o.weight": "attn.o.weight",
        "SelfAttention.relative_attention_bias.weight": "pos_embedding.embedding.weight",
        "layer_norm.weight": "norm1.weight",
    },
    "1": {
        "DenseReluDense.wi_0.weight": "ffn.gate.0.weight",
        "DenseReluDense.wi_1.weight": "ffn.fc1.weight",
        "DenseReluDense.wo.weight": "ffn.fc2.weight",
        "layer_norm.weight": "norm2.weight",
    },
}
_HF_BLOCK_RE = re.compile(r"(?:encoder\.)?block\.(\d+)\.layer\.([01])\.(.+)")


def hf_umt5_key_to_wan(name: str) -> str | None:
    """Map a ``UMT5EncoderModel`` weight name to :class:`WanUMT5Encoder`; ``None`` for unused keys."""
    if name in ("shared.weight", "encoder.embed_tokens.weight"):
        return "token_embedding.weight"
    if name == "encoder.final_layer_norm.weight":
        return "norm.weight"
    match = _HF_BLOCK_RE.fullmatch(name)
    if match is None:
        return None
    index, layer, rest = match.groups()
    target = _HF_LAYER_KEYS[layer].get(rest)
    return None if target is None else f"blocks.{index}.{target}"


def clean_prompt(text: str) -> str:
    """Wan's ``clean='whitespace'`` tokenizer preprocessing (without ftfy, as in the
    reference environment; LIBERO prompts are plain ASCII either way)."""
    text = html.unescape(html.unescape(text)).strip()
    return re.sub(r"\s+", " ", text).strip()
