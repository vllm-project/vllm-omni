# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import html
import math
import os
import re
from typing import Any

import ftfy
import torch
import torch.nn.functional as F
from torch import nn

from vllm_omni.diffusion.models.nava.config import NAVAConfig


class _GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3.0))))


class _T5LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            y = y.type_as(self.weight)
        return self.weight * y


class _T5Attention(nn.Module):
    def __init__(self, dim: int, dim_attn: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if dim_attn % num_heads != 0:
            raise ValueError("T5 attention dimension must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads
        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        pos_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, heads, head_dim = x.size(0), self.num_heads, self.head_dim
        q = self.q(x).view(bsz, -1, heads, head_dim)
        k = self.k(x).view(bsz, -1, heads, head_dim)
        v = self.v(x).view(bsz, -1, heads, head_dim)

        attn_bias = x.new_zeros(bsz, heads, q.size(1), k.size(1))
        if pos_bias is not None:
            attn_bias = attn_bias + pos_bias
        if mask is not None:
            if mask.ndim not in (2, 3):
                raise ValueError(f"NAVA T5 attention mask must be 2D or 3D, got {mask.ndim}D.")
            expanded_mask = mask.view(bsz, 1, 1, -1) if mask.ndim == 2 else mask.unsqueeze(1)
            attn_bias.masked_fill_(expanded_mask == 0, torch.finfo(x.dtype).min)

        attn = torch.einsum("binc,bjnc->bnij", q, k) + attn_bias
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        y = torch.einsum("bnij,bjnc->binc", attn, v)
        y = y.reshape(bsz, -1, heads * head_dim)
        return self.dropout(self.o(y))


class _T5FeedForward(nn.Module):
    def __init__(self, dim: int, dim_ffn: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim, dim_ffn, bias=False), _GELU())
        self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
        self.fc2 = nn.Linear(dim_ffn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc1(x) * self.gate(x)
        y = self.dropout(y)
        y = self.fc2(y)
        return self.dropout(y)


class _T5RelativeEmbedding(nn.Module):
    def __init__(self, num_buckets: int, num_heads: int, *, bidirectional: bool, max_dist: int = 128) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.max_dist = max_dist
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def forward(self, lq: int, lk: int) -> torch.Tensor:
        device = self.embedding.weight.device
        rel_pos = torch.arange(lk, device=device).unsqueeze(0) - torch.arange(lq, device=device).unsqueeze(1)
        rel_buckets = self._relative_position_bucket(rel_pos)
        rel_pos_embeds = self.embedding(rel_buckets)
        return rel_pos_embeds.permute(2, 0, 1).unsqueeze(0).contiguous()

    def _relative_position_bucket(self, rel_pos: torch.Tensor) -> torch.Tensor:
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            rel_buckets = (rel_pos > 0).long() * num_buckets
            rel_pos = torch.abs(rel_pos)
        else:
            num_buckets = self.num_buckets
            rel_buckets = 0
            rel_pos = -torch.min(rel_pos, torch.zeros_like(rel_pos))

        max_exact = num_buckets // 2
        rel_pos_large = (
            max_exact
            + (
                torch.log(rel_pos.float() / max_exact) / math.log(self.max_dist / max_exact) * (num_buckets - max_exact)
            ).long()
        )
        rel_pos_large = torch.min(rel_pos_large, torch.full_like(rel_pos_large, num_buckets - 1))
        return rel_buckets + torch.where(rel_pos < max_exact, rel_pos, rel_pos_large)


class _T5SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_attn: int,
        dim_ffn: int,
        num_heads: int,
        num_buckets: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = _T5LayerNorm(dim)
        self.attn = _T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = _T5LayerNorm(dim)
        self.ffn = _T5FeedForward(dim, dim_ffn, dropout)
        self.pos_embedding = _T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        pos_bias = self.pos_embedding(x.size(1), x.size(1))
        x = x + self.attn(self.norm1(x), mask=mask, pos_bias=pos_bias)
        if x.dtype == torch.float16 and torch.isinf(x).any():
            clamp = torch.finfo(x.dtype).max - 1000
            x = torch.clamp(x, min=-clamp, max=clamp)
        x = x + self.ffn(self.norm2(x))
        if x.dtype == torch.float16 and torch.isinf(x).any():
            clamp = torch.finfo(x.dtype).max - 1000
            x = torch.clamp(x, min=-clamp, max=clamp)
        return x


class _T5Encoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        dim: int,
        dim_attn: int,
        dim_ffn: int,
        num_heads: int,
        num_layers: int,
        num_buckets: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [_T5SelfAttentionBlock(dim, dim_attn, dim_ffn, num_heads, num_buckets, dropout) for _ in range(num_layers)]
        )
        self.norm = _T5LayerNorm(dim)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.dropout(self.token_embedding(ids))
        for block in self.blocks:
            x = block(x, mask)
        return self.dropout(self.norm(x))


class NAVAWanTextEncoder(nn.Module):
    def __init__(
        self,
        model_root: str,
        config: NAVAConfig,
        device: torch.device,
        *,
        compile_model: bool = False,
    ) -> None:
        super().__init__()
        from transformers import AutoTokenizer

        self.config = config
        self.device = device
        wan_root = os.path.join(model_root, config.wan_dir)
        tokenizer_path = os.path.join(wan_root, "google", "umt5-xxl")
        checkpoint_path = os.path.join(wan_root, "models_t5_umt5-xxl-enc-bf16.pth")
        if not os.path.isdir(tokenizer_path):
            raise FileNotFoundError(f"NAVA text tokenizer directory not found: {tokenizer_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"NAVA text encoder checkpoint not found: {checkpoint_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        self._ensure_tokenizer_padding(self.tokenizer)
        self.spk_token_id = int(self.tokenizer("<extra_id_2>", return_tensors="pt").input_ids[0, 0].item())
        self.model = _T5Encoder(
            vocab_size=256384,
            dim=config.text_embed_dim,
            dim_attn=config.text_embed_dim,
            dim_ffn=10240,
            num_heads=64,
            num_layers=24,
            num_buckets=32,
        )
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict, assign=True)
        self.model = self.model.to(device=device, dtype=config.target_dtype).eval()
        self.model.requires_grad_(False)
        if compile_model:
            self.model = torch.compile(self.model)

    @staticmethod
    def _ensure_tokenizer_padding(tokenizer: Any) -> None:
        if tokenizer.pad_token is not None:
            return
        fallback_pad_token = tokenizer.eos_token or tokenizer.unk_token
        if fallback_pad_token is not None:
            tokenizer.pad_token = fallback_pad_token
        else:
            tokenizer.pad_token_id = 0

    @torch.inference_mode()
    def encode(
        self,
        texts: list[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
        return_speaker_positions: bool = False,
    ) -> list[torch.Tensor] | tuple[list[torch.Tensor], list[list[int]]]:
        cleaned = [self._clean_whitespace(text) for text in texts]
        inputs = self.tokenizer(
            cleaned,
            padding="max_length",
            truncation=True,
            max_length=self.config.text_len,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = inputs.input_ids.to(device)
        mask = inputs.attention_mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.model(input_ids, mask).to(dtype=dtype)
        embeds = [item[: int(length.item())] for item, length in zip(context, seq_lens)]
        if return_speaker_positions:
            speaker_positions = [
                [
                    int(pos)
                    for pos in (row[: int(length.item())] == self.spk_token_id).nonzero(as_tuple=True)[0].tolist()
                ]
                for row, length in zip(input_ids, seq_lens)
            ]
            return embeds, speaker_positions
        return embeds

    @staticmethod
    def _clean_whitespace(text: str) -> str:
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return re.sub(r"\s+", " ", text).strip()
