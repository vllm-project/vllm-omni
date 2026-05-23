# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vendored DiT model from FunCineForge.

Adapted from ``funcineforge/models/modules/dit_flow_matching/dit_model.py``
with the following changes:
  - Replaced ``x_transformers.RotaryEmbedding`` with a self-contained
    implementation to remove the ``x_transformers`` dependency.
  - Replaced ``funcineforge.models.utils.masks.causal_block_mask`` with a
    self-contained implementation.
  - Replaced ``einops.repeat`` with plain ``torch.expand``.
  - Removed ``funcineforge.register`` decorators.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from vllm_omni.model_executor.models.funcineforge.vendor.dit_modules import (
    AdaLayerNormZero_Final,
    CausalConvPositionEmbedding,
    ConvNeXtV2Block,
    DiTBlock,
    TimestepEmbedding,
    get_pos_embed_indices,
    precompute_freqs_cis,
)

# ---------------------------------------------------------------------------
# Self-contained RotaryEmbedding (replaces x_transformers.RotaryEmbedding)
# ---------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    """Rotary position embedding compatible with FunCineForge's DiT."""

    def __init__(self, dim, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self._precomputed_len = 0
        self._freqs = None

    def _ensure_precomputed(self, seq_len: int, device: torch.device):
        if seq_len <= self._precomputed_len and self._freqs is not None and self._freqs.device == device:
            return
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
        positions = torch.arange(seq_len, device=device)
        self._freqs = torch.outer(positions, freqs).repeat_interleave(2, dim=-1).unsqueeze(0)
        self._precomputed_len = seq_len

    def forward_from_seq_len(self, seq_len: int, device: torch.device | None = None):
        """Return (freqs, xpos_scale) tuple for the given sequence length.

        The ``xpos_scale`` is set to ``None`` since FunCineForge's DiT
        doesn't use xpos scaling.
        """
        if device is None:
            device = self._freqs.device if self._freqs is not None else torch.device("cpu")
        self._ensure_precomputed(seq_len, device)
        freqs = self._freqs[:, :seq_len]
        return (freqs, None)


# ---------------------------------------------------------------------------
# Self-contained causal block mask
# ---------------------------------------------------------------------------


def causal_block_mask(seq_len: int, block_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create a causal block mask of shape (seq_len, seq_len).

    Each block of ``block_size`` tokens can attend to all tokens within
    the same block and all preceding blocks.
    """
    if block_size <= 0:
        return torch.ones(seq_len, seq_len, device=device, dtype=dtype)
    n_blocks = (seq_len + block_size - 1) // block_size
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    for i in range(n_blocks):
        start = i * block_size
        end = min((i + 1) * block_size, seq_len)
        # Block i can attend to blocks 0..i
        mask[start:end, :end] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Text embedding
# ---------------------------------------------------------------------------


class TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds, text_dim, conv_layers=0, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, text, seq_len, drop_text=False):
        batch, text_len = text.shape[0], text.shape[1]
        text = text + 1  # use 0 as filler token
        text = text[:, :seq_len]
        text = F.pad(text, (0, max(0, seq_len - text_len)), value=0)

        if drop_text:
            text = torch.zeros_like(text)

        text = self.text_embed(text)

        if self.extra_modeling:
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed
            text = self.text_blocks(text)

        return text


# ---------------------------------------------------------------------------
# Input embedding
# ---------------------------------------------------------------------------


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim, spk_dim=None):
        super().__init__()
        spk_dim = 0 if spk_dim is None else spk_dim
        self.spk_dim = spk_dim
        self.proj = nn.Linear(mel_dim * 2 + text_dim + spk_dim, out_dim)
        self.conv_pos_embed = CausalConvPositionEmbedding(dim=out_dim)

    def forward(self, x, cond, text_embed, spks):
        to_cat = [x, cond, text_embed]
        if self.spk_dim > 0:
            spks = spks.unsqueeze(1).expand(-1, x.shape[1], -1)  # b d -> b t d
            to_cat.append(spks)
        x = self.proj(torch.cat(to_cat, dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


# ---------------------------------------------------------------------------
# DiT
# ---------------------------------------------------------------------------


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=80,
        mu_dim=None,
        long_skip_connection=False,
        spk_dim=None,
        **kwargs,
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if mu_dim is None:
            mu_dim = mel_dim
        self.input_embed = InputEmbedding(mel_dim, mu_dim, dim, spk_dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout) for _ in range(depth)]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNormZero_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)
        self.causal_mask_type = kwargs.get("causal_mask_type", None)

    def build_mix_causal_mask(self, attn_mask, rand=None, ratio=None):
        b, _, _, t = attn_mask.shape
        if rand is None:
            rand = torch.rand((b, 1, 1, 1), device=attn_mask.device, dtype=torch.float32)
        mixed_mask = attn_mask.clone()
        for item in self.causal_mask_type:
            prob_min, prob_max = item["prob_min"], item["prob_max"]
            _ratio = 1
            if "ratio" in item:
                _ratio = item["ratio"]
            if ratio is not None:
                _ratio = ratio
            block_size = item["block_size"] * _ratio
            if block_size <= 0:
                causal_mask = attn_mask
            else:
                causal_mask = (
                    causal_block_mask(t, block_size, attn_mask.device, torch.float32).unsqueeze(0).unsqueeze(1)
                )
            flag = (prob_min <= rand) & (rand < prob_max)
            mixed_mask = mixed_mask * (~flag) + (causal_mask * attn_mask) * flag
        return mixed_mask

    def forward(
        self,
        x,  # noised input audio
        cond,  # masked cond audio
        mu,  # mu (codec embedding)
        spks,  # spk xvec
        time,  # time step
        return_hidden: bool = False,
        mask=None,
        mask_rand=None,
        **kwargs,
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        t = self.time_embed(time)
        x = self.input_embed(x, cond, mu, spks.squeeze(1))

        rope = self.rotary_embed.forward_from_seq_len(seq_len, device=x.device)

        if self.long_skip_connection is not None:
            residual = x

        mask = mask.unsqueeze(1)  # B,1,1,T
        if self.causal_mask_type is not None:
            mask = self.build_mix_causal_mask(mask, rand=mask_rand.unsqueeze(-1))

        for block in self.transformer_blocks:
            x = x * mask[:, 0, -1, :].unsqueeze(-1)
            x = block(x, t, mask=mask.bool(), rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        if return_hidden:
            return output, None

        return output
