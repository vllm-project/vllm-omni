"""Vendored TADA codec building blocks.

Adapted from HumeAI's ``hume-tada`` package (subset of ``tada/modules/encoder.py``
plus a local ``Snake1d``), which is MIT-licensed. Only the layers needed by the
codec *decoder* are vendored here; the audio encoder / aligner are not. Class and
parameter names are kept identical so pretrained checkpoints load without remapping.

SPDX-License-Identifier: MIT
Copyright (c) Hume AI.
"""

import math
from typing import Literal

import torch
import torch.nn as nn


class Snake1d(nn.Module):
    """Snake activation: x + (1/alpha) * sin(alpha * x) ** 2.

    Parameter ``alpha`` has shape ``[1, channels, 1]`` to match the DAC/tada-codec
    checkpoints (identical to ``transformers.models.dac.modeling_dac.Snake1d``).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        shape = x.shape
        x = x.reshape(shape[0], shape[1], -1)
        x = x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).pow(2)
        x = x.reshape(shape)
        return x


def WNConv1d(*args, **kwargs):
    return torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return torch.nn.utils.parametrizations.weight_norm(torch.nn.ConvTranspose1d(*args, **kwargs))


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


def _create_segment_attention_mask(
    text_token_mask: torch.Tensor, version: Literal["v1", "v2", "decoder_block_attention"] = "v1"
) -> torch.Tensor:
    """Create an attention mask based on block boundaries marked in text_token_mask.

    Only the ``v2`` path is exercised by the decoder; ``v1`` is kept for parity.
    Returns a boolean mask where True means *masked* (cannot attend).
    """
    if version == "v1":
        block_ids = torch.cumsum(text_token_mask, dim=1) - text_token_mask
        block_ids_i = block_ids.unsqueeze(2)
        block_ids_j = block_ids.unsqueeze(1)
        same_block = block_ids_j == block_ids_i
        block_ids_j_excluding_last = torch.where(text_token_mask.bool(), -10, block_ids_j[:, 0, :]).unsqueeze(1)
        next_block = block_ids_j_excluding_last == (block_ids_i + 1)
        can_attend = same_block | next_block
        return ~can_attend
    elif version == "v2":
        block_ids = torch.cumsum(text_token_mask, dim=1)
        block_ids_i = block_ids.unsqueeze(2)
        block_ids_j = block_ids.unsqueeze(1)
        same_block = block_ids_i == block_ids_j
        is_marked_i = text_token_mask.unsqueeze(2).bool()
        is_marked_j = text_token_mask.unsqueeze(1).bool()
        same_block_valid = same_block & (~is_marked_j | (is_marked_i & same_block))
        prev_block = block_ids_j == (block_ids_i - 1)
        prev_block_valid = prev_block & ~is_marked_j
        can_attend = same_block_valid | (is_marked_i & prev_block_valid)
        return ~can_attend
    else:
        raise ValueError(f"Unknown version: {version}")


class LocalSelfAttention(nn.Module):
    """Local self-attention with RoPE (rotary position embeddings)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
        causal: bool = False,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal
        self.use_flash_attn = use_flash_attn

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.register_buffer("rope_freqs", self._compute_rope_freqs(self.head_dim, max_seq_len))

        self.max_seq_len = max_seq_len
        self.register_buffer("_precomputed_mask", self._create_full_mask(max_seq_len))

    def _create_full_mask(self, max_seq_len: int) -> torch.Tensor:
        positions = torch.arange(max_seq_len).unsqueeze(0)
        positions_t = positions.transpose(0, 1)
        distances = positions - positions_t
        # NOTE: kept verbatim from upstream; this is a no-op mask (abs() >= 0 is
        # always True). The real masking comes from the external block mask passed
        # into forward(). Do not "fix" this — it would diverge from trained weights.
        mask = distances.abs() >= 0
        return mask

    def _compute_rope_freqs(self, head_dim: int, max_seq_len: int) -> torch.Tensor:
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        freqs_cos = freqs.cos()
        freqs_sin = freqs.sin()
        return torch.stack([freqs_cos, freqs_sin], dim=-1)

    def _apply_rope(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        batch, num_heads, seq_len, head_dim = x.shape
        freqs = self.rope_freqs[:seq_len]
        freqs_cos = freqs[..., 0]
        freqs_sin = freqs[..., 1]
        x_reshaped = x.reshape(batch, num_heads, seq_len, head_dim // 2, 2)
        x0 = x_reshaped[..., 0]
        x1 = x_reshaped[..., 1]
        x_rotated_0 = x0 * freqs_cos.unsqueeze(0).unsqueeze(0) - x1 * freqs_sin.unsqueeze(0).unsqueeze(0)
        x_rotated_1 = x0 * freqs_sin.unsqueeze(0).unsqueeze(0) + x1 * freqs_cos.unsqueeze(0).unsqueeze(0)
        x_rotated = torch.stack([x_rotated_0, x_rotated_1], dim=-1)
        return x_rotated.reshape(batch, num_heads, seq_len, head_dim)

    def create_local_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return self._precomputed_mask[:seq_len, :seq_len]

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self._apply_rope(q, seq_len)
        k = self._apply_rope(k, seq_len)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is None:
            local_mask = self.create_local_mask(seq_len, x.device)
            attn_scores = attn_scores.masked_fill(local_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        else:
            if mask.dim() == 2:
                attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            elif mask.dim() == 3:
                attn_scores = attn_scores.masked_fill(mask.unsqueeze(1), float("-inf"))
            else:
                raise ValueError(f"Mask should have 2 or 3 dimensions, got {mask.dim()}")

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, d_model)

        output = self.out_proj(attn_output)
        output = self.dropout(output)

        output = self.layer_norm(x + output)
        return output


class LocalAttentionEncoderLayer(nn.Module):
    """Transformer encoder layer with local self-attention and feed-forward network."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: int = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_seq_len: int = 8192,
        window_size: int = 1500,
        causal: bool = False,
        use_flash_attn: bool = True,
    ):
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model

        self.self_attn = LocalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            causal=causal,
            use_flash_attn=use_flash_attn,
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.self_attn(x, mask=mask)
        x = self.norm(x + self.ffn(x))
        return x


class LocalAttentionEncoder(nn.Module):
    """Stack of local attention encoder layers."""

    def __init__(
        self,
        d_model: int,
        d_input: int | None = None,
        num_layers: int = 4,
        num_heads: int = 8,
        d_ff: int | None = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_seq_len: int = 8192,
        causal: bool = False,
        use_flash_attn: bool = True,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                LocalAttentionEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                    max_seq_len=max_seq_len,
                    causal=causal,
                    use_flash_attn=use_flash_attn,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)

        if d_input is not None and d_input != d_model:
            self.input_proj: nn.Module = nn.Linear(d_input, d_model)
        else:
            self.input_proj = nn.Identity()

    def _forward_window(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.final_norm(x)

    def _forward_sliding_window(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        inference_window_size: int | None = None,
        inference_window_stride: int | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        window_size = int(inference_window_size * 50)
        stride = int(inference_window_stride * 50)

        if seq_len <= window_size:
            return self._forward_window(x, mask)

        output = torch.zeros(batch_size, seq_len, d_model, device=x.device, dtype=x.dtype)
        prev_end_idx = 0
        for start_idx in range(0, seq_len, stride):
            end_idx = min(start_idx + window_size, seq_len)
            x_window = x[:, start_idx:end_idx, :]

            mask_window = None
            if mask is not None:
                if mask.dim() == 2:
                    mask_window = mask[start_idx:end_idx, start_idx:end_idx]
                elif mask.dim() == 3:
                    mask_window = mask[:, start_idx:end_idx, start_idx:end_idx]

            output_window = self._forward_window(x_window, mask_window)

            if start_idx == 0:
                output[:, start_idx:end_idx, :] = output_window
            else:
                overlap_start = start_idx
                overlap_end = min(prev_end_idx, end_idx)
                overlap_length = overlap_end - overlap_start
                if overlap_length > 0:
                    mid_point = overlap_start + overlap_length // 2
                    window_offset = mid_point - start_idx
                    output[:, mid_point:overlap_end, :] = output_window[:, window_offset:overlap_length, :]
                if overlap_end < end_idx:
                    window_offset = overlap_end - start_idx
                    output[:, overlap_end:end_idx, :] = output_window[:, window_offset:, :]

            prev_end_idx = end_idx
            if end_idx >= seq_len:
                break

        return output

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        inference_window_size: int | None = None,
        inference_window_stride: int | None = None,
    ) -> torch.Tensor:
        if inference_window_size is not None and not self.training:
            return self._forward_sliding_window(x, mask, inference_window_size, inference_window_stride)
        else:
            return self._forward_window(x, mask)
