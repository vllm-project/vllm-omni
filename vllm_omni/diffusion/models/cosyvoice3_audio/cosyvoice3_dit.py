# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/FunAudioLLM/CosyVoice/tree/main/cosyvoice/flow/DiT
# Refactored to use vllm_omni diffusion infrastructure for optimized attention backends

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
from diffusers.models.normalization import AdaLayerNormZero
from einops import repeat
from torch import nn
from vllm.logger import init_logger
from x_transformers.x_transformers import RotaryEmbedding, apply_rotary_pos_emb

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention as DiffusionAttention
from vllm_omni.model_executor.layers.timestep_embedding import DiTTimestepEmbedding

logger = init_logger(__name__)

"""
in notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, theta_rescale_factor=1.0):
    """Precompute rotary embedding frequencies."""
    theta *= theta_rescale_factor ** (dim / (dim - 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return torch.cat([freqs_cos, freqs_sin], dim=-1)


def get_pos_embed_indices(start, length, max_pos, scale=1.0):
    """Get position embedding indices."""
    scale = scale * torch.ones_like(start, dtype=torch.float32)
    pos = (
        start.unsqueeze(1)
        + (torch.arange(length, device=start.device, dtype=torch.float32).unsqueeze(0) * scale.unsqueeze(1)).long()
    )
    pos = torch.where(pos < max_pos, pos, max_pos - 1)
    return pos


def _cosyvoice3_varlen_attention_enabled() -> bool:
    return os.environ.get("COSYVOICE3_VARLEN_ATTENTION", "0") not in ("0", "false", "False", "")


def _cosyvoice3_varlen_attention_dtype() -> torch.dtype | None:
    dtype_name = os.environ.get("COSYVOICE3_VARLEN_ATTENTION_DTYPE", "bf16").lower()
    if dtype_name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype_name in ("fp16", "float16", "half"):
        return torch.float16
    if dtype_name in ("none", "off", "0", "false"):
        return None
    logger.warning_once("Unsupported COSYVOICE3_VARLEN_ATTENTION_DTYPE=%s; disabling varlen attention", dtype_name)
    return None


def _normalize_attention_mask(mask: torch.Tensor | None, batch_size: int, seq_len: int) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.dim() == 2:
        attn_mask = mask
    elif mask.dim() == 3 and mask.shape[1] == 1:
        attn_mask = mask[:, 0]
    elif mask.dim() == 4:
        attn_mask = mask[:, 0, -1]
    else:
        return None
    if attn_mask.shape != (batch_size, seq_len):
        return None
    return attn_mask.bool()


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0, approximate: str = "none"):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        activation = nn.GELU(approximate=approximate)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), activation)
        self.ff = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.ff(x)


class DiTAttention(nn.Module):
    """Attention module using diffusion infrastructure for optimized backends.

    This replaces the original Attention class to leverage FlashAttention,
    SageAttention, or SDPA backends automatically.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = dim_head * heads
        self.dropout = dropout
        self.scale = 1.0 / math.sqrt(dim_head)

        # Q/K/V projections
        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)

        # Output projection
        self.to_out = nn.Sequential(nn.Linear(self.inner_dim, dim), nn.Dropout(dropout))

        # Diffusion attention backend (Flash/Sage/SDPA)
        self.attn = DiffusionAttention(
            num_heads=heads,
            head_size=dim_head,
            softmax_scale=self.scale,
            causal=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        rope=None,
    ) -> torch.Tensor:
        batch_size, seq_len = x.shape[0], x.shape[1]

        with torch.profiler.record_function("cosyvoice3_dit_attention_qkv_projection"):
            query = self.to_q(x)
            key = self.to_k(x)
            value = self.to_v(x)

        with torch.profiler.record_function("cosyvoice3_dit_attention_rope"):
            if rope is not None:
                freqs, xpos_scale = rope
                q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
                query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
                key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        with torch.profiler.record_function("cosyvoice3_dit_attention_backend"):
            # Reshape for attention: (batch, seq, heads, head_dim)
            query = query.view(batch_size, seq_len, self.heads, self.dim_head)
            key = key.view(batch_size, seq_len, self.heads, self.dim_head)
            value = value.view(batch_size, seq_len, self.heads, self.dim_head)

            # The diffusion Attention layer expects (batch, seq, heads, head_dim)
            attn_mask = _normalize_attention_mask(mask, batch_size, seq_len)
            attn_metadata = AttentionMetadata(attn_mask=attn_mask) if attn_mask is not None else None
            attn_dtype = _cosyvoice3_varlen_attention_dtype() if _cosyvoice3_varlen_attention_enabled() else None
            original_dtype = query.dtype
            if attn_dtype is not None and attn_metadata is not None:
                with torch.profiler.record_function(f"cosyvoice3_dit_attention_varlen_{attn_dtype}"):
                    out = self.attn(
                        query.to(dtype=attn_dtype),
                        key.to(dtype=attn_dtype),
                        value.to(dtype=attn_dtype),
                        attn_metadata=attn_metadata,
                    ).to(dtype=original_dtype)
            else:
                out = self.attn(query, key, value, attn_metadata=attn_metadata)

        with torch.profiler.record_function("cosyvoice3_dit_attention_output_projection"):
            # Some attention backends return a non-contiguous layout.
            out = out.reshape(batch_size, seq_len, self.inner_dim)
            out = out.to(x.dtype)
            out = self.to_out(out)

        # Apply mask if provided
        attn_mask = _normalize_attention_mask(mask, batch_size, seq_len)
        if attn_mask is not None:
            out = out.masked_fill(~attn_mask.unsqueeze(-1), 0.0)

        return out


class DiTBlock(nn.Module):
    """DiT block with AdaLayerNorm modulation."""

    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1):
        super().__init__()

        self.attn_norm = AdaLayerNormZero(dim)
        self.attn = DiTAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, t, mask=None, rope=None):
        with torch.profiler.record_function("cosyvoice3_dit_adalayernorm_attention"):
            norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        with torch.profiler.record_function("cosyvoice3_dit_attention"):
            attn_output = self.attn(x=norm, mask=mask, rope=rope)

        with torch.profiler.record_function("cosyvoice3_dit_attention_residual"):
            x = x + gate_msa.unsqueeze(1) * attn_output

        with torch.profiler.record_function("cosyvoice3_dit_layernorm_ffn"):
            ff_norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            ff_output = self.ff(ff_norm)
            x = x + gate_mlp.unsqueeze(1) * ff_output

        return x


class CausalConvPositionEmbedding(nn.Module):
    """Causal convolutional position embedding."""

    def __init__(self, dim, kernel_size=31, groups=16):
        super().__init__()
        assert kernel_size % 2 != 0
        self.kernel_size = kernel_size
        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=0),
            nn.Mish(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=0),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        with torch.profiler.record_function("cosyvoice3_dit_causal_conv_position_embedding"):
            if mask is not None:
                mask = mask[..., None]
                x = x.masked_fill(~mask, 0.0)

            x = x.permute(0, 2, 1)
            x = F.pad(x, (self.kernel_size - 1, 0, 0, 0))
            x = self.conv1(x)
            x = F.pad(x, (self.kernel_size - 1, 0, 0, 0))
            x = self.conv2(x)
            out = x.permute(0, 2, 1)

            if mask is not None:
                out = out.masked_fill(~mask, 0.0)

        return out


class AdaLayerNormZero_Final(nn.Module):
    """AdaLayerNormZero for final layer - returns only modulated x."""

    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class GRN(nn.Module):
    """Global Response Normalization layer."""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    """ConvNeXt-V2 Block."""

    def __init__(self, dim: int, intermediate_dim: int, dilation: int = 1):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=padding, groups=dim, dilation=dilation)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


class TextEmbedding(nn.Module):
    """Text embedding with optional ConvNeXt modeling."""

    def __init__(self, text_num_embeds, text_dim, conv_layers=0, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, text: torch.Tensor, seq_len, drop_text=False):
        batch, text_len = text.shape[0], text.shape[1]
        text = text + 1
        text = text[:, :seq_len]
        text = F.pad(text, (0, seq_len - text_len), value=0)

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


class InputEmbedding(nn.Module):
    """Input embedding combining noised audio, condition, text, and speaker."""

    def __init__(self, mel_dim, text_dim, out_dim, spk_dim=None):
        super().__init__()
        spk_dim = 0 if spk_dim is None else spk_dim
        self.spk_dim = spk_dim
        self.proj = nn.Linear(mel_dim * 2 + text_dim + spk_dim, out_dim)
        self.conv_pos_embed = CausalConvPositionEmbedding(dim=out_dim)

    def forward(self, x, cond, text_embed, spks):
        with torch.profiler.record_function("cosyvoice3_dit_input_projection"):
            to_cat = [x, cond, text_embed]
            if self.spk_dim > 0:
                spks = repeat(spks, "b c -> b t c", t=x.shape[1])
                to_cat.append(spks)

            x = self.proj(torch.cat(to_cat, dim=-1))
        with torch.profiler.record_function("cosyvoice3_dit_input_causal_conv"):
            x = self.conv_pos_embed(x) + x
        return x


class DiT(nn.Module):
    """Diffusion Transformer backbone using optimized attention backends.

    This is a drop-in replacement for the original DiT that uses the
    vllm_omni diffusion infrastructure for FlashAttention/SageAttention/SDPA.
    """

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
        out_channels=None,
        static_chunk_size=50,
        num_decoding_left_chunks=2,
    ):
        super().__init__()

        self.time_embed = DiTTimestepEmbedding(dim)
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
        self.out_channels = out_channels
        self.static_chunk_size = static_chunk_size
        self.num_decoding_left_chunks = num_decoding_left_chunks

    def forward(self, x, mask, mu, t, spks=None, cond=None):
        with torch.profiler.record_function("cosyvoice3_dit_prepare_inputs"):
            x = x.transpose(1, 2)
            mu = mu.transpose(1, 2)
            cond = cond.transpose(1, 2)
            spks = spks.unsqueeze(dim=1)
            batch, seq_len = x.shape[0], x.shape[1]
            if t.ndim == 0:
                t = t.repeat(batch)

        with torch.profiler.record_function("cosyvoice3_dit_time_embedding"):
            t = self.time_embed(t)
        x = self.input_embed(x, cond, mu, spks.squeeze(1))

        with torch.profiler.record_function("cosyvoice3_dit_rope_and_mask"):
            rope = self.rotary_embed.forward_from_seq_len(seq_len)
            attn_mask = mask[:, 0].bool() if mask.dim() == 3 and mask.shape[1] == 1 else mask.bool()

        if self.long_skip_connection is not None:
            residual = x

        with torch.profiler.record_function("cosyvoice3_dit_transformer_blocks"):
            for block_index, block in enumerate(self.transformer_blocks):
                with torch.profiler.record_function(f"cosyvoice3_dit_transformer_block_{block_index}"):
                    x = block(x, t, mask=attn_mask.bool(), rope=rope)

        if self.long_skip_connection is not None:
            with torch.profiler.record_function("cosyvoice3_dit_long_skip_projection"):
                x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        with torch.profiler.record_function("cosyvoice3_dit_final_adalayernorm"):
            x = self.norm_out(x, t)
        with torch.profiler.record_function("cosyvoice3_dit_final_projection"):
            output = self.proj_out(x).transpose(1, 2)
        return output
