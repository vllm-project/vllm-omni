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

# CosyVoice3 flow-decoder streaming controls. The model was trained
# chunked-causal (static_chunk_size / num_decoding_left_chunks) but the base
# forward path ran full bidirectional attention. These flags restore the
# chunked-causal pattern and enable an incremental streaming decode with a
# per-layer attention KV cache for the finalized prefix, so each streaming
# chunk only denoises its own new frames instead of the whole cumulative
# sequence. Default on; export COSYVOICE3_CHUNK_CAUSAL=0 / COSYVOICE3_DIT_KV_CACHE=0
# for the byte-identical full-sequence path.
CHUNK_CAUSAL = os.environ.get("COSYVOICE3_CHUNK_CAUSAL", "1") == "1"
KV_CACHE = os.environ.get("COSYVOICE3_DIT_KV_CACHE", "1") == "1"


def build_chunk_causal_mask(
    q_len: int,
    kv_len: int,
    chunk_size: int,
    device: torch.device,
    q_offset: int = 0,
) -> torch.Tensor:
    """Boolean chunked-causal attention mask, True == may attend.

    Query row ``i`` (absolute position ``q_offset + i``) may attend to key
    column ``j`` (absolute position ``j``) iff ``j`` lies in the same static
    chunk as the query or any earlier chunk, i.e.
    ``j < ((q_offset + i) // chunk_size + 1) * chunk_size``. All left chunks are
    visible (``num_decoding_left_chunks = -1``). Returns shape
    ``(1, 1, q_len, kv_len)`` so SDPA broadcasts it over batch and heads.
    """
    q_pos = torch.arange(q_len, device=device) + q_offset
    k_pos = torch.arange(kv_len, device=device)
    ending = ((q_pos // chunk_size) + 1) * chunk_size
    mask = k_pos.unsqueeze(0) < ending.unsqueeze(1)
    return mask.view(1, 1, q_len, kv_len)


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
        attn_mask: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        """Self-attention with optional chunked-causal mask and prefix KV cache.

        ``x`` holds only the query frames (the new streaming chunk when a cache
        is supplied, otherwise the full sequence). ``rope`` must already be
        sliced to the query frames' absolute positions. When ``kv_cache`` is
        given, the cached prefix K/V (finalized, at t=1) are concatenated before
        the freshly projected new-frame K/V. ``attn_mask`` is a boolean SDPA
        mask (True == attend) of shape ``(1, 1, q_len, kv_len)``.

        Returns ``(out, new_key, new_value)`` where ``new_key``/``new_value`` are
        the query frames' K/V (post-RoPE, pre-concat) so the caller can append
        them to the cache when the chunk is finalized.
        """
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Project to Q, K, V
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        # Apply rotary position embedding
        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale**-1.0) if xpos_scale is not None else (1.0, 1.0)
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        # Reshape for attention: (batch, seq, heads, head_dim)
        query = query.view(batch_size, seq_len, self.heads, self.dim_head)
        key = key.view(batch_size, seq_len, self.heads, self.dim_head)
        value = value.view(batch_size, seq_len, self.heads, self.dim_head)

        new_key, new_value = key, value
        if kv_cache is not None:
            cached_key, cached_value = kv_cache
            if cached_key is not None and cached_key.shape[1] > 0:
                key = torch.cat([cached_key, key], dim=1)
                value = torch.cat([cached_value, value], dim=1)

        # Use diffusion attention backend. On the fp32 flow this routes to SDPA,
        # which consumes attn_metadata.attn_mask directly for a 4D mask.
        attn_metadata = AttentionMetadata(attn_mask=attn_mask) if attn_mask is not None else None
        out = self.attn(query, key, value, attn_metadata=attn_metadata)

        # Reshape back: (batch, seq, dim). reshape, not view: an attention
        # backend may return a non-contiguous layout, which view rejects.
        out = out.reshape(batch_size, seq_len, self.inner_dim)
        out = out.to(query.dtype)

        # Output projection
        out = self.to_out(out)

        # Apply padding mask if provided (no-op for the unpadded batch here).
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            elif mask.dim() == 4:
                # (batch, heads, seq, seq) -> use last dim
                mask = mask[:, 0, -1].unsqueeze(-1)
            out = out.masked_fill(~mask.bool(), 0.0)

        return out, new_key, new_value


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

    def forward(self, x, t, mask=None, rope=None, attn_mask=None, kv_cache=None):
        # pre-norm & modulation for attention input
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)

        # attention (only the query frames of `x` are updated)
        attn_output, new_key, new_value = self.attn(
            x=norm, mask=mask, rope=rope, attn_mask=attn_mask, kv_cache=kv_cache
        )

        # process attention output for input x
        x = x + gate_msa.unsqueeze(1) * attn_output

        ff_norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(ff_norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output

        return x, new_key, new_value


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

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        cnn_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        """Causal conv position embedding with an optional left-context cache.

        Each of the two causal convs left-pads by ``kernel_size - 1`` frames. In
        streaming, those frames must be the tail of the previous chunk's conv
        input, not zeros, or the chunk boundary corrupts. ``cnn_cache`` supplies
        that tail per conv; the updated tails are returned for the next chunk.
        Passing ``cnn_cache=None`` reproduces the zero-padded (non-streaming)
        path exactly. Returns ``(out, new_cnn_cache)``.
        """
        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.0)

        pad = self.kernel_size - 1
        x = x.permute(0, 2, 1)

        if cnn_cache is not None:
            left1, left2 = cnn_cache
        else:
            left1 = x.new_zeros(x.shape[0], x.shape[1], pad)
            left2 = None  # filled below with zeros of conv1's output shape

        x1_in = torch.cat([left1, x], dim=-1)
        new_left1 = x1_in[..., -pad:]
        x = self.conv1(x1_in)

        if left2 is None:
            left2 = x.new_zeros(x.shape[0], x.shape[1], pad)
        x2_in = torch.cat([left2, x], dim=-1)
        new_left2 = x2_in[..., -pad:]
        x = self.conv2(x2_in)

        out = x.permute(0, 2, 1)

        if mask is not None:
            out = out.masked_fill(~mask, 0.0)

        return out, (new_left1, new_left2)


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

    def forward(self, x, cond, text_embed, spks, cnn_cache=None):
        to_cat = [x, cond, text_embed]
        if self.spk_dim > 0:
            spks = repeat(spks, "b c -> b t c", t=x.shape[1])
            to_cat.append(spks)

        x = self.proj(torch.cat(to_cat, dim=-1))
        conv_out, new_cnn_cache = self.conv_pos_embed(x, cnn_cache=cnn_cache)
        x = conv_out + x
        return x, new_cnn_cache


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
        x = x.transpose(1, 2)
        mu = mu.transpose(1, 2)
        cond = cond.transpose(1, 2)
        spks = spks.unsqueeze(dim=1)
        batch, seq_len = x.shape[0], x.shape[1]
        if t.ndim == 0:
            t = t.repeat(batch)

        # t: conditioning time, c: context (text + masked cond audio), x: noised input audio
        t = self.time_embed(t)
        x, _ = self.input_embed(x, cond, mu, spks.squeeze(1))

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        pad_mask = mask.bool().repeat(1, x.size(1), 1).unsqueeze(dim=1)

        # Chunked-causal attention: the model was trained with a static chunk
        # size and all left context visible; the base path ran full bidirectional
        # attention. Restoring the chunked-causal pattern also makes a finalized
        # prefix's K/V stable (it never attends to future chunks), which is what
        # the incremental streaming cache relies on.
        chunk_mask = None
        if CHUNK_CAUSAL or KV_CACHE:
            chunk_mask = build_chunk_causal_mask(seq_len, seq_len, self.static_chunk_size, x.device)

        for block in self.transformer_blocks:
            x, _, _ = block(x, t, mask=pad_mask.bool(), rope=rope, attn_mask=chunk_mask)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x).transpose(1, 2)
        return output

    def forward_chunk(
        self,
        x,
        mu,
        t,
        spks,
        cond,
        att_step,
        cnn_step,
        offset,
    ):
        """Incremental streaming forward over only the new chunk's frames.

        ``x``/``mu``/``cond`` are ``(B, mel_dim, N_new)`` for the new frames at
        one Euler step. ``offset`` is the number of finalized prefix frames.
        ``att_step`` is the per-layer prefix K/V captured at this *same* Euler
        step (a list of ``(K, V)`` or None on the first chunk); ``cnn_step`` is
        the prefix causal-conv tail at this step. The new frames attend, under a
        chunked-causal mask, to the whole prefix (at the matching denoising t)
        plus themselves.

        Returns ``(output, new_kv, new_cnn)`` — ``output`` is
        ``(B, mel_dim, N_new)`` and ``new_kv``/``new_cnn`` are this step's new
        frames' per-layer K/V and conv tail, to be stored at this step's cache
        slot.
        """
        if self.long_skip_connection is not None:
            raise NotImplementedError("long_skip_connection is unsupported on the streaming chunk path")

        x = x.transpose(1, 2)
        mu = mu.transpose(1, 2)
        cond = cond.transpose(1, 2)
        spks = spks.unsqueeze(dim=1)
        batch, n_new = x.shape[0], x.shape[1]
        if t.ndim == 0:
            t = t.repeat(batch)

        t = self.time_embed(t)
        x, new_cnn = self.input_embed(x, cond, mu, spks.squeeze(1), cnn_cache=cnn_step)

        total_len = offset + n_new
        freqs, xpos_scale = self.rotary_embed.forward_from_seq_len(total_len)
        # Slice RoPE to the new frames' absolute positions [offset, total_len).
        # freqs is (1, total_len, rot_dim): slice the sequence dim, not dim 0.
        freqs = freqs[:, offset:total_len, :]
        if torch.is_tensor(xpos_scale):
            xpos_scale = xpos_scale[:, offset:total_len, :]
        rope = (freqs, xpos_scale)

        chunk_mask = build_chunk_causal_mask(n_new, total_len, self.static_chunk_size, x.device, q_offset=offset)

        new_kv = []
        for i, block in enumerate(self.transformer_blocks):
            layer_cache = att_step[i] if att_step is not None else None
            x, new_key, new_value = block(x, t, mask=None, rope=rope, attn_mask=chunk_mask, kv_cache=layer_cache)
            new_kv.append((new_key, new_value))

        x = self.norm_out(x, t)
        output = self.proj_out(x).transpose(1, 2)
        return output, new_kv, new_cnn
