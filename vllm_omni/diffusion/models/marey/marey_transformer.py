# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Marey MMDiT (Multimodal Diffusion Transformer) for video generation.

Ported from the Flux/MMDiT architecture. The key difference from standard DiT
is the joint multimodal attention where text and visual token streams are
projected separately, concatenated for joint attention, then split back.

Architecture:
    Input latents -> PatchEmbed3D -> FluxBlocks (joint text+visual attn) -> FinalLayer -> Unpatchify
"""

from __future__ import annotations

import functools
import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput,
    SequenceParallelOutput,
)

logger = init_logger(__name__)

MAX_WAVELENGTH = 10_000


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def t2i_modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


def t2i_modulate_masked(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    mask: torch.Tensor | None = None,
    shift_t0: torch.Tensor | None = None,
    scale_t0: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mask-aware ``t2i_modulate`` for I2V.

    Positions where ``mask`` is True (noise frames) use ``(shift, scale)`` from
    ``t_emb``; positions where ``mask`` is False (conditioning frames) use the
    ``t0`` variant so conditioning tokens are modulated as if ``timestep=0``.
    Mirrors mv ``open_sora/opensora/models/stdit/dit3d.py:1165-1170``.
    """
    out = t2i_modulate(x, shift, scale)
    if mask is not None and shift_t0 is not None and scale_t0 is not None:
        out_t0 = t2i_modulate(x, shift_t0, scale_t0)
        out = torch.where(mask, out, out_t0)
    return out


def gate_masked(
    x: torch.Tensor,
    gate: torch.Tensor,
    mask: torch.Tensor | None = None,
    gate_t0: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mask-aware gate for I2V. Mirrors mv ``dit3d.py:1173-1177``."""
    out = x * gate
    if mask is not None and gate_t0 is not None:
        out_t0 = x * gate_t0
        out = torch.where(mask, out, out_t0)
    return out


def get_temporal_pos(
    x: torch.Tensor,
    num_t: int,
    num_s: int,
    cond_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Temporal positional indices for latent patches [B, T*S]."""
    if cond_offsets is not None:
        tf = cond_offsets.shape[0]
        index_tensor = torch.arange(num_t - tf, device=x.device, dtype=x.dtype)
        index_tensor = torch.cat([index_tensor, cond_offsets], dim=0)
    else:
        index_tensor = torch.arange(num_t, device=x.device, dtype=x.dtype)
    B = x.shape[0]
    index_tensor = index_tensor.unsqueeze(0).unsqueeze(-1)
    index_tensor = index_tensor.expand(B, num_t, num_s)
    return index_tensor.reshape(B, num_t * num_s)


def apply_rope(
    inputs: torch.Tensor,
    positions: torch.Tensor,
    max_wavelength: int = MAX_WAVELENGTH,
) -> torch.Tensor:
    """Apply RoPE to inputs [B, heads, seq, head_dim] using positions [B, seq]."""
    head_dim = inputs.shape[-1]
    fraction = 2 * torch.arange(0, head_dim // 2, device=inputs.device, dtype=torch.float32) / head_dim
    timescale = max_wavelength**fraction
    sinusoid_inp = positions[..., None].to(torch.float32) / timescale[None, :]
    sinusoid_inp = sinusoid_inp[..., None, :, :]
    sin = torch.sin(sinusoid_inp)
    cos = torch.cos(sinusoid_inp)
    first_half, second_half = torch.chunk(inputs, chunks=2, dim=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    return torch.cat([first_part, second_part], dim=-1).to(inputs.dtype)


def apply_rotary_emb(
    x: torch.Tensor,
    positions: torch.Tensor,
    channels: int,
    dim: int = -1,
) -> torch.Tensor:
    """Apply RoPE to the last `channels` of dimension `dim`."""
    x_left, x_right = x.split([x.shape[dim] - channels, channels], dim=dim)
    x_right = apply_rope(x_right, positions)
    return torch.cat([x_left, x_right], dim=dim)


# ---------------------------------------------------------------------------
# Norm layers
# ---------------------------------------------------------------------------


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)


class DistributedRMSNorm(nn.Module):
    """RMSNorm that computes global RMS across tensor parallel ranks."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = get_tensor_model_parallel_world_size()
        input_dtype = x.dtype
        x_float = x.float()
        local_sum_sq = (x_float**2).sum(dim=-1, keepdim=True)
        local_count = x.shape[-1]
        if tp_size > 1:
            global_sum_sq = local_sum_sq.clone()
            tensor_model_parallel_all_reduce(global_sum_sq)
            global_count = local_count * tp_size
        else:
            global_sum_sq = local_sum_sq
            global_count = local_count
        rms = torch.sqrt(global_sum_sq / global_count + self.eps)
        output = (x_float / rms) * self.weight.float()
        return output.to(input_dtype)


# ---------------------------------------------------------------------------
# Embedding layers
# ---------------------------------------------------------------------------


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep -> vector embedding."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        freqs = freqs.to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if t_freq.dtype != dtype:
            t_freq = t_freq.to(dtype)
        return self.mlp(t_freq)


class SizeEmbedder(TimestepEmbedder):
    """Embeds scalar values (fps, resolution) into vectors."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__(hidden_size=hidden_size, frequency_embedding_size=frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.outdim = hidden_size

    def forward(self, s: torch.Tensor, **kwargs) -> torch.Tensor:
        if s.ndim == 1:
            s = s[:, None]
        b, dims = s.shape[0], s.shape[1]
        s = s.reshape(b * dims)
        s_freq = self.timestep_embedding(s, self.frequency_embedding_size)
        param = next(self.parameters())
        s_freq = s_freq.to(param.dtype)
        s_emb = self.mlp(s_freq)
        s_emb = s_emb.reshape(b, dims * self.outdim)
        return s_emb


class LabelEmbedder(nn.Module):
    """Embeds categorical labels into vectors.

    Supports classifier-free guidance via a null class appended when
    dropout_prob > 0.  Labels equal to default_value are mapped to the
    null class during forward.
    """

    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float = 0.0, default_value: int = -1):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.default_value = default_value

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        drop_ids = labels == self.default_value
        labels = torch.where(drop_ids, self.num_classes, labels)
        return self.embedding_table(labels)


class OrderedEmbedder(nn.Module):
    """Embeds ordered categorical labels using separate lower/upper tables.

    During inference, lower table uses the label index and upper table
    defaults to num_classes - 1 (max).  Labels equal to default_value
    are mapped to the null class when dropout_prob > 0.
    """

    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float = 0.0, default_value: int = -1):
        super().__init__()
        use_null = int(dropout_prob > 0)
        half_dim = hidden_size // 2
        self.embedding_table_lower = nn.Embedding(num_classes + use_null, half_dim)
        self.embedding_table_upper = nn.Embedding(num_classes + use_null, hidden_size - half_dim)
        self.num_classes = num_classes
        self.default_value = default_value

    def forward(self, labels: torch.Tensor, labels_upper: torch.Tensor | None = None) -> torch.Tensor:
        if labels_upper is None:
            labels_upper = (self.num_classes - 1) * torch.ones_like(labels)
        drop_ids = labels == self.default_value
        null_idx = self.num_classes
        labels = torch.where(drop_ids, null_idx, labels)
        labels_upper = torch.where(drop_ids, null_idx, labels_upper)
        lower = self.embedding_table_lower(labels)
        upper = self.embedding_table_upper(labels_upper)
        return torch.cat([lower, upper], dim=-1)


class VectorEmbedder(nn.Module):
    """Embeds a continuous vector into hidden_size via a 2-layer MLP."""

    def __init__(self, hidden_size: int, vector_size: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(vector_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        return self.mlp(vec)


class _RoPEFreqsHolder(nn.Module):
    """Holds a rope.freqs buffer for checkpoint compatibility.

    The actual RoPE computation is done by apply_rope() directly; this module
    only exists so that the checkpoint's ``rope.freqs`` tensor has a matching
    entry in the model's buffer dict.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("freqs", torch.zeros(dim))


class PatchEmbed3D(nn.Module):
    """Video -> patch embedding via 3D convolution."""

    def __init__(
        self,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        in_channels: int = 4,
        embed_dim: int = 1536,
        flatten: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        return x


class PositionEmbedding2D(nn.Module):
    """Sinusoidal 2-D spatial position embedding (H x W grid)."""

    def __init__(self, dim: int, learned_pe: bool = False, theta: int = 10_000):
        super().__init__()
        self.dim = dim
        assert dim % 4 == 0
        half_dim = dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, half_dim, 2).float() / half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.learned_pe = learned_pe
        if learned_pe:
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim, bias=True),
                nn.SiLU(),
                nn.Linear(dim, dim, bias=True),
            )

    def _get_sin_cos_emb(self, t: torch.Tensor) -> torch.Tensor:
        out = torch.einsum("i,d->id", t, self.inv_freq)
        return torch.cat((torch.sin(out), torch.cos(out)), dim=-1)

    @functools.lru_cache(maxsize=512)
    def _get_cached_emb(
        self,
        device: torch.device,
        dtype: torch.dtype,
        h: int,
        w: int,
        scale: float = 1.0,
        base_size: int | None = None,
    ) -> torch.Tensor:
        grid_h = torch.arange(h, device=device) / scale
        grid_w = torch.arange(w, device=device) / scale
        if base_size is not None:
            grid_h *= base_size / h
            grid_w *= base_size / w
        grid_h, grid_w = torch.meshgrid(grid_w, grid_h, indexing="ij")
        grid_h = grid_h.t().reshape(-1)
        grid_w = grid_w.t().reshape(-1)
        emb_h = self._get_sin_cos_emb(grid_h)
        emb_w = self._get_sin_cos_emb(grid_w)
        return torch.concat([emb_h, emb_w], dim=-1).unsqueeze(0).to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        h: int,
        w: int,
        scale: float = 1.0,
        base_size: int | None = None,
    ) -> torch.Tensor:
        pe = self._get_cached_emb(x.device, x.dtype, h, w, scale, base_size)
        if self.learned_pe:
            pe = self.mlp(pe)
        return pe


class SeqProjector(nn.Module):
    """Projects sequences from input_size(s) to projection_dim (Linear + LayerNorm).

    Supports single or multiple input modalities (e.g. UL2 + ByT5).
    When multiple input_sizes are given, forward expects a list of tensors
    and returns their projections concatenated along the sequence dimension.

    Checkpoint naming: projections.{i}.0 = Linear, projections.{i}.1 = LayerNorm
    """

    def __init__(self, input_size: int | list[int], projection_dim: int):
        super().__init__()
        if isinstance(input_size, int):
            input_size = [input_size]
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(sz, projection_dim),
                    nn.LayerNorm(projection_dim),
                )
                for sz in input_size
            ]
        )

    def forward(self, x: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        if isinstance(x, list):
            return torch.cat([proj(t) for t, proj in zip(x, self.projections)], dim=1)
        return self.projections[0](x)


class CaptionEmbedder(nn.Module):
    """Projects text encoder outputs (seq_cond + vector_cond) into hidden_size.

    Supports multiple text encoders (e.g. UL2 + ByT5) via list-valued in_channels
    and token_num. Supports classifier-free guidance via null embedding dropout.
    """

    def __init__(
        self,
        in_channels: int | list[int],
        hidden_size: int,
        uncond_prob: float,
        token_num: int | list[int] = 300,
        vector_in_channels: int | None = None,
    ):
        super().__init__()
        self.multi_encoder = isinstance(in_channels, list)
        self.y_proj = SeqProjector(in_channels, hidden_size)

        if vector_in_channels is not None:
            self.vector_proj = SeqProjector(vector_in_channels, hidden_size)
        else:
            self.vector_proj = None

        if self.multi_encoder:
            if not isinstance(token_num, list):
                token_num = [token_num] * len(in_channels)
            for i, (tn, ch) in enumerate(zip(token_num, in_channels)):
                self.register_buffer(f"y_embedding_{i}", torch.randn(tn, ch) / ch**0.5)
        else:
            self.register_buffer("y_embedding", torch.randn(token_num, in_channels) / in_channels**0.5)

        if vector_in_channels is not None:
            self.register_buffer("vector_embedding", torch.randn(1, vector_in_channels) / vector_in_channels**0.5)
        else:
            self.vector_embedding = None

        self.uncond_prob = uncond_prob

    def forward(
        self,
        seq_cond: torch.Tensor | list[torch.Tensor],
        seq_cond_mask: torch.Tensor | list[torch.Tensor],
        vector_cond: torch.Tensor | None,
        train: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (seq_cond, seq_cond_mask, vector_cond) after projection."""
        if train and self.uncond_prob > 0:
            if self.multi_encoder:
                B = seq_cond[0].shape[0]
                drop_ids = torch.rand(B, device=seq_cond[0].device) < self.uncond_prob
                seq_cond = [
                    torch.where(drop_ids[:, None, None], getattr(self, f"y_embedding_{i}")[None], sc)
                    for i, sc in enumerate(seq_cond)
                ]
            else:
                B = seq_cond.shape[0]
                drop_ids = torch.rand(B, device=seq_cond.device) < self.uncond_prob
                seq_cond = torch.where(drop_ids[:, None, None], self.y_embedding[None], seq_cond)
            if self.vector_proj is not None and vector_cond is not None:
                vector_cond = torch.where(drop_ids[:, None], self.vector_embedding, vector_cond)

        seq_cond = self.y_proj(seq_cond)
        if self.multi_encoder and isinstance(seq_cond_mask, list):
            seq_cond_mask = torch.cat(seq_cond_mask, dim=1)

        if self.vector_proj is not None and vector_cond is not None:
            vector_cond = self.vector_proj(vector_cond)
        else:
            vector_cond = torch.zeros_like(seq_cond[:, 0])
        return seq_cond, seq_cond_mask, vector_cond


# ---------------------------------------------------------------------------
# SwiGLU MLP (TP-aware)
# ---------------------------------------------------------------------------


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network with TP support."""

    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        # SwiGLU needs 2x hidden for gating
        self.w1 = ColumnParallelLinear(in_features, hidden_features, bias=True, gather_output=False, return_bias=False)
        self.w2 = ColumnParallelLinear(in_features, hidden_features, bias=True, gather_output=False, return_bias=False)
        self.w3 = RowParallelLinear(hidden_features, in_features, bias=True, input_is_parallel=True, return_bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x2) * x1
        return self.w3(hidden)


# ---------------------------------------------------------------------------
# Flux Multimodal Attention
# ---------------------------------------------------------------------------


class MareyFluxAttention(nn.Module):
    """Multimodal attention for MMDiT: jointly attends over visual (x) and text (y) tokens.

    Separate Q/KV projections per stream, concat for joint attention, then split outputs back.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        share_weights: bool = False,
        rope_channels_ratio: float | None = 0.5,
        rope_dim: int = -1,
        num_kv_heads: int | None = None,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.share_weights = share_weights

        kv_dim = self.num_kv_heads * self.head_dim

        # Visual stream projections
        self.q_linear_x = ColumnParallelLinear(dim, dim, bias=qkv_bias, gather_output=False, return_bias=False)
        self.kv_linear_x = ColumnParallelLinear(dim, kv_dim * 2, bias=qkv_bias, gather_output=False, return_bias=False)
        tp_size = get_tensor_model_parallel_world_size()
        tp_num_heads = num_heads // tp_size
        tp_num_kv_heads = self.num_kv_heads // tp_size

        # Per-head QK norm (checkpoint stores weights of shape [head_dim])
        self.q_norm_x = LlamaRMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm_x = LlamaRMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj_x = RowParallelLinear(dim, dim, bias=True, input_is_parallel=True, return_bias=False)

        # Text stream projections (separate or shared with visual)
        if not share_weights:
            self.q_linear_y = ColumnParallelLinear(dim, dim, bias=qkv_bias, gather_output=False, return_bias=False)
            self.kv_linear_y = ColumnParallelLinear(
                dim, kv_dim * 2, bias=qkv_bias, gather_output=False, return_bias=False
            )
            self.q_norm_y = LlamaRMSNorm(self.head_dim) if qk_norm else nn.Identity()
            self.k_norm_y = LlamaRMSNorm(self.head_dim) if qk_norm else nn.Identity()
            self.proj_y = RowParallelLinear(dim, dim, bias=True, input_is_parallel=True, return_bias=False)

        self.tp_num_heads = tp_num_heads
        self.tp_num_kv_heads = tp_num_kv_heads

        # RoPE config
        if rope_channels_ratio is not None and rope_channels_ratio > 0.0:
            if rope_dim == -1:
                channels = self.head_dim
            elif rope_dim == -3:
                channels = num_heads
            else:
                raise ValueError(f"Unsupported rope_dim={rope_dim}")
            rope_channels = int(channels * rope_channels_ratio)
            rope_channels = (rope_channels // 2) * 2
            self.rope_channels = rope_channels
            self.rope_dim = rope_dim
        else:
            self.rope_channels = None
            self.rope_dim = None

        self.attn = Attention(
            num_heads=self.tp_num_heads,
            head_size=self.head_dim,
            num_kv_heads=self.tp_num_kv_heads,
            softmax_scale=self.scale,
            causal=False,
        )

    def _project_q(self, x: torch.Tensor, linear: nn.Module, norm: nn.Module) -> torch.Tensor:
        q = linear(x)
        q = q.unflatten(-1, (self.tp_num_heads, self.head_dim))
        return norm(q)

    def _project_kv(self, x: torch.Tensor, linear: nn.Module, norm: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        kv = linear(x)
        kv = kv.unflatten(-1, (2, self.tp_num_kv_heads, self.head_dim))
        k, v = kv.unbind(dim=-3)
        k = norm(k)
        return k, v

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        temporal_pos: torch.Tensor | None = None,
        y_temporal_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.share_weights:
            q_linear_y, kv_linear_y = self.q_linear_x, self.kv_linear_x
            q_norm_y, k_norm_y = self.q_norm_x, self.k_norm_x
            proj_y = self.proj_x
        else:
            q_linear_y, kv_linear_y = self.q_linear_y, self.kv_linear_y
            q_norm_y, k_norm_y = self.q_norm_y, self.k_norm_y
            proj_y = self.proj_y

        B, N_x, _ = x.shape
        _, N_y, _ = y.shape

        q_x = self._project_q(x, self.q_linear_x, self.q_norm_x)
        q_y = self._project_q(y, q_linear_y, q_norm_y)
        k_x, v_x = self._project_kv(x, self.kv_linear_x, self.k_norm_x)
        k_y, v_y = self._project_kv(y, kv_linear_y, k_norm_y)

        # Apply RoPE to visual tokens (and optionally text tokens)
        if self.rope_dim is not None and temporal_pos is not None:
            # Transpose to [B, heads, seq, head_dim] for RoPE
            q_x_t = q_x.transpose(1, 2)
            k_x_t = k_x.transpose(1, 2)
            q_x_t = apply_rotary_emb(q_x_t, temporal_pos, self.rope_channels, self.rope_dim)
            k_x_t = apply_rotary_emb(k_x_t, temporal_pos, self.rope_channels, self.rope_dim)
            q_x = q_x_t.transpose(1, 2)
            k_x = k_x_t.transpose(1, 2)

            if y_temporal_pos is not None:
                q_y_t = q_y.transpose(1, 2)
                k_y_t = k_y.transpose(1, 2)
                q_y_t = apply_rotary_emb(q_y_t, y_temporal_pos, self.rope_channels, self.rope_dim)
                k_y_t = apply_rotary_emb(k_y_t, y_temporal_pos, self.rope_channels, self.rope_dim)
                q_y = q_y_t.transpose(1, 2)
                k_y = k_y_t.transpose(1, 2)

        # Concatenate text + visual for joint attention: [B, N_x+N_y, heads, head_dim]
        # q = torch.cat([q_x, q_y], dim=1)
        # k = torch.cat([k_x, k_y], dim=1)
        # v = torch.cat([v_x, v_y], dim=1)

        attn_metadata = AttentionMetadata(
            joint_query=q_y,
            joint_key=k_y,
            joint_value=v_y,
            joint_strategy="rear",
        )

        # Attention
        out = self.attn(q_x, k_x, v_x, attn_metadata=attn_metadata)  # [B, N_x+N_y, heads, head_dim]
        out = out.flatten(2, 3)  # [B, N_x+N_y, dim]
        out = out.type_as(x)

        x_out, y_out = out.split([N_x, N_y], dim=1)
        x_out = self.proj_x(x_out)
        y_out = proj_y(y_out)
        return x_out, y_out


# ---------------------------------------------------------------------------
# Flux Transformer Block
# ---------------------------------------------------------------------------


class MareyFluxBlock(nn.Module):
    """MMDiT transformer block with AdaLN modulation for both visual (x) and text (y) streams.

    The last `depth_single_blocks` blocks share weights between the two streams.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        share_weights: bool = False,
        rope_channels_ratio: float | None = 0.5,
        rope_dim: int = -1,
        qk_norm: bool = True,
        num_kv_heads: int | None = None,
        add_spatial_pos_emb: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.share_weights = share_weights

        self.modulation_x = nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        self.norm1_x = LlamaRMSNorm(hidden_size)
        self.norm2_x = LlamaRMSNorm(hidden_size)
        self.mlp_x = SwiGLUFFN(hidden_size, int(hidden_size * mlp_ratio))

        self.attn = MareyFluxAttention(
            dim=hidden_size,
            num_heads=num_heads,
            share_weights=share_weights,
            rope_channels_ratio=rope_channels_ratio,
            rope_dim=rope_dim,
            qk_norm=qk_norm,
            num_kv_heads=num_kv_heads,
        )

        if not share_weights:
            self.modulation_y = nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            self.norm1_y = LlamaRMSNorm(hidden_size)
            self.norm2_y = LlamaRMSNorm(hidden_size)
            self.mlp_y = SwiGLUFFN(hidden_size, int(hidden_size * mlp_ratio))

        if add_spatial_pos_emb:
            self.pos_emb_alpha = nn.Parameter(torch.randn(hidden_size) / hidden_size**0.5)
        else:
            self.pos_emb_alpha = None

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t_x: torch.Tensor,
        t_y: torch.Tensor,
        temporal_pos: torch.Tensor | None = None,
        y_temporal_pos: torch.Tensor | None = None,
        spatial_pos_emb: torch.Tensor | None = None,
        x_t_mask: torch.Tensor | None = None,
        t0_x: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.share_weights:
            norm1_y, norm2_y = self.norm1_x, self.norm2_x
            mlp_y = self.mlp_x
            modulation_y = self.modulation_x
        else:
            norm1_y, norm2_y = self.norm1_y, self.norm2_y
            mlp_y = self.mlp_y
            modulation_y = self.modulation_y

        # Optional spatial position embedding
        if self.pos_emb_alpha is not None and spatial_pos_emb is not None:
            x = x + self.pos_emb_alpha[None, None, :] * spatial_pos_emb

        # Modulation: Linear projects t_emb to 6 shift/scale/gate params
        shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x = (
            self.modulation_x(t_x).unsqueeze(1).chunk(6, dim=-1)
        )
        shift_msa_y, scale_msa_y, gate_msa_y, shift_mlp_y, scale_mlp_y, gate_mlp_y = (
            modulation_y(t_y).unsqueeze(1).chunk(6, dim=-1)
        )

        # I2V: compute t0-modulation params for the x-branch (y-branch unused).
        # Reuses the same ``self.modulation_x`` Linear — no extra weights.
        shift_msa_t0_x = scale_msa_t0_x = gate_msa_t0_x = None
        shift_mlp_t0_x = scale_mlp_t0_x = gate_mlp_t0_x = None
        if t0_x is not None:
            (
                shift_msa_t0_x,
                scale_msa_t0_x,
                gate_msa_t0_x,
                shift_mlp_t0_x,
                scale_mlp_t0_x,
                gate_mlp_t0_x,
            ) = self.modulation_x(t0_x).unsqueeze(1).chunk(6, dim=-1)

        _bd = getattr(self, "_block_diag", False)
        if _bd:
            logger.info(
                f"  BLOCK_DIAG modulation_x: "
                f"shift_msa={shift_msa_x.float().mean():.4f}/{shift_msa_x.float().std():.4f} "
                f"scale_msa={scale_msa_x.float().mean():.4f}/{scale_msa_x.float().std():.4f} "
                f"gate_msa={gate_msa_x.float().mean():.4f}/{gate_msa_x.float().std():.4f} "
                f"shift_mlp={shift_mlp_x.float().mean():.4f}/{shift_mlp_x.float().std():.4f} "
                f"scale_mlp={scale_mlp_x.float().mean():.4f}/{scale_mlp_x.float().std():.4f} "
                f"gate_mlp={gate_mlp_x.float().mean():.4f}/{gate_mlp_x.float().std():.4f}"
            )

        # Pre-attention: norm + modulate (x-branch is mask-aware in I2V)
        x_m = t2i_modulate_masked(
            self.norm1_x(x),
            shift_msa_x,
            scale_msa_x,
            mask=x_t_mask,
            shift_t0=shift_msa_t0_x,
            scale_t0=scale_msa_t0_x,
        )
        y_m = t2i_modulate(norm1_y(y), shift_msa_y, scale_msa_y)

        if _bd:
            logger.info(
                f"  BLOCK_DIAG after_norm_mod: x_m std={x_m.float().std():.4f} "
                f"norm1_x std={self.norm1_x(x).float().std():.4f}"
            )

        # Multimodal attention
        x_attn, y_attn = self.attn(x_m, y_m, temporal_pos=temporal_pos, y_temporal_pos=y_temporal_pos)

        if _bd:
            logger.info(
                f"  BLOCK_DIAG attn: x_attn std={x_attn.float().std():.4f} y_attn std={y_attn.float().std():.4f}"
            )

        # Gate + residual (attention). x-branch: mask-aware gate.
        x = x + gate_masked(x_attn, gate_msa_x, mask=x_t_mask, gate_t0=gate_msa_t0_x)
        y = y + y_attn * gate_msa_y

        if _bd:
            logger.info(
                f"  BLOCK_DIAG after_attn_res: x std={x.float().std():.4f} "
                f"gated_attn std={(x_attn * gate_msa_x).float().std():.4f}"
            )

        # MLP: norm + modulate + MLP + gate + residual
        x_m = t2i_modulate_masked(
            self.norm2_x(x),
            shift_mlp_x,
            scale_mlp_x,
            mask=x_t_mask,
            shift_t0=shift_mlp_t0_x,
            scale_t0=scale_mlp_t0_x,
        )
        y_m = t2i_modulate(norm2_y(y), shift_mlp_y, scale_mlp_y)
        mlp_out_x = self.mlp_x(x_m)
        mlp_out_y = mlp_y(y_m)

        if _bd:
            logger.info(
                f"  BLOCK_DIAG mlp: mlp_out_x std={mlp_out_x.float().std():.4f} "
                f"gated_mlp std={(mlp_out_x * gate_mlp_x).float().std():.4f}"
            )

        x = x + gate_masked(mlp_out_x, gate_mlp_x, mask=x_t_mask, gate_t0=gate_mlp_t0_x)
        y = y + mlp_out_y * gate_mlp_y

        return x, y


# ---------------------------------------------------------------------------
# Final layer
# ---------------------------------------------------------------------------


class MareyFinalLayer(nn.Module):
    """Final layer: AdaLN modulation + linear projection."""

    def __init__(self, hidden_size: int, num_patch: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, num_patch * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(t).unsqueeze(1).chunk(2, dim=-1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        # Reshape to separate T,S
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, T, S, C]
        return x


# ---------------------------------------------------------------------------
# Main Transformer
# ---------------------------------------------------------------------------
class SPInputsWrap(nn.Module):
    """Prepares inputs to be sharded by _sp_plan"""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x_hidden_states: torch.Tensor,
        temporal_pos: torch.Tensor,
        spatial_pos_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare hidden_states for SP.

        Args:
            y_hidden_states: [batch, img_seq_len, channels]
            temporal_pos: Temporal position embeddings
            spatial_pos_emb: Spatial position embeddings
        """
        return x_hidden_states, temporal_pos, spatial_pos_emb


class SPOutputWrap(nn.Module):
    """Wraps output to be gathered by _sp_plan"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gather output from SP."""
        return x


def _is_marey_block(name: str, module: nn.Module) -> bool:
    return "blocks" in name and name.split(".")[-1].isdigit()


class MareyTransformer(nn.Module):
    """Marey MMDiT Transformer for video diffusion.

    This implements the Flux/MMDiT architecture where text and visual tokens
    are jointly processed through shared transformer blocks with multimodal attention.

    Sequence Parallelism:
        _sp_plan splits the visual sequence across GPUs:
        - blocks.0: Split hidden_states input
        - proj_out_layer: Gather output

    Args:
        in_channels: VAE latent channels.
        out_channels: Output channels (usually same as in_channels).
        hidden_size: Transformer hidden dimension.
        depth: Number of transformer blocks.
        depth_single_blocks: Number of trailing blocks with shared text/visual weights.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dim multiplier.
        patch_size: 3D patch size for video embedding.
        caption_channels: Text encoder output dimension.
        model_max_length: Maximum text sequence length.
        vector_cond_channels: Vector conditioning dimension (e.g. from CLIP pooled output).
        qk_norm: Whether to apply QK normalization.
        rope_channels_ratio: Fraction of head_dim used for RoPE.
        rope_dim: Dimension for RoPE (-1 for channel, -3 for head).
        add_pos_embed_at_every_block: Add spatial pos embed at every block.
        input_sq_size: Input spatial size for position embedding scaling.
        class_dropout_prob: Dropout probability for classifier-free guidance.
        num_kv_heads: Number of KV heads for GQA (None = MHA).
        learned_pe: Use learned spatial position embeddings.
    """

    _sp_plan = {
        "sp_inputs_wrap": {
            0: SequenceParallelInput(split_dim=2, expected_dims=4, split_output=True),
            1: SequenceParallelInput(split_dim=2, expected_dims=3, split_output=True),
            2: SequenceParallelInput(split_dim=2, expected_dims=4, split_output=True),
        },
        "sp_output_wrap": SequenceParallelOutput(gather_dim=2, expected_dims=4),
    }

    _hsdp_shard_conditions = [_is_marey_block]

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int | None = None,
        hidden_size: int = 1536,
        depth: int = 28,
        depth_single_blocks: int = 0,
        num_heads: int = 24,
        mlp_ratio: float = 4.0,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        caption_channels: int = 4096,
        model_max_length: int = 300,
        vector_cond_channels: int | None = None,
        qk_norm: bool = True,
        rope_channels_ratio: float = 0.5,
        rope_dim: int = -1,
        add_pos_embed_at_every_block: bool = False,
        input_sq_size: int = 512,
        class_dropout_prob: float = 0.1,
        num_kv_heads: int | None = None,
        learned_pe: bool = False,
        extra_features_config: dict[str, dict[str, Any]] | None = None,
        camera_dim: int | None = None,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.depth_single_blocks = depth_single_blocks
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.input_sq_size = input_sq_size
        self.add_pos_embed_at_every_block = add_pos_embed_at_every_block

        # Store config for pipeline compatibility
        self.config = type(
            "Config",
            (),
            {
                "patch_size": patch_size,
                "in_channels": in_channels,
                "out_channels": out_channels,
                "hidden_size": hidden_size,
                "depth": depth,
                "num_heads": num_heads,
            },
        )()

        # Embeddings
        self.x_embedder = PatchEmbed3D(patch_size, in_channels, hidden_size)
        self.pos_embed = PositionEmbedding2D(hidden_size, learned_pe=learned_pe)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
        )
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
            uncond_prob=class_dropout_prob,
            token_num=model_max_length,
            vector_in_channels=vector_cond_channels,
        )
        # Extra features embedders (AR, watermark, aesthetics, etc.)
        self.extra_features_embedders = nn.ModuleDict()
        if extra_features_config:
            for feat, params in extra_features_config.items():
                ftype = params.get("type", "")
                if ftype == "SizeEmbedder":
                    self.extra_features_embedders[feat] = SizeEmbedder(hidden_size)
                elif ftype == "LabelEmbedder":
                    self.extra_features_embedders[feat] = LabelEmbedder(
                        num_classes=params.get("num_classes", 2),
                        hidden_size=hidden_size,
                        dropout_prob=params.get("dropout_prob", 0.0),
                        default_value=params.get("default_value", -1),
                    )
                elif ftype == "OrderedEmbedder":
                    self.extra_features_embedders[feat] = OrderedEmbedder(
                        num_classes=params.get("num_classes", 10),
                        hidden_size=hidden_size,
                        dropout_prob=params.get("dropout_prob", 0.0),
                        default_value=params.get("default_value", -1),
                    )

        if "fps" not in self.extra_features_embedders:
            self.fps_embedder = SizeEmbedder(hidden_size)

        if camera_dim is not None:
            self.camera_embedder = VectorEmbedder(hidden_size, vector_size=camera_dim)

        rope_channels = int(hidden_size // num_heads * rope_channels_ratio)
        rope_channels = (rope_channels // 2) * 2
        self.rope = _RoPEFreqsHolder(rope_channels // 2)

        # Transformer blocks
        assert depth_single_blocks <= depth

        def _share_weights(i: int) -> bool:
            return i >= depth - depth_single_blocks

        self.blocks = nn.ModuleList(
            [
                MareyFluxBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    share_weights=_share_weights(i),
                    rope_channels_ratio=rope_channels_ratio,
                    rope_dim=rope_dim,
                    qk_norm=qk_norm,
                    num_kv_heads=num_kv_heads,
                    add_spatial_pos_emb=add_pos_embed_at_every_block,
                )
                for i in range(depth)
            ]
        )

        # Final layer
        self.final_layer = MareyFinalLayer(hidden_size, int(np.prod(patch_size)), out_channels)

        # Wrapper for SP
        self.sp_inputs_wrap = SPInputsWrap()
        self.sp_output_wrap = SPOutputWrap()

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def get_dynamic_size(self, x: torch.Tensor) -> tuple[int, int, int]:
        _, _, T, H, W = x.size()
        T = math.ceil(T / self.patch_size[0])
        H = math.ceil(H / self.patch_size[1])
        W = math.ceil(W / self.patch_size[2])
        return T, H, W

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        encoder_hidden_states_mask: torch.Tensor | list[torch.Tensor] | None = None,
        vector_cond: torch.Tensor | None = None,
        height: torch.Tensor | None = None,
        width: torch.Tensor | None = None,
        fps: torch.Tensor | None = None,
        extra_features: dict[str, torch.Tensor] | None = None,
        cond_frames: torch.Tensor | None = None,
        cond_offsets: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor | Transformer2DModelOutput:
        """
        Args:
            hidden_states: Noised latents [B, C, T, H, W].
            timestep: Diffusion timestep [B].
            encoder_hidden_states: Text embeddings [B, seq_len, text_dim].
            encoder_hidden_states_mask: Text mask [B, seq_len].
            vector_cond: Pooled text embedding [B, vector_dim] (e.g. CLIP).
            height: Video height [B] (for position embedding scaling).
            width: Video width [B].
            fps: Frames per second [B] (for FPS conditioning).
            cond_frames: (I2V) Conditioning latents [B, C, Tf, H, W]. When provided,
                these are patch-embedded and concatenated onto ``hidden_states``
                along the temporal axis; the appended positions are modulated as
                ``timestep=0`` via ``t0_emb`` and have their RoPE temporal positions
                set by ``cond_offsets`` so the model sees them at their true target
                frame locations. Mirrors mv ``dit3d.py:1020-1043``.
            cond_offsets: (I2V) Per-keyframe latent-time offsets of shape [Tf].
        """
        dtype = self.x_embedder.proj.weight.dtype
        B = hidden_states.size(0)
        hidden_states = hidden_states.to(dtype)
        timestep = timestep.to(dtype)

        # Compute spatial sizes
        T, H, W = self.get_dynamic_size(hidden_states)
        S = H * W

        # Spatial position embedding
        base_size = round(S**0.5)
        if height is not None and width is not None:
            resolution_sq = (height[0].item() * width[0].item()) ** 0.5
        else:
            resolution_sq = float(self.input_sq_size)
        scale = resolution_sq / self.input_sq_size
        spatial_pos_emb = self.pos_embed(hidden_states, H, W, scale=scale, base_size=base_size)
        spatial_pos_emb = spatial_pos_emb[None]  # [1, 1, S, C]

        # Text embedding
        seq_cond, seq_cond_mask, vec_cond = self.y_embedder(
            encoder_hidden_states, encoder_hidden_states_mask, vector_cond, train=self.training
        )

        _diag = getattr(self, "_diag", False)
        if _diag:
            _sf = seq_cond.float()
            logger.info(
                f"DIAG y_embedder: seq_cond {list(seq_cond.shape)} "
                f"mean={_sf.mean():.4f} std={_sf.std():.4f} "
                f"per_token_std={_sf.std(dim=-1).mean():.4f} "
                f"vec_cond mean={vec_cond.float().mean():.4f} std={vec_cond.float().std():.4f}"
            )

        # FPS conditioning mixed into vector_cond
        if fps is not None:
            if T == 1:
                fps = 120.0 * torch.ones_like(fps)
            fps_mod = (
                self.extra_features_embedders["fps"]
                if "fps" in self.extra_features_embedders
                else getattr(self, "fps_embedder", None)
            )
            if fps_mod is not None:
                vec_cond = vec_cond + fps_mod(fps)

        # Extra feature embeddings (AR, watermark, aesthetics, etc.)
        if extra_features and self.extra_features_embedders:
            for feat, val in extra_features.items():
                if feat == "fps":
                    continue
                if feat in self.extra_features_embedders:
                    vec_cond = vec_cond + self.extra_features_embedders[feat](val)
        elif hasattr(self, "_extra_vec_cond") and self._extra_vec_cond is not None:
            vec_cond = vec_cond + self._extra_vec_cond

        # Timestep embedding
        t_emb = self.t_embedder(timestep, dtype=dtype)
        t_emb = t_emb + vec_cond
        t_mlp = self.t_block(t_emb)  # [B, hidden_size]
        t_emb_final = t_emb

        # I2V: precompute the ``timestep=0`` path so cond-frame tokens can be
        # modulated with t0 inside the blocks. Same ``t_embedder`` / ``t_block``
        # weights (zero new params). Mirrors mv ``dit3d.py:1008``.
        t0_mlp: torch.Tensor | None = None
        if cond_frames is not None:
            t0_emb = self.t_embedder(torch.zeros_like(timestep), dtype=dtype) + vec_cond
            t0_mlp = self.t_block(t0_emb)

        if _diag:
            logger.info(
                f"DIAG t_emb: mean={t_emb.float().mean():.4f} std={t_emb.float().std():.4f} "
                f"t_mlp: mean={t_mlp.float().mean():.4f} std={t_mlp.float().std():.4f}"
            )

        # Patch embed + spatial pos
        x = self.x_embedder(hidden_states)  # [B, T*S, C]
        x = x.reshape(B, T, S, -1)
        if not self.add_pos_embed_at_every_block:
            x = x + spatial_pos_emb

        # I2V: embed conditioning frames the same way and concat along T.
        # Mirrors mv ``dit3d.py:1018-1043`` (cond_frames_concat_to_stack path).
        Tf = 0
        x_t_mask: torch.Tensor | None = None
        if cond_frames is not None:
            Tf_local, Hf, Wf = self.get_dynamic_size(cond_frames)
            assert Hf == H and Wf == W, f"cond_frames spatial dims ({Hf}x{Wf}) must match hidden_states ({H}x{W})"
            Tf = Tf_local
            cond_frames_emb = self.x_embedder(cond_frames.to(dtype))  # [B, Tf*S, C]
            cond_frames_emb = cond_frames_emb.reshape(B, Tf, S, -1)
            # Unlike ``x`` above, cond frames ALWAYS receive the initial
            # spatial_pos_emb, even when ``add_pos_embed_at_every_block=True``.
            # Mirrors mv ``dit3d.py:1033`` — the per-block pos_emb is then
            # added on top for both noise and cond tokens during the block
            # loop, giving cond tokens a double-strength positional signal
            # that the checkpoint was trained with.
            cond_frames_emb = cond_frames_emb + spatial_pos_emb
            x = torch.cat([x, cond_frames_emb], dim=1)  # [B, T+Tf, S, C]
            # Mask: True for noise frames (first T), False for cond frames (last Tf).
            mask_1d = torch.arange(T + Tf, device=x.device) < T
            x_t_mask = mask_1d.view(1, T + Tf, 1, 1).expand(B, T + Tf, S, 1).contiguous()
            T = T + Tf

        x = x.reshape(B, T * S, -1)

        if _diag:
            _xf = x.float()
            logger.info(
                f"DIAG after x_embed: mean={_xf.mean():.4f} std={_xf.std():.4f} spatial_var={_xf.var(dim=1).mean():.6f}"
            )

        # Prepare spatial_pos_emb for per-block use
        # Always create it because sp inputs needs to split it; if add_pos_embed_at_every_block is false,
        # it is set to None below.
        spatial_pos_emb_blocks = spatial_pos_emb.expand(-1, T, -1, -1).reshape(1, T * S, -1)

        # Temporal positions for RoPE (I2V: append cond_offsets for the appended cond frames).
        temporal_pos = get_temporal_pos(x, T, S, cond_offsets)

        # Pass through SP wrapper for _sp_plan to auto shard them
        temporal_pos = temporal_pos.reshape(temporal_pos.shape[0], T, S)  # [B, T, S]
        spatial_pos_emb_blocks = spatial_pos_emb_blocks.reshape(spatial_pos_emb.shape[0], T, S, -1)  # [1, T, S, C]

        x = x.reshape(B, T, S, -1)  # [B, T, S, C]
        x, temporal_pos, spatial_pos_emb_blocks = self.sp_inputs_wrap(x, temporal_pos, spatial_pos_emb_blocks)
        S_new = x.shape[2]
        S_full = S
        S = S_new
        x = x.reshape(B, T * S, -1)
        temporal_pos = temporal_pos.reshape(temporal_pos.shape[0], T * S)
        spatial_pos_emb_blocks = spatial_pos_emb_blocks.reshape(1, T * S, -1)

        # I2V: shard x_t_mask along the same S dim as x, bypassing the _sp_plan
        # framework (which would require x_t_mask to be non-None even for T2V).
        if x_t_mask is not None:
            from vllm_omni.diffusion.distributed.sp_sharding import sp_shard

            x_t_mask = sp_shard(x_t_mask, dim=2, validate=False)  # [1, T, S_local, 1]
            x_t_mask = x_t_mask.reshape(B, T * S, 1)

        if not self.add_pos_embed_at_every_block:
            spatial_pos_emb_blocks = None

        # Zero out padding text tokens so they don't pollute joint attention
        # (reference: Flux.rearrange_tokens_for_attention applies y * mask)
        if seq_cond_mask is not None:
            y = seq_cond * seq_cond_mask.unsqueeze(-1)
        else:
            y = seq_cond
        y_t_emb = t_mlp

        # Transformer blocks
        if _diag and len(self.blocks) > 0:
            self.blocks[0]._block_diag = True
        for bi, block in enumerate(self.blocks):
            x, y = block(
                x,
                y,
                t_x=t_mlp,
                t_y=y_t_emb,
                temporal_pos=temporal_pos,
                spatial_pos_emb=spatial_pos_emb_blocks,
                x_t_mask=x_t_mask,
                t0_x=t0_mlp,
            )
            if _diag and bi in (0, 20, len(self.blocks) - 1):
                _xf = x.float()
                logger.info(
                    f"DIAG block {bi}: x mean={_xf.mean():.4f} std={_xf.std():.4f} "
                    f"spatial_var={_xf.var(dim=1).mean():.6f} "
                    f"y mean={y.float().mean():.4f} std={y.float().std():.4f}"
                )

        if _diag and len(self.blocks) > 0:
            self.blocks[0]._block_diag = False

        # Final layer
        x = self.final_layer(x, t_emb_final)
        x = x.reshape(x.shape[0], T, S, -1)
        x = self.sp_output_wrap(x)

        # I2V: drop the appended cond-frame positions before unpatchify so the
        # returned tensor matches the input noise-latent shape. Mirrors mv
        # ``dit3d.py:1128-1132``.
        if Tf > 0:
            T = T - Tf
            x = x[:, :T]

        x = x.reshape(x.shape[0], T * S_full, -1)
        # Unpatchify
        x = self._unpatchify(x, T, H, W, hidden_states)

        x = x.to(torch.float32)

        if not return_dict:
            return (x,)
        return Transformer2DModelOutput(sample=x)

    def _unpatchify(
        self,
        x: torch.Tensor,
        n_t: int,
        n_h: int,
        n_w: int,
        original_input: torch.Tensor,
    ) -> torch.Tensor:
        """Convert patch sequence back to [B, C, T, H, W]."""
        t_p, h_p, w_p = self.patch_size
        _, _, r_t, r_h, r_w = original_input.shape
        x = x.reshape(
            x.shape[0],
            n_t,
            n_h,
            n_w,
            t_p,
            h_p,
            w_p,
            self.out_channels,
        )
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)  # [B, C, n_t, t_p, n_h, h_p, n_w, w_p]
        x = x.reshape(
            x.shape[0],
            self.out_channels,
            n_t * t_p,
            n_h * h_p,
            n_w * w_p,
        )
        # Crop to original size (remove padding)
        x = x[:, :, :r_t, :r_h, :r_w]
        return x

    _MLP_WEIGHT_MAP = {
        ".mlp_x.fc1_x.": ".mlp_x.w1.",
        ".mlp_x.fc1_g.": ".mlp_x.w2.",
        ".mlp_x.fc2.": ".mlp_x.w3.",
        ".mlp_y.fc1_x.": ".mlp_y.w1.",
        ".mlp_y.fc1_g.": ".mlp_y.w2.",
        ".mlp_y.fc2.": ".mlp_y.w3.",
    }

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with handling for TP sharding and projection remapping."""
        params_dict = dict(self.named_parameters())
        buffers_dict = dict(self.named_buffers())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            original_name = name

            for old, new in self._MLP_WEIGHT_MAP.items():
                if old in name:
                    name = name.replace(old, new)
                    break

            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(original_name)
            elif name in buffers_dict:
                buffers_dict[name].copy_(loaded_weight)
                loaded_params.add(original_name)
            else:
                logger.warning(f"Skipping weight {original_name}: not found in model")

        return loaded_params
