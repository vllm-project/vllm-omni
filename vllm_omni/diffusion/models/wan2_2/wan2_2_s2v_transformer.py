# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Wan2.2 Speech-to-Video (S2V) Transformer using vllm-omni ops.

Audio and motion modules use standard nn.Linear (no TP needed).
"""

import math
from collections.abc import Iterable
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import AdaLayerNorm
from diffusers.models.normalization import FP32LayerNorm
from einops import rearrange, repeat
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.platforms import current_omni_platform

from .wan2_2_transformer import DistributedRMSNorm, WanFeedForward

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# S2V-specific utility functions
# ---------------------------------------------------------------------------


def sinusoidal_embedding_1d(dim, position):
    if dim % 2 != 0:
        raise ValueError(f"dim ({dim}) must be even")
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast(current_omni_platform.device_type, enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    if dim % 2 != 0:
        raise ValueError(f"dim ({dim}) must be even")
    freqs = torch.outer(
        torch.arange(max_seq_len), 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def rope_precompute(x, grid_sizes, freqs, start=None):
    """Precompute complex-valued RoPE embeddings for S2V multi-grid positions."""
    b, s, n, c = x.size(0), x.size(1), x.size(2), x.size(3) // 2

    if isinstance(freqs, list):
        trainable_freqs = freqs[1]
        freqs = freqs[0]
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = torch.view_as_complex(x.detach().reshape(b, s, n, -1, 2).to(torch.float64))
    seq_bucket = [0]
    if not isinstance(grid_sizes, list):
        grid_sizes = [grid_sizes]
    for g in grid_sizes:
        if not isinstance(g, list):
            g = [torch.zeros_like(g), g]
        batch_size = g[0].shape[0]
        for i in range(batch_size):
            if start is None:
                f_o, h_o, w_o = g[0][i]
            else:
                f_o, h_o, w_o = start[i]

            f, h, w = g[1][i]
            t_f, t_h, t_w = g[2][i]
            seq_f, seq_h, seq_w = f - f_o, h - h_o, w - w_o
            seq_len = int(seq_f * seq_h * seq_w)
            if seq_len > 0:
                if t_f > 0:
                    if f_o >= 0:
                        f_sam = np.linspace(f_o.item(), (t_f + f_o).item() - 1, seq_f).astype(int).tolist()
                    else:
                        f_sam = np.linspace(-f_o.item(), (-t_f - f_o).item() + 1, seq_f).astype(int).tolist()
                    h_sam = np.linspace(h_o.item(), (t_h + h_o).item() - 1, seq_h).astype(int).tolist()
                    w_sam = np.linspace(w_o.item(), (t_w + w_o).item() - 1, seq_w).astype(int).tolist()

                    assert f_o * f >= 0 and h_o * h >= 0 and w_o * w >= 0
                    freqs_0 = freqs[0][f_sam] if f_o >= 0 else freqs[0][f_sam].conj()
                    freqs_0 = freqs_0.view(seq_f, 1, 1, -1)

                    freqs_i = torch.cat(
                        [
                            freqs_0.expand(seq_f, seq_h, seq_w, -1),
                            freqs[1][h_sam].view(1, seq_h, 1, -1).expand(seq_f, seq_h, seq_w, -1),
                            freqs[2][w_sam].view(1, 1, seq_w, -1).expand(seq_f, seq_h, seq_w, -1),
                        ],
                        dim=-1,
                    ).reshape(seq_len, 1, -1)
                elif t_f < 0:
                    freqs_i = trainable_freqs.unsqueeze(1)
                output[i, seq_bucket[-1] : seq_bucket[-1] + seq_len] = freqs_i
        seq_bucket.append(seq_bucket[-1] + seq_len)
    return output


@torch.amp.autocast(current_omni_platform.device_type, enabled=False)
def rope_apply(x, grid_sizes, freqs, start=None):
    """Apply RoPE using grid-based frequency computation (for motioner/init)."""
    n, c = x.size(2), x.size(3) // 2
    input_dtype = x.dtype

    if isinstance(freqs, list):
        trainable_freqs = freqs[1]
        freqs = freqs[0]
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = x.clone()
    seq_bucket = [0]
    if not isinstance(grid_sizes, list):
        grid_sizes = [grid_sizes]
    for g in grid_sizes:
        if not isinstance(g, list):
            g = [torch.zeros_like(g), g]
        batch_size = g[0].shape[0]
        for i in range(batch_size):
            if start is None:
                f_o, h_o, w_o = g[0][i]
            else:
                f_o, h_o, w_o = start[i]

            f, h, w = g[1][i]
            t_f, t_h, t_w = g[2][i]
            seq_f, seq_h, seq_w = f - f_o, h - h_o, w - w_o
            seq_len = int(seq_f * seq_h * seq_w)
            if seq_len > 0:
                if t_f > 0:
                    if f_o >= 0:
                        f_sam = np.linspace(f_o.item(), (t_f + f_o).item() - 1, seq_f).astype(int).tolist()
                    else:
                        f_sam = np.linspace(-f_o.item(), (-t_f - f_o).item() + 1, seq_f).astype(int).tolist()
                    h_sam = np.linspace(h_o.item(), (t_h + h_o).item() - 1, seq_h).astype(int).tolist()
                    w_sam = np.linspace(w_o.item(), (t_w + w_o).item() - 1, seq_w).astype(int).tolist()

                    assert f_o * f >= 0 and h_o * h >= 0 and w_o * w >= 0
                    freqs_0 = freqs[0][f_sam] if f_o >= 0 else freqs[0][f_sam].conj()
                    freqs_0 = freqs_0.view(seq_f, 1, 1, -1)

                    freqs_i = torch.cat(
                        [
                            freqs_0.expand(seq_f, seq_h, seq_w, -1),
                            freqs[1][h_sam].view(1, seq_h, 1, -1).expand(seq_f, seq_h, seq_w, -1),
                            freqs[2][w_sam].view(1, 1, seq_w, -1).expand(seq_f, seq_h, seq_w, -1),
                        ],
                        dim=-1,
                    ).reshape(seq_len, 1, -1)
                elif t_f < 0:
                    freqs_i = trainable_freqs.unsqueeze(1)
                x_i = torch.view_as_complex(
                    x[i, seq_bucket[-1] : seq_bucket[-1] + seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
                )
                x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
                output[i, seq_bucket[-1] : seq_bucket[-1] + seq_len] = x_i
        seq_bucket.append(seq_bucket[-1] + seq_len)
    return output.to(input_dtype)


@torch.compiler.disable
@torch.amp.autocast(current_omni_platform.device_type, enabled=False)
def rope_apply_s2v(x, grid_sizes, freqs, start=None):
    """Apply RoPE using precomputed complex freqs (for S2V main transformer).

    Under TP, x has fewer heads than freqs (which is precomputed with all heads).
    Since RoPE frequencies are identical across heads, we slice freqs to match.
    """
    n = x.size(2)
    input_dtype = x.dtype
    output = []
    for i, _ in enumerate(x):
        s = x.size(1)
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(s, n, -1, 2))
        freqs_i = freqs[i, :s, :n]
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])
        output.append(x_i)
    return torch.stack(output).to(input_dtype)


@torch.amp.autocast(current_omni_platform.device_type, enabled=False)
def rope_apply_usp(x, grid_sizes, freqs):
    """Apply RoPE for Ulysses sequence parallel (context parallel mode)."""
    n = x.size(2)
    input_dtype = x.dtype
    output = []
    for i, _ in enumerate(x):
        s = x.size(1)
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(s, n, -1, 2))
        freqs_i = freqs[i, :, :n]
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])
        output.append(x_i)
    return torch.stack(output).to(input_dtype)


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def torch_dfs(model, parent_name="root"):
    module_names, modules = [], []
    current_name = parent_name if parent_name else "root"
    module_names.append(current_name)
    modules.append(model)
    for name, child in model.named_children():
        if parent_name:
            child_name = f"{parent_name}.{name}"
        else:
            child_name = name
        child_modules, child_names = torch_dfs(child, child_name)
        module_names += child_names
        modules += child_modules
    return modules, module_names


# ---------------------------------------------------------------------------
# TP-enabled main transformer building blocks
# ---------------------------------------------------------------------------


class WanS2VSelfAttention(nn.Module):
    """S2V self-attention using vllm-omni ops (QKVParallelLinear + Attention)."""

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.inner_dim = num_heads * self.head_dim

        # Fused QKV projection
        self.to_qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=self.head_dim,
            total_num_heads=num_heads,
            bias=True,
            quant_config=None,
        )

        self.num_heads = self.to_qkv.num_heads
        self.num_kv_heads = self.to_qkv.num_kv_heads
        self.tp_inner_dim = self.num_heads * self.head_dim

        # QK normalization (TP-aware)
        self.norm_q = DistributedRMSNorm(self.tp_inner_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = DistributedRMSNorm(self.tp_inner_dim, eps=eps) if qk_norm else nn.Identity()

        # Output projection
        self.to_out = RowParallelLinear(
            self.inner_dim,
            dim,
            bias=True,
            input_is_parallel=True,
            return_bias=False,
        )

        # Unified attention layer
        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
        )

    def forward(self, x, seq_lens, grid_sizes, freqs):
        b, s = x.shape[:2]

        # Fused QKV
        qkv, _ = self.to_qkv(x)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)

        # QK normalization
        query = self.norm_q(query)
        key = self.norm_k(key)

        # Reshape to [B, S, N, D]
        query = query.view(b, s, self.num_heads, self.head_dim)
        key = key.view(b, s, self.num_kv_heads, self.head_dim)
        value = value.view(b, s, self.num_kv_heads, self.head_dim)

        # Apply S2V complex-valued RoPE
        query = rope_apply_s2v(query, grid_sizes, freqs)
        key = rope_apply_s2v(key, grid_sizes, freqs)

        # Reshape back after rope (rope_apply_s2v preserves [B, S, N, D])
        query = query.view(b, s, self.num_heads, self.head_dim)
        key = key.view(b, s, self.num_kv_heads, self.head_dim)

        # Attention — rope_apply_s2v preserves input dtype (bf16), so
        # query/key/value are all bf16 here, matching FlashAttention requirements.
        hidden_states = self.attn(query, key, value)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(x)

        # Output
        hidden_states = self.to_out(hidden_states)
        return hidden_states


class WanS2VCrossAttention(nn.Module):
    """S2V cross-attention using vllm-omni ops (ColumnParallelLinear + Attention)."""

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.inner_dim = num_heads * self.head_dim

        # Separate Q/K/V projections for cross-attention
        self.to_q = ColumnParallelLinear(dim, self.inner_dim, bias=True, gather_output=False, return_bias=False)
        self.to_k = ColumnParallelLinear(dim, self.inner_dim, bias=True, gather_output=False, return_bias=False)
        self.to_v = ColumnParallelLinear(dim, self.inner_dim, bias=True, gather_output=False, return_bias=False)

        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads = num_heads // tp_size
        self.tp_inner_dim = self.num_heads * self.head_dim

        # QK normalization (TP-aware)
        self.norm_q = DistributedRMSNorm(self.tp_inner_dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = DistributedRMSNorm(self.tp_inner_dim, eps=eps) if qk_norm else nn.Identity()

        # Output projection
        self.to_out = RowParallelLinear(self.inner_dim, dim, bias=True, input_is_parallel=True, return_bias=False)

        # Unified attention layer
        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            num_kv_heads=self.num_heads,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
            skip_sequence_parallel=True,
        )

    def forward(self, x, context, context_lens=None):
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.norm_q(self.to_q(x)).view(b, -1, n, d)
        k = self.norm_k(self.to_k(context)).view(b, -1, n, d)
        v = self.to_v(context).view(b, -1, n, d)

        x = self.attn(q, k, v)
        x = x.flatten(2, 3)
        x = x.type_as(q)

        x = self.to_out(x)
        return x


class WanS2VTransformerBlock(nn.Module):
    """S2V transformer block with segment-wise modulation.

    Key S2V feature: `e` is a tuple [e0_tensor, seg_idx]. The sequence is split
    into 2 segments at seg_idx (noisy tokens vs ref/motion tokens), and different
    modulation is applied to each segment.
    """

    def __init__(self, dim, ffn_dim, num_heads, window_size=(-1, -1), qk_norm=True, cross_attn_norm=False, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads

        # Self-attention (TP-enabled)
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.self_attn = WanS2VSelfAttention(dim, num_heads, window_size, qk_norm, eps)

        # Cross-attention (TP-enabled)
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanS2VCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)

        # FFN (TP-enabled, reused from wan2_2_transformer.py)
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.ffn = WanFeedForward(dim=dim, inner_dim=ffn_dim, dim_out=dim)

        # 6-way scale-shift modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
        seg_boundary = e[1]
        seg_idx = [0, min(max(0, seg_boundary), x.size(1)), x.size(1)]
        e = e[0]

        # Modulation parameters — self.modulation will be cast to match e's dtype
        e = (self.modulation.unsqueeze(2).to(e.dtype) + e).chunk(6, dim=1)
        e = [element.squeeze(1) for element in e]

        # Self-attention with per-segment modulation
        norm_x = self.norm1(x).type_as(x)
        parts = []
        for i in range(2):
            parts.append(norm_x[:, seg_idx[i] : seg_idx[i + 1]] * (1 + e[1][:, i : i + 1]) + e[0][:, i : i + 1])
        norm_x = torch.cat(parts, dim=1)

        y = self.self_attn(norm_x, seq_lens, grid_sizes, freqs)
        y_parts = []
        for i in range(2):
            y_parts.append(y[:, seg_idx[i] : seg_idx[i + 1]] * e[2][:, i : i + 1])
        x = x + torch.cat(y_parts, dim=1)

        # Cross-attention
        x = x + self.cross_attn(self.norm3(x).type_as(x), context, context_lens)

        # FFN with per-segment modulation
        norm2_x = self.norm2(x).type_as(x)
        parts = []
        for i in range(2):
            parts.append(norm2_x[:, seg_idx[i] : seg_idx[i + 1]] * (1 + e[4][:, i : i + 1]) + e[3][:, i : i + 1])
        norm2_x = torch.cat(parts, dim=1)
        y = self.ffn(norm2_x)
        y_parts = []
        for i in range(2):
            y_parts.append(y[:, seg_idx[i] : seg_idx[i + 1]] * e[5][:, i : i + 1])
        x = x + torch.cat(y_parts, dim=1)
        return x


class WanS2VHead(nn.Module):
    """S2V output head with modulation."""

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size

        out_dim_full = math.prod(patch_size) * out_dim
        self.norm = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim_full)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        e = (self.modulation.to(e.dtype) + e.unsqueeze(1)).chunk(2, dim=1)
        x = self.norm(x).type_as(x) * (1 + e[1]) + e[0]
        x = self.head(x)
        return x


# ---------------------------------------------------------------------------
# Audio modules (no TP, keep as nn.Linear)
# ---------------------------------------------------------------------------


class CausalConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size=3, stride=1, dilation=1, pad_mode="replicate", **kwargs):
        super().__init__()
        self.pad_mode = pad_mode
        self.time_causal_padding = (kernel_size - 1, 0)
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)


class MotionEncoder_tc(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads=4, need_global=True, dtype=None, device=None):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.num_heads = num_heads
        self.need_global = need_global

        self.conv1_local = CausalConv1d(in_dim, hidden_dim // 4 * num_heads, 3, stride=1)
        if need_global:
            self.conv1_global = CausalConv1d(in_dim, hidden_dim // 4, 3, stride=1)
        self.act = nn.SiLU()
        self.conv2 = CausalConv1d(hidden_dim // 4, hidden_dim // 2, 3, stride=2)
        self.conv3 = CausalConv1d(hidden_dim // 2, hidden_dim, 3, stride=2)

        if need_global:
            self.final_linear = nn.Linear(hidden_dim, hidden_dim, **factory_kwargs)

        self.norm1 = nn.LayerNorm(hidden_dim // 4, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.norm2 = nn.LayerNorm(hidden_dim // 2, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))

    def forward(self, x):
        x = rearrange(x, "b t c -> b c t")
        x_ori = x.clone()
        b, c, t = x.shape
        x = self.conv1_local(x)
        x = rearrange(x, "b (n c) t -> (b n) t c", n=self.num_heads)
        x = self.norm1(x)
        x = self.act(x)
        x = rearrange(x, "b t c -> b c t")
        x = self.conv2(x)
        x = rearrange(x, "b c t -> b t c")
        x = self.norm2(x)
        x = self.act(x)
        x = rearrange(x, "b t c -> b c t")
        x = self.conv3(x)
        x = rearrange(x, "b c t -> b t c")
        x = self.norm3(x)
        x = self.act(x)
        x = rearrange(x, "(b n) t c -> b t n c", b=b)
        padding = self.padding_tokens.repeat(b, x.shape[1], 1, 1)
        x = torch.cat([x, padding], dim=-2)
        x_local = x.clone()

        if not self.need_global:
            return x_local

        x = self.conv1_global(x_ori)
        x = rearrange(x, "b c t -> b t c")
        x = self.norm1(x)
        x = self.act(x)
        x = rearrange(x, "b t c -> b c t")
        x = self.conv2(x)
        x = rearrange(x, "b c t -> b t c")
        x = self.norm2(x)
        x = self.act(x)
        x = rearrange(x, "b t c -> b c t")
        x = self.conv3(x)
        x = rearrange(x, "b c t -> b t c")
        x = self.norm3(x)
        x = self.act(x)
        x = self.final_linear(x)
        x = rearrange(x, "(b n) t c -> b t n c", b=b)
        return x, x_local


class CausalAudioEncoder(nn.Module):
    def __init__(self, dim=5120, num_layers=25, out_dim=2048, video_rate=8, num_token=4, need_global=False):
        super().__init__()
        self.encoder = MotionEncoder_tc(in_dim=dim, hidden_dim=out_dim, num_heads=num_token, need_global=need_global)
        weight = torch.ones((1, num_layers, 1, 1)) * 0.01
        self.weights = torch.nn.Parameter(weight)
        self.act = torch.nn.SiLU()

    def forward(self, features):
        # weights param is float32; cast to features dtype for the weighted sum
        weights = self.act(self.weights).to(features.dtype)
        weights_sum = weights.sum(dim=1, keepdims=True)
        weighted_feat = ((features * weights) / weights_sum).sum(dim=1)
        weighted_feat = weighted_feat.permute(0, 2, 1)
        res = self.encoder(weighted_feat)
        return res


class AudioCrossAttention(nn.Module):
    """Audio cross-attention using nn.Linear (no TP needed)."""

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        self.attn = Attention(
            num_heads=num_heads,
            head_size=self.head_dim,
            num_kv_heads=num_heads,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor, context_lens: list[int] | None = None) -> torch.Tensor:
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        x = self.attn(q, k, v)
        x = x.flatten(2, 3)
        x = self.o(x)
        return x


class AudioInjector_WAN(nn.Module):
    def __init__(
        self,
        all_modules,
        all_modules_names,
        dim=2048,
        num_heads=32,
        inject_layer=None,
        root_net=None,
        enable_adain=False,
        adain_dim=2048,
        need_adain_ont=False,
    ):
        super().__init__()
        if inject_layer is None:
            inject_layer = [0, 27]

        self.injected_block_id = {}
        audio_injector_id = 0
        for mod_name, mod in zip(all_modules_names, all_modules):
            if isinstance(mod, WanS2VTransformerBlock):
                for inject_id in inject_layer:
                    if f"transformer_blocks.{inject_id}" in mod_name:
                        self.injected_block_id[inject_id] = audio_injector_id
                        audio_injector_id += 1

        self.injector = nn.ModuleList(
            [AudioCrossAttention(dim=dim, num_heads=num_heads, qk_norm=True) for _ in range(audio_injector_id)]
        )
        self.injector_pre_norm_feat = nn.ModuleList(
            [nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6) for _ in range(audio_injector_id)]
        )
        self.injector_pre_norm_vec = nn.ModuleList(
            [nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6) for _ in range(audio_injector_id)]
        )
        if enable_adain:
            self.injector_adain_layers = nn.ModuleList(
                [
                    AdaLayerNorm(output_dim=dim * 2, embedding_dim=adain_dim, chunk_dim=1)
                    for _ in range(audio_injector_id)
                ]
            )
            if need_adain_ont:
                self.injector_adain_output_layers = nn.ModuleList(
                    [nn.Linear(dim, dim) for _ in range(audio_injector_id)]
                )


# ---------------------------------------------------------------------------
# Motion modules (no TP, keep as nn.Linear)
# ---------------------------------------------------------------------------


class SimpleSelfAttention(nn.Module):
    """Simple self-attention for motioner (no TP, uses nn.Linear)."""

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        self.attn = Attention(
            num_heads=num_heads,
            head_size=self.head_dim,
            num_kv_heads=num_heads,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
        )

    def forward(
        self, x: torch.Tensor, seq_lens: list[int], grid_sizes: list[tuple[int, int, int]], freqs: torch.Tensor
    ) -> torch.Tensor:
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)

        # rope_apply preserves [B, S, N, D] shape and input dtype
        q = q.view(b, s, n, d)
        k = k.view(b, s, n, d)

        x = self.attn(q, k, v)
        x = x.flatten(2, 3)
        x = self.o(x)
        return x


class SwinSelfAttention(SimpleSelfAttention):
    def forward(
        self, x: torch.Tensor, seq_lens: list[int], grid_sizes: list[tuple[int, int, int]], freqs: torch.Tensor
    ) -> torch.Tensor:
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if b != 1:
            raise ValueError(
                f"SwinSelfAttention only supports batch_size=1, got batch_size={b}. "
                "Batched inference requires refactoring the frame reference and rearrange logic."
            )

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)
        q = q.view(b, s, n, d)
        k = k.view(b, s, n, d)
        T, H, W = grid_sizes[0].tolist()

        q = rearrange(q, "b (t h w) n d -> (b t) (h w) n d", t=T, h=H, w=W)
        k = rearrange(k, "b (t h w) n d -> (b t) (h w) n d", t=T, h=H, w=W)
        v = rearrange(v, "b (t h w) n d -> (b t) (h w) n d", t=T, h=H, w=W)

        # Skip ref_q (last frame query) - unused in current implementation
        q = q[:-1]

        ref_k = repeat(k[-1:], "1 s n d -> t s n d", t=k.shape[0] - 1)
        k = k[:-1]
        k = torch.cat([k[:1], k, k[-1:]])
        k = torch.cat([k[1:-1], k[2:], k[:-2], ref_k], dim=1)

        ref_v = repeat(v[-1:], "1 s n d -> t s n d", t=v.shape[0] - 1)
        v = v[:-1]
        v = torch.cat([v[:1], v, v[-1:]])
        v = torch.cat([v[1:-1], v[2:], v[:-2], ref_v], dim=1)

        out = self.attn(q, k, v)
        out = torch.cat([out, ref_v[:1]], axis=0)
        out = rearrange(out, "(b t) (h w) n d -> b (t h w) n d", t=T, h=H, w=W)
        x = out.flatten(2, 3)
        x = self.o(x)
        return x


class CausalSelfAttention(SimpleSelfAttention):
    def forward(
        self, x: torch.Tensor, seq_lens: list[int], grid_sizes: list[tuple[int, int, int]], freqs: torch.Tensor
    ) -> torch.Tensor:
        shifting = 3
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if b != 1:
            raise ValueError(
                f"CausalSelfAttention only supports batch_size=1, got batch_size={b}. "
                "Batched inference requires refactoring the causal masking and frame reference logic."
            )

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        T, H, W = grid_sizes[0].tolist()

        q = rearrange(q, "b (t h w) n d -> (b t) (h w) n d", t=T, h=H, w=W)
        k = rearrange(k, "b (t h w) n d -> (b t) (h w) n d", t=T, h=H, w=W)
        v = rearrange(v, "b (t h w) n d -> (b t) (h w) n d", t=T, h=H, w=W)

        # Skip ref_q (last frame query) - unused in current causal implementation
        q = q[:-1]

        grid_sizes_q = torch.tensor([[1, H, W]] * q.shape[0], dtype=torch.long)
        start = [[shifting, 0, 0]] * q.shape[0]
        q = rope_apply(q, grid_sizes_q, freqs, start=start)
        q = q.view(q.shape[0], -1, n, d)

        ref_k = k[-1:]
        grid_sizes_ref = torch.tensor([[1, H, W]], dtype=torch.long)
        start_ref = [[shifting + 10, 0, 0]]
        ref_k = rope_apply(ref_k, grid_sizes_ref, freqs, start_ref)
        ref_k = ref_k.view(1, -1, n, d)
        ref_k = repeat(ref_k, "1 s n d -> t s n d", t=k.shape[0] - 1)

        k = k[:-1]
        k = torch.cat([*([k[:1]] * shifting), k])
        cat_k = []
        for i in range(shifting):
            cat_k.append(k[i : i - shifting])
        cat_k.append(k[shifting:])
        k = torch.cat(cat_k, dim=1)

        grid_sizes_k = torch.tensor([[shifting + 1, H, W]] * (q.shape[0]), dtype=torch.long)
        k = rope_apply(k, grid_sizes_k, freqs)
        k = k.view(k.shape[0], -1, n, d)
        k = torch.cat([k, ref_k], dim=1)

        ref_v = repeat(v[-1:], "1 s n d -> t s n d", t=q.shape[0])
        v = v[:-1]
        v = torch.cat([*([v[:1]] * shifting), v])
        cat_v = []
        for i in range(shifting):
            cat_v.append(v[i : i - shifting])
        cat_v.append(v[shifting:])
        v = torch.cat(cat_v, dim=1)
        v = torch.cat([v, ref_v], dim=1)

        outs = []
        for i in range(q.shape[0]):
            out = self.attn(q[i : i + 1], k[i : i + 1], v[i : i + 1])
            outs.append(out)
        out = torch.cat(outs, dim=0)
        out = torch.cat([out, ref_v[:1]], axis=0)
        out = rearrange(out, "(b t) (h w) n d -> b (t h w) n d", t=T, h=H, w=W)
        x = out.flatten(2, 3)
        x = self.o(x)
        return x


class MotionerAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        self_attn_block="SelfAttention",
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads

        self.norm1 = FP32LayerNorm(dim, elementwise_affine=False, eps=eps)
        if self_attn_block == "SelfAttention":
            self.self_attn = SimpleSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        elif self_attn_block == "SwinSelfAttention":
            self.self_attn = SwinSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        elif self_attn_block == "CausalSelfAttention":
            self.self_attn = CausalSelfAttention(dim, num_heads, window_size, qk_norm, eps)

        self.norm2 = FP32LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))

    def forward(
        self, x: torch.Tensor, seq_lens: list[int], grid_sizes: list[tuple[int, int, int]], freqs: torch.Tensor
    ) -> torch.Tensor:
        y = self.self_attn(self.norm1(x).type_as(x), seq_lens, grid_sizes, freqs)
        x = x + y
        y = self.ffn(self.norm2(x).type_as(x))
        x = x + y
        return x


class MotionerHead(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        out_dim = math.prod(patch_size) * out_dim
        self.norm = FP32LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.head = nn.Linear(dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(self.norm(x))
        return x


class MotionerTransformers(nn.Module):
    def __init__(
        self,
        patch_size=(1, 2, 2),
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        self_attn_block="SelfAttention",
        motion_token_num=1024,
        enable_tsm=False,
        motion_stride=4,
        expand_ratio=2,
        trainable_token_pos_emb=False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size

        self.enable_tsm = enable_tsm
        self.motion_stride = motion_stride
        self.expand_ratio = expand_ratio
        self.sample_c = patch_size[0]

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.ModuleList(
            [
                MotionerAttentionBlock(
                    dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps, self_attn_block=self_attn_block
                )
                for _ in range(num_layers)
            ]
        )

        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        if (dim // num_heads) % 2 != 0:
            raise ValueError(f"dim // num_heads ({dim // num_heads}) must be even")
        d = dim // num_heads
        self.freqs = torch.cat(
            [rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))],
            dim=1,
        )

        self.gradient_checkpointing = False
        self.motion_side_len = int(math.sqrt(motion_token_num))
        if self.motion_side_len**2 != motion_token_num:
            raise ValueError(f"motion_token_num ({motion_token_num}) must be a perfect square")
        self.token = nn.Parameter(torch.zeros(1, motion_token_num, dim).contiguous())

        self.trainable_token_pos_emb = trainable_token_pos_emb
        if trainable_token_pos_emb:
            x = torch.zeros([1, motion_token_num, num_heads, d])
            x[..., ::2] = 1
            gride_sizes = [
                [
                    torch.tensor([0, 0, 0]).unsqueeze(0).repeat(1, 1),
                    torch.tensor([1, self.motion_side_len, self.motion_side_len]).unsqueeze(0).repeat(1, 1),
                    torch.tensor([1, self.motion_side_len, self.motion_side_len]).unsqueeze(0).repeat(1, 1),
                ]
            ]
            token_freqs = rope_apply(x, gride_sizes, self.freqs)
            token_freqs = token_freqs[0, :, 0].reshape(motion_token_num, -1, 2)
            token_freqs = token_freqs * 0.01
            self.token_freqs = torch.nn.Parameter(token_freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x[0].shape[1] contains motion frames count, but not used directly here
        device = self.patch_embedding.weight.device
        freqs = self.freqs
        if freqs.device != device:
            freqs = freqs.to(device)

        if self.trainable_token_pos_emb:
            with torch.amp.autocast(current_omni_platform.device_type, dtype=torch.float64):
                token_freqs = self.token_freqs.to(torch.float64)
                token_freqs = token_freqs / token_freqs.norm(dim=-1, keepdim=True)
                freqs = [freqs, torch.view_as_complex(token_freqs)]

        if self.enable_tsm:
            sample_idx = [
                _sample_indices(u.shape[1], stride=self.motion_stride, expand_ratio=self.expand_ratio, c=self.sample_c)
                for u in x
            ]
            x = [torch.flip(torch.flip(u, [1])[:, idx], [1]) for idx, u in zip(sample_idx, x)]

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        seq_f, seq_h, seq_w = x[0].shape[-3:]
        batch_size = len(x)
        if not self.enable_tsm:
            grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
            grid_sizes = [[torch.zeros_like(grid_sizes), grid_sizes, grid_sizes]]
            seq_f = 0
        else:
            grid_sizes = []
            for idx in sample_idx[0][::-1][:: self.sample_c]:
                tsm_frame_grid_sizes = [
                    [
                        torch.tensor([idx, 0, 0]).unsqueeze(0).repeat(batch_size, 1),
                        torch.tensor([idx + 1, seq_h, seq_w]).unsqueeze(0).repeat(batch_size, 1),
                        torch.tensor([1, seq_h, seq_w]).unsqueeze(0).repeat(batch_size, 1),
                    ]
                ]
                grid_sizes += tsm_frame_grid_sizes
            seq_f = sample_idx[0][-1] + 1

        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        x = torch.cat([u for u in x])
        batch_size = len(x)

        token_grid_sizes = [
            [
                torch.tensor([seq_f, 0, 0]).unsqueeze(0).repeat(batch_size, 1),
                torch.tensor([seq_f + 1, self.motion_side_len, self.motion_side_len])
                .unsqueeze(0)
                .repeat(batch_size, 1),
                torch.tensor([1 if not self.trainable_token_pos_emb else -1, seq_h, seq_w])
                .unsqueeze(0)
                .repeat(batch_size, 1),
            ]
        ]

        grid_sizes = grid_sizes + token_grid_sizes
        token_len = self.token.shape[1]
        token = self.token.clone().repeat(x.shape[0], 1, 1).contiguous()
        seq_lens = seq_lens + torch.tensor([t.size(0) for t in token], dtype=torch.long)
        x = torch.cat([x, token], dim=1)

        kwargs = dict(seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=freqs)
        for block in self.blocks:
            x = block(x, **kwargs)

        out = x[:, -token_len:]
        return out

    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out


class FramePackMotioner(nn.Module):
    def __init__(self, inner_dim=1024, num_heads=16, zip_frame_buckets=None, drop_mode="drop", *args, **kwargs):
        super().__init__(*args, **kwargs)
        if zip_frame_buckets is None:
            zip_frame_buckets = [1, 2, 16]

        self.proj = nn.Conv3d(16, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.proj_2x = nn.Conv3d(16, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.proj_4x = nn.Conv3d(16, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))
        self.zip_frame_buckets = torch.tensor(zip_frame_buckets, dtype=torch.long)

        self.inner_dim = inner_dim
        self.num_heads = num_heads

        if inner_dim % num_heads != 0:
            raise ValueError(f"inner_dim ({inner_dim}) must be divisible by num_heads ({num_heads})")
        if (inner_dim // num_heads) % 2 != 0:
            raise ValueError(f"inner_dim // num_heads ({inner_dim // num_heads}) must be even")
        d = inner_dim // num_heads
        self.freqs = torch.cat(
            [rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))],
            dim=1,
        )
        self.drop_mode = drop_mode

    def forward(self, motion_latents: torch.Tensor, add_last_motion: int = 2) -> torch.Tensor:
        mot = []
        mot_remb = []
        for m in motion_latents:
            lat_height, lat_width = m.shape[2], m.shape[3]
            pad_lat = torch.zeros(16, self.zip_frame_buckets.sum(), lat_height, lat_width).to(
                device=m.device, dtype=m.dtype
            )
            overlap_frame = min(pad_lat.shape[1], m.shape[1])
            if overlap_frame > 0:
                pad_lat[:, -overlap_frame:] = m[:, -overlap_frame:]

            if add_last_motion < 2 and self.drop_mode != "drop":
                zero_end_frame = self.zip_frame_buckets[: len(self.zip_frame_buckets) - add_last_motion - 1].sum()
                pad_lat[:, -zero_end_frame:] = 0

            pad_lat = pad_lat.unsqueeze(0)
            clean_latents_4x, clean_latents_2x, clean_latents_post = pad_lat[
                :, :, -self.zip_frame_buckets.sum() :, :, :
            ].split(list(self.zip_frame_buckets)[::-1], dim=2)

            clean_latents_post = self.proj(clean_latents_post).flatten(2).transpose(1, 2)
            clean_latents_2x = self.proj_2x(clean_latents_2x).flatten(2).transpose(1, 2)
            clean_latents_4x = self.proj_4x(clean_latents_4x).flatten(2).transpose(1, 2)

            if add_last_motion < 2 and self.drop_mode == "drop":
                clean_latents_post = clean_latents_post[:, :0] if add_last_motion < 2 else clean_latents_post
                clean_latents_2x = clean_latents_2x[:, :0] if add_last_motion < 1 else clean_latents_2x

            motion_lat = torch.cat([clean_latents_post, clean_latents_2x, clean_latents_4x], dim=1)

            # Rope
            start_time_id = -(self.zip_frame_buckets[:1].sum())
            end_time_id = start_time_id + self.zip_frame_buckets[0]
            grid_sizes = (
                []
                if add_last_motion < 2 and self.drop_mode == "drop"
                else [
                    [
                        torch.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1),
                        torch.tensor([end_time_id, lat_height // 2, lat_width // 2]).unsqueeze(0).repeat(1, 1),
                        torch.tensor([self.zip_frame_buckets[0], lat_height // 2, lat_width // 2])
                        .unsqueeze(0)
                        .repeat(1, 1),
                    ]
                ]
            )

            start_time_id = -(self.zip_frame_buckets[:2].sum())
            end_time_id = start_time_id + self.zip_frame_buckets[1] // 2
            grid_sizes_2x = (
                []
                if add_last_motion < 1 and self.drop_mode == "drop"
                else [
                    [
                        torch.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1),
                        torch.tensor([end_time_id, lat_height // 4, lat_width // 4]).unsqueeze(0).repeat(1, 1),
                        torch.tensor([self.zip_frame_buckets[1], lat_height // 2, lat_width // 2])
                        .unsqueeze(0)
                        .repeat(1, 1),
                    ]
                ]
            )

            start_time_id = -(self.zip_frame_buckets[:3].sum())
            end_time_id = start_time_id + self.zip_frame_buckets[2] // 4
            grid_sizes_4x = [
                [
                    torch.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1),
                    torch.tensor([end_time_id, lat_height // 8, lat_width // 8]).unsqueeze(0).repeat(1, 1),
                    torch.tensor([self.zip_frame_buckets[2], lat_height // 2, lat_width // 2])
                    .unsqueeze(0)
                    .repeat(1, 1),
                ]
            ]

            grid_sizes = grid_sizes + grid_sizes_2x + grid_sizes_4x

            motion_rope_emb = rope_precompute(
                motion_lat.detach().view(1, motion_lat.shape[1], self.num_heads, self.inner_dim // self.num_heads),
                grid_sizes,
                self.freqs,
                start=None,
            )

            mot.append(motion_lat)
            mot_remb.append(motion_rope_emb)
        return mot, mot_remb


def _sample_indices(num_frames, stride, expand_ratio, c):
    indices = []
    current_start = 0
    while current_start < num_frames:
        bucket_width = int(stride * (expand_ratio ** (len(indices) / stride)))
        interval = int(bucket_width / stride * c)
        current_end = min(num_frames, current_start + bucket_width)
        bucket_samples = []
        for i in range(current_end - 1, current_start - 1, -interval):
            for near in range(c):
                bucket_samples.append(i - near)
        indices += bucket_samples[::-1]
        current_start += bucket_width
    return indices


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class WanS2VTransformer3DModel(nn.Module):
    """
    Wan2.2 S2V Transformer using vllm-omni ops for the main transformer blocks.

    This replaces WanModel_S2V from wan_modules, using:
    - QKVParallelLinear for self-attention Q/K/V (fused)
    - ColumnParallelLinear for cross-attention Q/K/V
    - RowParallelLinear for output projections
    - DistributedRMSNorm for QK normalization (TP-aware)
    - WanFeedForward (ColumnParallelGELU + RowParallelLinear)
    - FP32LayerNorm for layer normalization
    - Attention layer for attention computation
    """

    _repeated_blocks = ["WanS2VTransformerBlock"]
    _layerwise_offload_blocks_attrs = ["blocks"]

    @staticmethod
    def _is_transformer_block(name, mod):
        """Match transformer blocks for HSDP sharding."""
        return "blocks" in name and name.split(".")[-1].isdigit()

    _hsdp_shard_conditions = [_is_transformer_block]

    def __init__(
        self,
        cond_dim=0,
        audio_dim=5120,
        num_audio_token=4,
        enable_adain=False,
        adain_mode="attn_norm",
        audio_inject_layers=None,
        zero_init=False,
        zero_timestep=False,
        enable_motioner=True,
        add_last_motion=True,
        enable_tsm=False,
        trainable_token_pos_emb=False,
        motion_token_num=1024,
        enable_framepack=False,
        framepack_drop_mode="drop",
        model_type="s2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        *args,
        **kwargs,
    ):
        super().__init__()

        if audio_inject_layers is None:
            audio_inject_layers = [0, 4, 8, 12, 16, 20, 24, 27]

        assert model_type == "s2v"
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # Embeddings
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # Transformer blocks (TP-enabled)
        self.blocks = nn.ModuleList(
            [
                WanS2VTransformerBlock(dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps)
                for _ in range(num_layers)
            ]
        )

        # Output head
        self.head = WanS2VHead(dim, out_dim, patch_size, eps)

        # RoPE base frequencies (complex-valued)
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        if (dim // num_heads) % 2 != 0:
            raise ValueError(f"dim // num_heads ({dim // num_heads}) must be even")
        d = dim // num_heads
        self.freqs = torch.cat(
            [rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))],
            dim=1,
        )

        self.use_context_parallel = False

        # Condition encoder
        if cond_dim > 0:
            self.cond_encoder = nn.Conv3d(cond_dim, dim, kernel_size=patch_size, stride=patch_size)

        # Audio modules
        self.enable_adain = enable_adain
        self.casual_audio_encoder = CausalAudioEncoder(
            dim=audio_dim, out_dim=dim, num_token=num_audio_token, need_global=enable_adain
        )

        all_modules, all_modules_names = torch_dfs(self.blocks, parent_name="root.transformer_blocks")
        self.audio_injector = AudioInjector_WAN(
            all_modules,
            all_modules_names,
            dim=dim,
            num_heads=num_heads,
            inject_layer=audio_inject_layers,
            root_net=self,
            enable_adain=enable_adain,
            adain_dim=dim,
            need_adain_ont=adain_mode != "attn_norm",
        )
        self.adain_mode = adain_mode

        self.trainable_cond_mask = nn.Embedding(3, dim)

        if zero_init:
            self._zero_init_weights()

        self.zero_timestep = zero_timestep
        self.add_last_motion = add_last_motion

        # Motion modules
        if enable_motioner and enable_framepack:
            raise ValueError("enable_motioner and enable_framepack are mutually exclusive")

        self.enable_motioner = enable_motioner
        if enable_motioner:
            motioner_dim = 2048
            self.motioner = MotionerTransformers(
                patch_size=(2, 4, 4),
                dim=motioner_dim,
                ffn_dim=motioner_dim,
                freq_dim=256,
                out_dim=16,
                num_heads=16,
                num_layers=13,
                window_size=(-1, -1),
                qk_norm=True,
                cross_attn_norm=False,
                eps=1e-6,
                motion_token_num=motion_token_num,
                enable_tsm=enable_tsm,
                motion_stride=4,
                expand_ratio=2,
                trainable_token_pos_emb=trainable_token_pos_emb,
            )
            self.zip_motion_out = nn.Sequential(
                FP32LayerNorm(motioner_dim, elementwise_affine=False), zero_module(nn.Linear(motioner_dim, dim))
            )

            self.trainable_token_pos_emb = trainable_token_pos_emb
            if trainable_token_pos_emb:
                d = dim // num_heads
                x = torch.zeros([1, motion_token_num, num_heads, d])
                x[..., ::2] = 1
                gride_sizes = [
                    [
                        torch.tensor([0, 0, 0]).unsqueeze(0).repeat(1, 1),
                        torch.tensor([1, self.motioner.motion_side_len, self.motioner.motion_side_len])
                        .unsqueeze(0)
                        .repeat(1, 1),
                        torch.tensor([1, self.motioner.motion_side_len, self.motioner.motion_side_len])
                        .unsqueeze(0)
                        .repeat(1, 1),
                    ]
                ]
                token_freqs = rope_apply(x, gride_sizes, self.freqs)
                token_freqs = token_freqs[0, :, 0].reshape(motion_token_num, -1, 2)
                token_freqs = token_freqs * 0.01
                self.token_freqs = nn.Parameter(token_freqs)

        self.enable_framepack = enable_framepack
        if enable_framepack:
            self.frame_packer = FramePackMotioner(
                inner_dim=dim, num_heads=num_heads, zip_frame_buckets=[1, 2, 16], drop_mode=framepack_drop_mode
            )

    def _zero_init_weights(self):
        with torch.no_grad():
            self.trainable_cond_mask = zero_module(self.trainable_cond_mask)
            if hasattr(self, "cond_encoder"):
                self.cond_encoder = zero_module(self.cond_encoder)
            for i in range(len(self.audio_injector.injector)):
                self.audio_injector.injector[i].o = zero_module(self.audio_injector.injector[i].o)
                if self.enable_adain:
                    self.audio_injector.injector_adain_layers[i].linear = zero_module(
                        self.audio_injector.injector_adain_layers[i].linear
                    )

    # ------------------------------------------------------------------
    # Motion processing
    # ------------------------------------------------------------------

    def process_motion(self, motion_latents, drop_motion_frames=False):
        if drop_motion_frames or motion_latents[0].shape[1] == 0:
            return [], []
        self.lat_motion_frames = motion_latents[0].shape[1]
        mot = [self.patch_embedding(m.unsqueeze(0)) for m in motion_latents]
        batch_size = len(mot)
        mot_remb = []
        flatten_mot = []
        for bs in range(batch_size):
            height, width = mot[bs].shape[3], mot[bs].shape[4]
            flat_mot = mot[bs].flatten(2).transpose(1, 2).contiguous()
            motion_grid_sizes = [
                [
                    torch.tensor([-self.lat_motion_frames, 0, 0]).unsqueeze(0).repeat(1, 1),
                    torch.tensor([0, height, width]).unsqueeze(0).repeat(1, 1),
                    torch.tensor([self.lat_motion_frames, height, width]).unsqueeze(0).repeat(1, 1),
                ]
            ]
            motion_rope_emb = rope_precompute(
                flat_mot.detach().view(1, flat_mot.shape[1], self.num_heads, self.dim // self.num_heads),
                motion_grid_sizes,
                self.freqs,
                start=None,
            )
            mot_remb.append(motion_rope_emb)
            flatten_mot.append(flat_mot)
        return flatten_mot, mot_remb

    def process_motion_frame_pack(self, motion_latents, drop_motion_frames=False, add_last_motion=2):
        flatten_mot, mot_remb = self.frame_packer(motion_latents, add_last_motion)
        if drop_motion_frames:
            return [m[:, :0] for m in flatten_mot], [m[:, :0] for m in mot_remb]
        else:
            return flatten_mot, mot_remb

    def process_motion_transformer_motioner(self, motion_latents, drop_motion_frames=False, add_last_motion=True):
        batch_size = len(motion_latents)
        height = motion_latents[0].shape[2] // self.patch_size[1]
        width = motion_latents[0].shape[3] // self.patch_size[2]

        freqs = self.freqs
        device = self.patch_embedding.weight.device
        if freqs.device != device:
            freqs = freqs.to(device)
        if self.trainable_token_pos_emb:
            with torch.amp.autocast(current_omni_platform.device_type, dtype=torch.float64):
                token_freqs = self.token_freqs.to(torch.float64)
                token_freqs = token_freqs / token_freqs.norm(dim=-1, keepdim=True)
                freqs = [freqs, torch.view_as_complex(token_freqs)]

        if not drop_motion_frames and add_last_motion:
            last_motion_latent = [u[:, -1:] for u in motion_latents]
            last_mot = [self.patch_embedding(m.unsqueeze(0)) for m in last_motion_latent]
            last_mot = [m.flatten(2).transpose(1, 2) for m in last_mot]
            last_mot = torch.cat(last_mot)
            gride_sizes = [
                [
                    torch.tensor([-1, 0, 0]).unsqueeze(0).repeat(batch_size, 1),
                    torch.tensor([0, height, width]).unsqueeze(0).repeat(batch_size, 1),
                    torch.tensor([1, height, width]).unsqueeze(0).repeat(batch_size, 1),
                ]
            ]
        else:
            last_mot = torch.zeros(
                [batch_size, 0, self.dim], device=motion_latents[0].device, dtype=motion_latents[0].dtype
            )
            gride_sizes = []

        zip_motion = self.motioner(motion_latents)
        zip_motion = self.zip_motion_out(zip_motion)
        if drop_motion_frames:
            zip_motion = zip_motion * 0.0
        zip_motion_grid_sizes = [
            [
                torch.tensor([-1, 0, 0]).unsqueeze(0).repeat(batch_size, 1),
                torch.tensor([0, self.motioner.motion_side_len, self.motioner.motion_side_len])
                .unsqueeze(0)
                .repeat(batch_size, 1),
                torch.tensor([1 if not self.trainable_token_pos_emb else -1, height, width])
                .unsqueeze(0)
                .repeat(batch_size, 1),
            ]
        ]

        mot = torch.cat([last_mot, zip_motion], dim=1)
        gride_sizes = gride_sizes + zip_motion_grid_sizes

        motion_rope_emb = rope_precompute(
            mot.detach().view(batch_size, mot.shape[1], self.num_heads, self.dim // self.num_heads),
            gride_sizes,
            freqs,
            start=None,
        )
        return [m.unsqueeze(0) for m in mot], [r.unsqueeze(0) for r in motion_rope_emb]

    def inject_motion(
        self, x, seq_lens, rope_embs, mask_input, motion_latents, drop_motion_frames=False, add_last_motion=True
    ):
        if self.enable_motioner:
            mot, mot_remb = self.process_motion_transformer_motioner(
                motion_latents, drop_motion_frames=drop_motion_frames, add_last_motion=add_last_motion
            )
        elif self.enable_framepack:
            mot, mot_remb = self.process_motion_frame_pack(
                motion_latents, drop_motion_frames=drop_motion_frames, add_last_motion=add_last_motion
            )
        else:
            mot, mot_remb = self.process_motion(motion_latents, drop_motion_frames=drop_motion_frames)

        if len(mot) > 0:
            x = [torch.cat([u, m], dim=1) for u, m in zip(x, mot)]
            seq_lens = seq_lens + torch.tensor([r.size(1) for r in mot], dtype=torch.long)
            rope_embs = [torch.cat([u, m], dim=1) for u, m in zip(rope_embs, mot_remb)]
            mask_input = [
                torch.cat([m, 2 * torch.ones([1, u.shape[1] - m.shape[1]], device=m.device, dtype=m.dtype)], dim=1)
                for m, u in zip(mask_input, x)
            ]
        return x, seq_lens, rope_embs, mask_input

    def after_transformer_block(self, block_idx, hidden_states):
        if block_idx in self.audio_injector.injected_block_id.keys():
            audio_attn_id = self.audio_injector.injected_block_id[block_idx]
            audio_emb = self.merged_audio_emb
            num_frames = audio_emb.shape[1]

            input_hidden_states = hidden_states[:, : self.original_seq_len].clone()
            input_hidden_states = rearrange(input_hidden_states, "b (t n) c -> (b t) n c", t=num_frames)

            if self.enable_adain and self.adain_mode == "attn_norm":
                audio_emb_global = self.audio_emb_global
                audio_emb_global = rearrange(audio_emb_global, "b t n c -> (b t) n c")
                adain_hidden_states = self.audio_injector.injector_adain_layers[audio_attn_id](
                    input_hidden_states, temb=audio_emb_global[:, 0]
                )
                attn_hidden_states = adain_hidden_states
            else:
                attn_hidden_states = self.audio_injector.injector_pre_norm_feat[audio_attn_id](input_hidden_states)

            audio_emb = rearrange(audio_emb, "b t n c -> (b t) n c", t=num_frames)
            attn_audio_emb = audio_emb
            residual_out = self.audio_injector.injector[audio_attn_id](
                x=attn_hidden_states,
                context=attn_audio_emb,
                context_lens=torch.ones(attn_hidden_states.shape[0], dtype=torch.long, device=attn_hidden_states.device)
                * attn_audio_emb.shape[1],
            )
            residual_out = rearrange(residual_out, "(b t) n c -> b (t n) c", t=num_frames)
            hidden_states[:, : self.original_seq_len] = hidden_states[:, : self.original_seq_len] + residual_out

        return hidden_states

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def encode_audio(self, audio_input: torch.Tensor, motion_frames: list[int]) -> dict[str, torch.Tensor]:
        """Precompute audio embeddings from raw audio features.

        Call this once per clip before the denoising loop, then pass the
        result to forward() via the ``audio_emb`` kwarg to avoid redundant
        encoding at every denoising step.

        Returns:
            dict with 'audio_emb' (and optionally 'audio_emb_global' when
            enable_adain is True).
        """
        audio_input = torch.cat(
            [audio_input[..., 0:1].repeat(1, 1, 1, motion_frames[0]), audio_input],
            dim=-1,
        )
        audio_emb_res = self.casual_audio_encoder(audio_input)
        result = {}
        if self.enable_adain:
            audio_emb_global, audio_emb = audio_emb_res
            result["audio_emb_global"] = audio_emb_global[:, motion_frames[1] :].clone()
        else:
            audio_emb = audio_emb_res
        result["audio_emb"] = audio_emb[:, motion_frames[1] :, :]
        return result

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        seq_len: int,
        ref_latents: torch.Tensor,
        motion_latents: torch.Tensor,
        cond_states: torch.Tensor,
        audio_input: torch.Tensor | None = None,
        motion_frames: list[int] = [17, 5],
        add_last_motion: int = 2,
        drop_motion_frames: bool = False,
        audio_emb: dict[str, torch.Tensor] | None = None,
        *extra_args: object,
        **extra_kwargs: object,
    ) -> torch.Tensor:
        """Forward pass for S2V transformer.

        Args:
            x: Noisy latents [B, C, T, H, W]
            t: Timesteps [B]
            context: Text embeddings [B, seq_len, dim]
            seq_len: Sequence length
            ref_latents: Reference image latents [B, C, 1, H, W]
            motion_latents: Motion context latents [B, C, T_motion, H, W]
            cond_states: Pose condition latents [B, C, T, H, W] — typically zero tensor
                        for no pose control. Iterates over batch dimension (dim 0).
            audio_input: Raw audio features (optional if audio_emb provided)
            motion_frames: [num_frames, num_latent_frames] for audio alignment
            add_last_motion: Number of last motion frames to repeat
            drop_motion_frames: Whether to drop first motion frame
            audio_emb: Precomputed audio embeddings from encode_audio()

        Returns:
            Predicted noise [B, C, T, H, W]
        """
        add_last_motion = self.add_last_motion * add_last_motion

        # Audio encoding — use precomputed embeddings if available
        if audio_emb is not None:
            self.merged_audio_emb = audio_emb["audio_emb"]
            if self.enable_adain:
                self.audio_emb_global = audio_emb["audio_emb_global"]
        else:
            audio_input = torch.cat(
                [audio_input[..., 0:1].repeat(1, 1, 1, motion_frames[0]), audio_input],
                dim=-1,
            )
            audio_emb_res = self.casual_audio_encoder(audio_input)
            if self.enable_adain:
                audio_emb_global, audio_emb_local = audio_emb_res
                self.audio_emb_global = audio_emb_global[:, motion_frames[1] :].clone()
            else:
                audio_emb_local = audio_emb_res
            self.merged_audio_emb = audio_emb_local[:, motion_frames[1] :, :]

        # Patch embedding + condition
        # NOTE: x and cond_states are 5D tensors [B, C, T, H, W]. Iterating over them
        # yields 4D tensors [C, T, H, W], which we unsqueeze(0) to restore batch dim.
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        cond = [self.cond_encoder(c.unsqueeze(0)) for c in cond_states]
        x = [x_ + pose for x_, pose in zip(x, cond)]

        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)

        original_grid_sizes = deepcopy(grid_sizes)
        grid_sizes = [[torch.zeros_like(grid_sizes), grid_sizes, grid_sizes]]

        # Reference image tokens
        self.lat_motion_frames = motion_latents[0].shape[1]
        ref = [self.patch_embedding(r.unsqueeze(0)) for r in ref_latents]
        batch_size = len(ref)
        height, width = ref[0].shape[3], ref[0].shape[4]
        ref_grid_sizes = [
            [
                torch.tensor([30, 0, 0]).unsqueeze(0).repeat(batch_size, 1),
                torch.tensor([31, height, width]).unsqueeze(0).repeat(batch_size, 1),
                torch.tensor([1, height, width]).unsqueeze(0).repeat(batch_size, 1),
            ]
        ]

        ref = [r.flatten(2).transpose(1, 2) for r in ref]
        self.original_seq_len = seq_lens[0].item()

        seq_lens = seq_lens + torch.tensor([r.size(1) for r in ref], dtype=torch.long)
        grid_sizes = grid_sizes + ref_grid_sizes
        x = [torch.cat([u, r], dim=1) for u, r in zip(x, ref)]

        # Masks: 0=noisy, 1=ref, 2=motion
        mask_input = [torch.zeros([1, u.shape[1]], dtype=torch.long, device=x[0].device) for u in x]
        for i in range(len(mask_input)):
            mask_input[i][:, self.original_seq_len :] = 1

        # Precompute RoPE
        x = torch.cat(x)
        b, s, n, d = x.size(0), x.size(1), self.num_heads, self.dim // self.num_heads
        self.pre_compute_freqs = rope_precompute(x.detach().view(b, s, n, d), grid_sizes, self.freqs, start=None)

        x = [u.unsqueeze(0) for u in x]
        self.pre_compute_freqs = [u.unsqueeze(0) for u in self.pre_compute_freqs]

        # Inject motion tokens
        x, seq_lens, self.pre_compute_freqs, mask_input = self.inject_motion(
            x,
            seq_lens,
            self.pre_compute_freqs,
            mask_input,
            motion_latents,
            drop_motion_frames=drop_motion_frames,
            add_last_motion=add_last_motion,
        )

        x = torch.cat(x, dim=0)
        self.pre_compute_freqs = torch.cat(self.pre_compute_freqs, dim=0)
        mask_input = torch.cat(mask_input, dim=0)

        x = x + self.trainable_cond_mask(mask_input).to(x.dtype)

        # Time embeddings — sinusoidal_embedding_1d uses float64 internally
        # for precision, then we cast to the model dtype (bf16) since the
        # downstream scale/shift modulation doesn't need float32.
        if self.zero_timestep:
            t = torch.cat([t, torch.zeros([1], dtype=t.dtype, device=t.device)])
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).to(x.dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        if self.zero_timestep:
            e = e[:-1]
            zero_e0 = e0[-1:]
            e0 = e0[:-1]
            e0 = torch.cat([e0.unsqueeze(2), zero_e0.unsqueeze(2).repeat(e0.size(0), 1, 1, 1)], dim=2)
            e0 = [e0, self.original_seq_len]
        else:
            e0 = e0.unsqueeze(2).repeat(1, 1, 2, 1)
            e0 = [e0, 0]

        # Context embedding
        context_lens = None
        context = self.text_embedding(
            torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
        )

        # Transformer blocks
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.pre_compute_freqs,
            context=context,
            context_lens=context_lens,
        )

        for idx, block in enumerate(self.blocks):
            x = block(x, **kwargs)
            x = self.after_transformer_block(idx, x)

        # Output
        x = x[:, : self.original_seq_len]
        x = self.head(x, e)
        x = self.unpatchify(x, original_grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with QKV fusion for main transformer blocks."""
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        # QKV fusion mapping (only for main transformer blocks)
        stacked_params_mapping = [
            (".self_attn.to_qkv", ".self_attn.q", "q"),
            (".self_attn.to_qkv", ".self_attn.k", "k"),
            (".self_attn.to_qkv", ".self_attn.v", "v"),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            original_name = name
            is_main_block = name.startswith("blocks.")

            if is_main_block:
                # Try QKV fusion for self-attention
                fused = False
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    lookup_name = name.replace(weight_name, param_name)
                    if lookup_name in params_dict:
                        param = params_dict[lookup_name]
                        weight_loader = param.weight_loader
                        weight_loader(param, loaded_weight, shard_id)
                        fused = True
                        break

                if fused:
                    loaded_params.add(original_name)
                    loaded_params.add(lookup_name)
                    continue

                # Name remapping for main blocks
                name = name.replace(".self_attn.o.", ".self_attn.to_out.")
                name = name.replace(".cross_attn.q.", ".cross_attn.to_q.")
                name = name.replace(".cross_attn.k.", ".cross_attn.to_k.")
                name = name.replace(".cross_attn.v.", ".cross_attn.to_v.")
                name = name.replace(".cross_attn.o.", ".cross_attn.to_out.")
                name = name.replace(".ffn.0.", ".ffn.net_0.proj.")
                name = name.replace(".ffn.2.", ".ffn.net_2.")

            if name not in params_dict:
                logger.warning(f"Skipping weight {original_name} -> {name}")
                continue

            param = params_dict[name]

            # TP-shard norm weights for main blocks
            if (
                tp_size > 1
                and is_main_block
                and any(
                    norm_name in name
                    for norm_name in [
                        ".self_attn.norm_q.",
                        ".self_attn.norm_k.",
                        ".cross_attn.norm_q.",
                        ".cross_attn.norm_k.",
                    ]
                )
            ):
                shard_size = loaded_weight.shape[0] // tp_size
                loaded_weight = loaded_weight[tp_rank * shard_size : (tp_rank + 1) * shard_size]

            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

            loaded_params.add(original_name)
            loaded_params.add(name)

        return loaded_params


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_s2v_transformer_from_config(config: dict) -> WanS2VTransformer3DModel:
    """Create WanS2VTransformer3DModel from a config dict (diffusers format)."""
    kwargs = {}

    # Map config keys to constructor parameters
    key_mapping = {
        "cond_dim": "cond_dim",
        "audio_dim": "audio_dim",
        "num_audio_token": "num_audio_token",
        "enable_adain": "enable_adain",
        "adain_mode": "adain_mode",
        "audio_inject_layers": "audio_inject_layers",
        "zero_init": "zero_init",
        "zero_timestep": "zero_timestep",
        "enable_motioner": "enable_motioner",
        "add_last_motion": "add_last_motion",
        "enable_tsm": "enable_tsm",
        "trainable_token_pos_emb": "trainable_token_pos_emb",
        "motion_token_num": "motion_token_num",
        "enable_framepack": "enable_framepack",
        "framepack_drop_mode": "framepack_drop_mode",
        "model_type": "model_type",
        "patch_size": "patch_size",
        "text_len": "text_len",
        "in_dim": "in_dim",
        "dim": "dim",
        "ffn_dim": "ffn_dim",
        "freq_dim": "freq_dim",
        "text_dim": "text_dim",
        "out_dim": "out_dim",
        "num_heads": "num_heads",
        "num_layers": "num_layers",
        "window_size": "window_size",
        "qk_norm": "qk_norm",
        "cross_attn_norm": "cross_attn_norm",
        "eps": "eps",
    }

    for config_key, param_name in key_mapping.items():
        if config_key in config:
            value = config[config_key]
            if config_key in ("patch_size", "window_size") and isinstance(value, list):
                value = tuple(value)
            kwargs[param_name] = value

    return WanS2VTransformer3DModel(**kwargs)
