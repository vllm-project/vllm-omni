# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from diffusers' transformer_qwenimage21.py (Qwen-Image 2.1 single-stream DiT).

from __future__ import annotations

import copy
import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import RMSNorm
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.cache.base import CachedTransformer
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.hsdp_utils import is_transformer_block_module
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput,
    SequenceParallelOutput,
)
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available
from vllm_omni.diffusion.layers.fused_norm_rope import FusedNormRope, RopeTables, prepare_rope_tables
from vllm_omni.diffusion.models.qwen_image_21.decode_graph import QwenImage21DecodeGraphManager
from vllm_omni.diffusion.models.qwen_image_21.qkv_norm_rope import use_mindiesd_qkv

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

logger = init_logger(__name__)

_FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

# Accepted values for the `prefix_kv_cache_dtype` switch (see QwenImage21Transformer2DModel).
_PREFIX_KV_FP8_ALIASES = {"fp8", "fp8_e4m3", "fp8_e4m3fn"}
_PREFIX_KV_FP8_V_ALIASES = {"fp8_v", "fp8_e4m3_v"}
_PREFIX_KV_FP8_V = "fp8_e4m3_v"


def _quantize_prefix_kv_fp8(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token-per-head symmetric FP8 E4M3 quantization of a (B, S, H, D) K/V tensor.

    One fp32 scale per (batch, token, head): it tracks the heavy-tailed per-token magnitude of
    post-RoPE keys and outlier-channel values far better than a per-tensor scale, and the scale
    tensor stays mergeable across the batch dim for batched decode (`_assemble_kv_cache`).
    """
    scale = t.abs().amax(dim=-1, keepdim=True).float().clamp_min(1e-12) / _FP8_E4M3_MAX
    quantized = (t / scale).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return quantized, scale


def _dequantize_prefix_kv_fp8(quantized: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (quantized.float() * scale).to(dtype)


def _normalize_prefix_kv_cache_dtype(value: Any) -> str | None:
    """Normalize the `prefix_kv_cache_dtype` switch: None/"auto" = off, "fp8*" = FP8 E4M3 storage."""
    if value is None or value == "auto":
        return None
    if value in _PREFIX_KV_FP8_ALIASES:
        return "fp8_e4m3"
    if value in _PREFIX_KV_FP8_V_ALIASES:
        # V-only FP8 storage: K stays in the native dtype. Measured nearly lossless on this
        # model (PSNR 40.9 dB vs bf16 vs 34.9 dB for K+V fp8) — post-RoPE K is the
        # precision-sensitive half of the cache.
        return _PREFIX_KV_FP8_V
    raise ValueError(
        f"Unknown prefix_kv_cache_dtype {value!r}; expected None, 'auto' or one of "
        f"{sorted(_PREFIX_KV_FP8_ALIASES | _PREFIX_KV_FP8_V_ALIASES)}."
    )


def _apply_qwen_image21_rotary_emb(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Rotate interleaved pairs with complex FP32 multiplication.

    Same contract as the reference `apply_rotary_emb_qwen(..., use_real=False)`:
    `x` is [B, S, H, D], `freqs` is the complex frequency tensor [S, D // 2].
    """
    paired = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(paired * freqs.unsqueeze(1)).flatten(3).to(x.dtype)


def _apply_qwen_image21_rotary_emb_native(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Platform fallback for `_apply_qwen_image21_rotary_emb` without complex tensors.

    Algebraically identical to the complex multiplication, computed in fp32:
    (x1 + i x2)(cos + i sin) interleaved back as [x1 cos - x2 sin, x1 sin + x2 cos, ...].
    """
    cos = torch.real(freqs).float().unsqueeze(1)
    sin = torch.imag(freqs).float().unsqueeze(1)
    x_pairs = x.float().unflatten(-1, (-1, 2))
    x1, x2 = x_pairs.unbind(-1)
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2).to(x.dtype)


def _normalize_qwen_image21_weight_name(name: str) -> str:
    name = name.removeprefix("transformer.")
    if ".to_out.0." in name:
        name = name.replace(".to_out.0.", ".to_out.")
    return name


def _resolve_qwen_image21_lookup_name(
    name: str,
    stacked_params_mapping: list[tuple[str, str, str]],
) -> tuple[str, str | None]:
    lookup_name = _normalize_qwen_image21_weight_name(name)
    for param_name, weight_name, shard_id in stacked_params_mapping:
        if weight_name not in lookup_name or param_name in lookup_name:
            continue
        return lookup_name.replace(weight_name, param_name), shard_id
    return lookup_name, None


def _select_modulation_rows(params: torch.Tensor, target_token_mask: torch.Tensor | None) -> torch.Tensor:
    r"""Broadcast per-sample modulation `params` over the token axis.

    With `causal_condition`, `params` holds `batch_size + 1` rows: rows `[0, batch_size)` come from the real timestep
    and the trailing row from `t = 0`. Text and condition-image tokens take the `t = 0` row, target-image tokens take
    their own sample's row.

    Ported from diffusers' transformer_qwenimage21._select_modulation_rows.
    """
    if target_token_mask is None:
        return params.unsqueeze(1)
    real, zero = params[:-1].unsqueeze(1), params[-1:].unsqueeze(0)
    return torch.where(target_token_mask.view(1, -1, 1), real, zero)


class QwenImage21TemporalTimesteps(nn.Module):
    r"""Sinusoidal timestep embedding. `cos` occupies the first half of the channels and `sin` the second."""

    def __init__(self, timestep_dim: int, max_period: int = 10000, time_factor: float = 1000.0):
        super().__init__()
        self.timestep_dim = timestep_dim
        self.time_factor = time_factor

        half = timestep_dim // 2
        # Compute the fixed table on CPU like Diffusers, then honor the loader's device.
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device="cpu") / half
        ).to(torch.get_default_device())
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        timestep = self.time_factor * timestep.float()
        args = timestep[:, None] * self.freqs[None].to(timestep.device)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.timestep_dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(timestep.dtype)


class QwenImage21TimestepEmbedder(nn.Module):
    """MLP over the sinusoidal timestep projection. No biases, matching the 2.1 checkpoint."""

    def __init__(self, in_dim: int, embedding_dim: int, prefix: str = ""):
        super().__init__()
        self.linear_1 = ReplicatedLinear(
            in_dim,
            embedding_dim,
            bias=False,
            return_bias=False,
            quant_config=None,
            prefix=f"{prefix}.linear_1",
        )
        self.act = nn.SiLU()
        self.linear_2 = ReplicatedLinear(
            embedding_dim,
            embedding_dim,
            bias=False,
            return_bias=False,
            quant_config=None,
            prefix=f"{prefix}.linear_2",
        )

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(sample)))


class QwenImage21TimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int, prefix: str = ""):
        super().__init__()
        self.time_proj = QwenImage21TemporalTimesteps(timestep_dim=256)
        self.timestep_embedder = QwenImage21TimestepEmbedder(
            in_dim=256, embedding_dim=embedding_dim, prefix=f"{prefix}.timestep_embedder"
        )

    def forward(self, timestep: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        return self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))


class QwenImage21ZeroCenterRMSNorm(nn.Module):
    r"""
    RMSNorm whose learnable weight is stored zero-centered: the effective scale is `weight + 1`, computed in fp32.
    Checkpoints therefore store `scale - 1`. The weight is loaded as-is (no re-centering) by `load_weights`.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        rrms = torch.rsqrt(torch.mean(hidden_states**2, dim=-1, keepdim=True) + self.eps)
        return (hidden_states * rrms * (self.weight.float() + 1)).to(input_dtype)


class QwenImage21TextProjection(nn.Module):
    def __init__(self, context_in_dim: int, hidden_size: int, eps: float = 1e-6, prefix: str = ""):
        super().__init__()
        self.text_norm = QwenImage21ZeroCenterRMSNorm(context_in_dim, eps=eps)
        self.in_layer = ReplicatedLinear(
            context_in_dim,
            hidden_size,
            bias=False,
            return_bias=False,
            quant_config=None,
            prefix=f"{prefix}.in_layer",
        )
        self.act = nn.GELU(approximate="tanh")
        self.out_layer = ReplicatedLinear(
            hidden_size,
            hidden_size,
            bias=False,
            return_bias=False,
            quant_config=None,
            prefix=f"{prefix}.out_layer",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.text_norm(hidden_states)
        hidden_states = self.in_layer(hidden_states)
        hidden_states = self.act(hidden_states)
        return self.out_layer(hidden_states)


class QwenImage21SwiGLUFeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        mlp_hidden_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.proj = ColumnParallelLinear(
            hidden_size,
            mlp_hidden_size,
            bias=False,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
        )
        self.gate_layer = ColumnParallelLinear(
            hidden_size,
            mlp_hidden_size,
            bias=False,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_layer",
        )
        self.out = RowParallelLinear(
            mlp_hidden_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.out",
        )
        self.activation_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.out(self.activation_fn(self.gate_layer(hidden_states)) * self.proj(hidden_states))


class QwenImage21AdaLayerNormContinuous(nn.Module):
    r"""
    Final adaptive norm. Scale only — this variant emits no shift, so `linear` maps to `embedding_dim` rather than
    `2 * embedding_dim`.
    """

    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int, eps: float = 1e-6, prefix: str = ""):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = ReplicatedLinear(
            conditioning_embedding_dim,
            embedding_dim,
            bias=False,
            return_bias=False,
            quant_config=None,
            prefix=f"{prefix}.linear",
        )
        self.norm = nn.LayerNorm(embedding_dim, eps, elementwise_affine=False, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning_embedding: torch.Tensor,
        target_token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scale = self.linear(self.silu(conditioning_embedding).to(hidden_states.dtype))
        scale = _select_modulation_rows(scale, target_token_mask)
        return self.norm(hidden_states) * (1 + scale)


class QwenImage21Rope(nn.Module):
    r"""
    3-axis (frame, height, width) rotary embedding over the joint text/image sequence.

    Text tokens advance a shared position on all three axes. Every image block freezes the frame axis at the position
    reached by the preceding text and lays its tokens out on a height/width grid centred on zero, so a block's spatial
    positions do not depend on where it sits in the sequence.

    Ported from diffusers' transformer_qwenimage21.QwenImage21Rope.
    """

    def __init__(self, theta: int, axes_dim: list[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

        # Match Diffusers' CPU initialization even when the loader sets a CUDA
        # default device; CPU and CUDA trigonometric kernels round differently.
        with torch.device("cpu"):
            pos_index = torch.arange(8192)
            neg_index = torch.arange(1024).flip(0) * -1 - 1
            self.freqs = [
                torch.cat([self.rope_params(pos_index, dim, theta), self.rope_params(neg_index, dim, theta)], dim=0)
                for dim in axes_dim
            ]

    def rope_params(self, index: torch.Tensor, dim: int, theta: int = 10000) -> torch.Tensor:
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        return torch.polar(torch.ones_like(freqs), freqs)

    def forward(
        self, img_shapes: list[tuple[int, int, int]], image_pad_mask: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        self.freqs = [freq.to(device) for freq in self.freqs]

        frame_index, height_index, width_index = [], [], []
        image_height_index, image_width_index = [], []
        cursor, position = 0, 0
        total_len = image_pad_mask.shape[-1]
        is_image_token = image_pad_mask.tolist()

        for _, height, width in img_shapes:
            block_start = is_image_token.index(True, cursor)
            text_len = block_start - cursor
            frame_index.extend(range(position, position + text_len))
            position += text_len

            cursor = block_start + height * width
            frame_index.extend([position] * (height * width))
            position += max(height, width)

            image_height_index.extend([h for h in range(-(height - height // 2), height // 2) for _ in range(width)])
            image_width_index.extend([w for _ in range(height) for w in range(-(width - width // 2), width // 2)])

        if cursor < total_len:
            frame_index.extend(range(position, position + total_len - cursor))

        frame_index = torch.tensor(frame_index, dtype=torch.long, device=device)
        height_index = frame_index.clone()
        width_index = frame_index.clone()
        height_index[image_pad_mask] = torch.tensor(image_height_index, dtype=torch.long, device=device)
        width_index[image_pad_mask] = torch.tensor(image_width_index, dtype=torch.long, device=device)

        height_index = torch.where(height_index < 0, height_index + self.freqs[1].shape[0], height_index)
        width_index = torch.where(width_index < 0, width_index + self.freqs[2].shape[0], width_index)
        return torch.cat(
            [
                self.freqs[0].index_select(0, frame_index),
                self.freqs[1].index_select(0, height_index),
                self.freqs[2].index_select(0, width_index),
            ],
            dim=-1,
        )


def _enable_pattern_ignored_layers(quant_config: QuantizationConfig | None) -> QuantizationConfig | None:
    """Switch vLLM FP8-style `ignored_layers` to pattern (substring) matching.

    vLLM's ``Fp8Config`` matches ``ignored_layers`` against full layer prefixes
    exactly, but the diffusion quantization docs and CLI describe them as name
    patterns (e.g. ``img_mlp``) — under exact matching such patterns silently
    skip nothing. Substring matching is a strict superset for exact full
    prefixes, so existing exact-prefix configs keep working.
    """
    if quant_config is None:
        return quant_config
    component_configs = getattr(quant_config, "component_configs", None)
    if component_configs is not None:
        configs = [*component_configs.values(), getattr(quant_config, "default_config", None)]
    else:
        configs = [quant_config]
    for config in configs:
        if config is None:
            continue
        if getattr(config, "ignored_layers", None) and hasattr(config, "ignored_layers_match_mode"):
            config.ignored_layers_match_mode = "substring"
    return quant_config


class QwenImage21Attention(nn.Module):
    r"""Single-stream attention for Qwen-Image 2.1.

    Q/K/V projections are fused into one `QKVParallelLinear` (checkpoint shards load via
    `packed_modules_mapping`); the output projection is a `RowParallelLinear`. The actual
    attention runs through vllm-omni's `Attention` layer with Q/K/V in (B, S, H, D) layout,
    so structural masking arrives via `AttentionMetadata` (piecewise `full_attn_spans` or a
    4D `attn_mask`) instead of a custom processor.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        eps: float = 1e-6,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        prefix_kv_cache_dtype: str | None = None,
    ):
        super().__init__()
        self.head_dim = dim_head
        self.prefix_kv_cache_dtype = prefix_kv_cache_dtype

        self.to_qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=dim_head,
            total_num_heads=heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_qkv",
        )
        self.num_heads = self.to_qkv.num_heads
        self.num_kv_heads = self.to_qkv.num_kv_heads

        # Diffusers normalizes in FP32, then casts before the learned scale.
        # Keep that rounding order for BF16 Q/K; TP shards whole heads.
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)
        self.fused_norm_rope = FusedNormRope()

        self.to_out = RowParallelLinear(
            heads * dim_head,
            dim,
            bias=False,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_out.0",
        )

        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=dim_head,
            softmax_scale=1.0 / (dim_head**0.5),
            causal=False,
            num_kv_heads=self.num_kv_heads,
        )

    def _apply_rotary_emb(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        if x.device.type == "cuda":
            return _apply_qwen_image21_rotary_emb(x, freqs)
        # Platform fallback without complex-tensor kernels (the CustomOp rope layer
        # dispatches by platform, not tensor device, so it cannot serve CPU tensors).
        return _apply_qwen_image21_rotary_emb_native(x, freqs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
        sp_prefix_len: int = 0,
        sp_decode: bool = False,
        kv_cache: dict[str, dict[str, torch.Tensor]] | None = None,
        cache_branch: str = "cond",
        cache_write_len: int | None = None,
        qkv_rope_tables: RopeTables | None = None,
    ) -> torch.Tensor:
        r"""
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, seq_len, dim)`):
                Modulated tokens. The layout is the full joint sequence normally; under sequence parallelism it is
                `[replicated prefix, local target shard]`, and in KV-cache decode it is target tokens only.
            freqs (`torch.Tensor`): Complex RoPE frequencies matching the local `hidden_states` layout.
            attn_metadata (`AttentionMetadata`, *optional*): Mask/span metadata for the attention backend. This module
                takes a shallow copy before attaching SP joint tensors, so callers may share one metadata object
                across blocks.
            sp_prefix_len (`int`): Number of leading tokens that are replicated across SP ranks. When > 0 they are
                handed to the `Attention` layer as joint (front) tensors and only the trailing shard is all-to-all'd.
            sp_decode (`bool`): KV-cache decode under sequence parallelism: the cached prefix K/V are replicated
                and join through the Ulysses joint mechanism with an empty joint query.
            kv_cache (dict, *optional*): Per-block cache keyed by CFG branch (`"cond"` / `"uncond"`); each branch
                stores `{"key", "value"}` of the timestep-independent prefix (text + condition images). With
                FP8 cache storage enabled, the branch additionally carries `{"key_scale", "value_scale"}`.
            cache_branch (`str`): CFG branch whose cache entry is read/written this forward.
            cache_write_len (`int`, *optional*): Prefill mode — cache the first `cache_write_len` K/V positions.
                `None` with a populated branch means decode — cached prefix K/V are prepended instead.
        """
        batch_size, seq_len, _ = hidden_states.shape

        qkv, _ = self.to_qkv(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)

        query = query.unflatten(-1, (self.num_heads, self.head_dim))
        key = key.unflatten(-1, (self.num_kv_heads, self.head_dim))
        value = value.unflatten(-1, (self.num_kv_heads, self.head_dim))

        if qkv_rope_tables is None:
            query = self.norm_q(query).to(value.dtype)
            key = self.norm_k(key).to(value.dtype)
            query = self._apply_rotary_emb(query, freqs)
            key = self._apply_rotary_emb(key, freqs)
        else:
            query, key, value = self.fused_norm_rope(
                query,
                key,
                value,
                self.norm_q.weight,
                self.norm_k.weight,
                self.norm_q.eps,
                qkv_rope_tables,
            )

        cached_key = cached_value = None
        if kv_cache is not None:
            branch_cache = kv_cache.setdefault(cache_branch, {})
            if cache_write_len is not None:
                # Prefill: cache the timestep-independent prefix K/V for later denoising steps.
                if self.prefix_kv_cache_dtype is not None:
                    # FP8 storage: quantize once at prefill; decode dequantizes per step.
                    # The fp32 scales ride along as extra branch entries so the pipeline's
                    # batched decode merge can concatenate them like the K/V tensors. The
                    # quantized tensors own their storage, so no clone is needed here.
                    if self.prefix_kv_cache_dtype == "fp8_e4m3":
                        prefix_key, branch_cache["key_scale"] = _quantize_prefix_kv_fp8(key[:, :cache_write_len])
                        branch_cache["key"] = prefix_key
                    else:
                        # V-only FP8 ("fp8_e4m3_v"): K stays in the native dtype.
                        branch_cache["key"] = key[:, :cache_write_len].clone()
                    prefix_value, branch_cache["value_scale"] = _quantize_prefix_kv_fp8(value[:, :cache_write_len])
                    branch_cache["value"] = prefix_value
                else:
                    # Own the prefix storage instead of retaining the full prefill tensors.
                    branch_cache["key"] = key[:, :cache_write_len].clone()
                    branch_cache["value"] = value[:, :cache_write_len].clone()
            else:
                cached_key = branch_cache["key"]
                cached_value = branch_cache["value"]
                if self.prefix_kv_cache_dtype is not None:
                    # Dequantize only entries that were actually quantized (scale present).
                    if "key_scale" in branch_cache:
                        cached_key = _dequantize_prefix_kv_fp8(cached_key, branch_cache["key_scale"], key.dtype)
                    if "value_scale" in branch_cache:
                        cached_value = _dequantize_prefix_kv_fp8(cached_value, branch_cache["value_scale"], value.dtype)

        metadata = copy.copy(attn_metadata) if attn_metadata is not None else None
        if sp_prefix_len > 0:
            # Ulysses joint attention: the prefix is replicated across SP ranks, so it is
            # concatenated (front) after the all-to-all instead of being sharded.
            if metadata is None:
                metadata = AttentionMetadata()
            metadata.joint_query = query[:, :sp_prefix_len]
            metadata.joint_key = key[:, :sp_prefix_len]
            metadata.joint_value = value[:, :sp_prefix_len]
            metadata.joint_strategy = "front"
            attn_output = self.attn(
                query[:, sp_prefix_len:], key[:, sp_prefix_len:], value[:, sp_prefix_len:], metadata
            )
        elif cached_key is not None:
            if sp_decode:
                # Decode under SP: cached prefix K/V are replicated; pass them as joint K/V with an
                # empty joint query so only the target shard goes through the all-to-all.
                if metadata is None:
                    metadata = AttentionMetadata()
                metadata.joint_query = query.new_zeros(batch_size, 0, self.num_heads, self.head_dim)
                metadata.joint_key = cached_key
                metadata.joint_value = cached_value
                metadata.joint_strategy = "front"
                attn_output = self.attn(query, key, value, metadata)
            else:
                # Decode: the block-causal mask degenerates to full attention for target rows.
                key = torch.cat([cached_key, key], dim=1)
                value = torch.cat([cached_value, value], dim=1)
                attn_output = self.attn(query, key, value, metadata)
        else:
            attn_output = self.attn(query, key, value, metadata)

        attn_output = attn_output.flatten(2, 3).type_as(hidden_states)
        return self.to_out(attn_output)


class QwenImage21TransformerBlock(nn.Module):
    r"""
    Single-stream block. Modulation is not learned per block — the parent model computes one shared `modulation`
    tensor and every block slices its own scales and gates out of it.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: int = 3,
        eps: float = 1e-6,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        prefix_kv_cache_dtype: str | None = None,
    ):
        super().__init__()
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = QwenImage21Attention(
            dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            eps=eps,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            prefix_kv_cache_dtype=prefix_kv_cache_dtype,
        )
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = QwenImage21SwiGLUFeedForward(
            hidden_size=dim,
            mlp_hidden_size=dim * mlp_ratio,
            quant_config=quant_config,
            prefix=f"{prefix}.img_mlp",
        )

    def _modulate(
        self,
        hidden_states: torch.Tensor,
        mod_params: torch.Tensor,
        target_token_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale, gate = mod_params.chunk(2, dim=-1)
        scale = _select_modulation_rows(scale, target_token_mask)
        gate = _select_modulation_rows(gate, target_token_mask)
        return hidden_states * (1 + scale), gate

    def forward(
        self,
        hidden_states: torch.Tensor,
        modulation: torch.Tensor,
        freqs: torch.Tensor,
        target_token_mask: torch.Tensor | None = None,
        attn_metadata: AttentionMetadata | None = None,
        sp_prefix_len: int = 0,
        sp_decode: bool = False,
        kv_cache: dict[str, dict[str, torch.Tensor]] | None = None,
        cache_branch: str = "cond",
        cache_write_len: int | None = None,
        qkv_rope_tables: RopeTables | None = None,
    ) -> torch.Tensor:
        mod1, mod2 = modulation.chunk(2, dim=-1)

        img_modulated, img_gate1 = self._modulate(self.img_norm1(hidden_states), mod1, target_token_mask)
        attn_output = self.attn(
            img_modulated,
            freqs,
            attn_metadata=attn_metadata,
            sp_prefix_len=sp_prefix_len,
            sp_decode=sp_decode,
            kv_cache=kv_cache,
            cache_branch=cache_branch,
            cache_write_len=cache_write_len,
            qkv_rope_tables=qkv_rope_tables,
        )
        hidden_states = hidden_states + img_gate1.tanh() * attn_output

        img_modulated2, img_gate2 = self._modulate(self.img_norm2(hidden_states), mod2, target_token_mask)
        hidden_states = hidden_states + img_gate2.tanh() * self.img_mlp(img_modulated2)

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states


class QwenImage21SequencePrepare(nn.Module):
    r"""Assemble the joint text/image sequence and split it into prefix and target parts.

    This module boundary lets `_sp_plan` shard the target-image tokens (and their RoPE
    frequencies) via `split_output=True` while the prefix (text + condition images) stays
    replicated across SP ranks for the Ulysses joint-attention mechanism.
    """

    def __init__(self, img_in: nn.Module, pos_embed: QwenImage21Rope):
        super().__init__()
        self.img_in = img_in
        self.pos_embed = pos_embed

    @staticmethod
    def expand_image_pad_mask(img_mask_row: torch.Tensor) -> torch.Tensor:
        """One VLM image slot stands for a 2x2 group of latent tokens; expand those positions four-fold."""
        repeats = torch.where(img_mask_row, 4, 1)
        return torch.repeat_interleave(img_mask_row, repeats)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        img_shapes: list[tuple[int, int, int]],
        img_mask: torch.Tensor,
        is_decode: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Returns:
            `tuple` of `(prefix_hidden_states, target_hidden_states, prefix_freqs, target_freqs)`.
            `prefix_*` cover text + condition-image tokens (replicated under SP); `target_*` cover the
            target image (sharded under SP via `_sp_plan`). In decode mode the prefix parts are empty.
        """
        batch_size = hidden_states.shape[0]
        image_pad_mask = self.expand_image_pad_mask(img_mask[0])
        rotary_emb = self.pos_embed(img_shapes, image_pad_mask, device=hidden_states.device)

        target_tokens = math.prod(img_shapes[-1])
        prefix_len = image_pad_mask.shape[0] - target_tokens

        if is_decode:
            # Only the target image's tokens are recomputed; the prefix comes from the KV cache.
            target_hidden_states = self.img_in(hidden_states[:, -target_tokens:])
            dim = target_hidden_states.shape[-1]
            prefix_hidden_states = target_hidden_states.new_zeros(batch_size, 0, dim)
            return prefix_hidden_states, target_hidden_states, rotary_emb[:0], rotary_emb[prefix_len:]

        hidden_states = self.img_in(hidden_states)

        joint_hidden_states = torch.cat(
            [
                encoder_hidden_states,
                encoder_hidden_states.new_zeros(batch_size, target_tokens // 4, encoder_hidden_states.shape[2]),
            ],
            dim=1,
        )
        repeats = torch.where(img_mask[0], 4, 1)
        joint_hidden_states = joint_hidden_states.repeat_interleave(repeats, dim=1)
        joint_hidden_states[:, image_pad_mask] = hidden_states

        return (
            joint_hidden_states[:, :prefix_len],
            joint_hidden_states[:, prefix_len:],
            rotary_emb[:prefix_len],
            rotary_emb[prefix_len:],
        )


class QwenImage21Transformer2DModel(CachedTransformer):
    r"""
    The single-stream Transformer used by Qwen-Image 2.1, ported to vllm-omni parallel layers.

    Text and image latents share one sequence: condition-image tokens are substituted into the text stream at the
    positions the vision-language encoder reserved for them, and the target image's tokens are appended. A single
    shared `modulation` projection feeds every block, so blocks hold no modulation parameters of their own.

    Two behaviours distinguish 2.1 from 2.0, both switched on by config and neither adding parameters:

    - `causal_block` — attention follows `(q_idx >= kv_idx) or same_image_block`, so the sequence is causal while
      every image block stays internally bidirectional. Implemented through `AttentionMetadata`: piecewise
      `full_attn_spans` (one span per image block; requires a piecewise-capable backend such as FLASH_ATTN) when the
      batch is homogeneous and unpadded, or a dense 4D boolean `attn_mask` otherwise.
    - `causal_condition` — text and condition-image tokens are modulated from `t = 0` instead of the sampled
      timestep, which also makes their activations timestep-independent and so cacheable across denoising steps
      (`kv_cache`). The cached prefix K/V are stored in the running dtype by default, or in FP8 E4M3 (with
      per-token-per-head fp32 scales) when `od_config.extras["prefix_kv_cache_dtype"]` is `"fp8"` (K and V)
      or `"fp8_v"` (V only — K stays in the native dtype).
    """

    # the small and frequently-repeated block(s) of a model
    # -- typically a transformer layer
    # used for torch compile optimizations
    _repeated_blocks = ["QwenImage21TransformerBlock"]
    _layerwise_offload_blocks_attrs = ["transformer_blocks"]
    packed_modules_mapping = {
        "to_qkv": ["to_q", "to_k", "to_v"],
    }

    _hsdp_shard_conditions = [is_transformer_block_module]

    # Sequence Parallelism plan (corresponds to diffusers' _cp_plan).
    #
    # The joint sequence is `[prefix (text + condition images) | target image]`. The prefix is
    # replicated across SP ranks and reaches the attention backend through the Ulysses joint
    # mechanism (`joint_strategy="front"`); only the target tokens and their RoPE frequencies
    # are sharded, via the `sequence_prepare` module boundary with auto-padding. SP padding
    # positions sit at the very end of the joint sequence (the target is the tail block) and
    # are masked out through the 4D-mask path built in `forward`.
    #
    # Limitation: prefix tokens are recomputed redundantly on every rank through all blocks
    # (their K/V are needed by the joint attention); only the target image is actually
    # sequence-parallel.
    _sp_plan = {
        # Shard sequence_prepare outputs: target hidden_states and target RoPE freqs.
        # Indices 0 (prefix hidden_states) and 2 (prefix freqs) stay replicated.
        "sequence_prepare": {
            1: SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True, auto_pad=True),
            3: SequenceParallelInput(split_dim=0, expected_dims=2, split_output=True, auto_pad=True),
        },
        # Gather the (target-only) output after proj_out.
        "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: int | None = 64,
        num_layers: int = 32,
        attention_head_dim: int = 128,
        num_attention_heads: int = 32,
        context_in_dim: int = 4096,
        mlp_ratio: int = 3,
        axes_dims_rope: tuple[int, int, int] = (16, 56, 56),
        eps: float = 1e-6,
        causal_condition: bool = True,
        causal_block: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "transformer",
        cuda_graph_max_decode_graphs: int = 8,
    ):
        super().__init__()
        self.parallel_config = od_config.parallel_config
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.patch_size = patch_size
        self.inner_dim = num_attention_heads * attention_head_dim
        self.causal_condition = causal_condition
        self.causal_block = causal_block
        self.quant_config = _enable_pattern_ignored_layers(quant_config)
        quant_config = self.quant_config

        # Storage dtype for the cross-step prefix KV cache, from the supplementary
        # model-specific config entry `extras["prefix_kv_cache_dtype"]` (None = bf16/fp32
        # native storage, unchanged behavior; "fp8" = FP8 E4M3 quantized storage, halving
        # prefix K/V memory). This is independent of `diffusion_kv_cache_dtype`, which
        # quantizes attention Q/K/V *compute* per forward on supported backends.
        extras = getattr(od_config, "extras", None) or {}
        self.prefix_kv_cache_dtype = _normalize_prefix_kv_cache_dtype(extras.get("prefix_kv_cache_dtype"))
        self.mindiesd_qkv_fusion = use_mindiesd_qkv(
            od_config,
            native_cache=self.prefix_kv_cache_dtype is None,
            unquantized=quant_config is None,
        )
        self.mindiesd_qkv_geometry = attention_head_dim == 128

        self.pos_embed = QwenImage21Rope(theta=10000, axes_dim=list(axes_dims_rope))
        self.time_text_embed = QwenImage21TimestepProjEmbeddings(
            embedding_dim=self.inner_dim, prefix=f"{prefix}.time_text_embed"
        )
        self.txt_in = QwenImage21TextProjection(context_in_dim, self.inner_dim, eps=eps, prefix=f"{prefix}.txt_in")

        # Entry projections are kept full precision — small sensitive layers at the network boundary.
        self.img_in = ReplicatedLinear(
            in_channels * patch_size * patch_size,
            self.inner_dim,
            bias=False,
            return_bias=False,
            quant_config=None,
            prefix=f"{prefix}.img_in",
        )

        # One shared modulation for every block: [mod1.scale, mod1.gate, mod2.scale, mod2.gate].
        self.modulation = nn.Sequential(
            nn.SiLU(),
            ReplicatedLinear(
                self.inner_dim,
                4 * self.inner_dim,
                bias=False,
                return_bias=False,
                quant_config=None,
                prefix=f"{prefix}.modulation.1",
            ),
        )

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImage21TransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    quant_config=quant_config,
                    prefix=f"{prefix}.transformer_blocks.{i}",
                    prefix_kv_cache_dtype=self.prefix_kv_cache_dtype,
                )
                for i in range(num_layers)
            ]
        )

        self.norm_out = QwenImage21AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, eps=eps, prefix=f"{prefix}.norm_out"
        )
        self.proj_out = ReplicatedLinear(
            self.inner_dim,
            patch_size * patch_size * self.out_channels,
            bias=False,
            return_bias=False,
            quant_config=None,
            prefix=f"{prefix}.proj_out",
        )

        # Module boundary where _sp_plan shards the target tokens + their RoPE freqs.
        self.sequence_prepare = QwenImage21SequencePrepare(self.img_in, self.pos_embed)

        # Use decode graphs unless the caller disables them or requests eager
        # execution. Model-level (sequential) offload registers its swap hook
        # on this top-level module rather than on individual blocks; graphs
        # stay eligible only when that hook keeps weights on persistent
        # staging storage (see QwenImage21DecodeGraphManager._offload_reason).
        self.enable_cuda_graph_decode = (
            getattr(od_config, "enable_cuda_graph_decode", True) and not od_config.enforce_eager
        )
        self._decode_graph_manager = (
            QwenImage21DecodeGraphManager(
                self,
                max_entries=cuda_graph_max_decode_graphs,
                model_level_offload=(
                    od_config.enable_cpu_offload or getattr(od_config, "enable_distributed_layerwise_offload", False)
                ),
            )
            if self.enable_cuda_graph_decode
            else None
        )

    @staticmethod
    def build_token_metadata(
        image_pad_mask: torch.Tensor, img_shapes: list[tuple[int, int, int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Label every token of the joint sequence with the image block it belongs to.

        Block boundaries come from the token counts in `img_shapes`, not from runs of `True` in `image_pad_mask`: two
        condition images that happen to sit next to each other with no text between them form one run but must stay
        separate blocks, otherwise they would attend to each other bidirectionally.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: `image_ids` `(seq_len,)` with `-1` at text positions and a unique id
            per image block, and `target_token_mask` `(seq_len,)` marking the target image's tokens.
        """
        image_positions = image_pad_mask.nonzero(as_tuple=True)[0]
        block_lengths = [math.prod(shape) for shape in img_shapes]
        if sum(block_lengths) != image_positions.numel():
            raise ValueError(
                f"img_shapes accounts for {sum(block_lengths)} image tokens but image_pad_mask marks "
                f"{image_positions.numel()}."
            )

        image_ids = torch.full_like(image_pad_mask, -1, dtype=torch.long)
        block_ids = torch.repeat_interleave(
            torch.arange(len(block_lengths), device=image_pad_mask.device),
            torch.tensor(block_lengths, device=image_pad_mask.device),
        )
        image_ids[image_positions] = block_ids

        target_token_mask = torch.zeros_like(image_pad_mask)
        target_token_mask[image_positions[-block_lengths[-1] :]] = True
        return image_ids, target_token_mask

    @staticmethod
    def _build_full_attn_spans(image_ids: torch.Tensor) -> list[tuple[int, int]]:
        """One `[start, end)` full-attention span per image block, in joint-sequence coordinates."""
        spans = []
        for block_id in range(int(image_ids.max()) + 1):
            positions = (image_ids == block_id).nonzero(as_tuple=True)[0]
            spans.append((int(positions[0]), int(positions[-1]) + 1))
        return spans

    def _build_block_causal_mask(
        self,
        image_ids: torch.Tensor,
        joint_key_valid: torch.Tensor | None,
        batch_size: int,
        padded_seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        r"""Dense 4D block-causal mask `(q_idx >= kv_idx) or same_image_block`, excluding padding keys.

        Fallback path for heterogeneous/padded batches and for backends without piecewise-span support.
        """
        seq_len = image_ids.shape[0]
        ids = image_ids
        if padded_seq_len > seq_len:
            ids = F.pad(ids, (0, padded_seq_len - seq_len), value=-1)
        q_idx = torch.arange(padded_seq_len, device=device).view(-1, 1)
        kv_idx = torch.arange(padded_seq_len, device=device).view(1, -1)
        same_image_block = (ids.view(-1, 1) == ids.view(1, -1)) & (ids.view(-1, 1) >= 0)
        mask = (q_idx >= kv_idx) | same_image_block
        mask = mask.view(1, 1, padded_seq_len, padded_seq_len).expand(batch_size, 1, -1, -1)
        if joint_key_valid is not None:
            mask = mask & joint_key_valid.view(batch_size, 1, 1, -1)
        return mask.contiguous()

    def _resolve_attn_path(
        self,
        attn_path_hint: str,
        has_padding: bool,
        sp_padding: bool,
    ) -> str:
        r"""Choose between the piecewise-span and dense-mask attention paths.

        Returns one of `"piecewise"` (block-causal via `full_attn_spans`), `"mask"` (dense 4D mask)
        or `"key_padding"` (no block-causal structure — `causal_block=False` or KV-cache decode).
        """
        if not self.causal_block:
            return "key_padding"
        if attn_path_hint in ("piecewise", "mask"):
            return attn_path_hint
        if attn_path_hint != "auto":
            raise ValueError(f"Unknown attn_path hint {attn_path_hint!r}; expected 'auto', 'piecewise' or 'mask'.")
        if has_padding or sp_padding:
            return "mask"
        backend = self.transformer_blocks[0].attn.attn.attn_backend
        if backend is not None and backend.supports_piecewise_spans:
            return "piecewise"
        return "mask"

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_shapes: list[list[tuple[int, int, int]]],
        img_mask: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        kv_cache: list[dict[str, dict[str, torch.Tensor]]] | None = None,
        cache_branch: str = "cond",
        return_dict: bool = True,
    ) -> torch.Tensor | Transformer2DModelOutput:
        r"""
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Packed latents, condition images first and the target image last.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, context_in_dim)`):
                Text embeddings from the vision-language encoder.
            timestep (`torch.Tensor`):
                Current denoising step, scaled to `[0, 1]`.
            img_shapes (`list[list[tuple[int, int, int]]]`):
                Per-sample list of `(frame, height, width)` in latent tokens, condition images first and the target
                image last. All samples must share a layout.
            img_mask (`torch.Tensor` of shape `(batch_size, vlm_sequence_length)`):
                `True` at the vision-language encoder's image slots, each standing for a `2x2` group of latent tokens.
            encoder_hidden_states_mask (`torch.Tensor`, *optional*):
                `(batch_size, text_sequence_length)` bool marking valid text tokens. Padded positions are excluded
                from attention.
            attention_kwargs (dict, *optional*): Extra options. Recognized key: `attn_path` — `"auto"` (default),
                `"piecewise"` or `"mask"` to force the block-causal attention implementation.
            kv_cache (list of dict, *optional*): One dict per block, keyed by CFG branch (`"cond"`/`"uncond"`).
                Empty dicts prefill the text and condition-image keys and values; populated dicts switch to decode,
                where only the target image's tokens are recomputed. Requires `causal_condition`.
            cache_branch (`str`): CFG branch of the KV cache to read/write this forward.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a `Transformer2DModelOutput` instead of a plain tuple.
        """
        attention_kwargs = attention_kwargs or {}
        attn_path_hint = attention_kwargs.get("attn_path", "auto")

        batch_size = hidden_states.shape[0]

        if kv_cache is not None and not self.causal_condition:
            raise ValueError(
                "kv_cache requires `causal_condition=True`. The cache is only valid because text and condition-image "
                "tokens modulate from t=0, which makes their activations independent of the denoising step."
            )

        # All samples must share a token layout (same interleaving of text and image blocks);
        # per-sample variation enters only through `encoder_hidden_states_mask` padding.
        layout = img_shapes[0]
        for sample_shapes in img_shapes[1:]:
            if list(sample_shapes) != list(layout):
                raise ValueError("All samples in a batch must share the same img_shapes layout.")
        if batch_size > 1 and not bool((img_mask == img_mask[:1]).all()):
            raise ValueError("All samples in a batch must share the same img_mask layout.")

        sp_active = self.parallel_config is not None and self.parallel_config.sequence_parallel_size > 1
        ctx = get_forward_context() if is_forward_context_available() else None
        sp_padding = 0
        if sp_active:
            # The prefix is replicated across SP ranks for the Ulysses joint mechanism.
            if ctx is not None:
                ctx.split_text_embed_in_sp = False
                if (
                    getattr(self.parallel_config, "mask_sp_padding", False)
                    and ctx.sp_original_seq_len is not None
                    and ctx.sp_padding_size > 0
                ):
                    sp_padding = int(ctx.sp_padding_size)

        is_decode = kv_cache is not None and len(kv_cache) > 0 and "key" in kv_cache[0].get(cache_branch, {})

        if is_decode and self._decode_graph_manager is not None:
            graph_output = self._decode_graph_manager.try_decode(
                hidden_states=hidden_states,
                timestep=timestep,
                kv_cache=kv_cache,
                cache_branch=cache_branch,
                img_shapes=layout,
                img_mask=img_mask,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
            )
            if graph_output is not None:
                if not return_dict:
                    return (graph_output,)
                return Transformer2DModelOutput(sample=graph_output)

        if is_decode:
            txt_hidden_states = None
        else:
            txt_hidden_states = self.txt_in(encoder_hidden_states)

        prefix_hidden_states, target_hidden_states, prefix_freqs, target_freqs = self.sequence_prepare(
            hidden_states, txt_hidden_states, layout, img_mask, is_decode
        )
        # Local layout is always [prefix, target (shard)]: full sequence when SP is off.
        joint_hidden_states = torch.cat([prefix_hidden_states, target_hidden_states], dim=1)
        freqs = torch.cat([prefix_freqs, target_freqs], dim=0)
        prefix_len = prefix_hidden_states.shape[1]
        sp_prefix_len = prefix_len if sp_active and not is_decode else 0

        image_pad_mask = self.sequence_prepare.expand_image_pad_mask(img_mask[0])
        _, target_token_mask = self.build_token_metadata(image_pad_mask, layout)

        timestep = timestep.to(hidden_states.dtype)
        if self.causal_condition:
            # Extra t=0 row; text and condition-image tokens modulate from it. `modulation_mask` selects which row
            # each token reads, and is `None` when every token shares the sampled timestep.
            timestep = torch.cat([timestep, timestep.new_zeros(1)], dim=0)
            modulation_mask = target_token_mask
        else:
            modulation_mask = None
        temb = self.time_text_embed(timestep, joint_hidden_states)
        modulation = self.modulation(temb)

        # Align the modulation mask with the local [prefix, target shard] layout.
        if modulation_mask is not None:
            local_mask = torch.cat(
                [
                    modulation_mask[:prefix_len],
                    torch.ones(target_hidden_states.shape[1], dtype=torch.bool, device=modulation_mask.device),
                ]
            )
        else:
            local_mask = None

        # Right-padded prompt positions must never be attended to, on any path. Text positions of the joint sequence
        # line up, in order, with the non-image positions of the vision-language sequence — the two are interleaved,
        # so the mask cannot be sliced off as a prefix.
        joint_key_valid = None
        if encoder_hidden_states_mask is not None:
            joint_key_valid = torch.ones(
                batch_size, image_pad_mask.shape[0], dtype=torch.bool, device=hidden_states.device
            )
            text_positions = (~image_pad_mask).nonzero(as_tuple=True)[0]
            vlm_text_positions = ~img_mask[0][: encoder_hidden_states_mask.shape[1]]
            joint_key_valid[:, text_positions] = encoder_hidden_states_mask.bool()[:, vlm_text_positions]

        image_ids, _ = self.build_token_metadata(image_pad_mask, layout)
        full_seq_len = image_pad_mask.shape[0]
        padded_seq_len = full_seq_len + sp_padding
        if sp_padding and joint_key_valid is None:
            joint_key_valid = torch.ones(batch_size, full_seq_len, dtype=torch.bool, device=hidden_states.device)
        if sp_padding:
            joint_key_valid = F.pad(joint_key_valid, (0, sp_padding), value=False)

        has_padding = joint_key_valid is not None and not bool(joint_key_valid.all())
        if not has_padding:
            # Match Diffusers and use the same mask-free dispatch in eager and graph decode.
            joint_key_valid = None
        attn_metadata: AttentionMetadata | None = None
        cache_write_len: int | None = None
        if is_decode:
            # Decode: block-causal degenerates to full attention for target rows; only key padding applies.
            if joint_key_valid is not None:
                attn_metadata = AttentionMetadata(attn_mask=joint_key_valid)
        else:
            attn_path = self._resolve_attn_path(attn_path_hint, has_padding, sp_padding > 0)
            if kv_cache is not None:
                cache_write_len = prefix_len
            if attn_path == "piecewise":
                spans = self._build_full_attn_spans(image_ids)
                attn_metadata = AttentionMetadata(full_attn_spans=[list(spans) for _ in range(batch_size)])
            elif attn_path == "mask":
                attn_metadata = AttentionMetadata(
                    attn_mask=self._build_block_causal_mask(
                        image_ids, joint_key_valid, batch_size, padded_seq_len, hidden_states.device
                    )
                )
            elif joint_key_valid is not None:
                attn_metadata = AttentionMetadata(attn_mask=joint_key_valid)

        qkv_rope_tables = None
        if self.mindiesd_qkv_fusion:
            if (
                joint_hidden_states.device.type == "npu"
                and joint_hidden_states.dtype == torch.bfloat16
                and batch_size == 1
                and self.mindiesd_qkv_geometry
            ):
                qkv_rope_tables = prepare_rope_tables(freqs, joint_hidden_states.dtype)
            else:
                logger.warning_once("Qwen2.1 MindIE-SD QKV fusion requires NPU BF16, batch=1 and head_dim=128")
        for index_block, block in enumerate(self.transformer_blocks):
            block_kv_cache = kv_cache[index_block] if kv_cache is not None else None
            joint_hidden_states = block(
                hidden_states=joint_hidden_states,
                modulation=modulation,
                freqs=freqs,
                target_token_mask=local_mask,
                attn_metadata=attn_metadata,
                sp_prefix_len=sp_prefix_len,
                sp_decode=sp_active and is_decode,
                kv_cache=block_kv_cache,
                cache_branch=cache_branch,
                cache_write_len=cache_write_len,
                qkv_rope_tables=qkv_rope_tables,
            )

        if (
            not is_decode
            and cache_write_len is not None
            and kv_cache is not None
            and self._decode_graph_manager is not None
        ):
            # Allocate graph scratch buffers; the request keeps its own prefix K/V.
            self._decode_graph_manager.register_prefill(
                kv_cache=kv_cache,
                cache_branch=cache_branch,
                prefix_len=cache_write_len,
                img_shapes=layout,
                target_freqs=target_freqs,
                joint_key_valid=joint_key_valid,
                dtype=hidden_states.dtype,
            )

        if sp_active and not is_decode:
            # Only target tokens feed proj_out under SP: the prefix is replicated on every
            # rank and gathering it would duplicate it. Output is the (gathered) target image.
            joint_hidden_states = joint_hidden_states[:, prefix_len:]
            if local_mask is not None:
                local_mask = local_mask[prefix_len:]

        joint_hidden_states = self.norm_out(joint_hidden_states, temb, local_mask)
        output = self.proj_out(joint_hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def _decode_graph_forward(self, entry) -> torch.Tensor:
        """The exact decode-step computation, captured into a CUDA graph.

        Mirrors the eager decode path of ``forward``: target tokens only,
        prefix K/V read from the entry's static buffers, the same mask-free
        attention dispatch as eager decode, and an all-ones modulation mask.
        All inputs live in the entry's fixed-address buffers.
        """
        hidden_states = self.img_in(entry.hidden)
        timestep = torch.cat([entry.timestep, entry.timestep.new_zeros(1)], dim=0)
        temb = self.time_text_embed(timestep, hidden_states)
        modulation = self.modulation(temb)
        for index_block, block in enumerate(self.transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                modulation=modulation,
                freqs=entry.freqs,
                target_token_mask=entry.target_token_mask,
                attn_metadata=entry.attn_metadata,
                sp_prefix_len=0,
                sp_decode=False,
                kv_cache=entry.block_caches[index_block],
                cache_branch=entry.branch,
                cache_write_len=None,
            )
        hidden_states = self.norm_out(hidden_states, temb, entry.target_token_mask)
        return self.proj_out(hidden_states)

    def release_captured_graphs(self) -> None:
        """Drop captured decode graphs and their static buffers (e.g. sleep mode)."""
        if self._decode_graph_manager is not None:
            self._decode_graph_manager.clear()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".to_qkv", ".to_q", "q"),
            (".to_qkv", ".to_k", "k"),
            (".to_qkv", ".to_v", "v"),
        ]
        # Expose packed shard mappings for LoRA handling of fused projections.
        self.stacked_params_mapping = stacked_params_mapping

        params_dict = dict(self.named_parameters())

        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            original_name = name.removeprefix("transformer.")
            lookup_name, shard_id = _resolve_qwen_image21_lookup_name(
                original_name,
                stacked_params_mapping,
            )

            if lookup_name.endswith(".bias") and lookup_name not in params_dict:
                continue

            param = params_dict.get(lookup_name)
            if param is None:
                logger.warning("Skipping unexpected Qwen-Image 2.1 transformer weight %s", original_name)
                continue

            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            if shard_id is None:
                weight_loader(param, loaded_weight)
            else:
                weight_loader(param, loaded_weight, shard_id)

            loaded_params.add(original_name)
            loaded_params.add(lookup_name)

        unloaded = sorted(set(params_dict) - set(loaded_params))
        if unloaded:
            logger.warning("Model parameters not loaded from checkpoint: %s", unloaded)
        return loaded_params
