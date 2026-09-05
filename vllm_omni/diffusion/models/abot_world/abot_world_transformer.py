# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Causal WanModel variant for ABot-World with KV-cache inference and action adapter."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Self, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.conv import Conv3dLayer
from vllm.model_executor.layers.linear import ColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.layers.norm import LayerNorm
from vllm_omni.diffusion.layers.rope import RotaryEmbeddingWan
from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import (
    ARDiffusionPagedLayerContext,
    ARDiffusionPagedLayerInputs,
    ar_diffusion_paged_attention,
    paged_write_attn,
)

# Re-export these symbols so registry detection works correctly.
ABOT_WORLD_WAN_PACKED_MODULES = {"qkv": ["q", "k", "v"]}


# ── Inline helpers (avoid importing private symbols from wan2_2_transformer) ──


def _sinusoidal_embedding(dim: int, timestep: torch.Tensor) -> torch.Tensor:
    if dim % 2:
        raise ValueError(f"freq_dim must be even, got {dim}.")
    half_dim = dim // 2
    timestep = timestep.to(torch.float64)
    frequencies = torch.pow(
        10000,
        -torch.arange(half_dim, device=timestep.device, dtype=torch.float64) / half_dim,
    )
    phase = torch.outer(timestep, frequencies)
    return torch.cat((phase.cos(), phase.sin()), dim=1)


def _rope_axis(max_seq_len: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    if dim == 0:
        empty = torch.empty(max_seq_len, 0, dtype=torch.float32)
        return empty, empty.clone()
    if dim % 2:
        raise ValueError(f"RoPE axis dimension must be even, got {dim}.")
    frequencies = 1.0 / torch.pow(
        10000,
        torch.arange(0, dim, 2, dtype=torch.float64) / dim,
    )
    phase = torch.outer(torch.arange(max_seq_len, dtype=torch.float64), frequencies)
    return phase.cos().float(), phase.sin().float()


def _projection_prefix(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


class _ABotRMSNorm(nn.Module):
    """RMSNorm over a tensor-parallel sharded projection."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})

    def weight_loader(self, param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        if param.shape == loaded_weight.shape:
            param.data.copy_(loaded_weight)
            return
        tp_size = get_tensor_model_parallel_world_size()
        if loaded_weight.shape[0] % tp_size != 0:
            raise ValueError(
                f"Cannot shard RMSNorm weight of shape {tuple(loaded_weight.shape)} across tp_size={tp_size}."
            )
        shard_size = loaded_weight.shape[0] // tp_size
        shard_start = get_tensor_model_parallel_rank() * shard_size
        shard = loaded_weight.narrow(0, shard_start, shard_size)
        if param.shape != shard.shape:
            raise ValueError(f"RMSNorm shard shape mismatch: param={tuple(param.shape)}, shard={tuple(shard.shape)}.")
        param.data.copy_(shard)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        tp_size = get_tensor_model_parallel_world_size()
        value_float = value.float()
        sum_of_squares = value_float.pow(2).sum(dim=-1, keepdim=True)
        element_count = value.shape[-1]
        if tp_size > 1:
            sum_of_squares = tensor_model_parallel_all_reduce(sum_of_squares)
            element_count *= tp_size
        rms = torch.sqrt(sum_of_squares / element_count + self.eps)
        return (value_float / rms * self.weight.float()).to(value.dtype)


@dataclass
class ABotAttentionCache:
    """K/V storage plus logical cursor metadata for a single attention layer."""

    key: torch.Tensor
    value: torch.Tensor
    end: int = 0
    absolute_end: int = 0
    last_start: int | None = None
    sink_end: int = 0


@dataclass
class ABotTransformerCache:
    """One request's per-layer video K/V and reusable text K/V."""

    self_attention: list[ABotAttentionCache | ARDiffusionPagedLayerContext | ARDiffusionPagedLayerInputs]
    cross_attention: list[ABotAttentionCache | None]


def allocate_abot_cache(
    *,
    batch_size: int,
    num_layers: int,
    max_tokens: int,
    num_local_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> ABotTransformerCache:
    shape = (batch_size, max_tokens, num_local_heads, head_dim)
    self_attention = [
        ABotAttentionCache(
            key=torch.zeros(shape, device=device, dtype=dtype),
            value=torch.zeros(shape, device=device, dtype=dtype),
        )
        for _ in range(num_layers)
    ]
    return ABotTransformerCache(
        self_attention=self_attention,
        cross_attention=[None for _ in range(num_layers)],
    )


class ABotCausalSelfAttention(nn.Module):
    """Block-causal self-attention for ABot-World with KV-cache support.

    Matches the checkpoint parameter names (to_qkv, to_out, norm_q, norm_k)
    so weights load directly from the HuggingFace safetensors file.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        head_dim: int,
        eps: float = 1e-5,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")

        tp_size = get_tensor_model_parallel_world_size()
        if num_heads % tp_size != 0:
            raise ValueError(f"num_heads={num_heads} must be divisible by tp_size={tp_size}.")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim

        self.qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=head_dim,
            total_num_heads=num_heads,
            bias=True,
            prefix=_projection_prefix(prefix, "qkv"),
        )
        self.num_local_heads = self.qkv.num_heads
        self.tp_inner_dim = self.num_local_heads * head_dim

        self.norm_q = _ABotRMSNorm(self.tp_inner_dim, eps)
        self.norm_k = _ABotRMSNorm(self.tp_inner_dim, eps)

        self.o = RowParallelLinear(
            self.inner_dim,
            dim,
            bias=True,
            input_is_parallel=True,
            return_bias=False,
            prefix=_projection_prefix(prefix, "o"),
        )
        self.rotary_embedding = RotaryEmbeddingWan(is_neox_style=False, half_head_dim=True)
        self.attn = Attention(
            num_heads=self.num_local_heads,
            head_size=head_dim,
            num_kv_heads=self.num_local_heads,
            softmax_scale=1.0 / (head_dim**0.5),
            causal=False,
            role="self",
            qkv_layout="BSND",
            prefix=prefix,
            skip_sequence_parallel=True,
        )

    def _update_cache(
        self,
        cache: ABotAttentionCache,
        key: torch.Tensor,
        value: torch.Tensor,
        current_start: int,
        *,
        sink_tokens: int,
        update_cache: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        chunk_tokens = key.shape[1]
        capacity = cache.key.shape[1]

        if cache.last_start is None:
            if chunk_tokens > capacity:
                raise ValueError(f"Current chunk has {chunk_tokens} tokens but cache can hold only {capacity}.")
            next_key = key
            next_value = value
        elif current_start == cache.last_start:
            if current_start + chunk_tokens != cache.absolute_end:
                raise ValueError("A repeated current_start must overwrite the same-size current chunk.")
            if chunk_tokens > cache.end:
                raise ValueError("The current chunk is no longer fully retained in the cache.")
            prefix_end = cache.end - chunk_tokens
            next_key = torch.cat((cache.key[:, :prefix_end], key), dim=1)
            next_value = torch.cat((cache.value[:, :prefix_end], value), dim=1)
        else:
            if current_start < cache.last_start:
                raise ValueError(f"current_start={current_start} precedes latest chunk start {cache.last_start}.")
            if current_start < cache.absolute_end:
                raise ValueError(
                    f"current_start={current_start} overlaps cached tokens ending at {cache.absolute_end}."
                )
            if current_start > cache.absolute_end:
                raise ValueError(
                    f"New chunks must be contiguous: current_start={current_start}, expected {cache.absolute_end}."
                )

            incoming_sink = max(0, min(chunk_tokens, sink_tokens - current_start))
            old_sink_key = cache.key[:, : cache.sink_end]
            old_sink_value = cache.value[:, : cache.sink_end]
            new_sink_key = key[:, :incoming_sink]
            new_sink_value = value[:, :incoming_sink]

            old_local_key = cache.key[:, cache.sink_end : cache.end]
            old_local_value = cache.value[:, cache.sink_end : cache.end]
            new_local_key = key[:, incoming_sink:]
            new_local_value = value[:, incoming_sink:]

            local_capacity = capacity - (cache.sink_end + incoming_sink)
            if new_local_key.shape[1] > local_capacity:
                raise ValueError("Configured cache cannot retain all sink tokens and the full current chunk.")
            retained = min(old_local_key.shape[1], local_capacity - new_local_key.shape[1])
            old_local_key = old_local_key[:, -retained:] if retained else old_local_key[:, :0]
            old_local_value = old_local_value[:, -retained:] if retained else old_local_value[:, :0]

            next_key = torch.cat((old_sink_key, new_sink_key, old_local_key, new_local_key), dim=1)
            next_value = torch.cat((old_sink_value, new_sink_value, old_local_value, new_local_value), dim=1)

        next_end = next_key.shape[1]
        if update_cache:
            cache.key[:, :next_end].copy_(next_key)
            cache.value[:, :next_end].copy_(next_value)
            cache.end = next_end
            cache.absolute_end = current_start + chunk_tokens
            cache.last_start = current_start
            cache.sink_end = cache.sink_end + max(0, min(chunk_tokens, sink_tokens - current_start))

        return next_key, next_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cache: ABotAttentionCache | ARDiffusionPagedLayerInputs,
        current_start: int,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        sink_tokens: int = 0,
        update_cache: bool = True,
    ) -> torch.Tensor:
        qkv, _ = self.qkv(hidden_states)
        q_size = self.tp_inner_dim
        kv_size = self.tp_inner_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)

        query = self.norm_q(query)
        key = self.norm_k(key)

        query = query.unflatten(2, (self.num_local_heads, self.head_dim))
        key = key.unflatten(2, (self.num_local_heads, self.head_dim))
        value = value.unflatten(2, (self.num_local_heads, self.head_dim))

        if rotary_emb is not None:
            cos, sin = rotary_emb
            query = self.rotary_embedding(query, cos, sin)
            key = self.rotary_embedding(key, cos, sin)

        if isinstance(cache, ARDiffusionPagedLayerInputs):
            if query.shape[0] != 1:
                raise RuntimeError("ABot AR-Diffusion paged attention requires batch_size=1.")
            hidden_states = paged_write_attn(
                cache, query[0], key[0], value[0], None, None, self.head_dim**-0.5
            ).unsqueeze(0)
        else:
            visible_key, visible_value = self._update_cache(
                cache,
                key,
                value,
                current_start,
                sink_tokens=sink_tokens,
                update_cache=update_cache,
            )
            if query.is_cuda and query.shape[0] == 1:
                block_size = key.shape[1]
                key_cache = visible_key[0].unflatten(0, (-1, block_size))
                value_cache = visible_value[0].unflatten(0, (-1, block_size))
                block_count = key_cache.shape[0]
                hidden_states = ar_diffusion_paged_attention(
                    query,
                    key_cache,
                    value_cache,
                    block_table=torch.arange(block_count, dtype=torch.int32, device=query.device).unsqueeze(0),
                    query_start_loc=torch.tensor([0, query.shape[1]], dtype=torch.int32, device=query.device),
                    seq_lens=torch.tensor([visible_key.shape[1]], dtype=torch.int32, device=query.device),
                    max_query_len=query.shape[1],
                    max_seq_len=visible_key.shape[1],
                    softmax_scale=self.head_dim**-0.5,
                )
            else:
                hidden_states = self.attn(query, visible_key, visible_value)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        return self.o(hidden_states)


class ABotCausalCrossAttention(nn.Module):
    """Cross-attention using the q/k/v/o names in the Wan checkpoint."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        head_dim: int,
        eps: float = 1e-5,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")

        tp_size = get_tensor_model_parallel_world_size()
        if num_heads % tp_size != 0:
            raise ValueError(f"num_heads={num_heads} must be divisible by tp_size={tp_size}.")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_local_heads = num_heads // tp_size
        self.tp_inner_dim = self.num_local_heads * head_dim

        self.q = ColumnParallelLinear(
            dim,
            dim,
            bias=True,
            gather_output=False,
            return_bias=False,
            prefix=_projection_prefix(prefix, "q"),
        )
        self.k = ColumnParallelLinear(
            dim,
            dim,
            bias=True,
            gather_output=False,
            return_bias=False,
            prefix=_projection_prefix(prefix, "k"),
        )
        self.v = ColumnParallelLinear(
            dim,
            dim,
            bias=True,
            gather_output=False,
            return_bias=False,
            prefix=_projection_prefix(prefix, "v"),
        )
        self.o = RowParallelLinear(
            dim,
            dim,
            bias=True,
            input_is_parallel=True,
            return_bias=False,
            prefix=_projection_prefix(prefix, "o"),
        )
        self.norm_q = _ABotRMSNorm(self.tp_inner_dim, eps)
        self.norm_k = _ABotRMSNorm(self.tp_inner_dim, eps)
        self.attn = Attention(
            num_heads=self.num_local_heads,
            head_size=head_dim,
            num_kv_heads=self.num_local_heads,
            softmax_scale=1.0 / (head_dim**0.5),
            causal=False,
            role="cross",
            qkv_layout="BSND",
            prefix=prefix,
            skip_sequence_parallel=True,
            disable_kv_quant=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        *,
        cache: ABotAttentionCache | None,
    ) -> tuple[torch.Tensor, ABotAttentionCache]:
        query = self.norm_q(self.q(hidden_states))
        query = query.unflatten(2, (self.num_local_heads, self.head_dim))

        if cache is None:
            if encoder_hidden_states is None:
                raise ValueError("encoder_hidden_states required when cross-attention cache is empty.")
            key = self.norm_k(self.k(encoder_hidden_states))
            value = self.v(encoder_hidden_states)
            key = key.unflatten(2, (self.num_local_heads, self.head_dim))
            value = value.unflatten(2, (self.num_local_heads, self.head_dim))
            cache = ABotAttentionCache(
                key=key,
                value=value,
                end=key.shape[1],
                absolute_end=key.shape[1],
                last_start=0,
            )
        else:
            key = cache.key[:, : cache.end]
            value = cache.value[:, : cache.end]

        output = self.attn(query, key, value)
        return self.o(output.flatten(2, 3)), cache


class _ABotResidualBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.conv2(self.relu(self.conv1(hidden_states)))


class ABotSimpleAdapter(nn.Module):
    """Action control adapter mapping 32-channel action features to DiT latent space.

    PixelUnshuffle(16) → 8192 channels, Conv2d(8192→dim, 2×2, stride=2),
    then ResidualBlock(dim).  The result is added to patch-embedded video tokens.
    """

    def __init__(
        self,
        dim: int,
        *,
        downscale_factor: int = 16,
        control_in_dim: int = 32,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.downscale_factor = downscale_factor
        self.control_in_dim = control_in_dim
        in_channels = control_in_dim * downscale_factor * downscale_factor
        self.conv = nn.Conv2d(
            in_channels,
            dim,
            kernel_size=2,
            stride=2,
        )
        self.residual_blocks = nn.Sequential(_ABotResidualBlock(dim))

    def forward(
        self,
        patch_tokens: torch.Tensor,
        action_condition: torch.Tensor,
        *,
        num_frames: int,
        spatial_tokens: int,
    ) -> torch.Tensor:
        """Apply action conditioning to patch tokens.

        Args:
            patch_tokens: [B, F*H*W, dim] video patch tokens.
            action_condition: [B, control_in_dim, F, H_pix, W_pix] raw action tensor.
            num_frames: number of temporal frames.
            spatial_tokens: tokens per frame (H*W after patching).
        """
        B, _, F_act, height, width = action_condition.shape
        frame_batch = action_condition.permute(0, 2, 1, 3, 4).reshape(B * F_act, self.control_in_dim, height, width)
        unshuffled = F.pixel_unshuffle(frame_batch, self.downscale_factor)
        features = self.residual_blocks(self.conv(unshuffled))
        # Reshape back to sequence: [B, F, dim, H', W'] → [B, F*H'*W', dim]
        _, dim, H_feat, W_feat = features.shape
        features = features.reshape(B, F_act, dim, H_feat, W_feat)
        features = features.permute(0, 1, 3, 4, 2).reshape(B, -1, dim)

        target_tokens = num_frames * spatial_tokens
        if features.shape[1] != target_tokens:
            raise ValueError(
                f"ABot action adapter geometry does not match video patches: {features.shape[1]} != {target_tokens}."
            )

        return patch_tokens + features


class ABotCausalAttentionBlock(nn.Module):
    """Per-block modulation matching checkpoint parameter names."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        head_dim: int,
        ffn_dim: int,
        eps: float = 1e-5,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.norm1 = LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.self_attn = ABotCausalSelfAttention(
            dim,
            num_heads,
            head_dim=head_dim,
            eps=eps,
            prefix=_projection_prefix(prefix, "self_attn"),
        )
        self.cross_attn = ABotCausalCrossAttention(
            dim,
            num_heads,
            head_dim=head_dim,
            eps=eps,
            prefix=_projection_prefix(prefix, "cross_attn"),
        )
        self.norm2 = LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = LayerNorm(dim, eps=eps, elementwise_affine=True)
        self.ffn = nn.Sequential(
            ColumnParallelLinear(
                dim,
                ffn_dim,
                bias=True,
                gather_output=False,
                return_bias=False,
                prefix=_projection_prefix(prefix, "ffn.0"),
            ),
            nn.GELU(approximate="tanh"),
            RowParallelLinear(
                ffn_dim,
                dim,
                bias=True,
                input_is_parallel=True,
                return_bias=False,
                prefix=_projection_prefix(prefix, "ffn.2"),
            ),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / math.sqrt(dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        temb: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        *,
        self_cache: ABotAttentionCache | ARDiffusionPagedLayerInputs,
        cross_cache: ABotAttentionCache | None,
        current_start: int,
        sink_tokens: int = 0,
        update_cache: bool = True,
    ) -> tuple[torch.Tensor, ABotAttentionCache]:
        modulation = (self.modulation.unsqueeze(1) + temb.float()).to(hidden_states.dtype)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            value.squeeze(2) for value in modulation.chunk(6, dim=2)
        )

        # Self-attention with per-frame modulation
        norm_hidden = self.norm1(hidden_states.float()).to(hidden_states.dtype)
        norm_hidden = norm_hidden * (1 + scale_msa) + shift_msa
        attn_output = self.self_attn(
            norm_hidden,
            cache=self_cache,
            current_start=current_start,
            rotary_emb=rotary_emb,
            sink_tokens=sink_tokens,
            update_cache=update_cache,
        )
        hidden_states = hidden_states + attn_output * gate_msa

        # Cross-attention
        norm_hidden = self.norm3(hidden_states)
        attn_output, cross_cache = self.cross_attn(
            norm_hidden,
            encoder_hidden_states,
            cache=cross_cache,
        )
        hidden_states = hidden_states + attn_output

        # Feed-forward with per-frame modulation
        norm_hidden = self.norm2(hidden_states.float()).to(hidden_states.dtype)
        norm_hidden = norm_hidden * (1 + c_scale_msa) + c_shift_msa
        ff_output = self.ffn(norm_hidden)
        hidden_states = (hidden_states + ff_output * c_gate_msa).to(hidden_states.dtype)

        return hidden_states, cast(ABotAttentionCache, cross_cache)


class ABotCausalHead(nn.Module):
    """Wan output head with checkpoint-compatible parameter names."""

    def __init__(self, dim: int, out_dim: int, eps: float) -> None:
        super().__init__()
        self.norm = LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / math.sqrt(dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        *,
        tokens_per_frame: int,
    ) -> torch.Tensor:
        batch_size, frames, _ = temb.shape
        hidden_states = hidden_states.unflatten(1, (frames, tokens_per_frame))
        modulation = (self.modulation.unsqueeze(1) + temb.unsqueeze(2).float()).to(hidden_states.dtype)
        shift, scale = (value.squeeze(2) for value in modulation.chunk(2, dim=2))
        norm_hidden = self.norm(hidden_states.float()).to(hidden_states.dtype)
        norm_hidden = norm_hidden * (1 + scale.unsqueeze(2)) + shift.unsqueeze(2)
        return self.head(norm_hidden).reshape(batch_size, frames * tokens_per_frame, -1)


class ABotWorldCausalTransformer3DModel(nn.Module):
    """Checkpoint-compatible causal Wan variant for ABot-World.

    Matches the checkpoint parameter layout::
        model.patch_embedding.*
        model.blocks.<n>.self_attn.to_qkv.*   (fused QKV)
        model.blocks.<n>.self_attn.norm_q.*
        model.blocks.<n>.self_attn.norm_k.*
        model.blocks.<n>.self_attn.to_out.*
        model.blocks.<n>.cross_attn.to_q/k/v/out.*
        model.blocks.<n>.cross_attn.norm_q/k.*
        model.blocks.<n>.ffn.0/2.*
        model.blocks.<n>.scale_shift_table
        model.act_control_adapter.conv.*
        model.act_control_adapter.residual_blocks.0.conv1/conv2.*
        model.head.norm.* / model.head.head.* / model.head.modulation
        model.time_embedding.*
        model.time_projection.*
        model.text_embedding.*
    """

    _repeated_blocks = ["ABotCausalAttentionBlock"]
    packed_modules_mapping = ABOT_WORLD_WAN_PACKED_MODULES
    _layerwise_offload_blocks_attrs = ["blocks"]

    def __init__(
        self,
        *,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        in_channels: int = 48,
        out_channels: int = 48,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 14336,
        num_layers: int = 30,
        eps: float = 1e-6,
        rope_max_seq_len: int = 1024,
        downscale_factor_control_adapter: int = 16,
        prefix: str = "",
    ) -> None:
        super().__init__()
        dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels
        self.dim = dim

        self.config = SimpleNamespace(
            patch_size=patch_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            text_dim=text_dim,
            freq_dim=freq_dim,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            eps=eps,
            rope_max_seq_len=rope_max_seq_len,
            downscale_factor_control_adapter=downscale_factor_control_adapter,
        )

        self.patch_embedding = Conv3dLayer(
            in_channels=in_channels,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Control adapter (action conditioning)
        self.act_control_adapter = ABotSimpleAdapter(dim, downscale_factor=downscale_factor_control_adapter)

        # Time and text condition embeddings
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                ABotCausalAttentionBlock(
                    dim,
                    num_attention_heads,
                    head_dim=attention_head_dim,
                    ffn_dim=ffn_dim,
                    eps=eps,
                    prefix=_projection_prefix(prefix, f"blocks.{i}"),
                )
                for i in range(num_layers)
            ]
        )

        # Output head
        self.head = ABotCausalHead(dim, out_channels * math.prod(patch_size), eps)

        # RoPE buffers
        temporal_dim = attention_head_dim - 4 * (attention_head_dim // 6)
        height_dim = width_dim = 2 * (attention_head_dim // 6)
        for axis, axis_dim in (("temporal", temporal_dim), ("height", height_dim), ("width", width_dim)):
            cosine, sine = _rope_axis(rope_max_seq_len, axis_dim)
            self.register_buffer(f"_rope_{axis}_cosine", cosine, persistent=False)
            self.register_buffer(f"_rope_{axis}_sine", sine, persistent=False)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        prefix: str = "",
    ) -> Self:
        patch_size = tuple(config.get("patch_size", (1, 2, 2)))
        num_heads = config.get("num_attention_heads") or config.get("num_heads") or 24
        dim = config.get("dim") or config.get("attention_head_dim", 128) * num_heads
        head_dim = dim // num_heads
        in_channels = config.get("in_channels") or config.get("in_dim") or 48
        out_channels = config.get("out_channels") or config.get("out_dim") or 48
        text_dim = config.get("text_dim") or 4096
        freq_dim = config.get("freq_dim", 256)
        ffn_dim = config.get("ffn_dim", 14336)
        num_layers = config.get("num_layers", 30)
        eps = config.get("eps", 1e-6)
        rope_max_seq_len = config.get("rope_max_seq_len", 1024)
        downscale_factor_control_adapter = config.get("downscale_factor_control_adapter", 16)

        return cls(
            patch_size=patch_size,
            num_attention_heads=num_heads,
            attention_head_dim=head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            text_dim=text_dim,
            freq_dim=freq_dim,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            eps=eps,
            rope_max_seq_len=rope_max_seq_len,
            downscale_factor_control_adapter=downscale_factor_control_adapter,
            prefix=prefix,
        )

    def _rotary_embedding(
        self,
        *,
        frames: int,
        height: int,
        width: int,
        start_frame: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if start_frame + frames > self.config.rope_max_seq_len:
            raise ValueError("Temporal RoPE positions exceed rope_max_seq_len.")

        def expand_axis(table: torch.Tensor, axis: str) -> torch.Tensor:
            if axis == "temporal":
                return (
                    table[start_frame : start_frame + frames].view(frames, 1, 1, -1).expand(frames, height, width, -1)
                )
            if axis == "height":
                return table[:height].view(1, height, 1, -1).expand(frames, height, width, -1)
            return table[:width].view(1, 1, width, -1).expand(frames, height, width, -1)

        cosine = torch.cat(
            (
                expand_axis(self._rope_temporal_cosine, "temporal"),
                expand_axis(self._rope_height_cosine, "height"),
                expand_axis(self._rope_width_cosine, "width"),
            ),
            dim=-1,
        )
        sine = torch.cat(
            (
                expand_axis(self._rope_temporal_sine, "temporal"),
                expand_axis(self._rope_height_sine, "height"),
                expand_axis(self._rope_width_sine, "width"),
            ),
            dim=-1,
        )
        return (
            cosine.reshape(frames * height * width, -1).to(device=device, dtype=dtype),
            sine.reshape(frames * height * width, -1).to(device=device, dtype=dtype),
        )

    def allocate_cache(
        self,
        *,
        batch_size: int,
        latent_height: int,
        latent_width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> ABotTransformerCache:
        patch_frames, patch_height, patch_width = self.config.patch_size
        window_frames = 21  # local_attn_size from ABot-World config
        post_patch_height = latent_height // patch_height
        post_patch_width = latent_width // patch_width
        max_tokens = int(window_frames * post_patch_height * post_patch_width)
        tp_size = get_tensor_model_parallel_world_size()
        num_local_heads = self.config.num_attention_heads // tp_size
        return allocate_abot_cache(
            batch_size=batch_size,
            num_layers=self.config.num_layers,
            max_tokens=max_tokens,
            num_local_heads=num_local_heads,
            head_dim=self.config.attention_head_dim,
            device=device,
            dtype=dtype,
        )

    def _timestep_embeddings(
        self,
        timestep: torch.Tensor,
        *,
        batch_size: int,
        frames: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if timestep.ndim == 0:
            timestep = timestep.reshape(1)
        if timestep.ndim == 1:
            timestep = timestep.unsqueeze(1).expand(batch_size, frames)
        if timestep.shape != (batch_size, frames):
            raise ValueError(
                "timestep must be scalar, [batch], or [batch, frames]; "
                f"got {tuple(timestep.shape)}, expected {(batch_size, frames)}."
            )
        freq_embed = _sinusoidal_embedding(self.config.freq_dim, timestep.reshape(-1)).to(dtype=dtype)
        temb = self.time_embedding(freq_embed).unflatten(0, (batch_size, frames))
        timestep_proj = self.time_projection(temb).unflatten(2, (6, self.dim))
        return temb, timestep_proj

    def _unpatchify(
        self,
        hidden_states: torch.Tensor,
        *,
        batch_size: int,
        frames: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        patch_frames, patch_height, patch_width = self.config.patch_size
        hidden_states = hidden_states.reshape(
            batch_size,
            frames,
            height,
            width,
            patch_frames,
            patch_height,
            patch_width,
            self.config.out_channels,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        cache: ABotTransformerCache,
        start_frame: int,
        update_cache: bool,
        action_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, channels, frames, height, width = hidden_states.shape
        patch_frames, patch_height, patch_width = self.config.patch_size

        if channels != self.config.in_channels:
            raise ValueError(f"hidden_states must have {self.config.in_channels} channels, got {channels}.")

        patched_frames = frames // patch_frames
        patched_height = height // patch_height
        patched_width = width // patch_width
        tokens_per_frame = patched_height * patched_width
        patched_start_frame = start_frame // patch_frames
        current_start = patched_start_frame * tokens_per_frame
        sink_tokens = 0

        rotary_emb = self._rotary_embedding(
            frames=patched_frames,
            height=patched_height,
            width=patched_width,
            start_frame=patched_start_frame,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # Patch embedding
        hidden_states = self.patch_embedding(hidden_states).flatten(2).transpose(1, 2)

        # Apply action control adapter
        if action_condition is not None:
            hidden_states = self.act_control_adapter(
                hidden_states,
                action_condition,
                num_frames=patched_frames,
                spatial_tokens=tokens_per_frame,
            )

        # Timestep embeddings
        temb, timestep_proj = self._timestep_embeddings(
            timestep,
            batch_size=batch_size,
            frames=patched_frames,
            dtype=hidden_states.dtype,
        )
        timestep_proj = timestep_proj.repeat_interleave(tokens_per_frame, dim=1)

        # Text embedding (cached across layers after first block)
        projected_text = (
            self.text_embedding(encoder_hidden_states) if any(c is None for c in cache.cross_attention) else None
        )

        if cache.self_attention and isinstance(cache.self_attention[0], ARDiffusionPagedLayerContext):
            if batch_size != 1:
                raise RuntimeError("ABot AR-Diffusion paged attention requires batch_size=1.")
            forward_context = cache.self_attention[0].forward_ctx
            expected_seq_len = patched_frames * tokens_per_frame
            if forward_context.seq_len != expected_seq_len:
                raise RuntimeError(
                    "ABot paged context token count does not match this block: "
                    f"{forward_context.seq_len} != {expected_seq_len}."
                )
            forward_context.prepare(
                device=hidden_states.device,
                action_len=0,
                query_len=hidden_states.shape[1],
            )
            cache.self_attention = [layer_context.to_layer_inputs() for layer_context in cache.self_attention]

        for idx, block in enumerate(self.blocks):
            hidden_states, cross_cache = block(
                hidden_states,
                projected_text if cache.cross_attention[idx] is None else None,
                timestep_proj,
                rotary_emb,
                self_cache=cache.self_attention[idx],
                cross_cache=cache.cross_attention[idx],
                current_start=current_start,
                sink_tokens=sink_tokens,
                update_cache=update_cache,
            )
            cache.cross_attention[idx] = cross_cache

        hidden_states = self.head(hidden_states, temb, tokens_per_frame=tokens_per_frame)

        return self._unpatchify(
            hidden_states,
            batch_size=batch_size,
            frames=patched_frames,
            height=patched_height,
            width=patched_width,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        qkv_fusion = (("q", "q"), ("k", "k"), ("v", "v"))

        for checkpoint_name, loaded_weight in weights:
            name = checkpoint_name
            for source_prefix in (
                "generator.model._fsdp_wrapped_module.",
                "generator.model.",
                "model._fsdp_wrapped_module.",
                "model.",
                "_fsdp_wrapped_module.",
                "module.",
            ):
                if name.startswith(source_prefix):
                    name = name[len(source_prefix) :]
                    break

            # Fuse self-attention Q/K/V → to_qkv
            shard_id = None
            for proj_name, proj_shard in qkv_fusion:
                marker = f".self_attn.{proj_name}."
                if marker in name:
                    name = name.replace(marker, ".self_attn.qkv.")
                    shard_id = proj_shard
                    break

            # The checkpoint may have an empty "" prefix key for root params
            # e.g. "patch_embedding.weight" matches model.patch_embedding.weight
            if name not in params:
                # Try without empty prefix if it has one
                stripped = name.lstrip(".")
                if stripped in params:
                    name = stripped

            if name not in params:
                raise KeyError(f"Unexpected ABot model weight name: {checkpoint_name}")

            param = params[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            if shard_id is None:
                weight_loader(param, loaded_weight)
            else:
                if weight_loader is default_weight_loader:
                    raise RuntimeError(f"Fused QKV parameter {name} has no stacked weight loader.")
                weight_loader(param, loaded_weight, shard_id)
            loaded.add(checkpoint_name)
            loaded.add(name)
        return loaded
