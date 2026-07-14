# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint-compatible causal DiT used by LingBot World v2."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Self

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.utils import set_weight_attrs

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.layers.rope import RotaryEmbeddingWan


@dataclass
class LingBotAttentionCache:
    """K/V storage plus its logical cursor metadata.

    ``end`` is occupied storage, ``absolute_end`` is the next global token
    position, and ``sink_end`` separates permanently retained prefix tokens
    from the sliding local window. ``last_start`` rejects overlapping or
    out-of-order causal chunks.
    """

    key: torch.Tensor
    value: torch.Tensor
    end: int = 0
    absolute_end: int = 0
    last_start: int | None = None
    sink_end: int = 0


@dataclass
class LingBotTransformerCache:
    """One request's per-layer video K/V and reusable text K/V."""

    self_attention: list[LingBotAttentionCache]
    cross_attention: list[LingBotAttentionCache | None]


def allocate_lingbot_cache(
    *,
    batch_size: int,
    num_layers: int,
    max_tokens: int,
    num_local_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> LingBotTransformerCache:
    # Cross-attention starts empty because its token count is known after text encoding.
    shape = (batch_size, max_tokens, num_local_heads, head_dim)
    self_attention = [
        LingBotAttentionCache(
            key=torch.zeros(shape, device=device, dtype=dtype),
            value=torch.zeros(shape, device=device, dtype=dtype),
        )
        for _ in range(num_layers)
    ]
    return LingBotTransformerCache(
        self_attention=self_attention,
        cross_attention=[None for _ in range(num_layers)],
    )


class _LingBotRMSNorm(nn.Module):
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


def _projection_prefix(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _validate_metadata_integer(name: str, value: int | None, *, allow_none: bool = False) -> None:
    if allow_none and value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Cache metadata {name} must be an integer{' or None' if allow_none else ''}.")


def _validate_current_start(current_start: int) -> None:
    if not isinstance(current_start, int) or isinstance(current_start, bool):
        raise ValueError("current_start must be a non-boolean integer.")
    if current_start < 0:
        raise ValueError(f"current_start must be non-negative, got {current_start}.")


def _select_rotary_chunk(
    rotary_emb: tuple[torch.Tensor, torch.Tensor],
    *,
    current_start: int,
    chunk_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    def select(table: torch.Tensor) -> torch.Tensor:
        current_token_count = table.numel() // table.shape[-1]
        if current_token_count == chunk_tokens:
            return table
        if table.ndim == 2 and table.shape[0] >= current_start + chunk_tokens:
            return table.narrow(0, current_start, chunk_tokens)
        if table.ndim == 3 and table.shape[0] == 1 and table.shape[1] >= current_start + chunk_tokens:
            return table.narrow(1, current_start, chunk_tokens)
        raise ValueError(
            "Rotary embeddings must describe the current chunk or provide a flat table covering its token offset."
        )

    cos, sin = rotary_emb
    return select(cos), select(sin)


class LingBotSelfAttention(nn.Module):
    """Block-causal self-attention over retained history and one full chunk."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        sink_tokens: int = 0,
        eps: float = 1e-6,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")
        if sink_tokens < 0:
            raise ValueError(f"sink_tokens must be non-negative, got {sink_tokens}.")

        tp_size = get_tensor_model_parallel_world_size()
        if num_heads % tp_size != 0:
            raise ValueError(f"num_heads={num_heads} must be divisible by tp_size={tp_size}.")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_local_heads = num_heads // tp_size
        self.tp_inner_dim = self.num_local_heads * self.head_dim
        self.sink_tokens = sink_tokens

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
        self.norm_q = _LingBotRMSNorm(self.tp_inner_dim, eps)
        self.norm_k = _LingBotRMSNorm(self.tp_inner_dim, eps)
        self.rotary_embedding = RotaryEmbeddingWan(is_neox_style=False, half_head_dim=True)
        self.attn = Attention(
            num_heads=self.num_local_heads,
            head_size=self.head_dim,
            num_kv_heads=self.num_local_heads,
            softmax_scale=self.head_dim**-0.5,
            causal=False,
            role="self",
            qkv_layout="BSND",
            prefix=prefix,
            skip_sequence_parallel=True,
        )

    def _validate_cache(
        self,
        cache: LingBotAttentionCache,
        *,
        sink_tokens: int,
    ) -> None:
        capacity = cache.key.shape[1]
        _validate_metadata_integer("end", cache.end)
        _validate_metadata_integer("absolute_end", cache.absolute_end)
        _validate_metadata_integer("last_start", cache.last_start, allow_none=True)
        _validate_metadata_integer("sink_end", cache.sink_end)

        if not 0 <= cache.end <= capacity:
            raise ValueError(f"Self-attention cache end={cache.end} must be within [0, {capacity}].")
        if not 0 <= cache.sink_end <= cache.end:
            raise ValueError("Self-attention cache sink_end must be within the retained cache span.")
        if cache.sink_end > sink_tokens:
            raise ValueError("Self-attention cache sink_end exceeds the configured sink token count.")
        if cache.absolute_end < 0:
            raise ValueError("Self-attention cache absolute_end must be non-negative.")

        # Cache layout: [permanent sink prefix | newest local-window tokens].
        if cache.last_start is None:
            if cache.end != 0 or cache.absolute_end != 0 or cache.sink_end != 0:
                raise ValueError("An uninitialized self-attention cache must have zeroed metadata.")
            return

        if cache.last_start < 0:
            raise ValueError("Self-attention cache last_start must be non-negative.")
        if cache.absolute_end <= cache.last_start:
            raise ValueError("Self-attention cache absolute_end must follow last_start.")
        if cache.end == 0 or cache.end > cache.absolute_end:
            raise ValueError("Initialized self-attention cache physical end is inconsistent with its absolute end.")
        latest_chunk_tokens = cache.absolute_end - cache.last_start
        if latest_chunk_tokens > cache.end:
            raise ValueError("Self-attention cache no longer retains the complete latest chunk.")

    def _update_cache(
        self,
        cache: LingBotAttentionCache,
        key: torch.Tensor,
        value: torch.Tensor,
        current_start: int,
        *,
        sink_tokens: int,
        update_cache: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_current_start(current_start)
        if key.shape[1] == 0:
            raise ValueError("The current attention chunk must contain at least one token.")

        chunk_tokens = key.shape[1]
        capacity = cache.key.shape[1]
        previous_end = cache.end
        next_sink_end = cache.sink_end

        if cache.last_start is None:
            if chunk_tokens > capacity:
                raise ValueError(
                    f"Current chunk has {chunk_tokens} tokens but the cache can hold only {capacity}; "
                    "the full current chunk must remain visible."
                )
            next_key = key
            next_value = value
            next_sink_end = max(0, min(chunk_tokens, sink_tokens - current_start))
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
                raise ValueError(f"current_start={current_start} precedes the latest chunk start {cache.last_start}.")
            if current_start < cache.absolute_end:
                raise ValueError(
                    f"current_start={current_start} overlaps cached tokens ending at {cache.absolute_end}."
                )
            if current_start > cache.absolute_end:
                raise ValueError(
                    f"New chunks must be contiguous: current_start={current_start}, expected {cache.absolute_end}."
                )

            incoming_sink_tokens = max(0, min(chunk_tokens, sink_tokens - current_start))
            old_sink_key = cache.key[:, : cache.sink_end]
            old_sink_value = cache.value[:, : cache.sink_end]
            new_sink_key = key[:, :incoming_sink_tokens]
            new_sink_value = value[:, :incoming_sink_tokens]
            next_sink_end = cache.sink_end + incoming_sink_tokens

            old_local_key = cache.key[:, cache.sink_end : cache.end]
            old_local_value = cache.value[:, cache.sink_end : cache.end]
            new_local_key = key[:, incoming_sink_tokens:]
            new_local_value = value[:, incoming_sink_tokens:]
            local_capacity = capacity - next_sink_end
            if new_local_key.shape[1] > local_capacity:
                raise ValueError("The configured cache cannot retain all sink tokens and the full current chunk.")
            retained_local_tokens = min(
                old_local_key.shape[1],
                local_capacity - new_local_key.shape[1],
            )
            if retained_local_tokens:
                old_local_key = old_local_key[:, -retained_local_tokens:]
                old_local_value = old_local_value[:, -retained_local_tokens:]
            else:
                old_local_key = old_local_key[:, :0]
                old_local_value = old_local_value[:, :0]

            next_key = torch.cat((old_sink_key, new_sink_key, old_local_key, new_local_key), dim=1)
            next_value = torch.cat((old_sink_value, new_sink_value, old_local_value, new_local_value), dim=1)

        next_end = next_key.shape[1]
        if update_cache:
            with torch.no_grad():
                cache.key[:, :next_end].copy_(next_key.detach())
                cache.value[:, :next_end].copy_(next_value.detach())
                if next_end < previous_end:
                    cache.key[:, next_end:previous_end].zero_()
                    cache.value[:, next_end:previous_end].zero_()
            cache.end = next_end
            cache.absolute_end = current_start + chunk_tokens
            cache.last_start = current_start
            cache.sink_end = next_sink_end
        return next_key, next_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cache: LingBotAttentionCache,
        current_start: int,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        sink_tokens: int | None = None,
        update_cache: bool = True,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("Self-attention hidden_states must be rank 3 [batch, tokens, dim].")
        _validate_current_start(current_start)
        if sink_tokens is None:
            sink_tokens = self.sink_tokens
        if not isinstance(sink_tokens, int) or isinstance(sink_tokens, bool) or sink_tokens < 0:
            raise ValueError("sink_tokens must be a non-negative integer.")
        if not isinstance(update_cache, bool):
            raise ValueError("update_cache must be a boolean.")
        self._validate_cache(cache, sink_tokens=sink_tokens)

        # Project logical [B, S, D] tokens into TP-local
        # [B, S, num_local_heads, head_dim].
        query = self.norm_q(self.q(hidden_states))
        key = self.norm_k(self.k(hidden_states))
        value = self.v(hidden_states)

        query = query.unflatten(2, (self.num_local_heads, self.head_dim))
        key = key.unflatten(2, (self.num_local_heads, self.head_dim))
        value = value.unflatten(2, (self.num_local_heads, self.head_dim))
        if rotary_emb is not None:
            cos, sin = _select_rotary_chunk(
                rotary_emb,
                current_start=current_start,
                chunk_tokens=hidden_states.shape[1],
            )
            query = self.rotary_embedding(query, cos, sin)
            key = self.rotary_embedding(key, cos, sin)

        # The attention kernel sees history + current K/V even when the caller
        # asks not to mutate persistent cache state.
        visible_key, visible_value = self._update_cache(
            cache,
            key,
            value,
            current_start,
            sink_tokens=sink_tokens,
            update_cache=update_cache,
        )
        output = self.attn(query, visible_key, visible_value)
        return self.o(output.flatten(2, 3))


class LingBotCrossAttention(nn.Module):
    """Cross-attention with caller-owned request-local encoder K/V reuse."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        eps: float = 1e-6,
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
        self.head_dim = dim // num_heads
        self.num_local_heads = num_heads // tp_size
        self.tp_inner_dim = self.num_local_heads * self.head_dim

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
        self.norm_q = _LingBotRMSNorm(self.tp_inner_dim, eps)
        self.norm_k = _LingBotRMSNorm(self.tp_inner_dim, eps)
        self.attn = Attention(
            num_heads=self.num_local_heads,
            head_size=self.head_dim,
            num_kv_heads=self.num_local_heads,
            softmax_scale=self.head_dim**-0.5,
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
        cache: LingBotAttentionCache | None,
    ) -> tuple[torch.Tensor, LingBotAttentionCache]:
        if hidden_states.ndim != 3:
            raise ValueError("Cross-attention hidden_states must be rank 3 [batch, tokens, dim].")
        query = self.norm_q(self.q(hidden_states))
        query = query.unflatten(2, (self.num_local_heads, self.head_dim))

        # Text K/V is constant within a request and is projected once per layer.
        if cache is None:
            if encoder_hidden_states is None:
                raise ValueError("encoder_hidden_states are required when the cross-attention cache is empty.")
            if encoder_hidden_states.ndim != 3:
                raise ValueError("encoder_hidden_states must be rank 3 [batch, tokens, dim].")
            if encoder_hidden_states.shape[0] != hidden_states.shape[0]:
                raise ValueError("encoder_hidden_states batch size must match hidden_states.")
            if encoder_hidden_states.shape[1] == 0:
                raise ValueError("encoder_hidden_states must contain at least one token.")
            if encoder_hidden_states.shape[2] != self.dim:
                raise ValueError(f"encoder_hidden_states width must equal dim={self.dim}.")
            if encoder_hidden_states.device != hidden_states.device:
                raise ValueError("encoder_hidden_states device must match hidden_states.")
            if encoder_hidden_states.dtype != hidden_states.dtype:
                raise ValueError("encoder_hidden_states dtype must match hidden_states.")
            key = self.norm_k(self.k(encoder_hidden_states))
            value = self.v(encoder_hidden_states)
            key = key.unflatten(2, (self.num_local_heads, self.head_dim))
            value = value.unflatten(2, (self.num_local_heads, self.head_dim))
            cache = LingBotAttentionCache(
                # The request cache owns its storage independently of encoder activations.
                key=key.detach().clone(),
                value=value.detach().clone(),
                end=key.shape[1],
                absolute_end=key.shape[1],
                last_start=0,
            )
        else:
            key = cache.key[:, : cache.end]
            value = cache.value[:, : cache.end]

        output = self.attn(query, key, value)
        return self.o(output.flatten(2, 3)), cache


class LingBotAttentionBlock(nn.Module):
    """Checkpoint-compatible LingBot block with causal video attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        ffn_dim: int | None = None,
        sink_tokens: int = 0,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        prefix: str = "",
    ) -> None:
        super().__init__()
        ffn_dim = ffn_dim or dim * 4
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.self_attn = LingBotSelfAttention(
            dim,
            num_heads,
            sink_tokens=sink_tokens,
            eps=eps,
            prefix=_projection_prefix(prefix, "self_attn"),
        )
        self.cross_attn = LingBotCrossAttention(
            dim,
            num_heads,
            eps=eps,
            prefix=_projection_prefix(prefix, "cross_attn"),
        )
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps) if cross_attn_norm else nn.Identity()
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
        self.cam_injector_layer1 = ColumnParallelLinear(
            dim,
            dim,
            bias=True,
            gather_output=False,
            return_bias=False,
            prefix=_projection_prefix(prefix, "cam_injector_layer1"),
        )
        self.cam_injector_layer2 = RowParallelLinear(
            dim,
            dim,
            bias=True,
            input_is_parallel=True,
            return_bias=False,
            prefix=_projection_prefix(prefix, "cam_injector_layer2"),
        )
        self.cam_scale_layer = nn.Linear(dim, dim)
        self.cam_shift_layer = nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        timestep_projection: torch.Tensor,
        camera_hidden_states: torch.Tensor,
        *,
        self_cache: LingBotAttentionCache,
        cross_cache: LingBotAttentionCache | None,
        current_start: int,
        sink_tokens: int,
        update_cache: bool,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, LingBotAttentionCache]:
        batch_size, token_count, dim = hidden_states.shape
        num_frames = timestep_projection.shape[1]
        if token_count % num_frames != 0:
            raise ValueError("The patched token count must be divisible by the timestep frame count.")
        tokens_per_frame = token_count // num_frames
        # Timestep, camera, and text remain separate conditioning paths.
        modulation = self.modulation.unsqueeze(1) + timestep_projection.float()
        shift_msa, scale_msa, gate_msa, shift_ffn, scale_ffn, gate_ffn = modulation.chunk(6, dim=2)

        hidden_grid = hidden_states.unflatten(1, (num_frames, tokens_per_frame))
        normalized = self.norm1(hidden_states.float()).unflatten(1, (num_frames, tokens_per_frame))
        normalized = (normalized * (1 + scale_msa) + shift_msa).flatten(1, 2).to(hidden_states.dtype)
        attention_output = self.self_attn(
            normalized,
            cache=self_cache,
            current_start=current_start,
            rotary_emb=rotary_emb,
            sink_tokens=sink_tokens,
            update_cache=update_cache,
        )
        hidden_grid = hidden_grid + attention_output.unflatten(1, (num_frames, tokens_per_frame)) * gate_msa
        hidden_states = hidden_grid.flatten(1, 2).to(hidden_states.dtype)

        camera_features = self.cam_injector_layer2(F.silu(self.cam_injector_layer1(camera_hidden_states)))
        camera_features = camera_features + camera_hidden_states
        camera_scale = self.cam_scale_layer(camera_features)
        camera_shift = self.cam_shift_layer(camera_features)
        hidden_states = ((1 + camera_scale) * hidden_states + camera_shift).to(hidden_states.dtype)

        attention_output, cross_cache = self.cross_attn(
            self.norm3(hidden_states),
            encoder_hidden_states,
            cache=cross_cache,
        )
        hidden_states = hidden_states + attention_output

        hidden_grid = hidden_states.unflatten(1, (num_frames, tokens_per_frame))
        normalized = self.norm2(hidden_states.float()).unflatten(1, (num_frames, tokens_per_frame))
        normalized = (normalized * (1 + scale_ffn) + shift_ffn).flatten(1, 2).to(hidden_states.dtype)
        ffn_output = self.ffn(normalized).unflatten(1, (num_frames, tokens_per_frame))
        hidden_states = (hidden_grid + ffn_output * gate_ffn).flatten(1, 2).to(hidden_states.dtype)
        if cross_cache is None:
            raise RuntimeError("Cross-attention must return a populated request cache.")
        return hidden_states, cross_cache


class _LingBotCameraPatchEmbedding(nn.Module):
    """Linear camera patchifier matching the checkpoint's direct parameter names."""

    def __init__(
        self,
        in_channels: int,
        dim: int,
        patch_size: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_features = in_channels * math.prod(patch_size)
        self.weight = nn.Parameter(torch.empty(dim, self.in_features))
        self.bias = nn.Parameter(torch.empty(dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 5:
            raise ValueError("camera_hidden_states must be rank 5 [batch, channels, frames, height, width].")
        batch_size, channels, frames, height, width = hidden_states.shape
        patch_frames, patch_height, patch_width = self.patch_size
        if frames % patch_frames or height % patch_height or width % patch_width:
            raise ValueError("camera_hidden_states dimensions must be divisible by patch_size.")
        # Preserve Conv3d patch order while retaining checkpoint weight/bias names.
        hidden_states = hidden_states.reshape(
            batch_size,
            channels,
            frames // patch_frames,
            patch_frames,
            height // patch_height,
            patch_height,
            width // patch_width,
            patch_width,
        )
        hidden_states = hidden_states.permute(0, 2, 4, 6, 1, 3, 5, 7)
        hidden_states = hidden_states.reshape(batch_size, -1, self.in_features)
        return F.linear(hidden_states, self.weight, self.bias)


class _LingBotHead(nn.Module):
    def __init__(
        self,
        dim: int,
        out_channels: int,
        patch_size: tuple[int, int, int],
        eps: float,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_channels * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / math.sqrt(dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_embedding: torch.Tensor,
    ) -> torch.Tensor:
        num_frames = timestep_embedding.shape[1]
        tokens_per_frame = hidden_states.shape[1] // num_frames
        modulation = self.modulation.unsqueeze(1) + timestep_embedding.unsqueeze(2).float()
        shift, scale = modulation.chunk(2, dim=2)
        normalized = self.norm(hidden_states.float()).unflatten(1, (num_frames, tokens_per_frame))
        normalized = (normalized * (1 + scale) + shift).flatten(1, 2).to(hidden_states.dtype)
        return self.head(normalized)


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


class CausalLingBotWorldTransformer3DModel(nn.Module):
    """Checkpoint-compatible causal LingBot World video transformer."""

    _layerwise_offload_blocks_attrs = ["blocks"]

    def __init__(
        self,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 36,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        image_dim: int | None = None,
        added_kv_proj_dim: int | None = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: int | None = None,
        qk_norm: str = "rms_norm_across_heads",
        sink_size: int = 9,
        num_frames_per_block: int = 3,
        sliding_window_num_frames: int = 18,
        local_attn_size: int = -1,
        quant_config: object | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if len(patch_size) != 3 or any(size <= 0 for size in patch_size):
            raise ValueError("patch_size must contain three positive integers.")
        if num_attention_heads <= 0 or attention_head_dim <= 0:
            raise ValueError("Attention head counts and dimensions must be positive.")
        if attention_head_dim % 2:
            raise ValueError("attention_head_dim must be even for RoPE.")
        if num_layers <= 0 or ffn_dim <= 0 or freq_dim <= 0:
            raise ValueError("num_layers, ffn_dim, and freq_dim must be positive.")
        if sink_size < 0 or num_frames_per_block <= 0 or sliding_window_num_frames <= 0:
            raise ValueError("Causal frame-window sizes must be positive, except sink_size which may be zero.")
        if local_attn_size != -1 and local_attn_size <= 0:
            raise ValueError("local_attn_size must be -1 or a positive frame count.")
        cache_window_frames = local_attn_size if local_attn_size != -1 else sliding_window_num_frames
        if cache_window_frames < sink_size + num_frames_per_block:
            raise ValueError("The causal cache window must fit all sink frames and one complete current block.")
        if qk_norm != "rms_norm_across_heads":
            raise ValueError(
                f"qk_norm must be 'rms_norm_across_heads' for the LingBot World v2 checkpoint, got {qk_norm!r}."
            )
        for field_name, field_value in (
            ("image_dim", image_dim),
            ("added_kv_proj_dim", added_kv_proj_dim),
            ("pos_embed_seq_len", pos_embed_seq_len),
        ):
            if field_value is not None:
                raise ValueError(f"{field_name} must be None because LingBot World v2 has no image embedding path.")
        if quant_config is not None:
            raise RuntimeError(
                "quant_config is not supported by the LingBot World transformer; construct the unquantized model."
            )

        dim = num_attention_heads * attention_head_dim
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
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            image_dim=image_dim,
            added_kv_proj_dim=added_kv_proj_dim,
            rope_max_seq_len=rope_max_seq_len,
            pos_embed_seq_len=pos_embed_seq_len,
            qk_norm=qk_norm,
            sink_size=sink_size,
            num_frames_per_block=num_frames_per_block,
            sliding_window_num_frames=sliding_window_num_frames,
            local_attn_size=local_attn_size,
        )

        self.patch_embedding = nn.Conv3d(
            in_channels,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.patch_embedding_wancamctrl = _LingBotCameraPatchEmbedding(6 * 8 * 8, dim, patch_size)
        self.c2ws_hidden_states_layer1 = ColumnParallelLinear(
            dim,
            dim,
            bias=True,
            gather_output=False,
            return_bias=False,
            prefix=_projection_prefix(prefix, "c2ws_hidden_states_layer1"),
        )
        self.c2ws_hidden_states_layer2 = RowParallelLinear(
            dim,
            dim,
            bias=True,
            input_is_parallel=True,
            return_bias=False,
            prefix=_projection_prefix(prefix, "c2ws_hidden_states_layer2"),
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6),
        )
        self.blocks = nn.ModuleList(
            [
                LingBotAttentionBlock(
                    dim,
                    num_attention_heads,
                    ffn_dim=ffn_dim,
                    sink_tokens=0,
                    cross_attn_norm=cross_attn_norm,
                    eps=eps,
                    prefix=_projection_prefix(prefix, f"blocks.{index}"),
                )
                for index in range(num_layers)
            ]
        )
        self.head = _LingBotHead(dim, out_channels, patch_size, eps)

        temporal_dim = attention_head_dim - 4 * (attention_head_dim // 6)
        height_dim = width_dim = 2 * (attention_head_dim // 6)
        for axis, axis_dim in (("temporal", temporal_dim), ("height", height_dim), ("width", width_dim)):
            cosine, sine = _rope_axis(rope_max_seq_len, axis_dim)
            self.register_buffer(f"_rope_{axis}_cosine", cosine, persistent=False)
            self.register_buffer(f"_rope_{axis}_sine", sine, persistent=False)

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype used by the transformer parameters."""

        return next(self.parameters()).dtype

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        quant_config: Any | None = None,
        prefix: str = "",
    ) -> Self:
        metadata_keys = {"_class_name", "_diffusers_version"}
        constructor_keys = {
            "patch_size",
            "num_attention_heads",
            "attention_head_dim",
            "in_channels",
            "out_channels",
            "text_dim",
            "freq_dim",
            "ffn_dim",
            "num_layers",
            "cross_attn_norm",
            "eps",
            "image_dim",
            "added_kv_proj_dim",
            "rope_max_seq_len",
            "pos_embed_seq_len",
            "qk_norm",
            "sink_size",
            "num_frames_per_block",
            "sliding_window_num_frames",
            "local_attn_size",
        }
        unexpected = set(config) - constructor_keys - metadata_keys
        if unexpected:
            raise ValueError(f"Unexpected LingBot config keys: {sorted(unexpected)}")
        missing = constructor_keys - set(config)
        if missing:
            raise ValueError(f"Missing LingBot checkpoint config keys: {sorted(missing)}")
        if config.get("_class_name") != "CausalLingBotWorldTransformer3DModel":
            raise ValueError(
                "_class_name must be 'CausalLingBotWorldTransformer3DModel' for the LingBot World v2 checkpoint."
            )
        kwargs = {name: value for name, value in config.items() if name in constructor_keys}
        if "patch_size" in kwargs:
            kwargs["patch_size"] = tuple(kwargs["patch_size"])
        checkpoint_contract = {
            "patch_size": (1, 2, 2),
            "num_attention_heads": 40,
            "attention_head_dim": 128,
            "in_channels": 36,
            "out_channels": 16,
            "text_dim": 4096,
            "freq_dim": 256,
            "ffn_dim": 13824,
            "num_layers": 40,
            "cross_attn_norm": True,
            "eps": 1e-6,
            "image_dim": None,
            "added_kv_proj_dim": None,
            "rope_max_seq_len": 1024,
            "pos_embed_seq_len": None,
            "qk_norm": "rms_norm_across_heads",
            "sink_size": 9,
            "num_frames_per_block": 3,
            "sliding_window_num_frames": 18,
            "local_attn_size": -1,
        }
        for name, expected in checkpoint_contract.items():
            if kwargs[name] != expected:
                raise ValueError(f"LingBot checkpoint config {name} must be {expected!r}, got {kwargs[name]!r}.")
        return cls(**kwargs, quant_config=quant_config, prefix=prefix)

    def allocate_cache(
        self,
        *,
        batch_size: int,
        latent_height: int,
        latent_width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> LingBotTransformerCache:
        """Allocate one caller-owned cache using this transformer's geometry."""

        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer for LingBot cache allocation.")
        if (
            isinstance(latent_height, bool)
            or not isinstance(latent_height, int)
            or latent_height <= 0
            or isinstance(latent_width, bool)
            or not isinstance(latent_width, int)
            or latent_width <= 0
        ):
            raise ValueError("latent_height and latent_width must be positive integers.")

        # Cache capacity is measured in post-patch tokens, not raw/latent
        # pixels: window_frames * (latent_height/patch_h) * (latent_width/patch_w).
        patch_frames, patch_height, patch_width = self.config.patch_size
        if patch_frames != 1 or latent_height % patch_height or latent_width % patch_width:
            raise ValueError("latent height/width must align with the configured LingBot patch size.")
        post_patch_height = latent_height // patch_height
        post_patch_width = latent_width // patch_width
        window_frames = (
            self.config.local_attn_size if self.config.local_attn_size != -1 else self.config.sliding_window_num_frames
        )
        max_tokens = int(window_frames * post_patch_height * post_patch_width)
        tp_size = get_tensor_model_parallel_world_size()
        num_local_heads = self.config.num_attention_heads // tp_size
        return allocate_lingbot_cache(
            batch_size=batch_size,
            num_layers=self.config.num_layers,
            max_tokens=max_tokens,
            num_local_heads=num_local_heads,
            head_dim=self.config.attention_head_dim,
            device=device,
            dtype=dtype,
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
        if height > self.config.rope_max_seq_len or width > self.config.rope_max_seq_len:
            raise ValueError("Spatial RoPE positions exceed rope_max_seq_len.")

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

    def _validate_inputs(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        camera_hidden_states: torch.Tensor,
        cache: LingBotTransformerCache,
        start_frame: int,
    ) -> tuple[int, int, int, int, int]:
        if hidden_states.ndim != 5:
            raise ValueError("hidden_states must be rank 5 [batch, channels, frames, height, width].")
        if hidden_states.shape[1] != self.config.in_channels:
            raise ValueError(f"hidden_states must have {self.config.in_channels} channels.")
        if not isinstance(start_frame, int) or isinstance(start_frame, bool) or start_frame < 0:
            raise ValueError("start_frame must be a non-negative integer latent-frame offset.")
        patch_frames, patch_height, patch_width = self.config.patch_size
        batch_size, _, frames, height, width = hidden_states.shape
        if frames % patch_frames or height % patch_height or width % patch_width:
            raise ValueError("hidden_states dimensions must be divisible by patch_size.")
        post_patch_frames = frames // patch_frames
        if post_patch_frames != self.config.num_frames_per_block:
            raise ValueError(
                f"Each LingBot forward must contain exactly {self.config.num_frames_per_block} post-patch frames, "
                f"got {post_patch_frames}."
            )
        if start_frame % patch_frames:
            raise ValueError("start_frame must align to the temporal patch size.")
        if encoder_hidden_states.ndim != 3:
            raise ValueError("encoder_hidden_states must be rank 3 [batch, tokens, text_dim].")
        if encoder_hidden_states.shape[0] != batch_size or encoder_hidden_states.shape[2] != self.config.text_dim:
            raise ValueError("encoder_hidden_states batch/width do not match the LingBot config.")
        if encoder_hidden_states.shape[1] == 0:
            raise ValueError("encoder_hidden_states must contain at least one token.")
        expected_camera_shape = (batch_size, 6 * 8 * 8, frames, height, width)
        if camera_hidden_states.shape != expected_camera_shape:
            raise ValueError(
                "camera_hidden_states must use folded [batch, 6*8*8, frames, height, width] layout "
                f"matching the video; expected {expected_camera_shape}, got {tuple(camera_hidden_states.shape)}."
            )
        if hidden_states.device != encoder_hidden_states.device or hidden_states.device != camera_hidden_states.device:
            raise ValueError("Video, text, and camera tensors must be on the same device.")
        if hidden_states.dtype != encoder_hidden_states.dtype or hidden_states.dtype != camera_hidden_states.dtype:
            raise ValueError("Video, text, and camera tensors must use the same dtype.")
        if timestep.device != hidden_states.device:
            raise ValueError("timestep must be on the same device as hidden_states.")
        if len(cache.self_attention) != self.config.num_layers or len(cache.cross_attention) != self.config.num_layers:
            raise ValueError("LingBotTransformerCache layer counts must match num_layers.")
        return batch_size, frames, height, width, patch_frames

    def _timestep_embeddings(
        self,
        timestep: torch.Tensor,
        *,
        batch_size: int,
        frames: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if timestep.ndim == 0:
            if batch_size != 1:
                raise ValueError("A scalar timestep is only valid for batch_size=1.")
            timestep = timestep.reshape(1)
        if timestep.ndim == 1:
            if timestep.shape[0] != batch_size:
                raise ValueError("A rank-1 timestep must contain one value per batch item.")
            timestep = timestep.unsqueeze(1).expand(batch_size, frames)
        elif timestep.ndim == 2:
            if timestep.shape != (batch_size, frames):
                raise ValueError(f"A rank-2 timestep must have shape {(batch_size, frames)}.")
        else:
            raise ValueError("timestep must be scalar, rank 1 [batch], or rank 2 [batch, frames].")

        frequency_embedding = _sinusoidal_embedding(self.config.freq_dim, timestep.reshape(-1)).to(dtype=dtype)
        timestep_embedding = self.time_embedding(frequency_embedding).unflatten(0, (batch_size, frames))
        timestep_projection = self.time_projection(timestep_embedding).unflatten(2, (6, self.dim))
        return timestep_embedding, timestep_projection

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
        return hidden_states.reshape(
            batch_size,
            self.config.out_channels,
            frames * patch_frames,
            height * patch_height,
            width * patch_width,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        camera_hidden_states: torch.Tensor,
        *,
        cache: LingBotTransformerCache,
        start_frame: int,
        update_cache: bool,
    ) -> torch.Tensor:
        if not isinstance(update_cache, bool):
            raise ValueError("update_cache must be a boolean.")
        batch_size, frames, height, width, patch_frames = self._validate_inputs(
            hidden_states,
            timestep,
            encoder_hidden_states,
            camera_hidden_states,
            cache,
            start_frame,
        )
        patch_height, patch_width = self.config.patch_size[1:]
        patched_frames = frames // patch_frames
        patched_height = height // patch_height
        patched_width = width // patch_width
        tokens_per_frame = patched_height * patched_width
        patched_start_frame = start_frame // patch_frames
        current_start = patched_start_frame * tokens_per_frame
        sink_tokens = self.config.sink_size * tokens_per_frame
        # Phase 1: create absolute 3D positions for this causal block. The
        # spatial coordinates restart for every frame; temporal coordinates
        # begin at ``start_frame`` so cached blocks retain their true ordering.
        rotary_emb = self._rotary_embedding(
            frames=patched_frames,
            height=patched_height,
            width=patched_width,
            start_frame=patched_start_frame,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        # Phase 2: independently patchify video and camera grids to identical
        # token layouts, then project timestep and text conditions.
        hidden_states = self.patch_embedding(hidden_states).flatten(2).transpose(1, 2)
        camera_hidden_states = self.patch_embedding_wancamctrl(camera_hidden_states)
        camera_hidden_states = camera_hidden_states + self.c2ws_hidden_states_layer2(
            F.silu(self.c2ws_hidden_states_layer1(camera_hidden_states))
        )
        if hidden_states.shape != camera_hidden_states.shape:
            raise ValueError("Patched video and camera token shapes must match.")

        timestep_embedding, timestep_projection = self._timestep_embeddings(
            timestep,
            batch_size=batch_size,
            frames=patched_frames,
            dtype=hidden_states.dtype,
        )
        encoder_hidden_states = self.text_embedding(encoder_hidden_states)
        # Phase 3: each layer receives its own cache entry. Text K/V is passed
        # only when absent; the returned cache is stored for subsequent DMD
        # steps and causal blocks in this request.
        for index, block in enumerate(self.blocks):
            hidden_states, cross_cache = block(
                hidden_states,
                encoder_hidden_states if cache.cross_attention[index] is None else None,
                timestep_projection,
                camera_hidden_states,
                self_cache=cache.self_attention[index],
                cross_cache=cache.cross_attention[index],
                current_start=current_start,
                sink_tokens=sink_tokens,
                update_cache=update_cache,
                rotary_emb=rotary_emb,
            )
            cache.cross_attention[index] = cross_cache

        # Phase 4: map tokens to per-patch 16-channel flow values and restore
        # [B, C, F, H, W] for the Pipeline's sampler update.
        hidden_states = self.head(hidden_states, timestep_embedding)
        return self._unpatchify(
            hidden_states,
            batch_size=batch_size,
            frames=patched_frames,
            height=patched_height,
            width=patched_width,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load exact checkpoint names, delegating TP sharding to parameters."""

        params = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, loaded_weight in weights:
            if name not in params:
                raise KeyError(f"Unexpected LingBot model weight name: {name}")
            param = params[name]
            weight_loader = getattr(param, "weight_loader", None)
            if weight_loader is not None:
                weight_loader(param, loaded_weight)
            else:
                if param.shape != loaded_weight.shape:
                    raise ValueError(
                        f"Weight shape mismatch for {name}: parameter={tuple(param.shape)}, "
                        f"checkpoint={tuple(loaded_weight.shape)}."
                    )
                with torch.no_grad():
                    param.copy_(loaded_weight)
            loaded.add(name)
        return loaded
