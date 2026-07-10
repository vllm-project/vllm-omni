# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
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
    key: torch.Tensor
    value: torch.Tensor
    end: int = 0
    absolute_end: int = 0
    last_start: int | None = None
    sink_end: int = 0


@dataclass
class LingBotTransformerCache:
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


def _validate_cache_storage(
    cache: LingBotAttentionCache,
    *,
    batch_size: int,
    num_local_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    cache_name: str,
) -> int:
    if cache.key.ndim != 4 or cache.value.ndim != 4:
        raise ValueError(f"{cache_name} cache key and value storage must both be rank 4.")
    if cache.key.shape != cache.value.shape:
        raise ValueError(f"{cache_name} cache key and value tensors must have identical shapes.")

    expected_shape = (batch_size, num_local_heads, head_dim)
    actual_shape = (cache.key.shape[0], cache.key.shape[2], cache.key.shape[3])
    if actual_shape != expected_shape:
        raise ValueError(
            f"{cache_name} cache batch/head shape {actual_shape} does not match attention shape {expected_shape}."
        )
    if cache.key.device != cache.value.device or cache.key.device != device:
        raise ValueError(f"{cache_name} cache key/value device must match the attention input device.")
    if cache.key.dtype != cache.value.dtype or cache.key.dtype != dtype:
        raise ValueError(f"{cache_name} cache key/value dtype must match the attention input dtype.")

    capacity = cache.key.shape[1]
    if capacity <= 0:
        raise ValueError(f"{cache_name} cache capacity must be positive.")
    if cache.key.untyped_storage().data_ptr() == cache.value.untyped_storage().data_ptr():
        raise ValueError(f"{cache_name} cache key and value must not share backing storage.")
    if (
        cache.key.requires_grad
        or cache.value.requires_grad
        or cache.key.grad_fn is not None
        or cache.value.grad_fn is not None
    ):
        raise ValueError(f"{cache_name} cache key and value must be detached from autograd.")
    return capacity


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

    def _validate_cache(self, cache: LingBotAttentionCache, hidden_states: torch.Tensor) -> None:
        capacity = _validate_cache_storage(
            cache,
            batch_size=hidden_states.shape[0],
            num_local_heads=self.num_local_heads,
            head_dim=self.head_dim,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
            cache_name="Self-attention",
        )
        _validate_metadata_integer("end", cache.end)
        _validate_metadata_integer("absolute_end", cache.absolute_end)
        _validate_metadata_integer("last_start", cache.last_start, allow_none=True)
        _validate_metadata_integer("sink_end", cache.sink_end)

        if not 0 <= cache.end <= capacity:
            raise ValueError(f"Self-attention cache end={cache.end} must be within [0, {capacity}].")
        if not 0 <= cache.sink_end <= cache.end:
            raise ValueError("Self-attention cache sink_end must be within the retained cache span.")
        if cache.sink_end > self.sink_tokens:
            raise ValueError("Self-attention cache sink_end exceeds the configured sink token count.")
        if cache.absolute_end < 0:
            raise ValueError("Self-attention cache absolute_end must be non-negative.")

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
            next_sink_end = max(0, min(chunk_tokens, self.sink_tokens - current_start))
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

            incoming_sink_tokens = max(0, min(chunk_tokens, self.sink_tokens - current_start))
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
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("Self-attention hidden_states must be rank 3 [batch, tokens, dim].")
        _validate_current_start(current_start)
        self._validate_cache(cache, hidden_states)

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

        visible_key, visible_value = self._update_cache(cache, key, value, current_start)
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

    def _validate_cache(self, cache: LingBotAttentionCache, hidden_states: torch.Tensor) -> None:
        capacity = _validate_cache_storage(
            cache,
            batch_size=hidden_states.shape[0],
            num_local_heads=self.num_local_heads,
            head_dim=self.head_dim,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
            cache_name="Cross-attention",
        )
        _validate_metadata_integer("end", cache.end)
        _validate_metadata_integer("absolute_end", cache.absolute_end)
        _validate_metadata_integer("last_start", cache.last_start, allow_none=True)
        _validate_metadata_integer("sink_end", cache.sink_end)

        if not 1 <= cache.end <= capacity:
            raise ValueError(f"Cross-attention cache end={cache.end} must be within [1, {capacity}].")
        if cache.absolute_end != cache.end:
            raise ValueError("Cross-attention cache absolute_end must equal end.")
        if cache.last_start != 0:
            raise ValueError("Cross-attention cache last_start must be zero.")
        if cache.sink_end != 0:
            raise ValueError("Cross-attention cache cannot contain sink tokens.")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        *,
        cache: LingBotAttentionCache | None,
    ) -> tuple[torch.Tensor, LingBotAttentionCache]:
        if hidden_states.ndim != 3:
            raise ValueError("Cross-attention hidden_states must be rank 3 [batch, tokens, dim].")
        if cache is not None:
            self._validate_cache(cache, hidden_states)

        query = self.norm_q(self.q(hidden_states))
        query = query.unflatten(2, (self.num_local_heads, self.head_dim))

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
    """Checkpoint namespace container for LingBot self- and cross-attention."""

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
