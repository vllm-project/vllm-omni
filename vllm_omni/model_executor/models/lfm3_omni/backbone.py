# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LFM3 attention with request-owned convolution state.

The checkpoint's modules define the parameters and normalization semantics.
Reference attention owns KV per request; the optional paged path uses the engine
cache. Request state is released by the model's request-finished hook. Prefix
caching and recompute preemption must be disabled.
"""

from copy import copy
from dataclasses import dataclass, field
from types import ModuleType

import torch
import torch.nn.functional as F
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.attention import Attention


@dataclass
class RequestState:
    conv: dict[int, torch.Tensor] = field(default_factory=dict)
    kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    next_audio_embed: torch.Tensor | None = None
    mode: str = "text"
    text_only: bool = False
    speaking: bool = False
    text_done: bool = False
    slots_left: int = 6
    audio_history: list[torch.Tensor] = field(default_factory=list)
    generator: torch.Generator | None = None
    last_was_audio: bool = False
    audio_temperature: float | None = 0.4
    audio_top_k: int | None = None
    audio_repetition_penalty: float = 1.2
    audio_repetition_window: int = 64


def causal_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    previous: torch.Tensor | None,
    reference: ModuleType | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A single request, including a chunk continuing an existing prompt.

    Keep K-1 inputs; this produces the same causal windows as the reference's
    K-element cache (which discards its oldest element on the next step).
    Where the checkpoint dispatches to its own width-7 kernels (CUDA), use
    them with the reference's K-element cache: their decode rounding depends
    on the cache width, and differs from a prompt's.
    """
    width = weight.shape[-1]
    x = x.transpose(0, 1).unsqueeze(0)
    kernels = reference is not None and reference.can_use_lfm3_conv(x, weight)
    if kernels and previous is not None and x.shape[-1] == 1:
        # Shifts the request-owned state in place, like the reference cache.
        result = reference.lfm3_causal_conv1d_update(x, previous, weight.squeeze(1), bias)
        return result.squeeze(0).transpose(0, 1), previous
    prefix = x.new_zeros((1, x.shape[1], width - 1)) if previous is None else previous
    window = torch.cat((prefix, x), dim=-1)
    if kernels:
        # Zero taps are exact in its fp32 sum, so the padded window matches.
        result = reference.lfm3_causal_conv1d(window, weight.squeeze(1), bias)[..., -x.shape[-1] :]
    else:
        result = F.conv1d(window, weight, bias, groups=x.shape[1])
    # The prefix must own its storage: retaining a view keeps the entire prompt.
    state = window[..., -(width if kernels else width - 1) :].contiguous()
    return result.squeeze(0).transpose(0, 1), state


class Lfm3TemporalModel(nn.Module):
    def __init__(self, vllm_config: VllmConfig, reference: ModuleType, prefix: str):
        super().__init__()
        config = vllm_config.model_config.hf_config.text_config
        self.config = config
        self.max_model_len = vllm_config.model_config.max_model_len
        self.attention_mode = getattr(vllm_config.model_config.hf_config, "lfm3_attention_backend", "reference")
        if self.attention_mode not in ("reference", "paged"):
            raise ValueError("lfm3_attention_backend must be reference or paged")
        self._reference_attention = reference.eager_attention_forward
        self._conv_kernels = reference if hasattr(reference, "can_use_lfm3_conv") else None
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(reference.Lfm3DecoderLayer(config, i) for i in range(config.num_hidden_layers))
        self.embedding_norm = reference.Lfm3RMSNorm(config.hidden_size, eps=config.norm_eps)
        # The reference initializes inverse frequencies on CPU. Computing the
        # powers on the accelerator changes rounding before BF16 decoding.
        with torch.device("cpu"):
            self.rotary_emb = reference.Lfm3RotaryEmbedding(config)
        self.rotary_emb.to(device=self.embed_tokens.weight.device)
        attention_cache_config = copy(vllm_config.cache_config)
        attention_cache_config.sliding_window = None
        for i, kind in enumerate(config.layer_types):
            if kind == "conv":
                continue
            self.layers[i].self_attn.attn = Attention(
                config.num_attention_heads,
                config.head_dim,
                config.head_dim**-0.5,
                num_kv_heads=config.num_key_value_heads,
                cache_config=attention_cache_config,
                per_layer_sliding_window=config.sliding_window if kind == "sliding_attention" else None,
                prefix=f"{prefix}.layers.{i}.self_attn.attn",
            )

    def model_local_kv_specs(self):
        from vllm_omni.model_executor.models.model_local_kv import ModelLocalKVScope, ModelLocalKVSpec, RowDriver

        if self.attention_mode != "reference":
            return []
        specs = []
        for kind, capacity in (
            ("full_attention", self.max_model_len),
            ("sliding_attention", self.config.sliding_window - 1),
        ):
            layers = self.config.layer_types.count(kind)
            if layers:
                specs.append(
                    ModelLocalKVSpec(
                        name=f"lfm3_{kind}",
                        layers=layers,
                        kv_heads=self.config.num_key_value_heads,
                        head_dim=self.config.head_dim,
                        dtype=self.embed_tokens.weight.dtype,
                        physical_capacity_positions=capacity,
                        capacity_source="max_model_len" if kind == "full_attention" else "sliding_window - 1",
                        scope=ModelLocalKVScope.REQUEST,
                        rows=RowDriver.MAX_NUM_SEQS,
                        allocation_note="per-request tensors; attention also temporarily concatenates old and new KV",
                    )
                )
        return specs

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # LFM3 rotates in fp32 and zeroes the *tail frequencies*. Using a
        # smaller rotary dimension changes the frequency progression.
        half = x.shape[-1] // 2
        rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1).float()
        return (x.float() * cos[:, None, :] + rotated * sin[:, None, :]).to(x.dtype)

    def _attention(self, index, attn, q, k, v, positions, requests):
        if self.attention_mode == "paged":
            return attn.attn(q.flatten(1), k.flatten(1), v)
        result = torch.zeros_like(q)
        v = v.view(-1, self.config.num_key_value_heads, self.config.head_dim)
        window = self.config.sliding_window if self.config.layer_types[index] == "sliding_attention" else None
        for start, end, state in requests:
            keys = k[start:end].transpose(0, 1).unsqueeze(0)
            values = v[start:end].transpose(0, 1).unsqueeze(0)
            previous = state.kv.get(index)
            past_length = 0 if previous is None else previous[0].shape[-2]
            if previous is not None:
                keys = torch.cat((previous[0], keys), dim=-2)
                values = torch.cat((previous[1], values), dim=-2)
            if window is None:
                state.kv[index] = (keys, values)
            else:
                state.kv[index] = (
                    keys[..., -(window - 1) :, :].contiguous(),
                    values[..., -(window - 1) :, :].contiguous(),
                )
            query_positions = positions[start:end, None]
            key_positions = torch.arange(keys.shape[-2], device=q.device) + positions[start] - past_length
            allowed = key_positions <= query_positions
            if window is not None:
                allowed &= key_positions > query_positions - window
            mask = torch.zeros(allowed.shape, device=q.device, dtype=q.dtype)
            mask.masked_fill_(~allowed, torch.finfo(q.dtype).min)
            output, _ = self._reference_attention(
                attn,
                q[start:end].transpose(0, 1).unsqueeze(0),
                keys,
                values,
                mask[None, None],
                scaling=attn.scaling,
                dropout=0.0,
            )
            result[start:end] = output[0]
        return result.flatten(1)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        requests: list[tuple[int, int, RequestState]],
    ) -> torch.Tensor:
        x = inputs_embeds
        ropes = {
            kind: self.rotary_emb(x.unsqueeze(0), positions.unsqueeze(0), layer_type=kind)
            for kind in self.rotary_emb.layer_types
        }
        for index, layer in enumerate(self.layers):
            residual = x
            normalized = layer.operator_norm(x)
            if layer.is_attention_layer:
                attn = layer.self_attn
                q = attn.q_norm(attn.q_proj(normalized).view(-1, self.config.num_attention_heads, attn.head_dim))
                k = attn.k_norm(attn.k_proj(normalized).view(-1, self.config.num_key_value_heads, attn.head_dim))
                v = attn.v_proj(normalized)
                cos, sin = ropes[self.config.layer_types[index]]
                q = self._rotate(q, cos[0], sin[0])
                k = self._rotate(k, cos[0], sin[0])
                attended = self._attention(index, attn, q, k, v, positions, requests)
                gate = attn.gate_proj(normalized).sigmoid().unsqueeze(-1)
                x = attn.o_proj((attended.view_as(q) * gate).flatten(1))
            else:
                conv = layer.conv
                b, c, value = conv.in_proj(normalized).chunk(3, dim=-1)
                gated = b * value
                convolved = torch.zeros_like(gated)
                for start, end, state in requests:
                    convolved[start:end], state.conv[index] = causal_conv(
                        gated[start:end], conv.conv.weight, conv.conv.bias, state.conv.get(index), self._conv_kernels
                    )
                x = conv.out_proj(c * convolved)
            if layer.use_sandwich_norm:
                x = layer.post_operator_norm(x)
            x = residual + x
            ffn = layer.feed_forward(layer.ffn_norm(x))
            if layer.use_sandwich_norm:
                ffn = layer.post_ffn_norm(ffn)
            x = x + ffn
        return self.embedding_norm(x)
