# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Frame-local KV reuse for the Qwen3-TTS residual codebooks.

The scratch buffer belongs to one predictor worker, just like its projection
buffer. Calls are serialized on the worker's CUDA stream. Every frame starts
by overwriting positions 0 and 1; later queries can only read that frame's
written positions. No request state survives between frames.
"""

from collections.abc import Sequence

import torch

from vllm_omni.model_executor.models.common.qwen3_code_predictor import (
    CodePredictorAttention,
    CodePredictorBaseModel,
    CodePredictorDecoderLayer,
    _rotate_half,
)

from .short_kv_attention import short_kv_attention


def _attention_cached(
    attn: CodePredictorAttention,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    write_positions: torch.Tensor,
) -> torch.Tensor:
    """Attend only the new positions, reusing earlier positions' keys and values.

    ``k_cache``/``v_cache`` are ``[B, num_kv_heads, max_seq, head_dim]``;
    the new positions' keys and values are written at ``write_positions``
    (``[n]``). The attention kernel masks keys after each query position,
    so unwritten slots are never read.
    """
    bsz, seq_len, _ = hidden_states.shape
    qkv = attn.qkv_proj(hidden_states)
    q_raw, k_raw, v_raw = attn._split_qkv(qkv)
    q = attn.q_norm(q_raw.view(bsz, seq_len, attn.num_heads, attn.head_dim)).transpose(1, 2)
    k = attn.k_norm(k_raw.view(bsz, seq_len, attn.num_kv_heads, attn.head_dim)).transpose(1, 2)
    v = v_raw.view(bsz, seq_len, attn.num_kv_heads, attn.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    k_cache.index_copy_(2, write_positions, k.to(k_cache.dtype))
    v_cache.index_copy_(2, write_positions, v.to(v_cache.dtype))
    attn_out = short_kv_attention(q, k_cache, v_cache, write_positions, attn.scaling)
    attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
    return attn.o_proj(attn_out)


def _layer_cached(
    layer: CodePredictorDecoderLayer,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    write_positions: torch.Tensor,
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)
    hidden_states = _attention_cached(
        layer.self_attn, hidden_states, position_embeddings, k_cache, v_cache, write_positions
    )
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    return residual + hidden_states


def forward_cached(
    model: CodePredictorBaseModel,
    inputs_embeds: torch.Tensor,
    position_ids: torch.Tensor,
    kv_caches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    write_positions: torch.Tensor,
) -> torch.Tensor:
    """``forward`` for the new positions only (``position_ids`` ``[B, n]``), reusing earlier K/V."""
    hidden_states = inputs_embeds
    with torch.amp.autocast(inputs_embeds.device.type, enabled=False):
        position_embeddings = model.rotary_emb(hidden_states, position_ids)
        for layer, (k_cache, v_cache) in zip(model.layers, kv_caches):
            hidden_states = _layer_cached(layer, hidden_states, position_embeddings, k_cache, v_cache, write_positions)
        hidden_states = model.norm(hidden_states)
    return hidden_states


class FrameLocalKVCache:
    """Non-module execution state; checkpoint parameters remain on the predictor."""

    def __init__(self, model: CodePredictorBaseModel, max_batch: int) -> None:
        self.model = model
        weight = next(model.parameters())
        attn = model.layers[0].self_attn
        self.buffer = torch.zeros(
            len(model.layers),
            2,
            max_batch,
            attn.num_kv_heads,
            attn.max_seq,
            attn.head_dim,
            device=weight.device,
            dtype=weight.dtype,
        )
        self.positions = {
            step: torch.arange(0 if step == 1 else step, step + 1, device=weight.device)
            for step in range(1, attn.max_seq - 1)
        }
        # Batch buckets share compiled kernels; outer MTP graphs still capture
        # fixed addresses. This helper never creates nested CUDA graphs.
        self.forward = torch.compile(forward_cached, dynamic=True, options={"epilogue_fusion": False})

    def __call__(self, proj_buf: torch.Tensor, batch: int, step: int) -> torch.Tensor:
        if batch > self.buffer.shape[2]:
            raise ValueError("Qwen3-TTS cached predictor batch exceeds its configured capacity")
        first = 0 if step == 1 else step
        caches = [(layer[0, :batch], layer[1, :batch]) for layer in self.buffer]
        positions = self.positions[step]
        return self.forward(
            self.model,
            proj_buf[:batch, first : step + 1],
            positions.unsqueeze(0).expand(batch, -1),
            caches,
            positions,
        )
