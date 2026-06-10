# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Patch Qwen3-TTS code predictor attention for 310P."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from vllm_omni.platforms.npu._310p.qwen3_tts_runtime import (
    aligned_code_predictor_seq_len,
    build_code_predictor_attention_mask,
    forward_code_predictor_attention,
)

_PATCHED = False
_code_predictor: Any | None = None
_original_attention_init: Callable[..., Any] | None = None
_original_attention_forward: Callable[..., Any] | None = None
_original_base_init: Callable[..., Any] | None = None
_original_base_forward: Callable[..., Any] | None = None


def apply_patch() -> None:
    global _PATCHED
    global _code_predictor
    global _original_attention_init
    global _original_attention_forward
    global _original_base_init
    global _original_base_forward

    if _PATCHED:
        return

    from vllm_omni.model_executor.models.common import qwen3_code_predictor as module

    _code_predictor = module
    _original_attention_init = module.CodePredictorAttention.__init__
    _original_attention_forward = module.CodePredictorAttention.forward
    _original_base_init = module.CodePredictorBaseModel.__init__
    _original_base_forward = module.CodePredictorBaseModel.forward

    module.CodePredictorAttention.__init__ = _attention_init_310p
    module.CodePredictorAttention.forward = _attention_forward_310p
    module.CodePredictorDecoderLayer.forward = _decoder_forward_310p
    module.CodePredictorBaseModel.__init__ = _base_init_310p
    module.CodePredictorBaseModel._get_310p_attention_mask_builder = _get_attention_mask_builder_310p
    module.CodePredictorBaseModel.forward = _base_forward_310p
    _PATCHED = True


def _attention_init_310p(self, *args, **kwargs) -> None:
    if _original_attention_init is None:
        raise RuntimeError("310P code predictor attention patch was not initialized.")
    _original_attention_init(self, *args, **kwargs)
    self._buffers.pop("_fusion_causal_mask", None)


def _attention_forward_310p(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask_builder=None,
) -> torch.Tensor:
    if hidden_states.device.type != "npu":
        if _original_attention_forward is None:
            raise RuntimeError("310P code predictor attention patch was not initialized.")
        return _original_attention_forward(self, hidden_states, position_embeddings)
    if _code_predictor is None:
        raise RuntimeError("310P code predictor module patch was not initialized.")

    bsz, seq_len, _ = hidden_states.shape
    hidden_shape_q = (bsz, seq_len, self.num_heads, self.head_dim)
    hidden_shape_kv = (bsz, seq_len, self.num_kv_heads, self.head_dim)

    q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape_q)).transpose(1, 2)
    k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape_kv)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(hidden_shape_kv).transpose(1, 2)

    cos, sin = position_embeddings
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q = (q * cos) + (_code_predictor._rotate_half(q) * sin)
    k = (k * cos) + (_code_predictor._rotate_half(k) * sin)

    attn_out = forward_code_predictor_attention(
        q,
        k,
        v,
        batch_size=bsz,
        seq_len=seq_len,
        num_heads=self.num_heads,
        num_kv_heads=self.num_kv_heads,
        head_dim=self.head_dim,
        scale=self.scaling,
        mask_builder=attention_mask_builder,
    )
    attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
    return self.o_proj(attn_out)


def _decoder_forward_310p(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask_builder=None,
) -> torch.Tensor:
    if attention_mask_builder is None:
        return _decoder_forward_with_original_attention(self, hidden_states, position_embeddings)

    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(
        hidden_states,
        position_embeddings,
        attention_mask_builder=attention_mask_builder,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states


def _base_init_310p(self, *args, **kwargs) -> None:
    if _original_base_init is None:
        raise RuntimeError("310P code predictor base patch was not initialized.")
    _original_base_init(self, *args, **kwargs)
    self._attention_mask_builder_310p_max_seq = aligned_code_predictor_seq_len(self.config.num_code_groups)
    self._attention_mask_builder_310p = None


def _get_attention_mask_builder_310p(self, device: torch.device):
    builder = getattr(self, "_attention_mask_builder_310p", None)
    if builder is not None and builder.device == device:
        return builder
    max_seq = getattr(self, "_attention_mask_builder_310p_max_seq", None)
    if max_seq is None:
        max_seq = aligned_code_predictor_seq_len(self.config.num_code_groups)
        self._attention_mask_builder_310p_max_seq = max_seq
    builder = build_code_predictor_attention_mask(device, max_seq)
    self._attention_mask_builder_310p = builder
    return builder


def _base_forward_310p(
    self,
    inputs_embeds: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    if inputs_embeds.device.type != "npu":
        if _original_base_forward is None:
            raise RuntimeError("310P code predictor base patch was not initialized.")
        return _original_base_forward(self, inputs_embeds, position_ids)

    input_dtype = inputs_embeds.dtype
    hidden_states = inputs_embeds
    with torch.amp.autocast(inputs_embeds.device.type, enabled=False, dtype=torch.float32):
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        attention_mask_builder = _get_attention_mask_builder_310p(self, inputs_embeds.device)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings,
                attention_mask_builder=attention_mask_builder,
            )
        hidden_states = self.norm(hidden_states)
    return hidden_states.to(input_dtype)


def _decoder_forward_with_original_attention(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    if _original_attention_forward is None:
        raise RuntimeError("310P code predictor attention patch was not initialized.")

    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = _original_attention_forward(self.self_attn, hidden_states, position_embeddings)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    return residual + hidden_states
