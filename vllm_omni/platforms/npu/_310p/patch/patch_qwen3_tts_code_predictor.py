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

TARGET_MODULE = "vllm_omni.model_executor.models.common.qwen3_code_predictor"

_code_predictor: Any | None = None
_original_attention_init: Callable[..., Any] | None = None
_original_decoder_forward: Callable[..., Any] | None = None
_original_base_init: Callable[..., Any] | None = None
_original_base_forward: Callable[..., Any] | None = None


def is_ready(module: Any) -> bool:
    return all(
        hasattr(module, name)
        for name in (
            "CodePredictorAttention",
            "CodePredictorBaseModel",
            "CodePredictorDecoderLayer",
        )
    )


def apply(module: Any) -> None:
    global _code_predictor
    global _original_attention_init
    global _original_decoder_forward
    global _original_base_init
    global _original_base_forward

    if _code_predictor is not None:
        return

    _code_predictor = module
    _original_attention_init = module.CodePredictorAttention.__init__
    _original_decoder_forward = module.CodePredictorDecoderLayer.forward
    _original_base_init = module.CodePredictorBaseModel.__init__
    _original_base_forward = module.CodePredictorBaseModel.forward

    module.CodePredictorAttention.__init__ = _attention_init_310p
    module.CodePredictorAttention.forward = _attention_forward_310p
    module.CodePredictorDecoderLayer.forward = _decoder_forward_310p
    module.CodePredictorBaseModel.__init__ = _base_init_310p
    module.CodePredictorBaseModel._get_310p_attention_mask_builder = _get_attention_mask_builder_310p
    module.CodePredictorBaseModel.forward = _base_forward_310p


def _attention_init_310p(self, *args, **kwargs) -> None:
    assert _original_attention_init is not None
    _original_attention_init(self, *args, **kwargs)
    self._buffers.pop("_fusion_causal_mask", None)


def _attention_forward_310p(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask_builder=None,
) -> torch.Tensor:
    assert _code_predictor is not None

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

    if q.device.type == "npu":
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
    else:
        attn_out = _code_predictor.F.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scaling,
            is_causal=True,
            enable_gqa=self.is_gqa,
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
        assert _original_decoder_forward is not None
        return _original_decoder_forward(self, hidden_states, position_embeddings)

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
    assert _original_base_init is not None
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
        assert _original_base_forward is not None
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
