# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. on 2025-09-30.
#
# Original file was released under Apache-2.0, with the full license text
# available at https://www.apache.org/licenses/LICENSE-2.0
#
# This modified file is released under the same license.
#
# --- Upstream header preserved below ---
#
# Copyright 2025 BAAI, The Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from einops import repeat

try:
    from transformers.modeling_flash_attention_utils import (  # type: ignore
        flash_attn_varlen_func,  # pyright: ignore[reportAttributeAccessIssue]
        is_flash_attn_available,
    )
except Exception:  # pragma: no cover - best-effort compatibility
    flash_attn_varlen_func = None  # type: ignore[assignment]

    def is_flash_attn_available() -> bool:  # type: ignore[override]
        return False


from .rope_real import apply_real_rotary_emb

_HAS_FLASH_ATTN_VARLEN = bool(is_flash_attn_available()) and flash_attn_varlen_func is not None


class AttnProcessor:
    def __init__(self) -> None:
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor requires PyTorch 2.0+ (F.scaled_dot_product_attention).")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        base_sequence_length: int | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape

        # Get Query-Key-Value Pair
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query_dim = query.shape[-1]
        inner_dim = key.shape[-1]
        head_dim = query_dim // attn.heads
        dtype = query.dtype

        # Get key-value heads
        kv_heads = inner_dim // head_dim

        # Reshape tensors for attention computation
        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, kv_heads, head_dim)
        value = value.view(batch_size, -1, kv_heads, head_dim)

        # Apply Query-Key normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply Rotary Position Embeddings
        if image_rotary_emb is not None:
            query = apply_real_rotary_emb(query, image_rotary_emb[0], image_rotary_emb[1])
            key = apply_real_rotary_emb(key, image_rotary_emb[0], image_rotary_emb[1])

        query, key = query.to(dtype), key.to(dtype)

        # Calculate attention scale
        if base_sequence_length is not None:
            softmax_scale = math.sqrt(math.log(sequence_length, base_sequence_length)) * attn.scale
        else:
            softmax_scale = attn.scale

        if _HAS_FLASH_ATTN_VARLEN and attention_mask is not None and hidden_states.is_cuda:
            # Flash-Attn varlen expects packed tokens + cu_seqlens. Here we only need
            # the self-attention case (q/k/v share the same padding mask).
            attention_mask = attention_mask.to(torch.bool)
            seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
            indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
            max_seqlen = int(seqlens.max().item())
            cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))

            query_states = query.reshape(batch_size * sequence_length, attn.heads, head_dim)[indices]
            key_states = key.reshape(batch_size * sequence_length, kv_heads, head_dim)[indices]
            value_states = value.reshape(batch_size * sequence_length, kv_heads, head_dim)[indices]

            if kv_heads < attn.heads:
                key_states = repeat(key_states, "l h c -> l (h k) c", k=attn.heads // kv_heads)
                value_states = repeat(value_states, "l h c -> l (h k) c", k=attn.heads // kv_heads)

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=0.0,
                causal=False,
                softmax_scale=softmax_scale,
            )

            out = torch.zeros(
                (batch_size * sequence_length, attn.heads, head_dim),
                device=attn_output_unpad.device,
                dtype=attn_output_unpad.dtype,
            )
            out[indices] = attn_output_unpad
            hidden_states = out.view(batch_size, sequence_length, attn.heads, head_dim).flatten(-2)
            hidden_states = hidden_states.type_as(query)
        else:
            # PyTorch SDPA path.
            attn_mask = None
            if attention_mask is not None:
                attention_mask = attention_mask.to(torch.bool)
                attn_mask = attention_mask.view(batch_size, 1, 1, -1)

            query = query.transpose(1, 2)  # [B, H, S, D]
            key = key.transpose(1, 2)  # [B, H_kv, S, D]
            value = value.transpose(1, 2)

            if kv_heads < attn.heads:
                key = key.repeat_interleave(attn.heads // kv_heads, dim=1)
                value = value.repeat_interleave(attn.heads // kv_heads, dim=1)

            hidden_states = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
                scale=softmax_scale,
            )

            if attention_mask is not None:
                # Keep padding tokens consistent with the flash-varlen path (zero output).
                hidden_states = hidden_states * attention_mask[:, None, :, None]

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.type_as(query)

        # Apply output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
