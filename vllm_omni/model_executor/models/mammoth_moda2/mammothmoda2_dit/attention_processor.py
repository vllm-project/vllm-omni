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
from transformers.modeling_flash_attention_utils import (
    flash_attn_varlen_func,
    is_flash_attn_available,
    pad_input,
    unpad_input,
)
from transformers.utils.import_utils import is_torch_npu_available

from mammothmoda2.model.mammothmoda2_dit.rope_real import apply_real_rotary_emb

if is_flash_attn_available():
    if is_torch_npu_available():
        from transformers.integrations.npu_flash_attention import index_first_axis
    else:
        from flash_attn.bert_padding import index_first_axis
else:
    error_msg = "No legal index_first_axis found. Please install flash-attn."
    raise ImportError(error_msg)


class AttnProcessorFlash2Varlen:
    """
    Processor for implementing scaled dot-product attention with flash attention and variable length sequences.

    This processor implements:
    - Flash attention with variable length sequences
    - Rotary position embeddings (RoPE)
    - Query-Key normalization
    - Proportional attention scaling

    Args:
        None
    """

    def __init__(self) -> None:
        """Initialize the attention processor."""
        if not is_flash_attn_available():
            raise ImportError("AttnProcessorFlash2Varlen requires flash_attn. Please install flash_attn.")

    def _upad_input(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: torch.Tensor,
        query_length: int,
        num_heads: int,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor], tuple[int, int]
    ]:
        """
        Unpad the input tensors for flash attention.

        Args:
            query_layer: Query tensor of shape (batch_size, seq_len, num_heads, head_dim)
            key_layer: Key tensor of shape (batch_size, seq_len, num_kv_heads, head_dim)
            value_layer: Value tensor of shape (batch_size, seq_len, num_kv_heads, head_dim)
            attention_mask: Attention mask tensor of shape (batch_size, seq_len)
            query_length: Length of the query sequence
            num_heads: Number of attention heads

        Returns:
            Tuple containing:
                - Unpadded query tensor
                - Unpadded key tensor
                - Unpadded value tensor
                - Query indices
                - Tuple of cumulative sequence lengths for query and key
                - Tuple of maximum sequence lengths for query and key
        """

        def _get_unpad_data(attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
            """Helper function to get unpadding data from attention mask."""
            seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
            indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
            max_seqlen_in_batch = seqlens_in_batch.max().item()
            cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
            return indices, cu_seqlens, max_seqlen_in_batch

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        # Unpad key and value layers
        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )

        # Handle different query length cases
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=query_layer.device)
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        base_sequence_length: int | None = None,
    ) -> torch.Tensor:
        """
        Process attention computation with flash attention.

        Args:
            attn: Attention module
            hidden_states: Hidden states tensor of shape (batch_size, seq_len, hidden_dim)
            encoder_hidden_states: Encoder hidden states tensor
            attention_mask: Optional attention mask tensor
            image_rotary_emb: Optional rotary embeddings for image tokens
            base_sequence_length: Optional base sequence length for proportional attention

        Returns:
            torch.Tensor: Processed hidden states after attention computation
        """
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

        # Unpad input for flash attention
        (
            query_states,
            key_states,
            value_states,
            indices_q,
            cu_seq_lens,
            max_seq_lens,
        ) = self._upad_input(query, key, value, attention_mask, sequence_length, attn.heads)

        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        # Handle different number of heads
        if kv_heads < attn.heads:
            key_states = repeat(key_states, "l h c -> l (h k) c", k=attn.heads // kv_heads)
            value_states = repeat(value_states, "l h c -> l (h k) c", k=attn.heads // kv_heads)

        # Apply flash attention
        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=0.0,
            causal=False,
            softmax_scale=softmax_scale,
        )

        # Pad output and apply final transformations
        hidden_states = pad_input(attn_output_unpad, indices_q, batch_size, sequence_length)
        hidden_states = hidden_states.flatten(-2)
        hidden_states = hidden_states.type_as(query)

        # Apply output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def scaled_dot_product_attention_torch(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
):
    """
    Implements torch-native scaled dot product attention (without using F.scaled_dot_product_attention).
    Args:
        query: (batch, num_heads, seq_len_q, head_dim)
        key:   (batch, num_heads, seq_len_k, head_dim)
        value: (batch, num_heads, seq_len_k, head_dim)
        attn_mask: (batch, 1, seq_len_q, seq_len_k) or (batch, seq_len_q, seq_len_k) or None
        dropout_p: dropout probability on attention weights
        is_causal: if True, apply causal mask
        scale: optional scaling factor (float)
    Returns:
        attn_output: (batch, num_heads, seq_len_q, head_dim)
    """
    B, num_heads, L_q, D = query.shape
    _, _, L_k, _ = key.shape

    # Compute attention scores
    attn_scores = torch.matmul(query, key.transpose(-2, -1))  # (B, num_heads, L_q, L_k)
    if scale is not None:
        attn_scores = attn_scores / scale
    else:
        attn_scores = attn_scores / (D**0.5)

    # Apply attention mask if provided
    if attn_mask is not None:
        # attn_mask should be broadcastable to (B, num_heads, L_q, L_k)
        if attn_mask.dtype == torch.bool:
            attn_scores = attn_scores.masked_fill(~attn_mask, float("-inf"))
        else:
            attn_scores = attn_scores + attn_mask

    # Apply causal mask if needed
    if is_causal:
        causal_mask = torch.tril(torch.ones((L_q, L_k), device=attn_scores.device, dtype=torch.bool))
        attn_scores = attn_scores.masked_fill(~causal_mask, float("-inf"))

    # Compute attention weights
    attn_weights = torch.softmax(attn_scores, dim=-1)
    if dropout_p > 0.0 and attn_weights.requires_grad:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # Compute output
    attn_output = torch.matmul(attn_weights, value)  # (B, num_heads, L_q, D)
    return attn_output


# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class AttnProcessor:
    """
    Processor for implementing scaled dot-product attention with flash attention and variable length sequences.

    This processor is optimized for PyTorch 2.0 and implements:
    - Flash attention with variable length sequences
    - Rotary position embeddings (RoPE)
    - Query-Key normalization
    - Proportional attention scaling

    Args:
        None

    Raises:
        ImportError: If PyTorch version is less than 2.0
    """

    def __init__(self) -> None:
        """Initialize the attention processor."""
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessorFlash2Varlen requires PyTorch 2.0. Please upgrade PyTorch to version 2.0 or later."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        base_sequence_length: int | None = None,
    ) -> torch.Tensor:
        """
        Process attention computation with flash attention.

        Args:
            attn: Attention module
            hidden_states: Hidden states tensor of shape (batch_size, seq_len, hidden_dim)
            encoder_hidden_states: Encoder hidden states tensor
            attention_mask: Optional attention mask tensor
            image_rotary_emb: Optional rotary embeddings for image tokens
            base_sequence_length: Optional base sequence length for proportional attention

        Returns:
            torch.Tensor: Processed hidden states after attention computation
        """
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

        # scaled_dot_product_attention expects attention_mask shape to be
        # (batch, heads, source_length, target_length)
        if attention_mask is not None:
            attention_mask = attention_mask.bool().view(batch_size, 1, 1, -1)  #

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # explicitly repeat key and value to match query length, otherwise using enable_gqa=True results in MATH backend of sdpa in our test of pytorch2.6
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

        # TODO: mask should be added
        hidden_states = scaled_dot_product_attention(query, key, value, scale=softmax_scale)  # attn_mask=attention_mask

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.type_as(query)
        # Apply output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
