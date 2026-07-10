# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LTX-2.3 connector customizations.

The connector module itself still comes from Diffusers. LTX-2.3 overrides only
the attention processor so Omni follows the official connector attention path.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from diffusers.pipelines.ltx2 import LTX2TextConnectors
from torch.nn.attention import SDPBackend, sdpa_kernel

from .ltx2_3_transformer import apply_interleaved_rotary_emb, apply_split_rotary_emb

_LTX23_OFFICIAL_SDPA_PRIORITY = [
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]


def ltx23_official_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    heads: int,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    batch_size, _, inner_dim = query.shape
    head_dim = inner_dim // heads
    query, key, value = (tensor.view(batch_size, -1, heads, head_dim).transpose(1, 2) for tensor in (query, key, value))

    if attention_mask is not None:
        if attention_mask.ndim == 2:
            attention_mask = attention_mask.unsqueeze(0)
        if attention_mask.ndim == 3:
            attention_mask = attention_mask.unsqueeze(1)

    with sdpa_kernel(_LTX23_OFFICIAL_SDPA_PRIORITY, set_priority=True):
        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
    return hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim)


class LTX23OfficialConnectorAttnProcessor:
    """Match the official LTX-2.3 text connector attention path."""

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        query_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        key_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        gate_logits = attn.to_gate_logits(hidden_states) if attn.to_gate_logits is not None else None

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if query_rotary_emb is not None:
            key_rotary_emb = key_rotary_emb if key_rotary_emb is not None else query_rotary_emb
            if attn.rope_type == "interleaved":
                query = apply_interleaved_rotary_emb(query, query_rotary_emb)
                key = apply_interleaved_rotary_emb(key, key_rotary_emb)
            elif attn.rope_type == "split":
                query = apply_split_rotary_emb(query, query_rotary_emb, head_dim=attn.head_dim)
                key = apply_split_rotary_emb(key, key_rotary_emb, head_dim=attn.head_dim)
            else:
                raise ValueError(f"Unsupported LTX-2.3 connector rope type: {attn.rope_type}")

        hidden_states = ltx23_official_sdpa(
            query,
            key,
            value,
            heads=attn.heads,
            attention_mask=attention_mask,
        )

        if gate_logits is not None:
            hidden_states = hidden_states.unflatten(2, (attn.heads, -1))
            gates = 2.0 * torch.sigmoid(gate_logits)
            hidden_states = hidden_states * gates.unsqueeze(-1)
            hidden_states = hidden_states.flatten(2, 3)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def install_ltx23_official_connector_processors(connectors: LTX2TextConnectors) -> None:
    for connector_name in ("video_connector", "audio_connector"):
        connector = getattr(connectors, connector_name, None)
        if connector is None:
            continue
        for block in getattr(connector, "transformer_blocks", ()):
            attn = getattr(block, "attn1", None)
            if attn is None:
                continue
            processor = LTX23OfficialConnectorAttnProcessor()
            if hasattr(attn, "set_processor"):
                attn.set_processor(processor)
            else:
                attn.processor = processor
