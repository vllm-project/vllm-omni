# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FunCineForge Talker (LM stage).

Wraps a Qwen2 backbone via ``VLLMQwen2Encoder`` and adds:
  - ``codec_embed``  — codec token embeddings (vocab = codec_unit)
  - ``timespk_embed`` — timespeaker tag embeddings (vocab = timespk_unit)
  - ``face_linear``  — face embedding projection (face_size → llm_dim)
  - ``codec_head``   — output head (llm_dim → codec_unit)

The embedding assembly follows FunCineForge's original logic:
  1. Compute text / timespk / codec flags from input_ids.
  2. Embed each region with the corresponding embedding table.
  3. Sum the three embedding streams.
  4. Insert face embeddings after the SOS token.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from vllm.config import VllmConfig


class VLLMQwen2Encoder(nn.Module):
    """Thin wrapper around vLLM's Qwen2Model for use as the LM backbone.

    Delegates ``forward`` to the internal ``Qwen2Model`` so that vLLM's
    PagedAttention / KV-cache machinery works transparently.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        from vllm.model_executor.models.qwen2 import Qwen2Model

        self.model = Qwen2Model(vllm_config=vllm_config, prefix=prefix)

    def forward(self, inputs_embeds: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # vLLM model expects flattened tensors [total_tokens, hidden_size]
        if inputs_embeds.dim() == 3:
            inputs_flat = inputs_embeds.view(-1, self.model.embed_tokens.embedding_dim)
        else:
            inputs_flat = inputs_embeds
        positions_flat = positions.view(-1)

        # KV cache managed externally via ForwardContext (set by GPUARModelRunner)
        # input_ids is required but ignored when inputs_embeds is provided
        hidden_states = self.model(
            input_ids=torch.zeros(inputs_flat.size(0), dtype=torch.long, device=inputs_flat.device),
            positions=positions_flat,
            intermediate_tensors=None,
            inputs_embeds=inputs_flat,
        )
        return hidden_states


class FunCineForgeTalker(nn.Module):
    """FunCineForge LM stage: Qwen2 backbone + codec/timespk/face embeddings + codec_head."""

    def __init__(
        self,
        codec_unit: int = 6761,
        timespk_unit: int = 1550,
        face_size: int = 512,
        llm: VLLMQwen2Encoder | None = None,
    ):
        super().__init__()
        self.llm = llm
        llm_dim = llm.model.embed_tokens.embedding_dim if llm is not None else 896

        self.codec_embed = nn.Embedding(codec_unit, llm_dim, padding_idx=0)
        self.timespk_embed = nn.Embedding(timespk_unit, llm_dim, padding_idx=0)
        self.face_linear = nn.Linear(face_size, llm_dim)

        self.codec_head = nn.Linear(llm_dim, codec_unit, bias=False)
