# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adopted from https://github.com/inclusionAI/Ming-omni-tts/blob/main/fm/dit.py
from __future__ import annotations

import torch
import torch.nn as nn
from x_transformers.x_transformers import RotaryEmbedding

from vllm_omni.model_executor.models.common.ming.dit import DiTBlock, FinalLayer


class Aggregator(nn.Module):
    def __init__(
        self,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        llm_input_dim=896,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads

        self.word_embedder = nn.Embedding(1, hidden_size)
        self.x_embedder = nn.Linear(in_channels, hidden_size)
        self.hidden_size = hidden_size
        self.rotary_embed = RotaryEmbedding(hidden_size // num_heads)
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **kwargs) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, llm_input_dim)

    def forward(self, x, mask=None):
        if x.ndim != 3:
            raise ValueError(f"Expected x rank-3 [Batch, Time, Dimension], got {tuple(x.shape)}")
        if x.shape[-1] != self.in_channels:
            raise ValueError(f"x feature dim mismatch: got {x.shape[-1]}, expected {self.in_channels}")

        # [Batch, Time, Dimension] -> [Batch, Time, Hidden].
        x = self.x_embedder(x)
        cls_embed = self.word_embedder(torch.zeros((x.shape[0], 1), dtype=torch.long, device=x.device))
        # Prepend a learned CLS token: [Batch, Time, Hidden] -> [Batch, Time + 1, Hidden].
        x = torch.cat([cls_embed, x], dim=1)

        rope = self.rotary_embed.forward_from_seq_len(x.shape[1])
        if mask is not None:
            if mask.ndim != 2:
                raise ValueError(f"Expected mask rank-2 [Batch, Time], got {tuple(mask.shape)}")
            if mask.shape[0] != x.shape[0] or mask.shape[1] != x.shape[1] - 1:
                raise ValueError(
                    f"Mask shape mismatch: got {tuple(mask.shape)}, expected {(x.shape[0], x.shape[1] - 1)}"
                )
            mask_pad = mask.clone().detach()[:, :1]
            mask = torch.cat([mask_pad, mask], dim=-1)
        for block in self.blocks:
            x = block(x, mask, rope)
        x = self.final_layer(x)
        # Keep the CLS projection only: [Batch, Time + 1, Hidden] -> [Batch, 1, Hidden].
        return x[:, :1, :]


__all__ = ["Aggregator"]
