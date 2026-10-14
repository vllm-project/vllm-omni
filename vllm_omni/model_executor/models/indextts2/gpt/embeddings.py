# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Learned position embeddings for IndexTTS2 GPT model."""

import torch
import torch.nn as nn


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len: int, model_dim: int, init: float = 0.02):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sl = x.shape[1]
        return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind: int, dev: torch.device) -> torch.Tensor:
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)
