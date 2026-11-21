# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import torch
import torch.nn as nn


class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embedding with additive combination.

    Args:
        max_seq_len: Maximum sequence length
        embedding_dim: Dimension of embeddings
        init_std: Standard deviation for weight initialization
    """

    def __init__(self, max_seq_len: int, embedding_dim: int, init_std: float = 0.02):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.embedding.weight.data.normal_(mean=0.0, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings to input.

        Args:
            x: Input tensor, shape (B, T, D)

        Returns:
            Input with positional embeddings added, shape (B, T, D)
        """
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        return x + self.embedding(positions).unsqueeze(0).expand(x.shape[0], -1, -1)
