# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from NAVER Cloud Corp. vision-decoder-api

"""
Vision Token Embedder for HyperCLOVAX Vision Decoder.

Converts discrete vision tokens to continuous embeddings.
"""

import numpy as np
import torch
import torch.nn as nn


class VisionTokenEmbedder(nn.Module):
    """
    Vision Token Embedder that converts discrete vision tokens to embeddings.

    This module embeds vision tokens (discrete vocabulary indices) into
    continuous vector representations for the VisionTransformer.

    Args:
        vocab_size: Size of the vision token vocabulary (default: 65536)
        embedding_dim: Dimension of the embedding vectors (default: 1536)
        token_length: Expected number of tokens per image (default: 729 for 27x27)
    """

    def __init__(
        self,
        vocab_size: int = 65536,
        embedding_dim: int = 1536,
        token_length: int = 729,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.token_length = token_length

        # Main vocabulary embeddings
        self.vocab_embeddings = nn.Parameter(torch.zeros(vocab_size, embedding_dim))

        # Unconditional embedding for classifier-free guidance
        self.uncond_embedding = nn.Parameter(torch.zeros(1, embedding_dim))

    def load_vocab_embeddings(self, embeddings: torch.Tensor) -> None:
        """Load vocabulary embeddings from a tensor."""
        if embeddings.shape != (self.vocab_size, self.embedding_dim):
            raise ValueError(
                f"Expected embeddings shape ({self.vocab_size}, {self.embedding_dim}), got {embeddings.shape}"
            )
        with torch.no_grad():
            self.vocab_embeddings.copy_(embeddings)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Convert vision tokens to embeddings.

        Args:
            tokens: Vision token IDs (B, L) where L is typically 729

        Returns:
            Dictionary with:
                - vision_last_hidden_state: (B, L, embedding_dim)
                - vision_pooler_output: (B, embedding_dim) - mean pooled
        """
        # Look up embeddings
        hidden_states = self.vocab_embeddings[tokens]

        # Mean pooling for pooler output
        pooler_output = hidden_states.mean(dim=1)

        return {
            "vision_last_hidden_state": hidden_states,
            "vision_pooler_output": pooler_output,
        }

    def get_uncond_embeddings(self, batch_size: int, token_length: int) -> dict[str, torch.Tensor]:
        """
        Get unconditional embeddings for classifier-free guidance.

        Args:
            batch_size: Batch size
            token_length: Number of tokens per sample

        Returns:
            Dictionary with unconditional hidden states and pooler output
        """
        uncond_hidden = self.uncond_embedding.expand(batch_size, token_length, -1)
        uncond_pooler = uncond_hidden.mean(dim=1)

        return {
            "vision_last_hidden_state": uncond_hidden,
            "vision_pooler_output": uncond_pooler,
        }

    @classmethod
    def from_numpy(cls, npy_path: str, token_length: int = 729) -> "VisionTokenEmbedder":
        """
        Create embedder from numpy file.

        Args:
            npy_path: Path to .npy file containing embeddings
            token_length: Number of tokens per image (inferred from model architecture;
                default 729 = 27×27 for HyperCLOVAX-SEED-Omni-8B)

        Returns:
            VisionTokenEmbedder instance with loaded embeddings
        """
        embeddings = torch.from_numpy(np.load(npy_path)).float()
        vocab_size, embedding_dim = embeddings.shape

        embedder = cls(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            token_length=token_length,
        )
        embedder.load_vocab_embeddings(embeddings)
        return embedder
