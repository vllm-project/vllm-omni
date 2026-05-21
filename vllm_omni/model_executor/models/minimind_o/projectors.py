# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Adapted from MiniMind-O repository

from collections.abc import Iterable

import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

logger = init_logger(__name__)


class MiniMindOAudioProjector(nn.Module):
    """2-layer MLP projector from SenseVoice output to LLM hidden space.

    Pattern from MiniMind-O: LayerNorm → Linear → GELU → Linear
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project audio features.

        Args:
            x: [seq_len, in_dim] or [B, seq_len, in_dim]

        Returns:
            Projected features with last dim = out_dim.
        """
        return self.mlp(x)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if not name.startswith("mlp."):
                name = f"mlp.{name}"
            if name not in params_dict:
                logger.warning("Skipping unknown audio projector weight: %s", name)
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class MiniMindOVisionProjector(nn.Module):
    """2-layer MLP projector from SigLIP2 output to LLM hidden space.

    Pattern from MiniMind-O: LayerNorm → Linear → GELU → Linear
    """

    def __init__(self, in_dim: int, out_dim: int, source_tokens: int = 64, target_tokens: int = 64):
        super().__init__()
        self.source_tokens = source_tokens
        self.target_tokens = target_tokens
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project vision features.

        Args:
            x: [seq_len, in_dim] or [B, seq_len, in_dim]

        Returns:
            Projected features with last dim = out_dim.
        """
        return self.mlp(x)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if not name.startswith("mlp."):
                name = f"mlp.{name}"
            if name not in params_dict:
                logger.warning("Skipping unknown vision projector weight: %s", name)
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
