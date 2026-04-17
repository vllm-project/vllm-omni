# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Copyright (c) Ant Group. All rights reserved.
# Adapted from Ming repository modeling_bailingmm2.py
# https://github.com/inclusionAI/Ming

from collections.abc import Iterable

import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

logger = init_logger(__name__)


class Transpose(nn.Module):
    """Used in nn.Sequential pipelines."""

    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(self.dim0, self.dim1)


class VisionProjector(nn.Module):
    """MLP projector from vision encoder output to LLM hidden space.

    Args:
        vision_dim: Vision encoder output dimension (out_hidden_size).
        llm_dim: LLM hidden dimension.
        mlp_depth: Number of linear layers (>= 1).
    """

    def __init__(self, vision_dim: int, llm_dim: int, mlp_depth: int = 1):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(vision_dim, llm_dim)]
        for _ in range(1, mlp_depth):
            layers.append(nn.GELU())
            layers.append(nn.Linear(llm_dim, llm_dim))
        self.proj = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project vision features.

        Args:
            x: [seq_len, vision_dim] or [B, seq_len, vision_dim]

        Returns:
            Projected features with last dim = llm_dim.
        """
        return self.proj(x)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if not name.startswith("proj."):
                name = f"proj.{name}"
            if name not in params_dict:
                logger.warning("Skipping unknown vision projector weight: %s", name)
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class AudioProjector(nn.Module):
    """Projector for audio features.

    Args:
        audio_dim: Audio encoder output dimension (n_state).
        llm_dim: LLM hidden dimension.
        ds_kernel_size: Conv1d kernel size for downsampling.
        ds_stride: Conv1d stride for downsampling.
        mlp_depth: Total number of projection layers (>= 1).
    """

    def __init__(
        self,
        audio_dim: int,
        llm_dim: int,
        ds_kernel_size: int = 3,
        ds_stride: int = 2,
        mlp_depth: int = 1,
    ):
        super().__init__()
        self.ds_kernel_size = ds_kernel_size
        self.ds_stride = ds_stride

        layers: list[nn.Module] = [
            nn.Conv1d(
                audio_dim,
                llm_dim,
                kernel_size=ds_kernel_size,
                stride=ds_stride,
                padding=ds_kernel_size // 2,
            ),
            Transpose(-1, -2),  # [B, llm_dim, T'] -> [B, T', llm_dim]
        ]
        for _ in range(1, mlp_depth):
            layers.append(nn.GELU())
            layers.append(nn.Linear(llm_dim, llm_dim))
        self.proj = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project audio features with temporal downsampling.

        Args:
            x: [B, T, audio_dim] audio encoder output (channel-last).

        Returns:
            [B, T', llm_dim] projected features (channel-last),
            where T' = (T - ds_kernel_size + 2*(ds_kernel_size//2)) // ds_stride + 1.
        """
        # Conv1d expects [B, C, T], so transpose input
        x = x.transpose(-1, -2)  # [B, audio_dim, T]
        return self.proj(x)

    def forward_packed(
        self,
        packed: torch.Tensor,
        encoded_lens: list[int],
    ) -> tuple[torch.Tensor, list[int]]:
        """Project packed audio features from the Whisper encoder.

        Args:
            packed: [total_T', audio_dim] packed encoder output.
            encoded_lens: Length of each clip after Whisper encoding.

        Returns:
            Tuple of:
                - [total_T'', llm_dim] packed projected features.
                - List of projected lengths per clip.
        """
        conv1d = self.proj[0]
        mlp = self.proj[2:]

        # Split packed tensor per clip for Conv1d
        segments = packed.split(encoded_lens)
        conv_segments = []
        proj_lens: list[int] = []
        for seg in segments:
            out = conv1d(seg.transpose(0, 1).unsqueeze(0))  # [1, llm_dim, T'_i]
            out = out.squeeze(0).transpose(0, 1)  # [T'_i, llm_dim]
            conv_segments.append(out)
            proj_lens.append(out.shape[0])

        packed_proj = torch.cat(conv_segments, dim=0)  # [total_T'', llm_dim]
        packed_proj = mlp(packed_proj)
        return packed_proj, proj_lens

    def compute_output_length(self, input_length: torch.Tensor) -> torch.Tensor:
        """Compute output sequence length after Conv1d downsampling.

        Args:
            input_length: Original mel spectrogram lengths.

        Returns:
            Output lengths after both convolutions.
        """
        length = (input_length - 3 + 2 * 1) // 2 + 1
        length = (length - self.ds_kernel_size + 2 * (self.ds_kernel_size // 2)) // self.ds_stride + 1
        return length

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if not name.startswith("proj."):
                name = f"proj.{name}"
            if name not in params_dict:
                logger.warning("Skipping unknown audio projector weight: %s", name)
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
