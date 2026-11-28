# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Action codec of CAE-FastWAM: the decoder of its suite-conditioned action VAE.

Adapted from AMD-AGI/faster-action-models ``FastWAM/src/fastwam/models/action_ae.py``
(Apache-2.0). CAE-FastWAM denoises ``num_tokens`` latent tokens of width
``latent_dim`` and decodes them into the ``[action_horizon, action_dim]`` chunk.
Only the decoder runs at inference; the encoder (and its variational head) is a
training-time component and its weights are not loaded.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from vllm_omni.diffusion.models.fastwam.config import ActionCodecConfig


def _factorise_ratio(ratio: int) -> list[int]:
    if ratio == 1:
        return [1]
    factors = []
    for factor in (4, 2, 3):
        while ratio % factor == 0:
            factors.append(factor)
            ratio //= factor
    if ratio > 1:
        factors.append(ratio)
    return factors


class ResidualConv1dBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.GroupNorm(min(8, channels), channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.GroupNorm(min(8, channels), channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ActionLatentDecoder(nn.Module):
    """[B, num_tokens, latent_dim] latents -> [B, action_horizon, action_dim] actions."""

    def __init__(self, cfg: ActionCodecConfig) -> None:
        super().__init__()
        self.cfg = cfg
        hidden = cfg.decoder_hidden
        self.input_proj = nn.Linear(cfg.latent_dim, hidden)
        layers: list[nn.Module] = []
        for stride in _factorise_ratio(cfg.action_horizon // cfg.num_tokens):
            layers.extend(
                [
                    nn.ConvTranspose1d(hidden, hidden, kernel_size=stride * 2, stride=stride, padding=stride // 2),
                    nn.GroupNorm(min(8, hidden), hidden),
                    nn.SiLU(),
                    ResidualConv1dBlock(hidden),
                ]
            )
        self.up = nn.Sequential(*layers)
        self.output_proj = nn.Linear(hidden, cfg.action_dim)
        self.suite_embed = nn.Embedding(cfg.num_suites, hidden) if cfg.num_suites > 0 else None

    def forward(self, z: torch.Tensor, suite_ids: torch.Tensor | None) -> torch.Tensor:
        x = self.input_proj(z)
        if self.suite_embed is not None:
            if suite_ids is None:
                raise ValueError("This action codec is suite-conditioned; a suite id is required.")
            x = x + self.suite_embed(suite_ids).unsqueeze(1)
        x = self.up(x.transpose(1, 2))
        target = self.cfg.action_horizon
        if x.shape[2] > target:
            x = x[:, :, :target]
        elif x.shape[2] < target:
            x = F.pad(x, (0, target - x.shape[2]))
        return self.output_proj(x.transpose(1, 2))

    @staticmethod
    def decoder_state_dict(action_vae_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Select the ``decoder.*`` tensors of an ``action_vae.pt`` ``model`` state dict."""
        prefix = "decoder."
        return {key[len(prefix) :]: value for key, value in action_vae_state.items() if key.startswith(prefix)}
