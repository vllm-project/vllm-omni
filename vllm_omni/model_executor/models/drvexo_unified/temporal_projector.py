# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The trained 4B connector; temporal convolutions never cross utterances."""

import torch
from torch import nn


class TemporalResidual(nn.Module):
    def __init__(self, audio_dim: int, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(audio_dim)
        self.down = nn.Linear(audio_dim, width)
        self.conv1 = nn.Conv1d(width, width, 5, padding=2, groups=width)
        self.mix1 = nn.Linear(width, width)
        self.conv2 = nn.Conv1d(width, width, 5, padding=4, dilation=2, groups=width)
        self.mix2 = nn.Linear(width, width)
        self.up = nn.Linear(width, audio_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("Process each unpadded utterance separately")
        hidden = torch.nn.functional.gelu(self.down(self.norm(features)))
        for conv, mix in ((self.conv1, self.mix1), (self.conv2, self.mix2)):
            hidden = hidden + mix(torch.nn.functional.gelu(conv(hidden.T[None])[0].T))
        return features + self.up(hidden)


class TemporalInputProjector(nn.Module):
    def __init__(self, audio_dim: int, language_dim: int, width: int):
        super().__init__()
        self.temporal = TemporalResidual(audio_dim, width)
        self.net = nn.Sequential(
            nn.LayerNorm(audio_dim),
            nn.Linear(audio_dim, language_dim),
            nn.GELU(),
            nn.Linear(language_dim, language_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(self.temporal(features))


def project_utterances(projector: nn.Module, features: torch.Tensor, lengths=None) -> tuple[torch.Tensor, ...]:
    if features.ndim != 2:
        raise ValueError("Audio encoder states must be flat unpadded frames")
    sizes = [len(features)] if lengths is None else [int(value) for value in lengths]
    if not sizes or any(not 0 < value <= 390 for value in sizes) or sum(sizes) != len(features):
        raise ValueError("Audio lengths must exactly cover 1–390 frames per utterance")
    dtype = next(projector.parameters()).dtype
    return tuple(projector(item.to(dtype)) for item in features.split(sizes))
