# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Copyright (c) 2022 OpenAI
#
# Shared Whisper encoder primitives used by multiple model implementations.
# Originally from the OpenAI Whisper codebase.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding."""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class Conv1d(nn.Conv1d):
    """Conv1d with automatic dtype casting for mixed precision inference."""

    def _conv_forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        return super()._conv_forward(x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype))


class ConvTranspose1d(nn.ConvTranspose1d):
    def _conv_forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        return super()._conv_forward(x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype))


class Linear(nn.Linear):
    """Linear layer with automatic dtype casting for mixed precision inference."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))
