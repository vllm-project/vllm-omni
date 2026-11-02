# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ming-3 audio projector with checkpoint-compatible module names."""

from collections.abc import Iterable, Sequence

import torch
import torch.nn as nn
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from .audio_lengths import apply_conv_stack


class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(self.dim0, self.dim1)


class Ming3AudioProjector(nn.Module):
    """Downsample Whisper features into the Bailing text hidden space."""

    def __init__(
        self,
        audio_dim: int,
        llm_dim: int,
        kernels: Sequence[int],
        strides: Sequence[int],
        paddings: Sequence[int],
    ) -> None:
        super().__init__()
        if not (len(kernels) == len(strides) == len(paddings)):
            raise ValueError("Audio projector convolution lists must have equal length")

        layers: list[nn.Module] = []
        input_dim = audio_dim
        for kernel, stride, padding in zip(kernels, strides, paddings, strict=True):
            layers.extend(
                [
                    nn.Conv1d(
                        input_dim,
                        llm_dim,
                        kernel_size=kernel,
                        stride=stride,
                        padding=padding,
                    ),
                    nn.GELU(),
                ]
            )
            input_dim = llm_dim
        layers.extend([Transpose(-1, -2), nn.Linear(llm_dim, llm_dim)])
        self.proj = nn.Sequential(*layers)
        self.kernels = tuple(kernels)
        self.strides = tuple(strides)
        self.paddings = tuple(paddings)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """Project [batch, time, audio_dim] to [batch, tokens, llm_dim]."""
        return self.proj(encoded.transpose(-1, -2))

    def output_length(self, encoded_length: int) -> int:
        return apply_conv_stack(encoded_length, self.kernels, self.strides, self.paddings)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if not name.startswith("proj."):
                name = f"proj.{name}"
            param = params_dict.get(name)
            if param is None:
                continue
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
