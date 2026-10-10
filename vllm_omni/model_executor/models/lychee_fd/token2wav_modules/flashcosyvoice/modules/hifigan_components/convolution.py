# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""HiFT-owned deterministic cuDNN convolution dispatch without global flags."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _use_local_cudnn(inputs: torch.Tensor) -> bool:
    # Preserve Torch's default implementation outside the released CUDA/cuDNN
    # synthesis path, including its decomposition for complex convolutions.
    return inputs.is_cuda and torch.backends.cudnn.enabled and not inputs.is_complex()


def _deterministic_convolution(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    stride,
    padding,
    dilation,
    groups: int,
    transposed: bool = False,
    output_padding=(0,),
) -> torch.Tensor:
    unbatched = inputs.ndim == 2
    if unbatched:
        inputs = inputs.unsqueeze(0)
    result = torch.ops.aten._convolution.default(
        inputs,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        False,
        True,
        torch.backends.cudnn.enabled,
        torch.backends.cudnn.allow_tf32,
    )
    return result.squeeze(0) if unbatched else result


class Conv1d(nn.Conv1d):
    """Keep nn.Conv1d state and padding semantics with local cuDNN determinism."""

    def _conv_forward(self, inputs: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        if not _use_local_cudnn(inputs):
            return super()._conv_forward(inputs, weight, bias)
        return self._deterministic_conv_forward(inputs, weight, bias)

    def _deterministic_conv_forward(
        self, inputs: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        padding = self.padding
        if self.padding_mode != "zeros":
            inputs = F.pad(inputs, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = (0,)
        elif isinstance(padding, str):
            if padding == "same":
                inputs = F.pad(inputs, self._reversed_padding_repeated_twice)
            padding = (0,)
        return _deterministic_convolution(
            inputs, weight, bias, stride=self.stride, padding=padding, dilation=self.dilation, groups=self.groups
        )


class ConvTranspose1d(nn.ConvTranspose1d):
    """Keep nn.ConvTranspose1d output_size semantics with local cuDNN determinism."""

    def forward(self, inputs: torch.Tensor, output_size: list[int] | None = None) -> torch.Tensor:
        if not _use_local_cudnn(inputs):
            return super().forward(inputs, output_size)
        return self._deterministic_forward(inputs, output_size)

    def _deterministic_forward(self, inputs: torch.Tensor, output_size: list[int] | None = None) -> torch.Tensor:
        if self.padding_mode != "zeros":
            raise ValueError("Only `zeros` padding mode is supported for ConvTranspose1d")
        output_padding = self._output_padding(
            inputs, output_size, self.stride, self.padding, self.kernel_size, 1, self.dilation
        )
        return _deterministic_convolution(
            inputs,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            transposed=True,
            output_padding=output_padding,
        )


__all__ = ["Conv1d", "ConvTranspose1d"]
