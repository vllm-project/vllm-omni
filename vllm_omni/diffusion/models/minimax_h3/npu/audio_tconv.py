# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Polyphase convolution for the BigVGAN stage upsamplers (PR #8813)."""

import types

import torch
from torch import nn
from torch.nn import functional as F


def _conv_transpose1d_as_conv1d(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, stride: int, padding: int
) -> torch.Tensor:
    """Compute the output phases as channels, then interleave them along time."""
    batch, _, length = x.shape
    in_channels, out_channels, kernel_size = weight.shape
    if kernel_size % stride:
        weight = F.pad(weight, (0, -kernel_size % stride))
    phase_taps = weight.shape[-1] // stride
    phase_weight = (
        weight.reshape(in_channels, out_channels, phase_taps, stride)
        .permute(1, 3, 0, 2)
        .flip(-1)
        .reshape(out_channels * stride, in_channels, phase_taps)
    )
    out = F.conv1d(x, phase_weight, padding=phase_taps - 1)
    out = out.reshape(batch, out_channels, stride, -1).transpose(2, 3).reshape(batch, out_channels, -1)
    output_length = (length - 1) * stride - 2 * padding + kernel_size
    out = out[..., padding : padding + output_length]
    return out if bias is None else out + bias.reshape(1, -1, 1)


def _stage_forward(self: nn.ConvTranspose1d, x: torch.Tensor, output_size: list[int] | None = None) -> torch.Tensor:
    """Run the phase-decomposed convolution for batched FP32 NPU inference."""
    if (
        x.device.type == "npu"
        and x.dtype == torch.float32
        and x.ndim == 3
        and not self.training
        and output_size is None
    ):
        weight = self.weight
        if (
            weight.dtype == x.dtype
            and x.numel()
            and (x.shape[-1] - 1) * self.stride[0] + weight.shape[-1] > 2 * self.padding[0]
        ):
            output = _conv_transpose1d_as_conv1d(x, weight, self.bias, self.stride[0], self.padding[0])
            self._h3_stage_polyphase_calls = getattr(self, "_h3_stage_polyphase_calls", 0) + 1
            return output
    return self._h3_stage_original_forward(x, output_size)


def install_bigvgan_upsamplers(decoder: nn.Module) -> int:
    """Replace only the stage upsamplers, not the FIR resamplers in AMP blocks."""
    installed = 0
    for stage in getattr(decoder, "ups", ()):
        for layer in stage:
            if (
                isinstance(layer, nn.ConvTranspose1d)
                and not hasattr(layer, "_h3_stage_original_forward")
                and layer.groups == 1
                and layer.dilation == (1,)
                and layer.output_padding == (0,)
            ):
                layer._h3_stage_original_forward = layer.forward
                layer.forward = types.MethodType(_stage_forward, layer)
                installed += 1
    return installed
