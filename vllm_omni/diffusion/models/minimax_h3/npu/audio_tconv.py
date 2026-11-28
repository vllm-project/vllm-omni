# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Polyphase convolution for the BigVGAN stage upsamplers (PR #8813)."""

import types
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

BIGVGAN_MIN_POLYPHASE_LATENT_FRAMES = 128


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


def _stage_forward(
    self: nn.ConvTranspose1d,
    x: torch.Tensor,
    output_size: list[int] | None = None,
    *,
    min_input_samples: int,
) -> torch.Tensor:
    """Use the phase path only for the validated long FP32 NPU inference regime."""
    if (
        x.device.type == "npu"
        and x.dtype == torch.float32
        and x.ndim == 3
        and x.shape[-1] >= min_input_samples
        and x.numel()
        and not self.training
        and output_size is None
    ):
        weight = self.weight
        if weight.dtype == x.dtype:
            output = _conv_transpose1d_as_conv1d(x, weight, self.bias, self.stride[0], self.padding[0])
            self._h3_stage_polyphase_calls = getattr(self, "_h3_stage_polyphase_calls", 0) + 1
            return output
    return self._h3_stage_original_forward(x, output_size)


def install_bigvgan_upsamplers(decoder: nn.Module) -> int:
    """Wrap stage convolutions, mapping the latent threshold into each stage's input length."""
    installed = 0
    min_input_samples = BIGVGAN_MIN_POLYPHASE_LATENT_FRAMES
    for stage in getattr(decoder, "ups", ()):
        for layer in stage:
            if not isinstance(layer, nn.ConvTranspose1d):
                continue
            stride, padding, kernel, dilation, output_padding = (
                layer.stride[0],
                layer.padding[0],
                layer.kernel_size[0],
                layer.dilation[0],
                layer.output_padding[0],
            )
            min_output_samples = (
                (min_input_samples - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1
            )
            if (
                not hasattr(layer, "_h3_stage_original_forward")
                and layer.groups == 1
                and dilation == 1
                and output_padding == 0
                and min_output_samples > 0
            ):
                layer._h3_stage_original_forward = layer.forward
                layer.forward = types.MethodType(partial(_stage_forward, min_input_samples=min_input_samples), layer)
                installed += 1
            min_input_samples = min_output_samples
    return installed
