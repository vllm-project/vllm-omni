# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""NPU-friendly replacements for the MiniMax-H3 audio VAE's transposed convolutions.

The audio VAE's BigVGAN decoder runs two families of ``conv_transpose1d`` on the
FP32 decode path:

* 7 full ``nn.ConvTranspose1d`` stage upsamplers (``BigVGAN.ups``), and
* 127 depthwise ``UpSample1d`` resamplers inside the anti-aliased
  ``Activation1d`` wrappers (up-sample -> SnakeBeta -> down-sample).

On Ascend both families dispatch to Conv3DTransposeV2, whose tiling is slow on
the FP32 streaming tails that the decoder produces (small channels over very
long sequences). Setting::

    VLLM_OMNI_MINIMAX_H3_AUDIO_TCONV_CONV1D=1

patches both families, at module-graph installation time, into a
phase-decomposed ``F.conv1d`` evaluation that is algebraically identical to the
transposed convolution: the kernel is split into per-phase taps, one grouped
convolution produces every phase stream at once, and an interleave restores the
time axis -- no zero insertion, no approximation. Each patched forward keeps the
original implementation for anything that is not (NPU, FP32), so CPU/GPU
semantics are untouched.
"""

from __future__ import annotations

import os
import types
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

logger = init_logger(__name__)

AUDIO_TCONV_CONV1D_ENV = "VLLM_OMNI_MINIMAX_H3_AUDIO_TCONV_CONV1D"


def _tconv_enabled_on_input(x: torch.Tensor) -> bool:
    """The replacement is only validated for NPU FP32; everything else stays native."""
    return x.device.type == "npu" and x.dtype == torch.float32


def _conv_transpose1d_as_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: int,
    padding: int,
) -> torch.Tensor:
    """Evaluate a strided full transposed conv as one grouped ``F.conv1d``.

    ``weight`` is the ``nn.ConvTranspose1d`` kernel ``(C_in, C_out, K)``. Each
    output phase ``t mod stride`` of the transposed convolution is an ordinary
    convolution against one phase of the kernel, so all ``stride`` phases can be
    produced at once by folding them into the conv output channels and
    interleaving them back onto the time axis. When ``K`` is not a multiple of
    ``stride`` the kernel is zero-padded; the extra taps only ever contribute to
    samples that the ``padding``-based crop removes.
    """
    batch, _, length = x.shape
    kernel_size = weight.shape[-1]
    if kernel_size % stride:
        weight = F.pad(weight, (0, -kernel_size % stride))
    phase_taps = weight.shape[-1] // stride
    in_channels, out_channels = weight.shape[0], weight.shape[1]
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
    if bias is not None:
        out = out + bias.reshape(1, -1, 1)
    return out


def _depthwise_upsample_as_conv1d(
    x: torch.Tensor,
    filter_: torch.Tensor,
    stride: int,
    pad: int,
    pad_left: int,
    pad_right: int,
    ratio: float,
) -> torch.Tensor:
    """Depthwise variant of the phase decomposition for the DAC ``UpSample1d``.

    Matches the original module exactly: replicate-pad, kaiser filter expanded
    per channel (no bias), ``ratio`` gain, and the original edge crops. Only the
    inner ``F.conv_transpose1d`` is rewritten as a grouped ``F.conv1d`` over the
    per-phase filter taps.
    """
    batch, channels, _ = x.shape
    kernel = filter_.expand(channels, -1, -1) if filter_.shape[0] == 1 else filter_
    kernel_size = kernel.shape[-1]
    if kernel_size % stride:
        kernel = F.pad(kernel, (0, -kernel_size % stride))
    phase_taps = kernel.shape[-1] // stride
    phase_weight = (
        kernel.reshape(channels, 1, phase_taps, stride)
        .permute(0, 3, 1, 2)
        .flip(-1)
        .reshape(channels * stride, 1, phase_taps)
    )
    x = F.pad(x, (pad, pad), mode="replicate")
    out = F.conv1d(x, phase_weight, padding=phase_taps - 1, groups=channels)
    out = out.reshape(batch, channels, stride, -1).transpose(2, 3).reshape(batch, channels, -1)
    out = out * ratio
    return out[..., pad_left : out.shape[-1] - pad_right]


def _patched_conv_transpose_forward(
    self: nn.ConvTranspose1d, x: torch.Tensor, output_size: list[int] | None = None
) -> torch.Tensor:
    del output_size
    if _tconv_enabled_on_input(x) and self.groups == 1 and self.dilation[0] == 1:
        return _conv_transpose1d_as_conv1d(x, self.weight, self.bias, self.stride[0], self.padding[0])
    return F.conv_transpose1d(
        x,
        self.weight,
        self.bias,
        self.stride,
        self.padding,
        self.output_padding,
        self.groups,
        self.dilation,
    )


def _patched_upsample1d_forward(self: Any, x: torch.Tensor) -> torch.Tensor:
    if _tconv_enabled_on_input(x):
        return _depthwise_upsample_as_conv1d(
            x,
            self.filter,
            self.stride,
            self.pad,
            self.pad_left,
            self.pad_right,
            self.ratio,
        )
    channels = x.shape[1]
    out = F.pad(x, (self.pad, self.pad), mode="replicate")
    out = self.ratio * F.conv_transpose1d(
        out, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels
    )
    return out[..., self.pad_left : out.shape[-1] - self.pad_right]


def _is_remote_upsample1d(module: nn.Module) -> bool:
    """Identify the checkpoint's ``UpSample1d`` without importing remote code."""
    return (
        type(module).__name__ == "UpSample1d"
        and isinstance(getattr(module, "filter", None), torch.Tensor)
        and getattr(module, "filter").dim() == 3
        and hasattr(module, "pad_left")
        and hasattr(module, "ratio")
    )


def _audio_tconv_env_enabled() -> bool:
    return os.environ.get(AUDIO_TCONV_CONV1D_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def install_audio_vae_tconv_conv1d(model: nn.Module) -> tuple[int, int]:
    """Patch the audio VAE graph when the env opt-in is set.

    Returns ``(upsample_patched, transpose_patched)``; ``(0, 0)`` when disabled.
    """
    if not _audio_tconv_env_enabled():
        return 0, 0
    upsample_patched = transpose_patched = 0
    for module in model.modules():
        if isinstance(module, nn.ConvTranspose1d):
            module.forward = types.MethodType(_patched_conv_transpose_forward, module)
            transpose_patched += 1
        elif _is_remote_upsample1d(module):
            module.forward = types.MethodType(_patched_upsample1d_forward, module)
            upsample_patched += 1
    logger.info(
        "MiniMax-H3 audio VAE transposed-conv replacement installed "
        "(%s=%s): %d stage ConvTranspose1d + %d UpSample1d resamplers now "
        "evaluate as phase-decomposed conv1d on NPU FP32",
        AUDIO_TCONV_CONV1D_ENV,
        os.environ.get(AUDIO_TCONV_CONV1D_ENV),
        transpose_patched,
        upsample_patched,
    )
    return upsample_patched, transpose_patched


__all__ = [
    "AUDIO_TCONV_CONV1D_ENV",
    "install_audio_vae_tconv_conv1d",
]
