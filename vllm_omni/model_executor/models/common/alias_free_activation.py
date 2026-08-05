# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Alias-free activation for BigVGAN-style speech decoders.

Provides anti-aliased activation (upsample → activation → downsample) with
Kaiser-windowed sinc filters.  Includes NPU bf16 workarounds for platforms
where F.pad(mode='replicate') does not support bfloat16.

Used by: Qwen2.5-Omni, Qwen3-TTS v1, CoVo-Audio, and future BigVGAN vocoders.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    """Generate a 1-D Kaiser-windowed sinc low-pass filter.

    Args:
        cutoff: Normalized cutoff frequency (0 to 0.5).
        half_width: Transition bandwidth.
        kernel_size: Number of filter taps.

    Returns:
        Filter tensor of shape ``(1, 1, kernel_size)``.
    """
    is_even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    delta_f = 4 * half_width
    attenuation = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95

    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0

    # torch.kaiser_window is not supported on NPU / XPU — compute on CPU.
    if current_omni_platform.is_npu():
        kaiser_window = torch.kaiser_window(
            kernel_size, beta=beta, periodic=False, dtype=torch.float32, device="cpu"
        ).to("npu")
    elif current_omni_platform.is_xpu():
        kaiser_window = torch.kaiser_window(
            kernel_size, beta=beta, periodic=False, dtype=torch.float32, device="cpu"
        ).to("xpu")
    else:
        kaiser_window = torch.kaiser_window(kernel_size, beta=beta, periodic=False, dtype=torch.float32)

    if is_even:
        time_indices = torch.arange(-half_size, half_size) + 0.5
    else:
        time_indices = torch.arange(kernel_size) - half_size

    if cutoff == 0:
        return torch.zeros((1, 1, kernel_size), dtype=torch.float32)

    sinc_filter = torch.sinc(2 * cutoff * time_indices)
    normalized_filter = 2 * cutoff * kaiser_window * sinc_filter
    normalized_filter /= normalized_filter.sum()

    return normalized_filter.view(1, 1, kernel_size)


def replication_pad_1d(hidden_states: torch.Tensor, pad_left: int, pad_right: int) -> torch.Tensor:
    """Manual replicate padding for NPU bf16 compatibility.

    ``F.pad(mode='replicate')`` does not support bfloat16 on NPU (and older
    CUDA).  This expand+cat fallback produces identical results.

    TODO: Remove when upstream PyTorch fixes replication_pad1d for bf16.
    """
    if pad_left == 0 and pad_right == 0:
        return hidden_states

    segments = []
    if pad_left > 0:
        left = hidden_states[..., :1].expand(*hidden_states.shape[:-1], pad_left)
        segments.append(left)

    segments.append(hidden_states)

    if pad_right > 0:
        right = hidden_states[..., -1:].expand(*hidden_states.shape[:-1], pad_right)
        segments.append(right)

    return torch.cat(segments, dim=-1)


# ---------------------------------------------------------------------------
# Up / Down sampling with Kaiser sinc filters
# ---------------------------------------------------------------------------


class UpSample1d(nn.Module):
    """Kaiser-sinc interpolation upsample (default: 2x, 12-tap filter)."""

    def __init__(self, ratio: int = 2, kernel_size: int | None = None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2

        filt = kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size)
        self.register_buffer("filter", filt, persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        channels = hidden_states.shape[1]
        if current_omni_platform.is_npu():
            input_dtype = hidden_states.dtype
            hidden_states = replication_pad_1d(hidden_states.to(self.filter.dtype), self.pad, self.pad)
            hidden_states = self.ratio * F.conv_transpose1d(
                hidden_states,
                self.filter.expand(channels, -1, -1),
                stride=self.stride,
                groups=channels,
            ).to(input_dtype)
        else:
            input_dtype = hidden_states.dtype
            hidden_states = F.pad(hidden_states, (self.pad, self.pad), mode="replicate").to(self.filter.dtype)
            hidden_states = self.ratio * F.conv_transpose1d(
                hidden_states,
                self.filter.expand(channels, -1, -1),
                stride=self.stride,
                groups=channels,
            ).to(input_dtype)
        hidden_states = hidden_states[..., self.pad_left : -self.pad_right]
        return hidden_states


class DownSample1d(nn.Module):
    """Kaiser-sinc anti-aliasing downsample (default: 2x, 12-tap filter)."""

    def __init__(self, ratio: int = 2, kernel_size: int | None = None):
        super().__init__()
        if kernel_size is None:
            kernel_size = int(6 * ratio // 2) * 2

        cutoff = 0.5 / ratio
        half_width = 0.6 / ratio

        if cutoff < 0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")

        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = ratio

        filt = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer("filter", filt, persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        channels = hidden_states.shape[1]
        if current_omni_platform.is_npu():
            input_dtype = hidden_states.dtype
            hidden_states = replication_pad_1d(hidden_states.to(self.filter.dtype), self.pad_left, self.pad_right)
            out = F.conv1d(
                hidden_states,
                self.filter.to(device=hidden_states.device, dtype=hidden_states.dtype).expand(channels, -1, -1),
                stride=self.stride,
                groups=channels,
            ).to(input_dtype)
        else:
            input_dtype = hidden_states.dtype
            hidden_states = F.pad(hidden_states, (self.pad_left, self.pad_right), mode="replicate").to(
                self.filter.dtype
            )
            out = F.conv1d(
                hidden_states,
                self.filter.expand(channels, -1, -1),
                stride=self.stride,
                groups=channels,
            ).to(input_dtype)
        return out


# ---------------------------------------------------------------------------
# Alias-free activation: upsample → activation → downsample
# ---------------------------------------------------------------------------


class AliasFreeActivation1d(nn.Module):
    """Anti-aliased activation following BigVGAN (ICLR 2023).

    Upsamples the signal, applies a pointwise activation (typically SnakeBeta),
    then downsamples with a low-pass filter to suppress aliasing from the
    high-frequency components introduced by the activation function.

    The ``activation`` should be an ``nn.Module`` instance — typically
    ``SnakeBeta`` from ``common.snake_activation``, which already dispatches
    between Triton and eager PyTorch internally.

    Args:
        activation: Pointwise activation module (e.g. ``SnakeBeta(channels)``).
        up_ratio: Upsample factor before activation.
        down_ratio: Downsample factor after activation.
        up_kernel_size: Kaiser sinc filter taps for upsampling.
        down_kernel_size: Kaiser sinc filter taps for downsampling.
    """

    def __init__(
        self,
        activation: nn.Module,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()
        if not callable(activation):
            raise TypeError("Activation function must be callable")
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.upsample(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.downsample(hidden_states)
        return hidden_states
