# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The vLLM-Omni team.
# Copyright (c) rednote-hilab. All rights reserved.
# Adapted from:
# https://github.com/rednote-hilab/dots.tts/tree/a393d2e/src/dots_tts/modules
#
# Merged from six upstream files (preserved as-is, inference behaviour
# unchanged; only internal imports were folded inline):
#   - modules/vocoder/config.py             -> AudioVAEConfig
#   - modules/backbone/layers.py            -> Conv1d, ConvTranspose1d
#                                              (Mlp/MultiHeadAttention/RotaryEmbedding
#                                              are not vendored here; they belong
#                                              to the DiT side-path module added later)
#   - modules/vocoder/alias_free_filter.py  -> sinc, kaiser_sinc_filter1d,
#                                              LowPassFilter1d
#   - modules/vocoder/alias_free_resample.py-> UpSample1d, DownSample1d
#   - modules/vocoder/alias_free_act.py     -> Snake, SnakeBeta, Activation1d
#   - modules/vocoder/bigvgan.py            -> AudioVAE and all its sub-modules
#
# Per-section upstream attributions are preserved at each section header.
# AudioVAEConfig is the only piece that diverges from upstream: pydantic
# ConfigBase replaced with a stdlib @dataclass (extra fields not exercised by
# any shipped checkpoint).

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import pow, sin
from torch.nn import Parameter
from torch.nn.utils import remove_weight_norm, weight_norm

# ============================================================================
# Configuration (adapted from modules/vocoder/config.py)
# ----------------------------------------------------------------------------
# Upstream subclasses a pydantic ``ConfigBase`` with ``extra="allow"``; both the
# dots.tts-soar and dots.tts-mf checkpoint ``config.json`` pass only the 17
# declared fields below, so the escape hatch is never exercised.  ``.get()``
# helper preserves bigvgan.py's ``h.get(...)`` call sites without modification.
# ``num_decoder_lookahead`` / ``num_encoder_lookahead`` are looked up via
# ``h.get(...)`` but not present in any shipped config; they fall back to 2.
# ============================================================================


@dataclass
class AudioVAEConfig:
    sample_rate: int = 24000
    upsample_rates: list[int] = field(default_factory=list)
    upsample_kernel_sizes: list[int] = field(default_factory=list)
    upsample_initial_channel: int = 1536
    resblock: str = "1"
    resblock_kernel_sizes: list[int] = field(default_factory=list)
    resblock_dilation_sizes: list[list[int]] = field(default_factory=list)
    downsample_rates: list[int] = field(default_factory=list)
    downsample_channels: list[int] = field(default_factory=list)
    activation: str = "snakebeta"
    snake_logscale: bool = True
    latent_dim: int = 128
    causal: bool = False
    mi_num_layers: int = 4
    causal_encoder: bool = False
    use_bias_at_final: bool = True
    use_tanh_at_final: bool = True

    num_decoder_lookahead: int = 2
    num_encoder_lookahead: int = 2

    def get(self, key: str, default=None):
        return getattr(self, key, default)


# ============================================================================
# Conv layers (vendored from modules/backbone/layers.py)
# ----------------------------------------------------------------------------
# Only the Conv1d / ConvTranspose1d helpers are needed by the AudioVAE.
# Mlp / MultiHeadAttention / RotaryEmbedding from the same upstream file are
# reserved for the DiT flow-matching head, which gets its own vendored module
# in a later step.
# ============================================================================


class Conv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        bias: bool = True,
        padding=None,
        causal: bool = False,
        **_kwargs,
    ):
        self.causal = causal
        if padding is None:
            if causal:
                padding = 0
                self.left_padding = dilation * (kernel_size - 1)
            else:
                padding = int((kernel_size * dilation - dilation) / 2)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
            bias=bias,
        )

        self.in_channels = in_channels

    def forward(self, x):
        if self.causal:
            x = F.pad(x.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)
        return super().forward(x)


class ConvTranspose1d(nn.ConvTranspose1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding=None,
        padding_mode: str = "zeros",
        causal: bool = False,
        **_kwargs,
    ):
        if padding is None:
            padding = 0 if causal else (kernel_size - stride) // 2
        if causal:
            assert padding == 0, "padding is not allowed in causal ConvTranspose1d."
            assert kernel_size == 2 * stride, "kernel_size must be equal to 2*stride in Causal ConvTranspose1d."

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
        )

        self.causal = causal
        self.stride = stride

    def forward(self, x):
        x = super().forward(x)
        if self.causal:
            x = x[:, :, : -self.stride]
        return x


# ============================================================================
# Alias-free low-pass filter (vendored from modules/vocoder/alias_free_filter.py)
# ----------------------------------------------------------------------------
# Adapted from https://github.com/junjun3518/alias-free-torch under
# the Apache License 2.0.  The sinc fallback is adopted from adefossez's
# julius.core.sinc (MIT).
# ============================================================================


if "sinc" in dir(torch):
    sinc = torch.sinc
else:

    def sinc(x: torch.Tensor):
        """
        Implementation of sinc, i.e. sin(pi * x) / (pi * x)
        __Warning__: Different to julius.sinc, the input is multiplied by `pi`!
        """
        return torch.where(
            x == 0,
            torch.tensor(1.0, device=x.device, dtype=x.dtype),
            torch.sin(math.pi * x) / math.pi / x,
        )


def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):  # return filter [1,1,kernel_size]
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    # For kaiser window
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.0:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.0:
        beta = 0.5842 * (A - 21) ** 0.4 + 0.07886 * (A - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    # ratio = 0.5/cutoff -> 2 * cutoff = 1 / ratio
    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * sinc(2 * cutoff * time)
        # Normalize filter to have sum = 1, otherwise we will have a small leakage
        # of the constant component in the input signal.
        filter_ /= filter_.sum()
        filter = filter_.view(1, 1, kernel_size)

    return filter


class LowPassFilter1d(nn.Module):
    def __init__(
        self,
        cutoff=0.5,
        half_width=0.6,
        stride: int = 1,
        padding: bool = True,
        padding_mode: str = "replicate",
        kernel_size: int = 12,
        channels: int = 1,
        causal: bool = True,
        fixed_filter: bool = False,
    ):
        # kernel_size should be even number for stylegan3 setup,
        # in this implementation, odd number is also possible.
        super().__init__()
        if cutoff < -0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.kernel_size = kernel_size
        if causal:
            self.pad_left = kernel_size - 1
            self.pad_right = 0
        else:
            self.even = kernel_size % 2 == 0
            self.pad_left = kernel_size // 2 - int(self.even)
            self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.fixed_filter = fixed_filter
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        if fixed_filter:
            self.register_buffer("filter", filter)
        else:
            self.filter = nn.Parameter(filter.expand(channels, -1, -1).clone())

    # input [B, C, T]
    def forward(self, x):
        _, C, _ = x.shape
        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        if self.fixed_filter:
            out = F.conv1d(x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)
        else:
            out = F.conv1d(x, self.filter, stride=self.stride, groups=C)
        return out


# ============================================================================
# Alias-free up/down sampling (vendored from modules/vocoder/alias_free_resample.py)
# ----------------------------------------------------------------------------
# Adapted from https://github.com/junjun3518/alias-free-torch (Apache 2.0).
# ============================================================================


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None, channels=None, causal=True, fixed_filter=False):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.channels = channels
        self.causal = causal
        self.fixed_filter = fixed_filter
        if causal:
            self.pad = 0
        else:
            self.pad = self.kernel_size // ratio - 1
            self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
            self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        filter = kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size)
        if self.fixed_filter:
            self.register_buffer("filter", filter)
        else:
            self.filter = nn.Parameter(filter.expand(channels, -1, -1).clone())

    # x: [B, C, T]
    def forward(self, x):
        _, C, _ = x.shape
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        if self.fixed_filter:
            x = self.ratio * F.conv_transpose1d(x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)
        else:
            x = self.ratio * F.conv_transpose1d(x, self.filter, stride=self.stride, groups=C)
        if self.causal:
            x = x[..., : -(self.kernel_size - self.stride)]
        else:
            x = x[..., self.pad_left : -self.pad_right]

        return x


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None, channels=None, causal=True, fixed_filter=False):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
            channels=channels,
            causal=causal,
            fixed_filter=fixed_filter,
        )

    def forward(self, x):
        return self.lowpass(x)


# ============================================================================
# Snake activations + alias-free Activation1d (vendored from
# modules/vocoder/alias_free_act.py)
# ----------------------------------------------------------------------------
# Adapted from https://github.com/junjun3518/alias-free-torch (Apache 2.0).
# Snake / SnakeBeta paper: https://arxiv.org/abs/2006.08195
# ============================================================================


class Activation1d(nn.Module):
    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
        causal=True,
        fixed_filter=False,
    ):
        super().__init__()
        # causal=False
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(
            up_ratio,
            up_kernel_size,
            activation.in_features,
            causal=causal,
            fixed_filter=fixed_filter,
        )
        self.downsample = DownSample1d(
            down_ratio,
            down_kernel_size,
            activation.in_features,
            causal=causal,
            fixed_filter=fixed_filter,
        )

    # x: [B,C,T]
    def forward(self, x):
        x = self.upsample(x)
        x = self.act(x)
        return self.downsample(x)


class Snake(nn.Module):
    """
    Implementation of a sine-based periodic activation function
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter
    """

    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        super().__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        # Snake := x + 1/a * sin^2 (xa)
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        return x + (1.0 / (alpha + self.no_div_by_zero)) * pow(sin(x * alpha), 2)


class SnakeBeta(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude
    of the periodic components.
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    """

    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        super().__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        # SnakeBeta := x + 1/b * sin^2 (xa)
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)


# ============================================================================
# BigVGAN AudioVAE (vendored from modules/vocoder/bigvgan.py)
# ----------------------------------------------------------------------------
# Copyright (c) 2022 NVIDIA CORPORATION.
#   Licensed under the MIT license.
# Upstream layout flattened: the only modification is that the three
# original ``from dots_tts.modules.{backbone,vocoder}.* import ...`` lines
# are dropped, since those names are already defined above in this file.
# ============================================================================


@dataclass(slots=True)
class BigVGANStreamState:
    lstm_hidden: tuple[torch.Tensor, torch.Tensor]
    decoder: DecoderStreamState


@dataclass(slots=True)
class DecoderStreamState:
    window: torch.Tensor
    chunk_size: int
    total_frames: int = 0
    emitted_frames: int = 0


def _empty_chunk(
    ref: torch.Tensor,
    *,
    channels: int | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return ref.new_zeros(
        (ref.size(0), channels or ref.size(1), 0),
        dtype=dtype or ref.dtype,
    )


def _module_state_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    for name in ("weight", "bias", "filter"):
        tensor = getattr(module, name, None)
        if tensor is not None:
            return tensor.device, tensor.dtype
    for tensor in itertools.chain(module.parameters(), module.buffers()):
        return tensor.device, tensor.dtype
    raise RuntimeError(f"Unable to infer state dtype/device for {type(module).__name__}.")


def _stream_state_zeros(
    batch_size: int,
    channels: int,
    length: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.zeros(
        (batch_size, channels, max(0, int(length))),
        device=device,
        dtype=dtype,
    )


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


class Conv1d_S(nn.Module):
    "Conv1d for spectral normalisation and orthogonal initialisation"

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        dilation=1,
        groups=1,
        causal=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.causal = causal
        pad = 0 if causal else dilation * (kernel_size - 1) // 2
        self.causal_pad = dilation * (kernel_size - 1) if causal else 0

        self.layer = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
                dilation=dilation,
                groups=groups,
            )
        )

    def forward(self, inputs):
        if self.causal and self.causal_pad > 0:
            inputs = F.pad(inputs, (self.causal_pad, 0))
        return self.layer(inputs)


class SLSTM(nn.Module):
    """
    LSTM without worrying about the hidden state, nor the layout of the data.
    Expects input as convolutional layout.
    """

    def __init__(
        self,
        dimension: int,
        num_layers: int = 2,
        skip: bool = True,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.skip = skip
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=dimension,
            hidden_size=dimension,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self._stream_num_layers = num_layers
        self._stream_weight_ih = tuple(getattr(self.lstm, f"weight_ih_l{layer_idx}") for layer_idx in range(num_layers))
        self._stream_weight_hh = tuple(getattr(self.lstm, f"weight_hh_l{layer_idx}") for layer_idx in range(num_layers))
        self._stream_bias_ih = tuple(getattr(self.lstm, f"bias_ih_l{layer_idx}") for layer_idx in range(num_layers))
        self._stream_bias_hh = tuple(getattr(self.lstm, f"bias_hh_l{layer_idx}") for layer_idx in range(num_layers))
        if self.bidirectional:
            self.proj_out = nn.Linear(dimension * 2, dimension)

    def forward(self, x):
        y, _ = self.lstm(x)
        if self.bidirectional:
            y = self.proj_out(y)
        if self.skip:
            y = y + x
        return y

    def stream_step(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if self.bidirectional:
            raise RuntimeError("Streaming only supports unidirectional SLSTM.")

        residual = x
        if hidden is None:
            hidden = self.init_stream_state(x.size(0))

        hidden_h, hidden_c = hidden
        next_hidden_h = []
        next_hidden_c = []
        for layer_idx in range(self._stream_num_layers):
            layer_input = x
            hx = hidden_h[layer_idx]
            cx = hidden_c[layer_idx]
            outputs = []
            weight_ih = self._stream_weight_ih[layer_idx]
            weight_hh = self._stream_weight_hh[layer_idx]
            bias_ih = self._stream_bias_ih[layer_idx]
            bias_hh = self._stream_bias_hh[layer_idx]

            for frame_idx in range(x.size(1)):
                gates = F.linear(layer_input[:, frame_idx, :], weight_ih, bias_ih)
                gates = gates + F.linear(hx, weight_hh, bias_hh)
                input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, dim=-1)
                input_gate = torch.sigmoid(input_gate)
                forget_gate = torch.sigmoid(forget_gate)
                cell_gate = torch.tanh(cell_gate)
                output_gate = torch.sigmoid(output_gate)
                cx = forget_gate * cx + input_gate * cell_gate
                hx = output_gate * torch.tanh(cx)
                outputs.append(hx)

            x = torch.stack(outputs, dim=1)
            next_hidden_h.append(hx)
            next_hidden_c.append(cx)

        y = x
        if self.skip:
            y = y + residual
        return y, (torch.stack(next_hidden_h, dim=0), torch.stack(next_hidden_c, dim=0))

    def init_stream_state(
        self,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_directions = 2 if self.bidirectional else 1
        state_shape = (
            self.lstm.num_layers * num_directions,
            batch_size,
            self.lstm.hidden_size,
        )
        weight = self.lstm.weight_hh_l0
        return (
            weight.new_zeros(state_shape),
            weight.new_zeros(state_shape),
        )


class ResStack(nn.Module):
    def __init__(self, channel, kernel_size=3, base=3, nums=4, causal=False):
        super().__init__()

        self.layers = nn.ModuleList([])
        for i in range(nums):
            dil = base**i
            pad1 = dil * (kernel_size - 1) if causal else dil
            pad2 = (kernel_size - 1) if causal else 1
            block = [
                nn.LeakyReLU(),
            ]
            if causal and pad1 > 0:
                block.append(nn.ConstantPad1d((pad1, 0), 0.0))
            block.append(
                nn.utils.weight_norm(
                    nn.Conv1d(
                        channel,
                        channel,
                        kernel_size=kernel_size,
                        dilation=dil,
                        padding=0 if causal else pad1,
                    )
                )
            )
            block.append(nn.LeakyReLU())
            if causal and pad2 > 0:
                block.append(nn.ConstantPad1d((pad2, 0), 0.0))
            block.append(
                nn.utils.weight_norm(
                    nn.Conv1d(
                        channel,
                        channel,
                        kernel_size=kernel_size,
                        dilation=1,
                        padding=0 if causal else pad2,
                    )
                )
            )
            self.layers.append(nn.Sequential(*block))

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=100,
        base_channels=12,
        proj_kernel_size=3,
        stack_kernel_size=3,
        stack_dilation_base=2,
        stacks=6,
        channels=(12, 24, 48, 96, 192, 384, 768),
        down_sample_factors=(2, 2, 2, 2, 4, 4),
        causal=False,
        lookahead=0,
    ):
        super().__init__()

        act_slope = 0.2
        layers = []
        # pre proj_layer
        layers += [
            Conv1d_S(
                in_channels,
                base_channels,
                kernel_size=proj_kernel_size,
                stride=1,
                causal=causal,
            ),
            nn.LeakyReLU(act_slope, True),
        ]

        # channels: [512, 256, 128, 64], upsample_factors: [5, 2, 2]
        for (in_c, out_c), down_f in zip(itertools.pairwise(channels), down_sample_factors, strict=True):
            layers += [
                Conv1d_S(in_c, out_c, kernel_size=down_f * 2, stride=down_f, causal=causal),
                ResStack(out_c, stack_kernel_size, stack_dilation_base, stacks, causal=causal),
                nn.LeakyReLU(act_slope, True),
            ]

        # post layers
        if lookahead > 0:
            layers += [
                Conv1d_S(
                    channels[-1],
                    out_channels,
                    kernel_size=lookahead * 2 + 1,
                    stride=1,
                    causal=False,
                ),
            ]
        else:
            layers += [
                Conv1d_S(
                    channels[-1],
                    out_channels,
                    kernel_size=proj_kernel_size,
                    stride=1,
                    causal=causal,
                ),
            ]
        self.generator = nn.Sequential(*layers)

    def forward(self, conditions, _z_inputs=None):
        return self.generator(conditions)


class AMPBlock1(torch.nn.Module):
    def __init__(
        self,
        h,
        channels,
        kernel_size=3,
        dilation=(1, 3, 5),
        activation=None,
        causal=True,
    ):
        super().__init__()
        self.h = h

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        causal=causal,
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        causal=causal,
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        causal=causal,
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, causal=causal)),
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, causal=causal)),
                weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, causal=causal)),
            ]
        )
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(self.convs2)  # total number of conv layers

        if activation == "snake":  # periodic nonlinearity with snake function and anti-aliasing
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=Snake(channels, alpha_logscale=h.snake_logscale),
                        causal=causal,
                        fixed_filter=True,
                    )
                    for _ in range(self.num_layers)
                ]
            )
        elif activation == "snakebeta":  # periodic nonlinearity with snakebeta function and anti-aliasing
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=SnakeBeta(channels, alpha_logscale=h.snake_logscale),
                        causal=causal,
                        fixed_filter=True,
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2, strict=True):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for layer in self.convs1:
            remove_weight_norm(layer)
        for layer in self.convs2:
            remove_weight_norm(layer)


class AMPBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3), activation=None, causal=True):
        super().__init__()
        self.h = h

        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        causal=causal,
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        causal=causal,
                    )
                ),
            ]
        )
        self.convs.apply(init_weights)

        self.num_layers = len(self.convs)  # total number of conv layers

        if activation == "snake":  # periodic nonlinearity with snake function and anti-aliasing
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=Snake(channels, alpha_logscale=h.snake_logscale),
                        causal=causal,
                        fixed_filter=True,
                    )
                    for _ in range(self.num_layers)
                ]
            )
        elif activation == "snakebeta":  # periodic nonlinearity with snakebeta function and anti-aliasing
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=SnakeBeta(channels, alpha_logscale=h.snake_logscale),
                        causal=causal,
                        fixed_filter=True,
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        for c, a in zip(self.convs, self.activations, strict=True):
            xt = a(x)
            xt = c(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for layer in self.convs:
            remove_weight_norm(layer)


class Decoder(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        causal = h.causal
        self._stream_window_sizes: dict[int, int] = {}

        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        num_decoder_lookahead = h.get("num_decoder_lookahead", 2)
        # pre conv
        self.conv_pre = weight_norm(
            Conv1d(
                h.latent_dim,
                h.upsample_initial_channel,
                kernel_size=2 * num_decoder_lookahead + 1,
                stride=1,
                causal=False,
            )
        )

        # define which AMPBlock to use. BigVGAN uses AMPBlock1 as default
        resblock = AMPBlock1 if h.resblock == "1" else AMPBlock2

        # transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes, strict=True)):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            ConvTranspose1d(
                                h.upsample_initial_channel // (2**i),
                                h.upsample_initial_channel // (2 ** (i + 1)),
                                k,
                                u,
                                causal=causal,
                            )
                        )
                    ]
                )
            )

        # residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes, strict=True):
                self.resblocks.append(resblock(h, ch, k, d, activation=h.activation, causal=causal))

        # post conv
        if h.activation == "snake":  # periodic nonlinearity with snake function and anti-aliasing
            activation_post = Snake(ch, alpha_logscale=h.snake_logscale)
            self.activation_post = Activation1d(activation=activation_post, causal=causal, fixed_filter=False)
        elif h.activation == "snakebeta":  # periodic nonlinearity with snakebeta function and anti-aliasing
            activation_post = SnakeBeta(ch, alpha_logscale=h.snake_logscale)
            self.activation_post = Activation1d(activation=activation_post, causal=causal, fixed_filter=False)
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, causal=causal, bias=h.get("use_bias_at_final", True)))

        # weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, z):
        # pre conv
        x = self.conv_pre(z)

        for i in range(self.num_upsamples):
            # upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # post conv
        x = self.activation_post(x)
        x = self.conv_post(x)
        if self.h.get("use_tanh_at_final", True):
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1.0, max=1.0)  # Bound the output to [-1, 1]

        return x

    @property
    def stream_lookahead(self) -> int:
        return (self.conv_pre.kernel_size[0] - 1) // 2

    @staticmethod
    def _conv1d_left_context(layer) -> int:
        dilation = layer.dilation[0] if isinstance(layer.dilation, tuple) else layer.dilation
        kernel_size = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
        if getattr(layer, "causal", False):
            return dilation * (kernel_size - 1)
        return layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding

    @staticmethod
    def _convtranspose1d_left_context(layer) -> int:
        stride = layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride
        kernel_size = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
        if not getattr(layer, "causal", False):
            raise NotImplementedError("Streaming only supports causal ConvTranspose1d.")
        if kernel_size != 2 * stride:
            raise ValueError(
                "Streaming ConvTranspose1d expects kernel_size == 2 * stride, got "
                f"kernel_size={kernel_size} stride={stride}."
            )
        return 1

    @classmethod
    def _activation_left_context(cls, activation: Activation1d) -> int:
        upsample = activation.upsample
        downsample = activation.downsample.lowpass
        if not upsample.causal or not downsample.padding or downsample.pad_right != 0:
            raise NotImplementedError("Streaming only supports causal alias-free activations.")
        ratio = int(upsample.ratio)
        if ratio != int(downsample.stride):
            raise ValueError(
                "Alias-free activation expects matched up/down ratios, got "
                f"up_ratio={ratio} down_ratio={downsample.stride}."
            )
        total_left = (upsample.kernel_size - 1) + (downsample.kernel_size - 1)
        return (total_left + ratio - 1) // ratio

    @classmethod
    def _ampblock_left_context(cls, block) -> int:
        if isinstance(block, AMPBlock1):
            left_context = 0
            acts1 = block.activations[::2]
            acts2 = block.activations[1::2]
            for conv1, conv2, act1, act2 in zip(
                block.convs1,
                block.convs2,
                acts1,
                acts2,
                strict=True,
            ):
                left_context += (
                    cls._activation_left_context(act1)
                    + cls._conv1d_left_context(conv1)
                    + cls._activation_left_context(act2)
                    + cls._conv1d_left_context(conv2)
                )
            return left_context
        if isinstance(block, AMPBlock2):
            left_context = 0
            for conv, activation in zip(block.convs, block.activations, strict=True):
                left_context += cls._activation_left_context(activation) + cls._conv1d_left_context(conv)
            return left_context
        raise TypeError(f"Unsupported resblock type: {type(block).__name__}.")

    def _stream_left_context(self) -> int:
        left_context = Fraction(self._conv1d_left_context(self.conv_pre), 1)
        current_scale = Fraction(1, 1)
        for stage_idx, upsample_layers in enumerate(self.ups):
            for upsample in upsample_layers:
                left_context += current_scale * self._convtranspose1d_left_context(upsample)
                stride = upsample.stride[0] if isinstance(upsample.stride, tuple) else upsample.stride
                current_scale /= int(stride)

            stage_start = stage_idx * self.num_kernels
            stage_end = stage_start + self.num_kernels
            stage_context = max(self._ampblock_left_context(block) for block in self.resblocks[stage_start:stage_end])
            left_context += current_scale * stage_context

        left_context += current_scale * self._activation_left_context(self.activation_post)
        left_context += current_scale * self._conv1d_left_context(self.conv_post)
        return int(left_context.__ceil__())

    def stream_window_size(self, chunk_size: int) -> int:
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")
        cached = self._stream_window_sizes.get(chunk_size)
        if cached is not None:
            return cached

        window_size = chunk_size + self.stream_lookahead + self._stream_left_context()
        self._stream_window_sizes[chunk_size] = window_size
        return window_size

    def remove_weight_norm(self):
        for upsample_layers in self.ups:
            for upsample_layer in upsample_layers:
                remove_weight_norm(upsample_layer)
        for resblock in self.resblocks:
            resblock.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class AudioVAE(nn.Module):
    def __init__(self, h: AudioVAEConfig):
        super().__init__()
        self.config = h

        self.h = h
        self.hop_size = int(np.prod(h.downsample_rates))
        self.sample_rate = h.sample_rate
        self.decoder_lookahead = int(h.get("num_decoder_lookahead", 2))

        self.audio_encoder = Encoder(
            out_channels=h.latent_dim,
            down_sample_factors=h.downsample_rates,
            channels=h.downsample_channels,
            causal=h.causal_encoder,
            lookahead=h.get("num_encoder_lookahead", 2),
        )

        intermediate_size = h.latent_dim * 4
        self.enc_mi_layer = nn.Sequential(
            nn.Linear(h.latent_dim, intermediate_size),
            SLSTM(intermediate_size, num_layers=h.mi_num_layers),
            nn.Linear(intermediate_size, h.latent_dim),
        )
        self.dec_mi_layer = nn.Sequential(
            nn.Linear(h.latent_dim, intermediate_size),
            SLSTM(intermediate_size, num_layers=h.mi_num_layers),
            nn.Linear(intermediate_size, h.latent_dim),
        )
        self.pre_proj = Conv1d(
            in_channels=h.latent_dim,
            out_channels=h.latent_dim * 2,
            kernel_size=1,
            stride=1,
        )
        self.post_proj = Conv1d(in_channels=h.latent_dim, out_channels=h.latent_dim, kernel_size=1, stride=1)

        self.decoder = Decoder(h)

    def inference(self, data):
        latents = self.extract_latents(data["sample"])
        return {"sample": self.inference_from_latents(latents)}

    @torch.autocast(enabled=False, device_type="cuda")
    def extract_latents(self, x, do_sample=False):
        x = x.float()
        x = self.audio_encoder(x)
        x = x.permute(0, 2, 1)
        x = self.enc_mi_layer(x)
        x = x.permute(0, 2, 1)
        x = self.pre_proj(x)
        if do_sample:
            m_q, logs_q = torch.split(x, self.h.latent_dim, dim=1)
            x = m_q + torch.randn_like(m_q) * torch.exp(logs_q)
        return x

    def inference_from_latents(self, x, do_sample=True, noise_scale=1.0):
        if do_sample:
            assert x.size(1) == self.h.latent_dim * 2, f"Input must be like [B, D, H], got {x.shape}"
            m_q, logs_q = torch.split(x, self.h.latent_dim, dim=1)
            x = m_q + torch.randn_like(m_q) * torch.exp(logs_q) * noise_scale
        else:
            assert x.size(1) == self.h.latent_dim, f"Input must be like [B, D, H], got {x.shape}"
        x = self.post_proj(x)
        x = x.permute(0, 2, 1)
        x = self.dec_mi_layer(x)
        x = x.permute(0, 2, 1)
        return self.decoder(x)

    def _validate_stream_latents(self, latents: torch.Tensor) -> None:
        if latents.ndim != 3:
            raise ValueError(
                f"Streaming latents must have shape [batch, latent_dim, frames], got {tuple(latents.shape)}."
            )
        if latents.size(1) != self.h.latent_dim:
            raise ValueError(f"Streaming latent_dim must be {self.h.latent_dim}, got {latents.size(1)}.")

    def init_stream_state(
        self,
        batch_size: int = 1,
        chunk_size: int = 8,
    ) -> BigVGANStreamState:
        if not self.h.causal:
            raise RuntimeError("Strict streaming requires a causal vocoder.")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
        window_size = self.decoder.stream_window_size(chunk_size)
        device, dtype = _module_state_device_dtype(self.decoder.conv_pre)
        return BigVGANStreamState(
            lstm_hidden=self.dec_mi_layer[1].init_stream_state(batch_size),
            decoder=DecoderStreamState(
                window=_stream_state_zeros(
                    batch_size,
                    self.h.latent_dim,
                    window_size,
                    device=device,
                    dtype=dtype,
                ),
                chunk_size=int(chunk_size),
            ),
        )

    def _decode_stream_latents(
        self,
        latents: torch.Tensor,
        lstm_hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        self._validate_stream_latents(latents)
        latents = latents.float()
        x = self.post_proj(latents)
        x = x.permute(0, 2, 1)
        x = self.dec_mi_layer[0](x)
        x, lstm_hidden = self.dec_mi_layer[1].stream_step(x, lstm_hidden)
        x = self.dec_mi_layer[2](x)
        decoder_dtype = self.decoder.conv_pre.weight.dtype
        return x.permute(0, 2, 1).to(dtype=decoder_dtype), lstm_hidden

    def _prepare_stream_decoder_input(
        self,
        latents: torch.Tensor,
        state: BigVGANStreamState,
    ) -> torch.Tensor:
        decoder_input, state.lstm_hidden = self._decode_stream_latents(
            latents,
            state.lstm_hidden,
        )
        return decoder_input

    def _append_stream_decoder_input_tensor(
        self,
        decoder_input: torch.Tensor,
        window: torch.Tensor,
        valid_frames: torch.Tensor,
    ) -> torch.Tensor:
        if window.dtype != decoder_input.dtype:
            window = window.to(dtype=decoder_input.dtype)
        chunk_size = int(decoder_input.size(-1))
        if chunk_size >= window.size(-1):
            raise ValueError(f"decoder window size {window.size(-1)} must be larger than chunk_size {chunk_size}.")
        positions = torch.arange(
            window.size(-1),
            device=window.device,
            dtype=valid_frames.dtype,
        )
        clipped_valid = valid_frames.clamp(min=0, max=window.size(-1))
        combined = torch.cat(
            [window, decoder_input.new_zeros(window.size(0), window.size(1), chunk_size)],
            dim=-1,
        )
        insert_index = clipped_valid + torch.arange(
            chunk_size,
            device=window.device,
            dtype=valid_frames.dtype,
        )
        combined.scatter_(
            -1,
            insert_index.view(1, 1, -1).expand_as(decoder_input),
            decoder_input,
        )
        new_valid = (clipped_valid + chunk_size).clamp(max=window.size(-1))
        start = (clipped_valid + chunk_size - window.size(-1)).clamp(min=0)
        gather_index = (start + positions).clamp(max=combined.size(-1) - 1)
        gathered = combined.gather(
            -1,
            gather_index.view(1, 1, -1).expand_as(window),
        )
        mask = (positions < new_valid).to(dtype=window.dtype).view(1, 1, -1)
        return gathered * mask

    def _append_stream_decoder_input(
        self,
        decoder_input: torch.Tensor,
        state: BigVGANStreamState,
    ) -> torch.Tensor:
        decoder_state = state.decoder
        chunk_size = int(decoder_input.size(-1))
        if chunk_size != decoder_state.chunk_size:
            raise ValueError(f"Streaming chunk_size must stay fixed at {decoder_state.chunk_size}, got {chunk_size}.")
        window = decoder_state.window
        valid_frames = min(decoder_state.total_frames, window.size(-1))
        valid_frames_tensor = window.new_tensor(valid_frames, dtype=torch.int64)
        new_window = self._append_stream_decoder_input_tensor(
            decoder_input,
            window,
            valid_frames_tensor,
        )
        decoder_state.window = new_window
        decoder_state.total_frames += chunk_size
        return new_window

    def compiled_stream_step(
        self,
        latents: torch.Tensor,
        hidden_h: torch.Tensor,
        hidden_c: torch.Tensor,
        window: torch.Tensor,
        valid_frames: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        decoder_input, (hidden_h, hidden_c) = self._decode_stream_latents(
            latents,
            (hidden_h, hidden_c),
        )
        new_window = self._append_stream_decoder_input_tensor(
            decoder_input,
            window,
            valid_frames,
        )
        audio_window = self.decode_stream_window(new_window)
        return audio_window, hidden_h, hidden_c, new_window

    def decode_stream_window(self, window: torch.Tensor) -> torch.Tensor:
        decoder_dtype = self.decoder.conv_pre.weight.dtype
        if window.dtype != decoder_dtype:
            window = window.to(dtype=decoder_dtype)
        return self.decoder(window)

    def _slice_stream_audio_window(
        self,
        audio_window: torch.Tensor,
        state: BigVGANStreamState,
        *,
        final: bool,
    ) -> torch.Tensor:
        decoder_state = state.decoder
        stable_end = (
            decoder_state.total_frames if final else max(0, decoder_state.total_frames - self.decoder.stream_lookahead)
        )
        if stable_end <= decoder_state.emitted_frames:
            return _empty_chunk(audio_window, channels=1)

        window_size = decoder_state.window.size(-1)
        valid_frames = min(decoder_state.total_frames, window_size)
        window_start = decoder_state.total_frames - valid_frames
        if decoder_state.emitted_frames < window_start:
            raise RuntimeError("Decoder stream window is too short for fixed-graph decoding.")

        local_start = decoder_state.emitted_frames - window_start
        local_end = stable_end - window_start
        sample_start = local_start * self.hop_size
        sample_end = local_end * self.hop_size
        decoder_state.emitted_frames = stable_end
        return audio_window[..., sample_start:sample_end]

    def stream_step(
        self,
        latents: torch.Tensor,
        state: BigVGANStreamState,
    ) -> torch.Tensor:
        decoder_input = self._prepare_stream_decoder_input(latents, state)
        window = self._append_stream_decoder_input(decoder_input, state)
        audio_window = self.decode_stream_window(window)
        return self._slice_stream_audio_window(audio_window, state, final=False)

    def stream_flush(self, state: BigVGANStreamState) -> torch.Tensor:
        audio_window = self.decode_stream_window(state.decoder.window)
        return self._slice_stream_audio_window(audio_window, state, final=True)

    def remove_weight_norm(self):
        self.decoder.remove_weight_norm()


__all__ = ["AudioVAEConfig", "AudioVAE"]
