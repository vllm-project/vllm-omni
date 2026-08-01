# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Derived from the Apache-2.0 LTX2 vocoder implementation in Diffusers,
# with the model wrapper localized for NAVA production code.

from __future__ import annotations

import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    delta_f = 4 * half_width
    half_size = kernel_size // 2
    amplitude = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if amplitude > 50.0:
        beta = 0.1102 * (amplitude - 8.7)
    elif amplitude >= 21.0:
        beta = 0.5842 * (amplitude - 21) ** 0.4 + 0.07886 * (amplitude - 21.0)
    else:
        beta = 0.0

    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    even = kernel_size % 2 == 0
    time = torch.arange(-half_size, half_size) + 0.5 if even else torch.arange(kernel_size) - half_size

    if cutoff == 0.0:
        low_pass_filter = torch.zeros_like(time)
    else:
        time = 2 * cutoff * time
        sinc = torch.where(
            time == 0,
            torch.ones_like(time),
            torch.sin(math.pi * time) / math.pi / time,
        )
        low_pass_filter = 2 * cutoff * window * sinc
        low_pass_filter = low_pass_filter / low_pass_filter.sum()
    return low_pass_filter


class DownSample1d(nn.Module):
    """1D low-pass filter for antialias downsampling."""

    def __init__(
        self,
        ratio: int = 2,
        kernel_size: int | None = None,
        use_padding: bool = True,
        padding_mode: str = "replicate",
        persistent: bool = True,
    ) -> None:
        super().__init__()
        self.ratio = ratio
        self.kernel_size = kernel_size or int(6 * ratio // 2) * 2
        self.pad_left = self.kernel_size // 2 + (self.kernel_size % 2) - 1
        self.pad_right = self.kernel_size // 2
        self.use_padding = use_padding
        self.padding_mode = padding_mode

        cutoff = 0.5 / ratio
        half_width = 0.6 / ratio
        low_pass_filter = kaiser_sinc_filter1d(cutoff, half_width, self.kernel_size)
        self.register_buffer("filter", low_pass_filter.view(1, 1, self.kernel_size), persistent=persistent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_channels = x.shape[1]
        if self.use_padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        low_pass_filter = self.filter.to(dtype=x.dtype, device=x.device).expand(num_channels, -1, -1)
        return F.conv1d(x, low_pass_filter, stride=self.ratio, groups=num_channels)


class UpSample1d(nn.Module):
    def __init__(
        self,
        ratio: int = 2,
        kernel_size: int | None = None,
        window_type: str = "kaiser",
        padding_mode: str = "replicate",
        persistent: bool = True,
    ) -> None:
        super().__init__()
        self.ratio = ratio
        self.padding_mode = padding_mode

        if window_type == "hann":
            rolloff = 0.99
            lowpass_filter_width = 6
            width = math.ceil(lowpass_filter_width / rolloff)
            self.kernel_size = 2 * width * ratio + 1
            self.pad = width
            self.pad_left = 2 * width * ratio
            self.pad_right = self.kernel_size - ratio

            time_axis = (torch.arange(self.kernel_size) / ratio - width) * rolloff
            time_clamped = time_axis.clamp(-lowpass_filter_width, lowpass_filter_width)
            window = torch.cos(time_clamped * math.pi / lowpass_filter_width / 2) ** 2
            sinc_filter = (torch.sinc(time_axis) * window * rolloff / ratio).view(1, 1, -1)
        else:
            self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
            self.pad = self.kernel_size // ratio - 1
            self.pad_left = self.pad * self.ratio + (self.kernel_size - self.ratio) // 2
            self.pad_right = self.pad * self.ratio + (self.kernel_size - self.ratio + 1) // 2

            sinc_filter = kaiser_sinc_filter1d(
                cutoff=0.5 / ratio,
                half_width=0.6 / ratio,
                kernel_size=self.kernel_size,
            )

        self.register_buffer("filter", sinc_filter.view(1, 1, self.kernel_size), persistent=persistent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_channels = x.shape[1]
        x = F.pad(x, (self.pad, self.pad), mode=self.padding_mode)
        low_pass_filter = self.filter.to(dtype=x.dtype, device=x.device).expand(num_channels, -1, -1)
        x = self.ratio * F.conv_transpose1d(x, low_pass_filter, stride=self.ratio, groups=num_channels)
        return x[..., self.pad_left : -self.pad_right]


class SnakeBeta(nn.Module):
    """Snake/SnakeBeta activation used by the LTX2 vocoder blocks."""

    def __init__(
        self,
        channels: int,
        alpha: float = 1.0,
        eps: float = 1e-9,
        trainable_params: bool = True,
        logscale: bool = True,
        use_beta: bool = True,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.logscale = logscale
        self.use_beta = use_beta

        self.alpha = nn.Parameter(torch.zeros(channels) if self.logscale else torch.ones(channels) * alpha)
        self.alpha.requires_grad = trainable_params
        if use_beta:
            self.beta = nn.Parameter(torch.zeros(channels) if self.logscale else torch.ones(channels) * alpha)
            self.beta.requires_grad = trainable_params

    def forward(self, hidden_states: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
        broadcast_shape = [1] * hidden_states.ndim
        broadcast_shape[channel_dim] = -1
        alpha = self.alpha.view(broadcast_shape)
        if self.use_beta:
            beta = self.beta.view(broadcast_shape)

        if self.logscale:
            alpha = torch.exp(alpha)
            if self.use_beta:
                beta = torch.exp(beta)

        amplitude = beta if self.use_beta else alpha
        return hidden_states + (1.0 / (amplitude + self.eps)) * torch.sin(hidden_states * alpha).pow(2)


class AntiAliasAct1d(nn.Module):
    def __init__(
        self,
        act_fn: str | nn.Module,
        ratio: int = 2,
        kernel_size: int = 12,
        **kwargs,
    ) -> None:
        super().__init__()
        self.upsample = UpSample1d(ratio=ratio, kernel_size=kernel_size)
        if isinstance(act_fn, str):
            if act_fn == "snakebeta":
                act_fn = SnakeBeta(use_beta=True, **kwargs)
            elif act_fn == "snake":
                act_fn = SnakeBeta(use_beta=False, **kwargs)
            else:
                act_fn = nn.LeakyReLU(**kwargs)
        self.act = act_fn
        self.downsample = DownSample1d(ratio=ratio, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.act(x)
        return self.downsample(x)


class ResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilations: tuple[int, ...] | list[int] = (1, 3, 5),
        act_fn: str = "leaky_relu",
        leaky_relu_negative_slope: float = 0.1,
        antialias: bool = False,
        antialias_ratio: int = 2,
        antialias_kernel_size: int = 12,
        padding_mode: str = "same",
    ) -> None:
        super().__init__()
        self.dilations = dilations

        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(channels, channels, kernel_size, stride=stride, dilation=dilation, padding=padding_mode)
                for dilation in dilations
            ]
        )
        self.acts1 = nn.ModuleList()
        for _ in range(len(self.convs1)):
            act = _make_activation(act_fn, channels, leaky_relu_negative_slope)
            if antialias:
                act = AntiAliasAct1d(act, ratio=antialias_ratio, kernel_size=antialias_kernel_size)
            self.acts1.append(act)

        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(channels, channels, kernel_size, stride=stride, dilation=1, padding=padding_mode)
                for _ in range(len(dilations))
            ]
        )
        self.acts2 = nn.ModuleList()
        for _ in range(len(self.convs2)):
            act = _make_activation(act_fn, channels, leaky_relu_negative_slope)
            if antialias:
                act = AntiAliasAct1d(act, ratio=antialias_ratio, kernel_size=antialias_kernel_size)
            self.acts2.append(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for act1, conv1, act2, conv2 in zip(self.acts1, self.convs1, self.acts2, self.convs2):
            residual = act1(x)
            residual = conv1(residual)
            residual = act2(residual)
            residual = conv2(residual)
            x = x + residual
        return x


def _make_activation(act_fn: str, channels: int, leaky_relu_negative_slope: float) -> nn.Module:
    if act_fn == "snakebeta":
        return SnakeBeta(channels, use_beta=True)
    if act_fn == "snake":
        return SnakeBeta(channels, use_beta=False)
    return nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)


class LTX2Vocoder(nn.Module):
    """LTX2 vocoder for converting generated mel spectrograms to waveforms."""

    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 1024,
        out_channels: int = 2,
        upsample_kernel_sizes: list[int] = [16, 15, 8, 4, 4],
        upsample_factors: list[int] = [6, 5, 2, 2, 2],
        resnet_kernel_sizes: list[int] = [3, 7, 11],
        resnet_dilations: list[list[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        act_fn: str = "leaky_relu",
        leaky_relu_negative_slope: float = 0.1,
        antialias: bool = False,
        antialias_ratio: int = 2,
        antialias_kernel_size: int = 12,
        final_act_fn: str | None = "tanh",
        final_bias: bool = True,
        output_sampling_rate: int = 24000,
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            upsample_kernel_sizes=upsample_kernel_sizes,
            upsample_factors=upsample_factors,
            resnet_kernel_sizes=resnet_kernel_sizes,
            resnet_dilations=resnet_dilations,
            act_fn=act_fn,
            leaky_relu_negative_slope=leaky_relu_negative_slope,
            antialias=antialias,
            antialias_ratio=antialias_ratio,
            antialias_kernel_size=antialias_kernel_size,
            final_act_fn=final_act_fn,
            final_bias=final_bias,
            output_sampling_rate=output_sampling_rate,
        )
        self.output_sampling_rate = output_sampling_rate
        self.num_upsample_layers = len(upsample_kernel_sizes)
        self.resnets_per_upsample = len(resnet_kernel_sizes)
        self.out_channels = out_channels
        self.total_upsample_factor = math.prod(upsample_factors)
        self.act_fn = act_fn
        self.negative_slope = leaky_relu_negative_slope
        self.final_act_fn = final_act_fn

        if self.num_upsample_layers != len(upsample_factors):
            raise ValueError(
                "`upsample_kernel_sizes` and `upsample_factors` should be lists of the same length but are length"
                f" {self.num_upsample_layers} and {len(upsample_factors)}, respectively."
            )

        if self.resnets_per_upsample != len(resnet_dilations):
            raise ValueError(
                "`resnet_kernel_sizes` and `resnet_dilations` should be lists of the same length but are length"
                f" {self.resnets_per_upsample} and {len(resnet_dilations)}, respectively."
            )

        supported_act_fns = ["snakebeta", "snake", "leaky_relu"]
        if self.act_fn not in supported_act_fns:
            raise ValueError(
                f"Unsupported activation function: {self.act_fn}. Currently supported values of `act_fn` are "
                f"{supported_act_fns}."
            )

        self.conv_in = nn.Conv1d(in_channels, hidden_channels, kernel_size=7, stride=1, padding=3)

        self.upsamplers = nn.ModuleList()
        self.resnets = nn.ModuleList()
        input_channels = hidden_channels
        for stride, kernel_size in zip(upsample_factors, upsample_kernel_sizes):
            output_channels = input_channels // 2
            self.upsamplers.append(
                nn.ConvTranspose1d(
                    input_channels,
                    output_channels,
                    kernel_size,
                    stride=stride,
                    padding=(kernel_size - stride) // 2,
                )
            )

            for resnet_kernel_size, dilations in zip(resnet_kernel_sizes, resnet_dilations):
                self.resnets.append(
                    ResBlock(
                        channels=output_channels,
                        kernel_size=resnet_kernel_size,
                        dilations=dilations,
                        act_fn=act_fn,
                        leaky_relu_negative_slope=leaky_relu_negative_slope,
                        antialias=antialias,
                        antialias_ratio=antialias_ratio,
                        antialias_kernel_size=antialias_kernel_size,
                    )
                )
            input_channels = output_channels

        if act_fn in {"snakebeta", "snake"}:
            act_out = SnakeBeta(channels=output_channels, use_beta=True)
            self.act_out = AntiAliasAct1d(act_out, ratio=antialias_ratio, kernel_size=antialias_kernel_size)
        else:
            self.act_out = nn.LeakyReLU()

        self.conv_out = nn.Conv1d(output_channels, out_channels, 7, stride=1, padding=3, bias=final_bias)

    def forward(self, hidden_states: torch.Tensor, time_last: bool = False) -> torch.Tensor:
        if not time_last:
            hidden_states = hidden_states.transpose(2, 3)
        hidden_states = hidden_states.flatten(1, 2)

        hidden_states = self.conv_in(hidden_states)

        for i in range(self.num_upsample_layers):
            if self.act_fn == "leaky_relu":
                hidden_states = F.leaky_relu(hidden_states, negative_slope=self.negative_slope)
            hidden_states = self.upsamplers[i](hidden_states)

            start = i * self.resnets_per_upsample
            end = (i + 1) * self.resnets_per_upsample
            resnet_outputs = torch.stack([self.resnets[j](hidden_states) for j in range(start, end)], dim=0)
            hidden_states = torch.mean(resnet_outputs, dim=0)

        hidden_states = self.act_out(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        if self.final_act_fn == "tanh":
            hidden_states = torch.tanh(hidden_states)
        elif self.final_act_fn == "clamp":
            hidden_states = torch.clamp(hidden_states, -1, 1)

        return hidden_states


class CausalSTFT(nn.Module):
    """
    Causal STFT shell whose precomputed bases are loaded from the checkpoint.
    """

    def __init__(self, filter_length: int = 512, hop_length: int = 80, window_length: int = 512) -> None:
        super().__init__()
        self.hop_length = hop_length
        self.window_length = window_length
        n_freqs = filter_length // 2 + 1

        self.register_buffer("forward_basis", torch.zeros(n_freqs * 2, 1, filter_length), persistent=True)
        self.register_buffer("inverse_basis", torch.zeros(n_freqs * 2, 1, filter_length), persistent=True)

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)

        left_pad = max(0, self.window_length - self.hop_length)
        waveform = F.pad(waveform, (left_pad, 0))

        spec = F.conv1d(waveform, self.forward_basis, stride=self.hop_length, padding=0)
        n_freqs = spec.shape[1] // 2
        real, imag = spec[:, :n_freqs], spec[:, n_freqs:]
        magnitude = torch.sqrt(real**2 + imag**2)
        phase = torch.atan2(imag.float(), real.float()).to(dtype=real.dtype)
        return magnitude, phase


class MelSTFT(nn.Module):
    """Causal log-mel spectrogram shell whose filterbank is loaded from the checkpoint."""

    def __init__(
        self,
        filter_length: int = 512,
        hop_length: int = 80,
        window_length: int = 512,
        num_mel_channels: int = 64,
    ) -> None:
        super().__init__()
        self.stft_fn = CausalSTFT(filter_length, hop_length, window_length)

        num_freqs = filter_length // 2 + 1
        self.register_buffer("mel_basis", torch.zeros(num_mel_channels, num_freqs), persistent=True)

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        magnitude, phase = self.stft_fn(waveform)
        energy = torch.norm(magnitude, dim=1)
        mel = torch.matmul(self.mel_basis.to(magnitude.dtype), magnitude)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        return log_mel, magnitude, phase, energy


class LTX2VocoderWithBWE(nn.Module):
    """LTX2 vocoder with bandwidth extension upsampling."""

    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 1536,
        out_channels: int = 2,
        upsample_kernel_sizes: list[int] = [11, 4, 4, 4, 4, 4],
        upsample_factors: list[int] = [5, 2, 2, 2, 2, 2],
        resnet_kernel_sizes: list[int] = [3, 7, 11],
        resnet_dilations: list[list[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        act_fn: str = "snakebeta",
        leaky_relu_negative_slope: float = 0.1,
        antialias: bool = True,
        antialias_ratio: int = 2,
        antialias_kernel_size: int = 12,
        final_act_fn: str | None = None,
        final_bias: bool = False,
        bwe_in_channels: int = 128,
        bwe_hidden_channels: int = 512,
        bwe_out_channels: int = 2,
        bwe_upsample_kernel_sizes: list[int] = [12, 11, 4, 4, 4],
        bwe_upsample_factors: list[int] = [6, 5, 2, 2, 2],
        bwe_resnet_kernel_sizes: list[int] = [3, 7, 11],
        bwe_resnet_dilations: list[list[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        bwe_act_fn: str = "snakebeta",
        bwe_leaky_relu_negative_slope: float = 0.1,
        bwe_antialias: bool = True,
        bwe_antialias_ratio: int = 2,
        bwe_antialias_kernel_size: int = 12,
        bwe_final_act_fn: str | None = None,
        bwe_final_bias: bool = False,
        filter_length: int = 512,
        hop_length: int = 80,
        window_length: int = 512,
        num_mel_channels: int = 64,
        input_sampling_rate: int = 16000,
        output_sampling_rate: int = 48000,
    ) -> None:
        super().__init__()
        self.input_sampling_rate = input_sampling_rate
        self.output_sampling_rate = output_sampling_rate
        self.hop_length = hop_length
        self.config = SimpleNamespace(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            upsample_kernel_sizes=upsample_kernel_sizes,
            upsample_factors=upsample_factors,
            resnet_kernel_sizes=resnet_kernel_sizes,
            resnet_dilations=resnet_dilations,
            act_fn=act_fn,
            leaky_relu_negative_slope=leaky_relu_negative_slope,
            antialias=antialias,
            antialias_ratio=antialias_ratio,
            antialias_kernel_size=antialias_kernel_size,
            final_act_fn=final_act_fn,
            final_bias=final_bias,
            bwe_in_channels=bwe_in_channels,
            bwe_hidden_channels=bwe_hidden_channels,
            bwe_out_channels=bwe_out_channels,
            bwe_upsample_kernel_sizes=bwe_upsample_kernel_sizes,
            bwe_upsample_factors=bwe_upsample_factors,
            bwe_resnet_kernel_sizes=bwe_resnet_kernel_sizes,
            bwe_resnet_dilations=bwe_resnet_dilations,
            bwe_act_fn=bwe_act_fn,
            bwe_leaky_relu_negative_slope=bwe_leaky_relu_negative_slope,
            bwe_antialias=bwe_antialias,
            bwe_antialias_ratio=bwe_antialias_ratio,
            bwe_antialias_kernel_size=bwe_antialias_kernel_size,
            bwe_final_act_fn=bwe_final_act_fn,
            bwe_final_bias=bwe_final_bias,
            filter_length=filter_length,
            hop_length=hop_length,
            window_length=window_length,
            num_mel_channels=num_mel_channels,
            input_sampling_rate=input_sampling_rate,
            output_sampling_rate=output_sampling_rate,
        )

        self.vocoder = LTX2Vocoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            upsample_kernel_sizes=upsample_kernel_sizes,
            upsample_factors=upsample_factors,
            resnet_kernel_sizes=resnet_kernel_sizes,
            resnet_dilations=resnet_dilations,
            act_fn=act_fn,
            leaky_relu_negative_slope=leaky_relu_negative_slope,
            antialias=antialias,
            antialias_ratio=antialias_ratio,
            antialias_kernel_size=antialias_kernel_size,
            final_act_fn=final_act_fn,
            final_bias=final_bias,
            output_sampling_rate=input_sampling_rate,
        )
        self.bwe_generator = LTX2Vocoder(
            in_channels=bwe_in_channels,
            hidden_channels=bwe_hidden_channels,
            out_channels=bwe_out_channels,
            upsample_kernel_sizes=bwe_upsample_kernel_sizes,
            upsample_factors=bwe_upsample_factors,
            resnet_kernel_sizes=bwe_resnet_kernel_sizes,
            resnet_dilations=bwe_resnet_dilations,
            act_fn=bwe_act_fn,
            leaky_relu_negative_slope=bwe_leaky_relu_negative_slope,
            antialias=bwe_antialias,
            antialias_ratio=bwe_antialias_ratio,
            antialias_kernel_size=bwe_antialias_kernel_size,
            final_act_fn=bwe_final_act_fn,
            final_bias=bwe_final_bias,
            output_sampling_rate=output_sampling_rate,
        )

        self.mel_stft = MelSTFT(
            filter_length=filter_length,
            hop_length=hop_length,
            window_length=window_length,
            num_mel_channels=num_mel_channels,
        )

        self.resampler = UpSample1d(
            ratio=output_sampling_rate // input_sampling_rate,
            window_type="hann",
            persistent=False,
        )

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        x = self.vocoder(mel_spec)
        _, num_channels, num_samples = x.shape

        remainder = num_samples % self.hop_length
        if remainder != 0:
            x = F.pad(x, (0, self.hop_length - remainder))

        mel, _, _, _ = self.mel_stft(x.flatten(0, 1))
        mel = mel.unflatten(0, (-1, num_channels))

        mel_for_bwe = mel.transpose(2, 3)
        residual = self.bwe_generator(mel_for_bwe)

        skip = self.resampler(x)
        waveform = torch.clamp(residual + skip, -1, 1)
        output_samples = num_samples * self.output_sampling_rate // self.input_sampling_rate
        return waveform[..., :output_samples]
