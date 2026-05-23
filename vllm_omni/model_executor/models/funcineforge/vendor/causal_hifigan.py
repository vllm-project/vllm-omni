# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vendored Causal HiFiGAN from FunCineForge.

This module re-exports the ``CausalHiFTGenerator``, ``CausalConvRNNF0Predictor``,
and ``CausalHifiGan`` classes from the FunCineForge codebase with minimal
adaptations (removing ``funcineforge.register`` decorators and
``funcineforge.utils`` dependencies).

For the initial integration we vendor the key classes directly.  The full
vocoder code is self-contained and does not depend on external packages
beyond PyTorch and standard scientific Python.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import get_window
from torch.nn.utils.parametrizations import weight_norm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weight init helper (from FunCineForge's hifigan/__init__.py)
# ---------------------------------------------------------------------------


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


# ---------------------------------------------------------------------------
# Snake activation (from FunCineForge's hifigan/activations.py)
# ---------------------------------------------------------------------------


class Snake(nn.Module):
    def __init__(self, channels, alpha_logscale=False):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        if alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(channels) * 0.1)
        else:
            self.alpha = nn.Parameter(torch.ones(channels))

    def forward(self, x):
        if self.alpha_logscale:
            alpha = torch.exp(self.alpha)
        else:
            alpha = self.alpha
        # Reshape for broadcasting: (C,) -> (1, C, 1) for (B, C, T) input
        alpha = alpha.unsqueeze(0).unsqueeze(-1)
        return x + (1.0 / (alpha + 1e-5)) * torch.sin(alpha * x) ** 2


# ---------------------------------------------------------------------------
# Causal convolution layers
# ---------------------------------------------------------------------------


class LookRightConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **kwargs,
        )
        assert stride == 1
        self.causal_padding = kernel_size - 1

    def forward(self, x, context=torch.zeros(0, 0, 0)):
        if context.size(2) == 0:
            x = F.pad(x, (0, self.causal_padding), value=0.0)
        else:
            assert context.size(2) == self.causal_padding
            x = torch.concat([x, context], dim=2)
        return super().forward(x)


class LookLeftConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **kwargs,
        )
        assert stride == 1 and dilation == 1
        self.causal_padding = kernel_size - 1

    def forward(self, x, cache=torch.zeros(0, 0, 0)):
        if cache.size(2) == 0:
            x = F.pad(x, (self.causal_padding, 0), value=0.0)
        else:
            assert cache.size(2) == self.causal_padding
            x = torch.concat([cache, x], dim=2)
        if self.causal_padding == 0:
            cache_new = x[:, :, :0]
        else:
            cache_new = x[:, :, -self.causal_padding :]
        x = super().forward(x)
        return x, cache_new


class LookLeftConvTranspose1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__(
            in_channels, out_channels, kernel_size, 1, padding=0, dilation=dilation, groups=groups, bias=bias, **kwargs
        )
        assert dilation == 1 and stride != 1
        self.causal_padding = kernel_size - 1
        self.upsample = nn.Upsample(scale_factor=stride, mode="nearest")

    def forward(self, x, cache=torch.zeros(0, 0, 0)):
        x = self.upsample(x)
        if cache.size(2) == 0:
            x = F.pad(x, (self.causal_padding, 0), value=0.0)
        else:
            assert cache.size(2) == self.causal_padding
            x = torch.concat([cache, x], dim=2)
        cache_new = x[:, :, -self.causal_padding :]
        x = super().forward(x)
        return x, cache_new


class LookLeftConv1dWithStride(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **kwargs,
        )
        assert stride != 1 and dilation == 1
        assert kernel_size % stride == 0
        self.causal_padding = stride - 1

    def forward(self, x, cache=torch.zeros(0, 0, 0)):
        if cache.size(2) == 0:
            x = F.pad(x, (self.causal_padding, 0), value=0.0)
        else:
            assert cache.size(2) == self.causal_padding
            x = torch.concat([cache, x], dim=2)
        cache_new = x[:, :, -self.causal_padding :]
        x = super().forward(x)
        return x, cache_new


class LookLeftConv1dWithDilation(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True, **kwargs):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **kwargs,
        )
        assert kernel_size // 2 * dilation * 2 == int((kernel_size * dilation - dilation) / 2) * 2
        self.causal_padding = int((kernel_size * dilation - dilation) / 2) * 2

    def forward(self, x, cache=torch.zeros(0, 0, 0)):
        if cache.size(2) == 0:
            x = F.pad(x, (self.causal_padding, 0), value=0.0)
        else:
            assert cache.size(2) == self.causal_padding
            x = torch.concat([cache, x], dim=2)
        cache_new = x[:, :, -self.causal_padding :]
        x = super().forward(x)
        return x, cache_new


# ---------------------------------------------------------------------------
# ResBlock
# ---------------------------------------------------------------------------


class ResBlock(nn.Module):
    def __init__(self, channels=512, kernel_size=3, dilations=None):
        super().__init__()
        if dilations is None:
            dilations = [1, 3, 5]
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        for dilation in dilations:
            self.convs1.append(
                weight_norm(
                    LookLeftConv1dWithDilation(channels, channels, kernel_size, 1, dilation=dilation)
                    if dilation != 1
                    else LookLeftConv1d(channels, channels, kernel_size, 1, dilation=dilation)
                )
            )
            self.convs2.append(weight_norm(LookLeftConv1d(channels, channels, kernel_size, 1, dilation=1)))
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)
        self.activations1 = nn.ModuleList([Snake(channels, alpha_logscale=False) for _ in range(len(self.convs1))])
        self.activations2 = nn.ModuleList([Snake(channels, alpha_logscale=False) for _ in range(len(self.convs2))])

    def forward(self, x, cache=torch.zeros(0, 0, 0, 0, 0)):
        for idx in range(len(self.convs1)):
            xt = self.activations1[idx](x)
            xt, _ = self.convs1[idx](xt)
            xt = self.activations2[idx](xt)
            xt, _ = self.convs2[idx](xt)
            x = xt + x
        return x, cache

    def remove_weight_norm(self):
        for idx in range(len(self.convs1)):
            try:
                nn.utils.remove_weight_norm(self.convs1[idx])
                nn.utils.remove_weight_norm(self.convs2[idx])
            except Exception:
                from torch.nn.utils.parametrize import remove_parametrizations

                remove_parametrizations(self.convs1[idx], "weight")
                remove_parametrizations(self.convs2[idx], "weight")


# ---------------------------------------------------------------------------
# F0 Predictor
# ---------------------------------------------------------------------------


class CausalConvRNNF0Predictor(nn.Module):
    def __init__(self, num_class=1, in_channels=80, cond_channels=512):
        super().__init__()
        self.num_class = num_class
        self.condnet = nn.Sequential(
            weight_norm(LookRightConv1d(in_channels, cond_channels, kernel_size=4)),
            nn.ELU(),
            weight_norm(LookLeftConv1d(cond_channels, cond_channels, kernel_size=3)),
            nn.ELU(),
            weight_norm(LookLeftConv1d(cond_channels, cond_channels, kernel_size=3)),
            nn.ELU(),
            weight_norm(LookLeftConv1d(cond_channels, cond_channels, kernel_size=3)),
            nn.ELU(),
            weight_norm(LookLeftConv1d(cond_channels, cond_channels, kernel_size=3)),
            nn.ELU(),
        )
        self.classifier = nn.Linear(in_features=cond_channels, out_features=self.num_class)

    def forward(self, x, cache=torch.zeros(0, 0, 0, 0), finalize=True):
        if finalize is False:
            x, context = x[:, :, : -self.condnet[0].causal_padding], x[:, :, -self.condnet[0].causal_padding :]
        else:
            x, context = x, x[:, :, :0]
        x = self.condnet[0](x, context)
        x = self.condnet[1](x)
        if cache.size(0) != 0:
            x, cache[0] = self.condnet[2](x, cache[0])
        else:
            x, _ = self.condnet[2](x)
        x = self.condnet[3](x)
        if cache.size(0) != 0:
            x, cache[1] = self.condnet[4](x, cache[1])
        else:
            x, _ = self.condnet[4](x)
        x = self.condnet[5](x)
        if cache.size(0) != 0:
            x, cache[2] = self.condnet[6](x, cache[2])
        else:
            x, _ = self.condnet[6](x)
        x = self.condnet[7](x)
        if cache.size(0) != 0:
            x, cache[3] = self.condnet[8](x, cache[3])
        else:
            x, _ = self.condnet[8](x)
        x = self.condnet[9](x)
        x = x.transpose(1, 2)
        x = torch.abs(self.classifier(x).squeeze(-1))
        return x, cache

    def init_cache(self, device):
        return torch.zeros(4, 1, 512, 2).to(device)

    def remove_weight_norm(self):
        for idx in [0, 2, 4, 6, 8]:
            try:
                nn.utils.remove_weight_norm(self.condnet[idx])
            except Exception:
                from torch.nn.utils.parametrize import remove_parametrizations

                remove_parametrizations(self.condnet[idx], "weight")


# ---------------------------------------------------------------------------
# SineGen / SourceModuleHnNSF
# ---------------------------------------------------------------------------


class SineGen(nn.Module):
    def __init__(
        self,
        samp_rate,
        upsample_scale,
        harmonic_num=0,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0,
        flag_for_pulse=False,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
        self.upsample_scale = upsample_scale
        self.rand_ini = torch.rand(1, 9)
        self.rand_ini[:, 0] = 0
        self.sine_waves = torch.rand(1, 300 * 24000, 9)

    def _f02uv(self, f0):
        return (f0 > self.voiced_threshold).type(torch.float32)

    def _f02sine(self, f0_values):
        rad_values = (f0_values / self.sampling_rate) % 1
        rad_values[:, 0, :] = rad_values[:, 0, :] + self.rand_ini.to(rad_values.device)
        if not self.flag_for_pulse:
            rad_values = F.interpolate(
                rad_values.transpose(1, 2), scale_factor=1 / self.upsample_scale, mode="linear"
            ).transpose(1, 2)
            phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi
            phase = F.interpolate(
                phase.transpose(1, 2) * self.upsample_scale, scale_factor=self.upsample_scale, mode="nearest"
            ).transpose(1, 2)
            sines = torch.sin(phase)
        else:
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)
            sines = torch.cos(i_phase * 2 * np.pi)
        return sines

    def forward(self, f0):
        fn = torch.multiply(f0, torch.FloatTensor([[range(1, self.harmonic_num + 2)]]).to(f0.device))
        sine_waves = self._f02sine(fn) * self.sine_amp
        uv = self._f02uv(f0)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        # Expand noise buffer if streaming accumulation exceeds pre-allocated size
        needed = sine_waves.shape[1]
        if needed > self.sine_waves.shape[1]:
            self.sine_waves = torch.rand(1, needed, self.sine_waves.shape[2])
        noise = noise_amp * self.sine_waves[:, :needed].to(sine_waves.device)
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(nn.Module):
    def __init__(
        self, sampling_rate, upsample_scale, harmonic_num=0, sine_amp=0.1, add_noise_std=0.003, voiced_threshold=0
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen(sampling_rate, upsample_scale, harmonic_num, sine_amp, add_noise_std, voiced_threshold)
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()
        self.uv = torch.rand(1, 300 * 24000, 1)

    def forward(self, x):
        with torch.no_grad():
            sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        # Expand noise buffer if streaming accumulation exceeds pre-allocated size
        needed = uv.shape[1]
        if needed > self.uv.shape[1]:
            self.uv = torch.rand(1, needed, 1)
        noise = self.uv[:, :needed] * self.sine_amp / 3
        return sine_merge, noise, uv


# ---------------------------------------------------------------------------
# CausalHiFTGenerator
# ---------------------------------------------------------------------------


class CausalHiFTGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int = 80,
        base_channels: int = 512,
        nb_harmonics: int = 8,
        sampling_rate: int = 22050,
        nsf_alpha: float = 0.1,
        nsf_sigma: float = 0.003,
        nsf_voiced_threshold: float = 10,
        upsample_rates: list[int] = None,
        upsample_kernel_sizes: list[int] = None,
        istft_params: dict[str, int] = None,
        resblock_kernel_sizes: list[int] = None,
        resblock_dilation_sizes: list[list[int]] = None,
        source_resblock_kernel_sizes: list[int] = None,
        source_resblock_dilation_sizes: list[list[int]] = None,
        lrelu_slope: float = 0.1,
        audio_limit: float = 0.99,
        f0_predictor=None,
    ):
        super().__init__()
        if upsample_rates is None:
            upsample_rates = [8, 8]
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = [16, 16]
        if istft_params is None:
            istft_params = {"n_fft": 16, "hop_len": 4}
        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = [3, 7, 11]
        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        if source_resblock_kernel_sizes is None:
            source_resblock_kernel_sizes = [7, 11]
        if source_resblock_dilation_sizes is None:
            source_resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5]]

        self.out_channels = 1
        self.nb_harmonics = nb_harmonics
        self.sampling_rate = sampling_rate
        self.istft_params = istft_params
        self.lrelu_slope = lrelu_slope
        self.audio_limit = audio_limit

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.m_source = SourceModuleHnNSF(
            sampling_rate=sampling_rate,
            upsample_scale=int(np.prod(upsample_rates)) * istft_params["hop_len"],
            harmonic_num=nb_harmonics,
            sine_amp=nsf_alpha,
            add_noise_std=nsf_sigma,
            voiced_threshold=nsf_voiced_threshold,
        )
        self.f0_upsamp = nn.Upsample(
            scale_factor=int(np.prod(upsample_rates)) * istft_params["hop_len"], mode="nearest"
        )

        self.conv_pre = weight_norm(LookRightConv1d(in_channels, base_channels, 5, 1))

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(LookLeftConvTranspose1d(base_channels // (2**i), base_channels // (2 ** (i + 1)), k, u))
            )

        self.source_downs = nn.ModuleList()
        self.source_resblocks = nn.ModuleList()
        downsample_rates = [1] + upsample_rates[::-1][:-1]
        downsample_cum_rates = np.cumprod(downsample_rates).tolist()
        for i, (u, k, d) in enumerate(
            zip(downsample_cum_rates[::-1], source_resblock_kernel_sizes, source_resblock_dilation_sizes)
        ):
            if u == 1:
                self.source_downs.append(
                    LookLeftConv1d(istft_params["n_fft"] + 2, base_channels // (2 ** (i + 1)), 1, 1)
                )
            else:
                self.source_downs.append(
                    LookLeftConv1dWithStride(istft_params["n_fft"] + 2, base_channels // (2 ** (i + 1)), u * 2, u)
                )
            self.source_resblocks.append(ResBlock(base_channels // (2 ** (i + 1)), k, d))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = base_channels // (2 ** (i + 1))
            for _, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d))

        ch = base_channels // (2 ** len(upsample_rates))
        self.conv_post = weight_norm(LookLeftConv1d(ch, istft_params["n_fft"] + 2, 7, 1))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.stft_window = torch.from_numpy(get_window("hann", istft_params["n_fft"], fftbins=True).astype(np.float32))
        self.f0_predictor = f0_predictor
        self.context_size = 8

    def remove_weight_norm(self):
        for layer in self.ups:
            try:
                nn.utils.remove_weight_norm(layer)
            except Exception:
                from torch.nn.utils.parametrize import remove_parametrizations

                remove_parametrizations(layer, "weight")
        for layer in self.resblocks:
            layer.remove_weight_norm()
        try:
            nn.utils.remove_weight_norm(self.conv_pre)
            nn.utils.remove_weight_norm(self.conv_post)
        except Exception:
            from torch.nn.utils.parametrize import remove_parametrizations

            remove_parametrizations(self.conv_pre, "weight")
            remove_parametrizations(self.conv_post, "weight")
        self.f0_predictor.remove_weight_norm()
        for layer in self.source_resblocks:
            layer.remove_weight_norm()

    def _stft(self, x):
        spec = torch.stft(
            x,
            self.istft_params["n_fft"],
            self.istft_params["hop_len"],
            self.istft_params["n_fft"],
            window=self.stft_window.to(x.device),
            return_complex=True,
        )
        spec = torch.view_as_real(spec)
        return spec[..., 0], spec[..., 1]

    def _istft(self, magnitude, phase):
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        img = magnitude * torch.sin(phase)
        return torch.istft(
            torch.complex(real, img),
            self.istft_params["n_fft"],
            self.istft_params["hop_len"],
            self.istft_params["n_fft"],
            window=self.stft_window.to(magnitude.device),
        )

    def decode(self, x, s=torch.zeros(0, 0, 0), finalize=True):
        s_stft_real, s_stft_imag = self._stft(s.squeeze(1))
        if finalize is False:
            s_stft_real = s_stft_real[:, :, : -int(480 * 4 / self.istft_params["hop_len"])]
            s_stft_imag = s_stft_imag[:, :, : -int(480 * 4 / self.istft_params["hop_len"])]
            x = self.conv_pre(x[:, :, :-4], x[:, :, -4:])
        else:
            x = self.conv_pre(x)
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.lrelu_slope)
            x, _ = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            si, _ = self.source_downs[i](s_stft)
            si, _ = self.source_resblocks[i](si)
            x = x + si
            xs = None
            for j in range(self.num_kernels):
                this_xs, _ = self.resblocks[i * self.num_kernels + j](x)
                if xs is None:
                    xs = this_xs
                else:
                    xs += this_xs
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x, _ = self.conv_post(x)
        magnitude = torch.exp(x[:, : self.istft_params["n_fft"] // 2 + 1, :])
        phase = torch.sin(x[:, self.istft_params["n_fft"] // 2 + 1 :, :])
        x = self._istft(magnitude, phase)
        if finalize is False:
            x = x[:, :-480]
        x = torch.clamp(x, -self.audio_limit, self.audio_limit)
        return x

    @torch.inference_mode()
    def inference(self, speech_feat, f0_cpu=False, finalize=True):
        if f0_cpu:
            self.f0_predictor.to("cpu")
            f0, _ = self.f0_predictor(speech_feat.cpu(), finalize=finalize)
            f0 = f0.to(speech_feat.device)
        else:
            self.f0_predictor.to(speech_feat.device)
            f0, _ = self.f0_predictor(speech_feat, finalize=finalize)
        s = self.f0_upsamp(f0[:, None]).transpose(1, 2)
        s, _, _ = self.m_source(s)
        s = s.transpose(1, 2)
        if finalize is False:
            generated_speech = self.decode(speech_feat[:, :, :-3], s, finalize=finalize)
        else:
            generated_speech = self.decode(speech_feat, s, finalize=finalize)
        return generated_speech, []
