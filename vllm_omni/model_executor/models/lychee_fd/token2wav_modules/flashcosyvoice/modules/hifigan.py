# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Kai Hu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HIFI-GAN"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import get_window
from torch.nn.utils import remove_weight_norm

try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:
    from torch.nn.utils import weight_norm  # noqa

from .hifigan_components.convolution import Conv1d, ConvTranspose1d
from .hifigan_components.f0_math import F0ELU, F0Conv1d
from .hifigan_components.layers import (
    ResBlock,
    SourceModuleHnNSF,
    SourceModuleHnNSF2,
    init_weights,
)
from .hifigan_components.preconv_math import PreconvConv1d


class ConvRNNF0Predictor(nn.Module):
    def __init__(self, num_class: int = 1, in_channels: int = 80, cond_channels: int = 512):
        super().__init__()

        self.num_class = num_class
        self.condnet = nn.Sequential(
            weight_norm(  # noqa
                F0Conv1d(in_channels, cond_channels, kernel_size=3, padding=1)
            ),
            F0ELU(),
            weight_norm(  # noqa
                F0Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)
            ),
            F0ELU(),
            weight_norm(  # noqa
                F0Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)
            ),
            F0ELU(),
            weight_norm(  # noqa
                F0Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)
            ),
            F0ELU(),
            weight_norm(  # noqa
                F0Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)
            ),
            F0ELU(),
        )
        self.classifier = nn.Linear(in_features=cond_channels, out_features=self.num_class)

    def shutdown(self) -> None:
        failure = None
        for module in self.condnet:
            if isinstance(module, F0Conv1d):
                try:
                    module.clear_released_plans()
                except Exception as error:
                    failure = failure or error
        if failure is not None:
            raise failure

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.condnet(x)
        x = x.transpose(1, 2)
        return torch.abs(self.classifier(x).squeeze(-1))


class HiFTGenerator(nn.Module):
    """
    HiFTNet Generator: Neural Source Filter + ISTFTNet
    https://arxiv.org/abs/2309.09493
    """

    def shutdown(self) -> None:
        failure = None
        for owner, method in (
            (self.f0_predictor, "shutdown"),
            (getattr(self, "conv_pre", None), "clear_released_plans"),
        ):
            cleanup = getattr(owner, method, None)
            if cleanup is not None:
                try:
                    cleanup()
                except Exception as error:
                    failure = failure or error
        if failure is not None:
            raise failure

    def __init__(
        self,
        in_channels: int = 80,
        base_channels: int = 512,
        nb_harmonics: int = 8,
        sampling_rate: int = 24000,
        nsf_alpha: float = 0.1,
        nsf_sigma: float = 0.003,
        nsf_voiced_threshold: float = 10,
        upsample_rates: list[int] = [8, 5, 3],  # noqa
        upsample_kernel_sizes: list[int] = [16, 11, 7],  # noqa
        istft_params: dict[str, int] = {"n_fft": 16, "hop_len": 4},  # noqa
        resblock_kernel_sizes: list[int] = [3, 7, 11],  # noqa
        resblock_dilation_sizes: list[list[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],  # noqa
        source_resblock_kernel_sizes: list[int] = [7, 7, 11],  # noqa
        source_resblock_dilation_sizes: list[list[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],  # noqa
        lrelu_slope: float = 0.1,
        audio_limit: float = 0.99,
        f0_predictor: torch.nn.Module = None,
    ):
        super().__init__()

        self.out_channels = 1
        self.nb_harmonics = nb_harmonics
        self.sampling_rate = sampling_rate
        self.istft_params = istft_params
        self.lrelu_slope = lrelu_slope
        self.audio_limit = audio_limit

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        # NOTE in CosyVoice2, we use the original SourceModuleHnNSF implementation
        this_SourceModuleHnNSF = SourceModuleHnNSF if self.sampling_rate == 22050 else SourceModuleHnNSF2
        self.m_source = this_SourceModuleHnNSF(
            sampling_rate=sampling_rate,
            upsample_scale=np.prod(upsample_rates) * istft_params["hop_len"],
            harmonic_num=nb_harmonics,
            sine_amp=nsf_alpha,
            add_noise_std=nsf_sigma,
            voiced_threshod=nsf_voiced_threshold,
        )
        self.f0_upsamp = torch.nn.Upsample(scale_factor=np.prod(upsample_rates) * istft_params["hop_len"])

        self.conv_pre = weight_norm(  # noqa
            PreconvConv1d(in_channels, base_channels, 7, 1, padding=3)
        )

        # Up
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(  # noqa
                    ConvTranspose1d(
                        base_channels // (2**i),
                        base_channels // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        # Down
        self.source_downs = nn.ModuleList()
        self.source_resblocks = nn.ModuleList()
        downsample_rates = [1] + upsample_rates[::-1][:-1]
        downsample_cum_rates = np.cumprod(downsample_rates)
        for i, (u, k, d) in enumerate(
            zip(downsample_cum_rates[::-1], source_resblock_kernel_sizes, source_resblock_dilation_sizes)
        ):
            if u == 1:
                self.source_downs.append(Conv1d(istft_params["n_fft"] + 2, base_channels // (2 ** (i + 1)), 1, 1))
            else:
                self.source_downs.append(
                    Conv1d(istft_params["n_fft"] + 2, base_channels // (2 ** (i + 1)), u * 2, u, padding=(u // 2))
                )

            self.source_resblocks.append(ResBlock(base_channels // (2 ** (i + 1)), k, d))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = base_channels // (2 ** (i + 1))
            for _, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, istft_params["n_fft"] + 2, 7, 1, padding=3))  # noqa
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.stft_window = torch.from_numpy(get_window("hann", istft_params["n_fft"], fftbins=True).astype(np.float32))
        self.f0_predictor = ConvRNNF0Predictor() if f0_predictor is None else f0_predictor

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for up in self.ups:
            remove_weight_norm(up)
        for resblock in self.resblocks:
            resblock.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
        self.m_source.remove_weight_norm()
        for source_down in self.source_downs:
            remove_weight_norm(source_down)
        for source_resblock in self.source_resblocks:
            source_resblock.remove_weight_norm()

    def _stft(self, x):
        spec = torch.stft(
            x,
            self.istft_params["n_fft"],
            self.istft_params["hop_len"],
            self.istft_params["n_fft"],
            window=self.stft_window.to(x.device),
            return_complex=True,
        )
        spec = torch.view_as_real(spec)  # [B, F, TT, 2]
        return spec[..., 0], spec[..., 1]

    def _istft(self, magnitude, phase):
        magnitude = torch.clip(magnitude, max=1e2)
        real = magnitude * torch.cos(phase)
        img = magnitude * torch.sin(phase)
        inverse_transform = torch.istft(
            torch.complex(real, img),
            self.istft_params["n_fft"],
            self.istft_params["hop_len"],
            self.istft_params["n_fft"],
            window=self.stft_window.to(magnitude.device),
        )
        return inverse_transform

    def decode(self, x: torch.Tensor, s: torch.Tensor = torch.zeros(1, 1, 0)) -> torch.Tensor:
        s_stft_real, s_stft_imag = self._stft(s.squeeze(1))
        s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)

        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x)

            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)

            # fusion
            si = self.source_downs[i](s_stft)
            si = self.source_resblocks[i](si)
            x = x + si

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        magnitude = torch.exp(x[:, : self.istft_params["n_fft"] // 2 + 1, :])
        phase = torch.sin(x[:, self.istft_params["n_fft"] // 2 + 1 :, :])  # actually, sin is redundancy

        x = self._istft(magnitude, phase)
        x = torch.clamp(x, -self.audio_limit, self.audio_limit)
        return x

    @torch.inference_mode()
    def forward(self, speech_feat: torch.Tensor, cache_source: torch.Tensor = torch.zeros(1, 1, 0)) -> torch.Tensor:
        # mel->f0
        f0 = self.f0_predictor(speech_feat)
        # f0->source
        s = self.f0_upsamp(f0[:, None]).transpose(1, 2)  # bs,n,t
        s, _, _ = self.m_source(s)
        s = s.transpose(1, 2)
        # use cache_source to avoid glitch
        if cache_source.shape[2] != 0:
            s[:, :, : cache_source.shape[2]] = cache_source
        generated_speech = self.decode(x=speech_feat, s=s)
        return generated_speech, s
