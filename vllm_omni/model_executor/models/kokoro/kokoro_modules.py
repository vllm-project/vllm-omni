# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Kokoro-82M network: StyleTTS 2 text/prosody encoders + iSTFTNet decoder.

Ported from the reference ``kokoro`` package (hexgrad/kokoro v0.9.4, Apache-2.0,
``model.py`` / ``modules.py`` / ``istftnet.py``), which adapts
https://github.com/yl4579/StyleTTS2. Differences from the reference, none of
which change the computed audio:

- Weight norm is folded into plain conv weights at load time
  (:func:`fold_weight_norm`) instead of being recomputed every forward.
- ``AdaIN1d``'s ``InstanceNorm1d`` affine parameters are absent from the
  checkpoint (the reference declares ``affine=True`` only to dodge an old ONNX
  exporter bug), so :meth:`KokoroNetwork.reset_adain_affine` pins them to the
  identity the reference leaves them at. They stay affine because the affine
  kernel rounds differently from the non-affine one.
- The sine/noise excitation draws from an explicit ``torch.Generator`` so each
  request owns its RNG stream. Draw order matches the reference, so seeding the
  generator with ``s`` reproduces ``torch.manual_seed(s)`` in the reference.
- Training-only branches (pulse-train sine generation, ONNX ``CustomSTFT``) are
  dropped.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AlbertConfig, AlbertModel

from vllm_omni.model_executor.models.kokoro.configuration_kokoro import KokoroConfig

_INV_SQRT2 = 2**-0.5


def fold_weight_norm(state_dict: dict[str, torch.Tensor], device: torch.device | str) -> dict[str, torch.Tensor]:
    """Replace legacy ``weight_g``/``weight_v`` pairs with the folded ``weight``.

    Uses the same ``torch._weight_norm`` kernel (``dim=0``) the reference's
    ``torch.nn.utils.weight_norm`` hook calls on every forward, on the device
    the network runs on: a CPU fold rounds the norm differently, and the
    vocoder's phase accumulation amplifies that into audible sample drift.
    """
    folded: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        leaf = name.rsplit(".", 1)[-1]
        if leaf == "weight_g":
            continue
        if leaf == "weight_v":
            prefix = name[: -len("weight_v")]
            g = state_dict[prefix + "weight_g"].to(device)
            folded[prefix + "weight"] = torch._weight_norm(tensor.to(device), g, 0)
        else:
            folded[name] = tensor
    return folded


class LayerNorm(nn.Module):
    """Channel LayerNorm over ``[B, C, T]`` inputs."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim: int, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)


class AdaIN1d(nn.Module):
    def __init__(self, style_dim: int, num_features: int):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=True)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


def _get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


class AdaINResBlock1(nn.Module):
    """HiFi-GAN ResBlock1 with AdaIN conditioning and Snake activations."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation=(1, 3, 5), style_dim: int = 64):
        super().__init__()
        self.convs1 = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=_get_padding(kernel_size, d))
            for d in dilation
        )
        self.convs2 = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=_get_padding(kernel_size, 1))
            for _ in dilation
        )
        self.adain1 = nn.ModuleList(AdaIN1d(style_dim, channels) for _ in dilation)
        self.adain2 = nn.ModuleList(AdaIN1d(style_dim, channels) for _ in dilation)
        self.alpha1 = nn.ParameterList(nn.Parameter(torch.ones(1, channels, 1)) for _ in dilation)
        self.alpha2 = nn.ParameterList(nn.Parameter(torch.ones(1, channels, 1)) for _ in dilation)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        for c1, c2, n1, n2, a1, a2 in zip(self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2):
            xt = n1(x, s)
            xt = xt + (1 / a1) * (torch.sin(a1 * xt) ** 2)  # Snake1D
            xt = c1(xt)
            xt = n2(xt, s)
            xt = xt + (1 / a2) * (torch.sin(a2 * xt) ** 2)  # Snake1D
            xt = c2(xt)
            x = xt + x
        return x


class TorchSTFT(nn.Module):
    def __init__(self, filter_length: int, hop_length: int, win_length: int):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        # Built on CPU like the reference: a GPU-computed Hann window rounds
        # differently, and the model is constructed under a CUDA device context.
        self._cpu_window = torch.hann_window(win_length, periodic=True, dtype=torch.float32, device="cpu")
        self._device_windows: dict[torch.device, torch.Tensor] = {}

    def _window(self, device: torch.device) -> torch.Tensor:
        window = self._device_windows.get(device)
        if window is None:
            window = self._device_windows[device] = self._cpu_window.to(device)
        return window

    def transform(self, input_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        forward_transform = torch.stft(
            input_data,
            self.filter_length,
            self.hop_length,
            self.win_length,
            window=self._window(input_data.device),
            return_complex=True,
        )
        return torch.abs(forward_transform), torch.angle(forward_transform)

    def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        inverse_transform = torch.istft(
            magnitude * torch.exp(phase * 1j),
            self.filter_length,
            self.hop_length,
            self.win_length,
            window=self._window(magnitude.device),
        )
        return inverse_transform.unsqueeze(-2)


class SineGen(nn.Module):
    """Harmonic sine excitation from F0 (hn-NSF), voiced/unvoiced noise mix."""

    def __init__(
        self,
        samp_rate: int,
        upsample_scale: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.upsample_scale = upsample_scale
        self.register_buffer(
            "harmonics", torch.arange(1, harmonic_num + 2, dtype=torch.float32).view(1, 1, -1), persistent=False
        )

    def _f02sine(self, f0_values: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        # The integer part of f0/sr is dropped: 2*pi*n does not affect phase.
        rad_values = (f0_values / self.sampling_rate) % 1
        # Random initial phase for every harmonic except the fundamental.
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device, generator=generator)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        rad_values = F.interpolate(
            rad_values.transpose(1, 2), scale_factor=1 / self.upsample_scale, mode="linear"
        ).transpose(1, 2)
        phase = torch.cumsum(rad_values, dim=1) * 2 * torch.pi
        phase = F.interpolate(
            phase.transpose(1, 2) * self.upsample_scale, scale_factor=self.upsample_scale, mode="linear"
        ).transpose(1, 2)
        return torch.sin(phase)

    def forward(self, f0: torch.Tensor, generator: torch.Generator | None) -> tuple[torch.Tensor, torch.Tensor]:
        fn = f0 * self.harmonics
        sine_waves = self._f02sine(fn, generator) * self.sine_amp
        uv = (f0 > self.voiced_threshold).type(torch.float32)
        # Unvoiced noise std is sine_amp / 3 (peak ~sine_amp); voiced is noise_std.
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        # empty_like(...).normal_() is randn_like with a generator: it keeps the
        # (transposed) strides, so values land in the reference's memory order.
        noise = noise_amp * torch.empty_like(sine_waves).normal_(generator=generator)
        return sine_waves * uv + noise, uv


class SourceModuleHnNSF(nn.Module):
    def __init__(
        self,
        sampling_rate: int,
        upsample_scale: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshold: float = 0,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.l_sin_gen = SineGen(sampling_rate, upsample_scale, harmonic_num, sine_amp, add_noise_std, voiced_threshold)
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        sine_wavs, uv = self.l_sin_gen(x, generator)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        # The reference also draws a noise-branch tensor here that the decoder
        # never reads. Keep the draw so multi-chunk requests consume the RNG
        # stream exactly like the reference does.
        torch.empty_like(uv).normal_(generator=generator)
        return sine_merge


class Generator(nn.Module):
    """iSTFTNet generator: upsampling convs + harmonic source, then iSTFT."""

    def __init__(
        self,
        style_dim: int,
        resblock_kernel_sizes: list[int],
        upsample_rates: list[int],
        upsample_initial_channel: int,
        resblock_dilation_sizes: list[list[int]],
        upsample_kernel_sizes: list[int],
        gen_istft_n_fft: int,
        gen_istft_hop_size: int,
        sampling_rate: int,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        upsample_scale = math.prod(upsample_rates) * gen_istft_hop_size
        self.m_source = SourceModuleHnNSF(
            sampling_rate=sampling_rate, upsample_scale=upsample_scale, harmonic_num=8, voiced_threshold=10
        )
        self.f0_upsamp = nn.Upsample(scale_factor=upsample_scale)
        self.noise_convs = nn.ModuleList()
        self.noise_res = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(
                    upsample_initial_channel // (2**i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    k,
                    u,
                    padding=(k - u) // 2,
                )
            )
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(AdaINResBlock1(ch, k, d, style_dim))
            if i + 1 < len(upsample_rates):
                stride_f0 = math.prod(upsample_rates[i + 1 :])
                self.noise_convs.append(
                    nn.Conv1d(
                        gen_istft_n_fft + 2,
                        ch,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=(stride_f0 + 1) // 2,
                    )
                )
                self.noise_res.append(AdaINResBlock1(ch, 7, [1, 3, 5], style_dim))
            else:
                self.noise_convs.append(nn.Conv1d(gen_istft_n_fft + 2, ch, kernel_size=1))
                self.noise_res.append(AdaINResBlock1(ch, 11, [1, 3, 5], style_dim))
        self.post_n_fft = gen_istft_n_fft
        self.conv_post = nn.Conv1d(ch, self.post_n_fft + 2, 7, 1, padding=3)
        self.reflection_pad = nn.ReflectionPad1d((1, 0))
        self.stft = TorchSTFT(filter_length=gen_istft_n_fft, hop_length=gen_istft_hop_size, win_length=gen_istft_n_fft)

    def forward(
        self, x: torch.Tensor, s: torch.Tensor, f0: torch.Tensor, generator: torch.Generator | None
    ) -> torch.Tensor:
        f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)
        har_source = self.m_source(f0, generator).transpose(1, 2).squeeze(1)
        har_spec, har_phase = self.stft.transform(har_source)
        har = torch.cat([har_spec, har_phase], dim=1)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, negative_slope=0.1)
            x_source = self.noise_convs[i](har)
            x_source = self.noise_res[i](x_source, s)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                block_out = self.resblocks[i * self.num_kernels + j](x, s)
                xs = block_out if xs is None else xs + block_out
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        spec = torch.exp(x[:, : self.post_n_fft // 2 + 1, :])
        phase = torch.sin(x[:, self.post_n_fft // 2 + 1 :, :])
        return self.stft.inverse(spec, phase)


class AdainResBlk1d(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        style_dim: int = 64,
        upsample: bool = False,
    ):
        super().__init__()
        self.actv = nn.LeakyReLU(0.2)
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self.conv1 = nn.Conv1d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv1d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False)
        if upsample:
            self.pool = nn.ConvTranspose1d(
                dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in, padding=1, output_padding=1
            )
        else:
            self.pool = nn.Identity()

    def _shortcut(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.pool(x)
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        return self.conv2(x)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return (self._residual(x, s) + self._shortcut(x)) * _INV_SQRT2


class Decoder(nn.Module):
    def __init__(self, dim_in: int, style_dim: int, sampling_rate: int, istftnet: dict[str, Any]):
        super().__init__()
        self.encode = AdainResBlk1d(dim_in + 2, 1024, style_dim)
        self.decode = nn.ModuleList(
            [
                AdainResBlk1d(1024 + 2 + 64, 1024, style_dim),
                AdainResBlk1d(1024 + 2 + 64, 1024, style_dim),
                AdainResBlk1d(1024 + 2 + 64, 1024, style_dim),
                AdainResBlk1d(1024 + 2 + 64, 512, style_dim, upsample=True),
            ]
        )
        self.F0_conv = nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1)
        self.N_conv = nn.Conv1d(1, 1, kernel_size=3, stride=2, groups=1, padding=1)
        self.asr_res = nn.Sequential(nn.Conv1d(512, 64, kernel_size=1))
        self.generator = Generator(style_dim, sampling_rate=sampling_rate, **istftnet)

    def forward(
        self,
        asr: torch.Tensor,
        f0_curve: torch.Tensor,
        noise_curve: torch.Tensor,
        s: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        f0 = self.F0_conv(f0_curve.unsqueeze(1))
        noise = self.N_conv(noise_curve.unsqueeze(1))
        x = torch.cat([asr, f0, noise], dim=1)
        x = self.encode(x, s)
        asr_res = self.asr_res(asr)
        res = True
        for block in self.decode:
            if res:
                x = torch.cat([x, asr_res, f0, noise], dim=1)
            x = block(x, s)
            if block.upsample:
                res = False
        return self.generator(x, s, f0_curve, generator)


class TextEncoder(nn.Module):
    def __init__(self, channels: int, kernel_size: int, depth: int, n_symbols: int):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)
        padding = (kernel_size - 1) // 2
        self.cnn = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
                LayerNorm(channels),
                nn.LeakyReLU(0.2),
            )
            for _ in range(depth)
        )
        self.lstm = nn.LSTM(channels, channels // 2, 1, batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x).transpose(1, 2)  # [B, C, T]
        m = m.unsqueeze(1)
        x.masked_fill_(m, 0.0)
        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)
        x = x.transpose(1, 2)  # [B, T, C]
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths.cpu(), batch_first=True, enforce_sorted=False)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = x.transpose(-1, -2)
        x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]], device=x.device)
        x_pad[:, :, : x.shape[-1]] = x
        x_pad.masked_fill_(m, 0.0)
        return x_pad


class DurationEncoder(nn.Module):
    def __init__(self, sty_dim: int, d_model: int, nlayers: int):
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(nlayers):
            self.lstms.append(
                nn.LSTM(d_model + sty_dim, d_model // 2, num_layers=1, batch_first=True, bidirectional=True)
            )
            self.lstms.append(AdaLayerNorm(sty_dim, d_model))

    def forward(
        self, x: torch.Tensor, style: torch.Tensor, text_lengths: torch.Tensor, m: torch.Tensor
    ) -> torch.Tensor:
        masks = m
        x = x.permute(2, 0, 1)
        s = style.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat([x, s], dim=-1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
        x = x.transpose(0, 1).transpose(-1, -2)
        lengths = text_lengths.cpu()
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm):
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, 2, 0)], dim=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                x = x.transpose(-1, -2)
                x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
                block.flatten_parameters()
                x, _ = block(x)
                x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
                x = x.transpose(-1, -2)
                x_pad = torch.zeros([x.shape[0], x.shape[1], m.shape[-1]], device=x.device)
                x_pad[:, :, : x.shape[-1]] = x
                x = x_pad
        return x.transpose(-1, -2)


class ProsodyPredictor(nn.Module):
    def __init__(self, style_dim: int, d_hid: int, nlayers: int, max_dur: int):
        super().__init__()
        self.text_encoder = DurationEncoder(sty_dim=style_dim, d_model=d_hid, nlayers=nlayers)
        self.lstm = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        # Checkpoint keeps the reference's LinearNorm wrapper name.
        self.duration_proj = nn.Module()
        self.duration_proj.linear_layer = nn.Linear(d_hid, max_dur)
        self.shared = nn.LSTM(d_hid + style_dim, d_hid // 2, 1, batch_first=True, bidirectional=True)
        self.F0 = nn.ModuleList(
            [
                AdainResBlk1d(d_hid, d_hid, style_dim),
                AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True),
                AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim),
            ]
        )
        self.N = nn.ModuleList(
            [
                AdainResBlk1d(d_hid, d_hid, style_dim),
                AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True),
                AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim),
            ]
        )
        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1, 1, 0)

    def F0Ntrain(self, x: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, _ = self.shared(x.transpose(-1, -2))
        F0 = x.transpose(-1, -2)
        for block in self.F0:
            F0 = block(F0, s)
        F0 = self.F0_proj(F0)
        N = x.transpose(-1, -2)
        for block in self.N:
            N = block(N, s)
        N = self.N_proj(N)
        return F0.squeeze(1), N.squeeze(1)


class KokoroNetwork(nn.Module):
    """Phoneme ids + style vector -> 24 kHz waveform (one chunk, batch 1)."""

    def __init__(self, config: KokoroConfig):
        super().__init__()
        plbert = dict(config.plbert)
        self.bert = AlbertModel(AlbertConfig(vocab_size=config.n_token, **plbert))
        self.bert_encoder = nn.Linear(plbert["hidden_size"], config.hidden_dim)
        self.predictor = ProsodyPredictor(
            style_dim=config.style_dim, d_hid=config.hidden_dim, nlayers=config.n_layer, max_dur=config.max_dur
        )
        self.text_encoder = TextEncoder(
            channels=config.hidden_dim,
            kernel_size=config.text_encoder_kernel_size,
            depth=config.n_layer,
            n_symbols=config.n_token,
        )
        self.decoder = Decoder(
            dim_in=config.hidden_dim,
            style_dim=config.style_dim,
            sampling_rate=config.sample_rate,
            istftnet=dict(config.istftnet),
        )
        self.style_dim = config.style_dim

    @staticmethod
    def is_adain_affine_param(name: str) -> bool:
        """Whether ``name`` is an AdaIN instance-norm affine param (not in the checkpoint)."""
        return name.endswith((".norm.weight", ".norm.bias"))

    def reset_adain_affine(self) -> None:
        for module in self.modules():
            if isinstance(module, AdaIN1d):
                nn.init.ones_(module.norm.weight)
                nn.init.zeros_(module.norm.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        ref_s: torch.Tensor,
        speed: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(audio [T], pred_dur [L])`` for ``input_ids [1, L]``.

        ``input_ids`` already carries the boundary pad id (0) on both ends.
        ``ref_s`` is the ``[1, 2 * style_dim]`` voicepack row for this chunk:
        the first half conditions the decoder, the second half prosody.
        """
        device = input_ids.device
        input_lengths = torch.full((input_ids.shape[0],), input_ids.shape[-1], device=device, dtype=torch.long)
        text_mask = torch.arange(input_lengths.max(), device=device).unsqueeze(0).expand(input_lengths.shape[0], -1)
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))
        bert_dur = self.bert(input_ids, attention_mask=(~text_mask).int()).last_hidden_state
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, self.style_dim :]
        d = self.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = self.predictor.lstm(d)
        duration = self.predictor.duration_proj.linear_layer(x)
        duration = torch.sigmoid(duration).sum(dim=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        # Expand each phoneme to its predicted number of frames.
        indices = torch.repeat_interleave(torch.arange(input_ids.shape[1], device=device), pred_dur)
        pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=device)
        pred_aln_trg[indices, torch.arange(indices.shape[0], device=device)] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)
        en = d.transpose(-1, -2) @ pred_aln_trg
        F0_pred, N_pred = self.predictor.F0Ntrain(en, s)
        t_en = self.text_encoder(input_ids, input_lengths, text_mask)
        asr = t_en @ pred_aln_trg
        audio = self.decoder(asr, F0_pred, N_pred, ref_s[:, : self.style_dim], generator).squeeze()
        return audio, pred_dur
