# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from LongCat-AudioDiT (https://github.com/meituan-longcat/LongCat-AudioDiT)

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def _wn_conv1d(in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=0, bias=True):
    return weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias))


def _wn_conv_transpose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def _get_vae_activation(name: str, channels: int = 0):
    if name == "elu":
        return nn.ELU()
    elif name == "snake":
        return _Snake1d(channels)
    else:
        raise ValueError(f"Unknown activation: {name}")


class _Snake1d(nn.Module):
    def __init__(self, channels, alpha_logscale: bool = True):
        super().__init__()
        self.channels = channels
        self.alpha_logscale = alpha_logscale
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return x + (1.0 / (beta + 1e-9)) * torch.sin(x * alpha).pow(2)


def _pixel_unshuffle_1d(x: torch.Tensor, factor: int) -> torch.Tensor:
    b, c, w = x.size()
    return x.view(b, c, w // factor, factor).permute(0, 1, 3, 2).contiguous().view(b, c * factor, w // factor)


def _pixel_shuffle_1d(x: torch.Tensor, factor: int) -> torch.Tensor:
    b, c, w = x.size()
    return x.view(b, c // factor, factor, w).permute(0, 1, 3, 2).contiguous().view(b, c // factor, w * factor)


class _DownsampleShortcut(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: int):
        super().__init__()
        self.factor = factor
        self.group_size = in_channels * factor // out_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _pixel_unshuffle_1d(x, self.factor)
        b, c, n = x.shape
        return x.view(b, self.out_channels, self.group_size, n).mean(dim=2)


class _UpsampleShortcut(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor: int):
        super().__init__()
        self.factor = factor
        self.repeats = out_channels * factor // in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        return _pixel_shuffle_1d(x, self.factor)


class _VaeResidualUnit(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, dilation: int, kernel_size: int = 7, use_snake: bool = False
    ):
        super().__init__()
        padding = (dilation * (kernel_size - 1)) // 2
        act = "snake" if use_snake else "elu"
        self.layers = nn.Sequential(
            _get_vae_activation(act, channels=out_channels),
            _wn_conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding),
            _get_vae_activation(act, channels=out_channels),
            _wn_conv1d(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class _VaeEncoderBlock(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, stride: int, use_snake: bool = False, downsample_shortcut: str = "none"
    ):
        super().__init__()
        layers = []
        for d in [1, 3, 9]:
            layers.append(_VaeResidualUnit(in_ch, in_ch, dilation=d, use_snake=use_snake))
        act = "snake" if use_snake else "elu"
        layers.append(_get_vae_activation(act, channels=in_ch))
        layers.append(_wn_conv1d(in_ch, out_ch, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)))
        self.layers = nn.Sequential(*layers)
        self.res = _DownsampleShortcut(in_ch, out_ch, stride) if downsample_shortcut == "averaging" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.res is not None:
            return self.layers(x) + self.res(x)
        return self.layers(x)


class _VaeDecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, use_snake: bool = False, upsample_shortcut: str = "none"):
        super().__init__()
        act = "snake" if use_snake else "elu"
        layers = [
            _get_vae_activation(act, channels=in_ch),
            _wn_conv_transpose1d(in_ch, out_ch, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)),
        ]
        for d in [1, 3, 9]:
            layers.append(_VaeResidualUnit(out_ch, out_ch, dilation=d, use_snake=use_snake))
        self.layers = nn.Sequential(*layers)
        self.res = _UpsampleShortcut(in_ch, out_ch, stride) if upsample_shortcut == "duplicating" else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.res is not None:
            return self.layers(x) + self.res(x)
        return self.layers(x)


class AudioDiTVaeEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        channels: int = 128,
        c_mults=None,
        strides=None,
        latent_dim: int = 64,
        encoder_latent_dim: int = 128,
        use_snake: bool = True,
        downsample_shortcut: str = "averaging",
        out_shortcut: str = "averaging",
    ):
        super().__init__()
        c_mults = [1] + (c_mults or [1, 2, 4, 8, 16])
        ch = channels
        layers = [_wn_conv1d(in_channels, c_mults[0] * ch, kernel_size=7, padding=3)]
        for i in range(len(c_mults) - 1):
            layers.append(
                _VaeEncoderBlock(
                    c_mults[i] * ch,
                    c_mults[i + 1] * ch,
                    strides[i] if strides else 2,
                    use_snake=use_snake,
                    downsample_shortcut=downsample_shortcut,
                )
            )
        layers.append(_wn_conv1d(c_mults[-1] * ch, encoder_latent_dim, kernel_size=3, padding=1))
        self.layers = nn.Sequential(*layers)

        if out_shortcut == "averaging":
            self.shortcut = _DownsampleShortcut(c_mults[-1] * ch, encoder_latent_dim, 1)
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shortcut is None:
            return self.layers(x)
        x = self.layers[:-1](x)
        return self.layers[-1](x) + self.shortcut(x)


class AudioDiTVaeDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        channels: int = 128,
        c_mults=None,
        strides=None,
        latent_dim: int = 64,
        use_snake: bool = True,
        in_shortcut: str = "duplicating",
        final_tanh: bool = False,
        upsample_shortcut: str = "duplicating",
    ):
        super().__init__()
        c_mults = [1] + (c_mults or [1, 2, 4, 8, 16])
        ch = channels

        if in_shortcut == "duplicating":
            self.shortcut = _UpsampleShortcut(latent_dim, c_mults[-1] * ch, 1)
        else:
            self.shortcut = None

        layers = [_wn_conv1d(latent_dim, c_mults[-1] * ch, kernel_size=7, padding=3)]
        for i in range(len(c_mults) - 1, 0, -1):
            layers.append(
                _VaeDecoderBlock(
                    c_mults[i] * ch,
                    c_mults[i - 1] * ch,
                    strides[i - 1] if strides else 2,
                    use_snake=use_snake,
                    upsample_shortcut=upsample_shortcut,
                )
            )
        act = "snake" if use_snake else "elu"
        layers.append(_get_vae_activation(act, channels=c_mults[0] * ch))
        layers.append(_wn_conv1d(c_mults[0] * ch, in_channels, kernel_size=7, padding=3, bias=False))
        if final_tanh:
            layers.append(nn.Tanh())
        else:
            layers.append(nn.Identity())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shortcut is None:
            return self.layers(x)
        x_short = self.shortcut(x) + self.layers[0](x)
        return self.layers[1:](x_short)


class LongCatAudioDiTVae(nn.Module):
    """WAV-VAE audio autoencoder for LongCat-AudioDiT.

    Supports encode/decode between waveform and latent space.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: int = 128,
        c_mults=None,
        strides=None,
        latent_dim: int = 64,
        encoder_latent_dim: int = 128,
        use_snake: bool = True,
        downsample_shortcut: str = "averaging",
        upsample_shortcut: str = "duplicating",
        out_shortcut: str = "averaging",
        in_shortcut: str = "duplicating",
        final_tanh: bool = False,
        downsampling_ratio: int = 2048,
        sample_rate: int = 24000,
        scale: float = 0.71,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.downsampling_ratio = downsampling_ratio
        self.sample_rate = sample_rate
        self.scale = scale

        self.encoder = AudioDiTVaeEncoder(
            in_channels=in_channels,
            channels=channels,
            c_mults=c_mults,
            strides=strides,
            latent_dim=latent_dim,
            encoder_latent_dim=encoder_latent_dim,
            use_snake=use_snake,
            downsample_shortcut=downsample_shortcut,
            out_shortcut=out_shortcut,
        )

        self.decoder = AudioDiTVaeDecoder(
            in_channels=in_channels,
            channels=channels,
            c_mults=c_mults,
            strides=strides,
            latent_dim=latent_dim,
            use_snake=use_snake,
            in_shortcut=in_shortcut,
            final_tanh=final_tanh,
            upsample_shortcut=upsample_shortcut,
        )

    def to_half(self):
        self.encoder.half()
        self.decoder.half()
        return self

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        is_half = next(self.encoder.parameters()).dtype == torch.float16
        if is_half:
            audio = audio.half()
        latents = self.encoder(audio)
        mean, scale_param = latents.chunk(2, dim=1)
        stdev = F.softplus(scale_param) + 1e-4
        latents = torch.randn_like(mean) * stdev + mean
        if is_half:
            latents = latents.float()
        return latents / self.scale

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        z = latents * self.scale
        is_half = next(self.decoder.parameters()).dtype == torch.float16
        if is_half:
            z = z.half()
        decoded = self.decoder(z)
        if is_half:
            decoded = decoded.float()
        return decoded
