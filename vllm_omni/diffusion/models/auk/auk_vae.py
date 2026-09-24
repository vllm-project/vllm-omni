# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# Adapted from the AuK reference release (MIT), which in turn adapts NVIDIA
# BigVGAN (MIT), jik876/hifi-gan (MIT), junjun3518/alias-free-torch
# (Apache-2.0), EdwardDixon/snake (MIT) and adefossez/julius (MIT).
"""BigVGAN-flow VAE for AuK: 24 kHz mono waveform <-> 50 Hz, 64-channel latents.

Inference only. The reference module also carries a residual coupling flow, a KL
term and a discriminator; all three are training-only, so the released
``flow.*`` tensors are skipped (see :meth:`AuKVAE.load_weights`).

Parameter and buffer names follow the reference release, so the published
``vae.safetensors`` loads without a key map. That is also why the convolutions
are built with the legacy ``weight_norm`` parametrization (``weight_g`` /
``weight_v``): :meth:`remove_weight_norm` folds them into plain weights once
the checkpoint is in.
"""

from __future__ import annotations

import inspect
import math
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn
from torch.nn.utils import remove_weight_norm as _fold_weight_norm
from torch.nn.utils import weight_norm as _apply_weight_norm
from vllm.logger import init_logger

logger = init_logger(__name__)

__all__ = ["AuKVAE"]

# Checkpoint prefix owned by the training-only coupling flow.
TRAINING_ONLY_PREFIX = "flow."


def _weight_norm(module: nn.Module) -> nn.Module:
    """Legacy ``weight_g``/``weight_v`` split, which is how the release stores these tensors."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*", category=FutureWarning)
        return _apply_weight_norm(module)


def _same_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def _kaiser_sinc_filter(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    """Kaiser-windowed sinc low-pass, returned as a ``[1, 1, kernel_size]`` conv kernel."""
    half_size = kernel_size // 2
    attenuation = 2.285 * (half_size - 1) * math.pi * (4 * half_width) + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21.0) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    if kernel_size % 2 == 0:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        return torch.zeros_like(time).view(1, 1, kernel_size)
    taps = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    # Normalize to unit sum so a constant input passes through unscaled.
    return (taps / taps.sum()).view(1, 1, kernel_size)


class Conv(nn.Conv1d):
    """1-D convolution over ``[B, C, T]``; ``causal`` swaps centred padding for a left pad."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        causal: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0 if causal else _same_padding(kernel_size, dilation),
            dilation=dilation,
            bias=bias,
        )
        self.causal = causal
        self.left_padding = dilation * (kernel_size - 1) if causal else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.causal:
            x = F.pad(x, (self.left_padding, 0))
        return super().forward(x)


class ConvTranspose(nn.ConvTranspose1d):
    """Transposed 1-D convolution; ``causal`` drops the trailing ``stride`` samples instead of padding."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int, causal: bool = False
    ) -> None:
        if causal and kernel_size != 2 * stride:
            raise ValueError(f"causal ConvTranspose needs kernel_size == 2 * stride, got {kernel_size} and {stride}")
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0 if causal else (kernel_size - stride) // 2,
        )
        self.causal = causal
        self.trim = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        return x[:, :, : -self.trim] if self.causal else x


class NormConv(nn.Module):
    """Weight-normalized conv held under ``layer``; the attribute name is load-bearing for the checkpoint."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, stride: int = 1) -> None:
        super().__init__()
        self.layer = _weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=(kernel_size - 1) // 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class ResStack(nn.Module):
    """Stack of dilated residual conv pairs. The leaky slope is torch's 0.01 default, as upstream."""

    def __init__(self, channels: int, kernel_size: int = 3, base: int = 3, nums: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LeakyReLU(),
                    _weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=base**i, padding=base**i)),
                    nn.LeakyReLU(),
                    _weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=1, padding=1)),
                )
                for i in range(nums)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)
        return x


class Encoder(nn.Module):
    """Strided conv encoder over ``[B, 1, T]``, emitting ``[B, 2 * latent_dim, T / hop]`` mean/log-std pairs."""

    def __init__(
        self,
        latent_dim: int,
        channels: tuple[int, ...],
        down_sample_factors: tuple[int, ...],
        in_channels: int = 1,
        proj_kernel_size: int = 3,
        stack_kernel_size: int = 3,
        stack_dilation_base: int = 2,
        stacks: int = 6,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            NormConv(in_channels, channels[0], kernel_size=proj_kernel_size),
            nn.LeakyReLU(0.2, True),
        ]
        for in_c, out_c, factor in zip(channels[:-1], channels[1:], down_sample_factors, strict=True):
            layers += [
                NormConv(in_c, out_c, kernel_size=factor * 2, stride=factor),
                ResStack(out_c, stack_kernel_size, stack_dilation_base, stacks),
                nn.LeakyReLU(0.2, True),
            ]
        layers.append(NormConv(channels[-1], latent_dim * 2, kernel_size=proj_kernel_size))
        self.generator = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)


class SnakeBeta(nn.Module):
    """``x + sin^2(alpha * x) / beta`` with per-channel learned alpha and beta (log-scale in AuK)."""

    def __init__(self, channels: int, alpha_logscale: bool = False) -> None:
        super().__init__()
        self.alpha_logscale = alpha_logscale
        init = torch.zeros(channels) if alpha_logscale else torch.ones(channels)
        self.alpha = nn.Parameter(init.clone())
        self.beta = nn.Parameter(init.clone())
        self.eps = 1e-9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return x + (1.0 / (beta + self.eps)) * torch.sin(x * alpha).pow(2)


class LowPass(nn.Module):
    """Fixed FIR low-pass with replicate padding, applied per channel."""

    def __init__(
        self,
        cutoff: float = 0.5,
        half_width: float = 0.6,
        stride: int = 1,
        kernel_size: int = 12,
        causal: bool = False,
    ) -> None:
        super().__init__()
        if not 0.0 <= cutoff <= 0.5:
            raise ValueError(f"cutoff must lie in [0, 0.5], got {cutoff}")
        if causal:
            self.pad_left, self.pad_right = kernel_size - 1, 0
        else:
            even = kernel_size % 2 == 0
            self.pad_left = kernel_size // 2 - int(even)
            self.pad_right = kernel_size // 2
        self.stride = stride
        self.register_buffer("filter", _kaiser_sinc_filter(cutoff, half_width, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.size(1)
        x = F.pad(x, (self.pad_left, self.pad_right), mode="replicate")
        return F.conv1d(x, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels)


class Upsample(nn.Module):
    """Band-limited interpolation by ``ratio``. Always non-causal, matching how AuK builds it."""

    def __init__(self, ratio: int = 2, kernel_size: int = 12) -> None:
        super().__init__()
        self.ratio = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * ratio + (kernel_size - ratio) // 2
        self.pad_right = self.pad * ratio + (kernel_size - ratio + 1) // 2
        self.register_buffer("filter", _kaiser_sinc_filter(0.5 / ratio, 0.6 / ratio, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.size(1)
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(x, self.filter.expand(channels, -1, -1), stride=self.ratio, groups=channels)
        return x[..., self.pad_left : -self.pad_right]


class Downsample(nn.Module):
    """Decimation by ``ratio`` behind a low-pass; ``lowpass`` holds the filter buffer."""

    def __init__(self, ratio: int = 2, kernel_size: int = 12, causal: bool = False) -> None:
        super().__init__()
        self.lowpass = LowPass(
            cutoff=0.5 / ratio, half_width=0.6 / ratio, stride=ratio, kernel_size=kernel_size, causal=causal
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lowpass(x)


class AliasFreeActivation(nn.Module):
    """Oversample, apply the pointwise activation, decimate, so snake harmonics do not fold back."""

    def __init__(
        self,
        activation: nn.Module,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.act = activation
        self.upsample = Upsample(up_ratio, up_kernel_size)
        self.downsample = Downsample(down_ratio, down_kernel_size, causal=causal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(self.act(self.upsample(x)))


class AmpBlock(nn.Module):
    """Anti-aliased multi-periodicity residual block: alternating alias-free snake and dilated convs."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: tuple[int, ...],
        snake_logscale: bool = True,
        causal: bool = True,
        act_causal: bool = False,
    ) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList(
            [_weight_norm(Conv(channels, channels, kernel_size, 1, dilation=d, causal=causal)) for d in dilations]
        )
        self.convs2 = nn.ModuleList(
            [_weight_norm(Conv(channels, channels, kernel_size, 1, dilation=1, causal=causal)) for _ in dilations]
        )
        self.activations = nn.ModuleList(
            [
                AliasFreeActivation(SnakeBeta(channels, alpha_logscale=snake_logscale), causal=act_causal)
                for _ in range(len(self.convs1) + len(self.convs2))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for conv1, conv2, act1, act2 in zip(self.convs1, self.convs2, acts1, acts2, strict=True):
            x = conv2(act2(conv1(act1(x)))) + x
        return x


class AuKVAE(nn.Module):
    """AuK's audio codec: :meth:`encode` waveform to normalized latents, :meth:`decode` back.

    Build from the released ``model.vae.model_init_kwargs`` with :meth:`from_config`, then
    :meth:`load_weights`. Latents are normalized by the checkpoint's global statistics, which is
    the space the rectified-flow transformer works in.
    """

    def __init__(
        self,
        *,
        upsample_rates: tuple[int, ...] = (5, 4, 3, 2, 2, 2),
        upsample_kernel_sizes: tuple[int, ...] = (10, 8, 6, 4, 4, 4),
        upsample_initial_channel: int = 1536,
        resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11),
        resblock_dilation_sizes: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        downsample_rates: tuple[int, ...] = (2, 2, 2, 3, 4, 5),
        downsample_channels: tuple[int, ...] = (12, 24, 48, 96, 192, 384, 768),
        snake_logscale: bool = True,
        latent_dim: int = 64,
        use_vae: bool = True,
        causal: bool = True,
        act_causal: bool = True,
        sample_rate: int = 24000,
    ) -> None:
        super().__init__()
        if not use_vae:
            raise ValueError("AuKVAE requires use_vae=True: the encoder head emits mean/log-std pairs")
        self.latent_dim = latent_dim
        self.sample_rate = sample_rate
        self.hop_size = math.prod(downsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self._norm_folded = False

        self.register_buffer("global_mean", torch.zeros(latent_dim, dtype=torch.float32))
        self.register_buffer("global_log_std", torch.ones(latent_dim, dtype=torch.float32))

        self.audio_encoder = Encoder(
            latent_dim=latent_dim,
            channels=tuple(downsample_channels),
            down_sample_factors=tuple(downsample_rates),
        )

        self.conv_pre = _weight_norm(Conv(latent_dim, upsample_initial_channel, 7, 1, causal=False))
        self.ups = nn.ModuleList()
        for i, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes, strict=True)):
            # One-element inner list: the checkpoint keys are ``ups.<i>.0.*``.
            self.ups.append(
                nn.ModuleList(
                    [
                        _weight_norm(
                            ConvTranspose(
                                upsample_initial_channel // (2**i),
                                upsample_initial_channel // (2 ** (i + 1)),
                                kernel,
                                rate,
                                causal=causal,
                            )
                        )
                    ]
                )
            )

        self.resblocks = nn.ModuleList()
        channels = upsample_initial_channel
        for i in range(self.num_upsamples):
            channels = upsample_initial_channel // (2 ** (i + 1))
            for kernel, dilations in zip(resblock_kernel_sizes, resblock_dilation_sizes, strict=True):
                self.resblocks.append(
                    AmpBlock(
                        channels,
                        kernel,
                        tuple(dilations),
                        snake_logscale=snake_logscale,
                        causal=causal,
                        act_causal=act_causal,
                    )
                )

        self.activation_post = AliasFreeActivation(
            SnakeBeta(channels, alpha_logscale=snake_logscale), causal=act_causal
        )
        self.conv_post = _weight_norm(Conv(channels, 1, 7, 1, causal=causal, bias=False))

    @classmethod
    def from_config(cls, cfg: dict) -> AuKVAE:
        """Build from the released ``model_init_kwargs`` mapping, ignoring training-only entries."""
        accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
        return cls(**{key: value for key, value in cfg.items() if key in accepted})

    def load_weights(
        self, path: str | Path, *, device: str | torch.device | None = None, fold_weight_norm: bool = True
    ) -> tuple[list[str], list[str]]:
        """Load a released ``vae.safetensors``, then fold weight norm away. Returns (missing, unexpected).

        ``device`` defaults to where the module already lives, which is the placement that matters:
        the fold reduces ``weight_v`` to a norm, and a CPU reduction lands up to one ULP away from the
        accelerator's. The encoder amplifies its activations roughly fiftyfold, so that one ULP grows
        into a ~4e-4 relative drift by the latent head. Move the module first, then load.
        """
        target = self.global_mean.device if device is None else torch.device(device)
        missing, unexpected = self.load_state_dict(load_file(str(path), device=str(target)), strict=False)
        missing, unexpected = list(missing), list(unexpected)
        stray = [key for key in missing + unexpected if not key.startswith(TRAINING_ONLY_PREFIX)]
        if stray:
            raise ValueError(f"AuK VAE weights at {path} do not match this module: {sorted(stray)[:8]}")
        logger.info(
            "Loaded AuK VAE from %s (skipped %d training-only %s* tensors)", path, len(unexpected), TRAINING_ONLY_PREFIX
        )
        if fold_weight_norm:
            self.remove_weight_norm()
        return missing, unexpected

    def remove_weight_norm(self) -> None:
        """Fold ``weight_g``/``weight_v`` into plain weights, on whichever device the module is on.

        Idempotent, and meant to run once the checkpoint is in. See :meth:`load_weights` for why the
        device this runs on is a numeric decision rather than a convenience.
        """
        if self._norm_folded:
            return
        folded = 0
        for module in self.modules():
            if hasattr(module, "weight_g"):
                _fold_weight_norm(module)
                folded += 1
        self._norm_folded = True
        logger.debug("Folded weight norm on %d AuK VAE convolutions", folded)

    def encode(
        self, wav: torch.Tensor, *, sample: bool = False, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Encode a mono ``[1, T]`` waveform at :attr:`sample_rate` into ``[1, T / hop_size, latent_dim]``.

        ``sample=False`` takes the posterior mean, which is what a reproducible pipeline wants;
        ``sample=True`` draws ``mean + randn * exp(log_std)`` from ``generator``. Latents come back
        normalized by the checkpoint's global statistics.
        """
        if wav.dim() != 2 or wav.size(0) != 1:
            raise ValueError(f"encode expects a mono [1, T] waveform, got {tuple(wav.shape)}")
        # Statistics stay in fp32 even under an ambient autocast, as in the reference.
        with torch.autocast(device_type=wav.device.type, enabled=False):
            mean, log_std = self.audio_encoder(wav.unsqueeze(1)).chunk(2, dim=1)
            if sample:
                latents = mean + self._noise_like(mean, generator) * torch.exp(log_std)
            else:
                latents = mean
            latents = latents.transpose(1, 2).float()
            return (latents - self.global_mean.float()) / torch.sqrt(self.global_log_std.float())

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode normalized ``[B, Np, latent_dim]`` latents into a ``[B, Np * hop_size]`` waveform in [-1, 1]."""
        if latents.dim() != 3 or latents.size(-1) != self.latent_dim:
            raise ValueError(f"decode expects [B, Np, {self.latent_dim}] latents, got {tuple(latents.shape)}")
        x = latents.float() * torch.sqrt(self.global_log_std.float()) + self.global_mean.float()
        x = self.conv_pre(x.permute(0, 2, 1))
        for i in range(self.num_upsamples):
            for up in self.ups[i]:
                x = up(x)
            first = i * self.num_kernels
            residual = self.resblocks[first](x)
            for j in range(1, self.num_kernels):
                residual = residual + self.resblocks[first + j](x)
            x = residual / self.num_kernels
        x = self.conv_post(self.activation_post(x))
        return torch.clamp(x, min=-1.0, max=1.0).squeeze(1)

    @staticmethod
    def _noise_like(reference: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        """Draw standard normal noise shaped like ``reference``; ``generator`` decides which RNG stream."""
        device = reference.device if generator is None else generator.device
        noise = torch.randn(reference.shape, generator=generator, dtype=reference.dtype, device=device)
        return noise.to(reference.device)
