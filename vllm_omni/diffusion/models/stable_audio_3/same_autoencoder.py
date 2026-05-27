# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SAME (Semantic-Acoustic Music Encoder) autoencoder for Stable Audio 3.

PORT_FROM: stable_audio_3/models/autoencoders.py (638 lines)
           stable_audio_3/models/bottleneck.py (71 lines, ported in full below)
           stable_audio_3/models/pretransforms.py (84 lines, ported in full below)
           stable_audio_3/models/blocks.py (parts: ResidualUnit, WNConv1d)

The autoencoder is a 4-layer composition:

  AutoencoderPretransform           ← outer wrapper, exposes chunked decode
    └─ AudioAutoencoder              ← top-level facade
         ├─ encoder = SAMEEncoder   ← transformer-based downsampling
         ├─ bottleneck = SoftNormBottleneck  ← learnable latent scaling
         ├─ decoder = SAMEDecoder   ← transformer-based upsampling
         └─ pretransform = PatchedPretransform  ← channel↔time folding

Variants per model_configs.py:
  SAME-S (433M small variants): smaller encoder/decoder
  SAME-L (medium variant):       1.4B-class, used by stable-audio-3-medium

Chunked decode defaults (from upstream decode_audio at line 596):
  chunk_size = 128 latents
  overlap    = 32 latents
"""

from __future__ import annotations

from typing import Any

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from torchaudio.transforms import Resample


# ---------------------------------------------------------------------------
# Small helpers (PORT_FROM: blocks.py)
# ---------------------------------------------------------------------------


def WNConv1d(*args, **kwargs):
    """PORT_FROM: blocks.py:16-17"""
    return weight_norm(nn.Conv1d(*args, **kwargs))


def get_activation(activation: str, channels: int | None = None) -> nn.Module:
    """PORT_FROM: blocks.py:6-14"""
    if activation == "elu":
        return nn.ELU()
    if activation == "none":
        return nn.Identity()
    raise ValueError(f"Unknown activation {activation}")


class ResidualUnit(nn.Module):
    """PORT_FROM: blocks.py:19-39"""

    def __init__(self, in_channels: int, out_channels: int, dilation: int, depthwise: bool = False, bias: bool = True) -> None:
        super().__init__()
        self.dilation = dilation
        padding = (dilation * (7 - 1)) // 2
        self.layers = nn.Sequential(
            get_activation("elu"),
            WNConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                dilation=dilation,
                padding=padding,
                groups=1 if not depthwise else out_channels,
                bias=bias,
            ),
            get_activation("elu"),
            WNConv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


# ---------------------------------------------------------------------------
# Bottleneck (PORT_FROM: bottleneck.py — full file, 71 lines)
# ---------------------------------------------------------------------------


class SoftNormBottleneck(nn.Module):
    """Learnable scaling/bias of the latent space, with optional auto-scale + noise augment.

    PORT_FROM: bottleneck.py SoftNormBottleneck (lines 4-71)
    """

    def __init__(
        self,
        dim: int = 32,
        noise_augment_dim: int = 0,
        noise_regularize: bool = False,
        auto_scale: bool = False,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        self.noise_augment_dim = noise_augment_dim
        self.noise_regularize = noise_regularize
        self.freeze = freeze

        self.scaling_factor = nn.Parameter(torch.ones(1, dim, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1))
        self.noise_scaling_factor = nn.Parameter(torch.ones(1, noise_augment_dim, 1))

        if freeze:
            self.scaling_factor.requires_grad = False
            self.bias.requires_grad = False
            self.noise_scaling_factor.requires_grad = False
        if auto_scale:
            self.register_parameter("running_std", nn.Parameter(torch.ones(1), requires_grad=False))

    def encode(self, x: torch.Tensor, return_info: bool = False, **kwargs):
        # PORT_FROM: bottleneck.py:23-50 (encode body)
        raise NotImplementedError

    def decode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # PORT_FROM: bottleneck.py:52-71 (decode body)
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Pretransforms (PORT_FROM: pretransforms.py — full file, 84 lines)
# ---------------------------------------------------------------------------


class AutoencoderPretransform(nn.Module):
    """Outer wrapper around AudioAutoencoder. Adds chunked-decode + scale.

    PORT_FROM: pretransforms.py AutoencoderPretransform (lines 7-25)
    """

    def __init__(self, model: nn.Module, scale: float = 1.0, iterate_batch: bool = False, chunked: bool = False) -> None:
        super().__init__()
        self.model = model
        self.model.requires_grad_(False).eval()
        self.scale = scale
        self.downsampling_ratio = model.downsampling_ratio
        self.io_channels = model.io_channels
        self.enable_grad = False
        self.iterate_batch = iterate_batch
        self.chunked = chunked

    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model.encode_audio(x, chunked=self.chunked, iterate_batch=self.iterate_batch, **kwargs) / self.scale

    def decode(self, z: torch.Tensor, chunked: bool | None = None, **kwargs) -> torch.Tensor:
        chunked = self.chunked if chunked is None else chunked
        return self.model.decode_audio(z * self.scale, chunked=chunked, iterate_batch=self.iterate_batch, **kwargs)


class PatchedPretransform(nn.Module):
    """Channel↔time folding: [B,C,L*H] ↔ [B,C*H,L].

    PORT_FROM: pretransforms.py PatchedPretransform (lines 37-84)
    """

    def __init__(
        self,
        channels: int,
        patch_size: int,
        oversampling: int = 1,
        postfilter_channels: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.patch_size = patch_size
        self.oversampling = oversampling
        self.downsampling_ratio = patch_size
        self.io_channels = channels
        self.encoded_channels = channels * patch_size
        self.enable_grad = False

        if oversampling > 1:
            self.input_upsampler = Resample(1, oversampling)
            self.output_downsampler = Resample(oversampling, 1)

        if postfilter_channels > 0:
            # PORT_FROM: pretransforms.py:51-65 — postfilter Sequential
            raise NotImplementedError

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-1]
        pad_len = (self.patch_size - (seq_len % self.patch_size)) % self.patch_size
        if pad_len > 0:
            x = torch.cat([x, torch.zeros_like(x[:, :, :pad_len])], dim=-1)
        return x

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.oversampling > 1:
            x = self.input_upsampler(x)
        x = self._pad(x)
        x = rearrange(x, "b c (l h) -> b (c h) l", h=self.patch_size)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b (c h) l -> b c (l h)", h=self.patch_size)
        if hasattr(self, "postfilter"):
            x = self.postfilter(x)
        if self.oversampling > 1:
            x = self.output_downsampler(x)
        return x


# ---------------------------------------------------------------------------
# SAMEEncoder / SAMEDecoder (PORT_FROM: autoencoders.py)
# Transformer-based U-Net-style audio (de)compressors with stride [2,4,8,8].
# ---------------------------------------------------------------------------


class TransformerResamplingBlock(nn.Module):
    """Learned resampling block with attention. Used by both SAMEEncoder and SAMEDecoder.

    PORT_FROM: autoencoders.py:34-223 (very long class)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        sliding_window: int | None = None,
        chunk_size: int = 128,
        chunk_midpoint_shift: bool = False,
        type: str = "encoder",
        transformer_depth: int = 3,
        checkpointing: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        # PORT_FROM: autoencoders.py:34-220
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SAMEEncoder(nn.Module):
    """Multi-stage transformer-based audio encoder.

    PORT_FROM: autoencoders.py:225-288
    """

    def __init__(
        self,
        in_channels: int = 2,
        channels: int = 128,
        latent_dim: int = 32,
        c_mults: list[int] | None = None,
        strides: list[int] | None = None,
        transformer_depths: list[int] | None = None,
        sliding_window: int | None = None,
        checkpointing: bool = False,
        conformer: bool = False,
        layer_scale: bool = False,
        causal: bool = False,
        differential: bool = True,
        variable_stride: bool = False,
        mask_noise: float = 0.0,
        conv_mapping: bool = False,
        freeze_backbone: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.strides = strides or [2, 4, 8, 8]
        # PORT_FROM: autoencoders.py:225-288
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SAMEDecoder(nn.Module):
    """Multi-stage transformer-based audio decoder. Mirror of SAMEEncoder.

    PORT_FROM: autoencoders.py:290-349
    """

    def __init__(
        self,
        out_channels: int = 2,
        channels: int = 128,
        latent_dim: int = 32,
        c_mults: list[int] | None = None,
        strides: list[int] | None = None,
        transformer_depths: list[int] | None = None,
        sliding_window: int | None = None,
        checkpointing: bool = False,
        conformer: bool = False,
        layer_scale: bool = False,
        causal: bool = False,
        differential: bool = True,
        variable_stride: bool = False,
        sinusoidal_blocks: list[int] | None = None,
        mask_noise: float = 0.0,
        conv_mapping: bool = False,
        freeze_backbone: bool = False,
        soft_clip: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        # PORT_FROM: autoencoders.py:290-349
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# AudioAutoencoder — top-level facade (PORT_FROM: autoencoders.py:351-638)
# Exposes .encode_audio() / .decode_audio() with optional chunked decode.
# ---------------------------------------------------------------------------


class AudioAutoencoder(nn.Module):
    """SAME autoencoder facade.

    Defaults from upstream decode_audio (line 596):
      chunked decode: chunk_size=128 latents, overlap=32 latents.
    """

    def __init__(
        self,
        encoder: SAMEEncoder,
        decoder: SAMEDecoder,
        io_channels: int = 2,
        latent_dim: int = 32,
        downsampling_ratio: int = 512,
        sample_rate: int = 44100,
        bottleneck: SoftNormBottleneck | None = None,
        pretransform: PatchedPretransform | None = None,
        in_channels: int | None = None,
        out_channels: int | None = None,
        soft_clip: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bottleneck = bottleneck
        self.pretransform = pretransform
        self.io_channels = io_channels
        self.latent_dim = latent_dim
        self.downsampling_ratio = downsampling_ratio
        self.sample_rate = sample_rate
        self.soft_clip = soft_clip

        # vllm-omni accessor: pipeline reads `self.vae.config.sampling_rate`
        class _Cfg:
            pass

        self.config = _Cfg()
        self.config.sampling_rate = sample_rate
        self.config.latent_channels = latent_dim
        self.config.downsampling_ratio = downsampling_ratio

    def encode(self, x: torch.Tensor, iterate_batch: bool = False, **kwargs) -> torch.Tensor:
        # PORT_FROM: autoencoders.py:~410-450 (encode)
        raise NotImplementedError

    def decode(self, latents: torch.Tensor, iterate_batch: bool = False, return_loss: bool = False, **kwargs) -> torch.Tensor:
        # PORT_FROM: autoencoders.py:451-495 (decode)
        raise NotImplementedError

    def encode_audio(self, audio: torch.Tensor, chunked: bool = False, overlap: int = 32, chunk_size: int = 128, **kwargs) -> torch.Tensor:
        # PORT_FROM: autoencoders.py: encode_audio (mirrors decode_audio structure)
        raise NotImplementedError

    def decode_audio(
        self,
        latents: torch.Tensor,
        chunked: bool = False,
        overlap: int = 32,
        chunk_size: int = 128,
        **kwargs,
    ) -> torch.Tensor:
        """Decode latents → waveform. Chunked decode caps VRAM for long clips.

        PORT_FROM: autoencoders.py:596-638 — verbatim port. Defaults match upstream.
        """
        raise NotImplementedError
