# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adopted from the fish-speech 0.1.0 PyPI release (Apache-2.0)
# https://pypi.org/project/fish-speech/0.1.0/
# Copyright (c) Fish Audio
"""DAC codec model (encoder / quantizer / decoder).

Differences from upstream ``fish_speech.models.dac.modded_dac.DAC``, none of
which affect the parameter tree or the inference math:

- Base class is plain ``nn.Module`` instead of ``(audiotools BaseModel,
  descript CodecMixin)`` -- both bases only add training/CLI conveniences.
- ``self.delay = self.get_delay()`` (a CodecMixin helper) is dropped; nothing
  in the inference paths reads it.
- The training-only ``forward``/``preprocess`` methods are dropped; only
  ``encode`` and ``decode`` are used at inference time.
"""

import math

import torch
from torch import nn

from vllm_omni.model_executor.models.fish_speech.dac_modules.conv import (
    CausalWNConv1d,
    CausalWNConvTranspose1d,
)
from vllm_omni.model_executor.models.fish_speech.dac_modules.layers import (
    Snake1d,
    WNConv1d,
    WNConvTranspose1d,
)
from vllm_omni.model_executor.models.fish_speech.dac_modules.transformer import (
    ModelArgs,
    WindowLimitedTransformer,
)


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1, causal: bool = False):
        super().__init__()
        conv_class = CausalWNConv1d if causal else WNConv1d
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            conv_class(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            conv_class(dim, dim, kernel_size=1),
        )
        self.causal = causal

    def forward(self, x):
        y = self.block(x)
        pad = x.shape[-1] - y.shape[-1]
        if pad > 0:
            if self.causal:
                x = x[..., :-pad]
            else:
                x = x[..., pad // 2 : -pad // 2]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int = 16,
        stride: int = 1,
        causal: bool = False,
        n_t_layer: int = 0,
        transformer_general_config=None,
    ):
        super().__init__()
        conv_class = CausalWNConv1d if causal else WNConv1d
        transformer_module = (
            nn.Identity()
            if n_t_layer == 0
            else (
                WindowLimitedTransformer(
                    causal=causal,
                    input_dim=dim,
                    window_size=512,
                    config=transformer_general_config(
                        n_layer=n_t_layer,
                        n_head=dim // 64,
                        dim=dim,
                        intermediate_size=dim * 3,
                    ),
                )
            )
        )
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1, causal=causal),
            ResidualUnit(dim // 2, dilation=3, causal=causal),
            ResidualUnit(dim // 2, dilation=9, causal=causal),
            Snake1d(dim // 2),
            conv_class(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            transformer_module,
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
        n_transformer_layers: list = [0, 0, 4, 4],
        transformer_general_config: ModelArgs = None,
        causal: bool = False,
    ):
        super().__init__()
        conv_class = CausalWNConv1d if causal else WNConv1d
        # Create first convolution
        self.block = [conv_class(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride, n_t_layer in zip(strides, n_transformer_layers):
            d_model *= 2
            self.block += [
                EncoderBlock(
                    d_model,
                    stride=stride,
                    causal=causal,
                    n_t_layer=n_t_layer,
                    transformer_general_config=transformer_general_config,
                )
            ]

        # Create last convolution
        self.block += [
            Snake1d(d_model),
            conv_class(d_model, d_latent, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 8,
        stride: int = 1,
        causal: bool = False,
        n_t_layer: int = 0,
        transformer_general_config=None,
    ):
        super().__init__()
        conv_trans_class = CausalWNConvTranspose1d if causal else WNConvTranspose1d
        # NOTE: upstream constructs a WindowLimitedTransformer from
        # ``n_t_layer`` here but never registers it (the line placing it in
        # ``self.block`` is commented out upstream), so decoder blocks carry no
        # transformer parameters. It is not built here at all -- same module
        # tree, without allocating throwaway weights.
        del n_t_layer, transformer_general_config
        self.block = nn.Sequential(
            Snake1d(input_dim),
            conv_trans_class(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnit(output_dim, dilation=1, causal=causal),
            ResidualUnit(output_dim, dilation=3, causal=causal),
            ResidualUnit(output_dim, dilation=9, causal=causal),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
        causal: bool = False,
        n_transformer_layers: list = [0, 0, 0, 0],
        transformer_general_config=None,
    ):
        super().__init__()
        conv_class = CausalWNConv1d if causal else WNConv1d
        # Add first conv layer
        layers = [conv_class(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, (stride, n_t_layer) in enumerate(zip(rates, n_transformer_layers)):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [
                DecoderBlock(
                    input_dim,
                    output_dim,
                    stride,
                    causal=causal,
                    n_t_layer=n_t_layer,
                    transformer_general_config=transformer_general_config,
                )
            ]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            conv_class(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DAC(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: list[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: list[int] = [8, 8, 4, 2],
        quantizer: torch.nn.Module = None,
        sample_rate: int = 44100,
        causal: bool = True,
        encoder_transformer_layers: list[int] = [0, 0, 0, 0],
        decoder_transformer_layers: list[int] = [0, 0, 0, 0],
        transformer_general_config=None,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim

        self.hop_length = int(math.prod(encoder_rates))
        self.encoder = Encoder(
            encoder_dim,
            encoder_rates,
            latent_dim,
            causal=causal,
            n_transformer_layers=encoder_transformer_layers,
            transformer_general_config=transformer_general_config,
        )

        self.quantizer = quantizer

        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
            causal=causal,
            n_transformer_layers=decoder_transformer_layers,
            transformer_general_config=transformer_general_config,
        )
        self.sample_rate = sample_rate
        self.apply(init_weights)

        self.frame_length = self.hop_length * 4

    def encode(
        self,
        audio_data: torch.Tensor,
        audio_lengths: torch.Tensor = None,
        n_quantizers: int = None,
        **kwargs,
    ):
        """Encode given audio data and return quantized latent codes.

        Parameters
        ----------
        audio_data : Tensor[B x 1 x T]
            Audio data to encode
        audio_lengths : Tensor[B], optional
            Valid sample counts per batch entry
        n_quantizers : int, optional
            Number of quantizers to use, by default None (all quantizers)

        Returns
        -------
        Tuple of (codes Tensor[B x N x F], code lengths Tensor[B]).
        """
        # pad to multiple of self.frame_length
        if audio_data.ndim == 2:
            audio_data = audio_data.unsqueeze(1)
        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.frame_length) * self.frame_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))
        if audio_lengths is None:
            audio_lengths = torch.LongTensor([length + right_pad]).to(audio_data.device)

        z = self.encoder(audio_data)
        vq_results = self.quantizer(z, n_quantizers, **kwargs)
        indices = vq_results.codes
        indices_lens = torch.ceil(audio_lengths / self.frame_length).long()
        return indices, indices_lens

    def decode(self, indices: torch.Tensor, feature_lengths):
        """Decode codebook indices [B x N x F] to waveform [B x 1 x T]."""
        if indices.ndim == 2:
            indices = indices[None]

        z = self.quantizer.decode(indices)
        audio_lengths = feature_lengths * self.frame_length
        return self.decoder(z), audio_lengths
