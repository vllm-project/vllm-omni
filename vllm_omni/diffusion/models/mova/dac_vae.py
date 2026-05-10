# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/OpenMOSS/MOVA
# Originally from https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley
"""
DAC (Descript Audio Codec) VAE for MOVA.

Inference-only implementation: supports continuous latent mode (encode/decode).
Removed: audiotools dependency, BaseModel, CodecMixin, compress/decompress,
DACFile, VQ training code, and __main__ test script.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from vllm.logger import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Weight-normed convolutions
# ---------------------------------------------------------------------------


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


# ---------------------------------------------------------------------------
# Snake activation
# ---------------------------------------------------------------------------


@torch.jit.script
def snake(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return snake(x, self.alpha)


# ---------------------------------------------------------------------------
# Distributions (for continuous VAE mode)
# ---------------------------------------------------------------------------


class DiagonalGaussianDistribution:
    """Diagonal Gaussian distribution for VAE reparameterization."""

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self) -> torch.Tensor:
        return self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)

    def kl(self, other=None) -> torch.Tensor:
        if self.deterministic:
            return torch.tensor([0.0])
        if other is None:
            return 0.5 * torch.mean(
                torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                dim=[1, 2],
            )
        return 0.5 * torch.mean(
            torch.pow(self.mean - other.mean, 2) / other.var + self.var / other.var - 1.0 - self.logvar + other.logvar,
            dim=[1, 2],
        )

    def mode(self) -> torch.Tensor:
        return self.mean


# ---------------------------------------------------------------------------
# Encoder / Decoder building blocks
# ---------------------------------------------------------------------------


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list[int] | None = None,
        d_latent: int = 64,
    ):
        super().__init__()
        if strides is None:
            strides = [2, 4, 8, 8]

        block: list[nn.Module] = [WNConv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            block.append(EncoderBlock(d_model, stride=stride))
        block.extend(
            [
                Snake1d(d_model),
                WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
            ]
        )
        self.block = nn.Sequential(*block)
        self.enc_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel: int,
        channels: int,
        rates: list[int] | None = None,
        d_out: int = 1,
    ):
        super().__init__()
        if rates is None:
            rates = [8, 8, 4, 2]

        layers: list[nn.Module] = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers.append(DecoderBlock(input_dim, output_dim, stride))
        layers.extend(
            [
                Snake1d(output_dim),
                WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
                nn.Tanh(),
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# VectorQuantize / ResidualVectorQuantize (kept for weight loading compat)
# ---------------------------------------------------------------------------


class VectorQuantize(nn.Module):
    """Factorized, L2-normalized vector quantization."""

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, z: torch.Tensor):
        z_e = self.in_proj(z)
        encodings = z_e.permute(0, 2, 1).reshape(-1, self.codebook_dim)
        codebook = self.codebook.weight
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = (-dist).max(1)[1].reshape(z.size(0), -1)
        z_q = F.embedding(indices, self.codebook.weight).permute(0, 2, 1)
        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])
        z_q = z_e + (z_q - z_e).detach()
        z_q = self.out_proj(z_q)
        return z_q, commitment_loss, codebook_loss, indices, z_e

    def decode_code(self, embed_id: torch.Tensor) -> torch.Tensor:
        return F.embedding(embed_id, self.codebook.weight).transpose(1, 2)


class ResidualVectorQuantize(nn.Module):
    """Multi-codebook residual vector quantization."""

    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: int | list = 8,
        quantizer_dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]
        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.quantizers = nn.ModuleList(
            [VectorQuantize(input_dim, codebook_size, codebook_dim[i]) for i in range(n_codebooks)]
        )
        self.quantizer_dropout = quantizer_dropout

    def forward(self, z: torch.Tensor, n_quantizers: int | None = None):
        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0
        codebook_indices = []
        latents = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks

        for i, quantizer in enumerate(self.quantizers):
            if i >= n_quantizers:
                break
            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(residual)
            z_q = z_q + z_q_i
            residual = residual - z_q_i
            commitment_loss += commitment_loss_i.mean()
            codebook_loss += codebook_loss_i.mean()
            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        codes = torch.stack(codebook_indices, dim=1)
        latents = torch.cat(latents, dim=1)
        return z_q, codes, latents, commitment_loss, codebook_loss

    def from_codes(self, codes: torch.Tensor):
        z_q = 0.0
        z_p = []
        for i in range(codes.shape[1]):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_p.append(z_p_i)
            z_q = z_q + self.quantizers[i].out_proj(z_p_i)
        return z_q, torch.cat(z_p, dim=1), codes


# ---------------------------------------------------------------------------
# Main DAC model (inference-only)
# ---------------------------------------------------------------------------


class DAC(nn.Module):
    """
    DAC (Descript Audio Codec) VAE.

    Inference-only: supports continuous latent mode used by MOVA.
    When continuous=True, encode() returns DiagonalGaussianDistribution,
    decode() uses post_quant_conv before decoding.
    """

    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: list[int] | None = None,
        latent_dim: int | None = None,
        decoder_dim: int = 1536,
        decoder_rates: list[int] | None = None,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: int | list = 8,
        quantizer_dropout: bool = False,
        sample_rate: int = 44100,
        continuous: bool = False,
        use_weight_norm: bool = True,
    ):
        super().__init__()

        if encoder_rates is None:
            encoder_rates = [2, 4, 8, 8]
        if decoder_rates is None:
            decoder_rates = [8, 8, 4, 2]

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate
        self.continuous = continuous
        self.use_weight_norm = use_weight_norm

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))
        self.latent_dim = latent_dim

        self.hop_length = int(np.prod(encoder_rates))
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)

        if not continuous:
            self.n_codebooks = n_codebooks
            self.codebook_size = codebook_size
            self.codebook_dim = codebook_dim
            self.quantizer = ResidualVectorQuantize(
                input_dim=latent_dim,
                n_codebooks=n_codebooks,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                quantizer_dropout=quantizer_dropout,
            )
        else:
            self.quant_conv = nn.Conv1d(latent_dim, 2 * latent_dim, 1)
            self.post_quant_conv = nn.Conv1d(latent_dim, latent_dim, 1)

        self.decoder = Decoder(latent_dim, decoder_dim, decoder_rates)
        self.apply(init_weights)

        if not self.use_weight_norm:
            self.remove_weight_norm()

    def preprocess(self, audio_data: torch.Tensor, sample_rate: int | None = None) -> torch.Tensor:
        if sample_rate is None:
            sample_rate = self.sample_rate
        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        return F.pad(audio_data, (0, right_pad))

    def encode(self, audio_data: torch.Tensor, n_quantizers: int | None = None):
        """
        Encode audio data to latent space.

        Returns:
            In continuous mode: (DiagonalGaussianDistribution, None, None, 0, 0)
            In discrete mode: (z_q, codes, latents, commitment_loss, codebook_loss)
        """
        z = self.encoder(audio_data)
        if not self.continuous:
            z, codes, latents, commitment_loss, codebook_loss = self.quantizer(z, n_quantizers)
        else:
            z = self.quant_conv(z)
            z = DiagonalGaussianDistribution(z)
            codes, latents, commitment_loss, codebook_loss = None, None, 0, 0
        return z, codes, latents, commitment_loss, codebook_loss

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent codes to audio waveform."""
        if not self.continuous:
            return self.decoder(z)
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, audio_data: torch.Tensor, sample_rate: int | None = None, n_quantizers: int | None = None):
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        if not self.continuous:
            z, codes, latents, commitment_loss, codebook_loss = self.encode(audio_data, n_quantizers)
            x = self.decode(z)
            return {"audio": x[..., :length], "z": z, "codes": codes, "latents": latents}
        posterior, _, _, _, _ = self.encode(audio_data, n_quantizers)
        z = posterior.sample()
        x = self.decode(z)
        kl_loss = posterior.kl().mean()
        return {"audio": x[..., :length], "z": z, "kl_loss": kl_loss}

    def remove_weight_norm(self) -> None:
        """Remove weight_norm from all modules (fuses weight_g and weight_v)."""
        from torch.nn.utils import remove_weight_norm as _remove_weight_norm

        num_removed = 0
        for name, module in list(self.named_modules()):
            if hasattr(module, "_forward_pre_hooks"):
                for hook in list(module._forward_pre_hooks.values()):
                    if "WeightNorm" in str(type(hook)):
                        try:
                            _remove_weight_norm(module)
                            num_removed += 1
                        except ValueError:
                            logger.warning("Failed to remove weight_norm from %s", name)
        if num_removed > 0:
            logger.info("Removed weight_norm from %d modules", num_removed)
            self.use_weight_norm = False
