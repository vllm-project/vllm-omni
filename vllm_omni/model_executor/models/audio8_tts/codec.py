# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Audio8 TTS neural audio codec (44.1 kHz, 10 codebooks).

Vendored from the reference ``modeling_arktts_codec.py`` shipped with
``Audio8/Audio8-TTS-Preview-0.6b`` (Apache-2.0). The module tree and attribute
names are reproduced verbatim so ``codec.pth`` loads under ``strict=True``.

Inference-only deviations: no ``torch.jit.script`` on Snake (it fights
``torch.compile`` / CUDA graphs), RoPE cached in a plain dict instead of a
non-persistent buffer (the pattern that leaves the reference LM with
uninitialised RoPE under ``from_pretrained`` on transformers >= 5), and
:func:`build_arktts_codec` prunes the unused encoder or decoder half per stage.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import weight_norm as legacy_weight_norm
from torch.nn.utils.parametrizations import weight_norm

# The codec's own transformer blocks always use base 10000, independent of the
# 1e6 rope_base of the DualAR language model.
_CODEC_ROPE_BASE = 10000.0


def _rope_table(length: int, head_dim: int, base: float, device: torch.device) -> Tensor:
    frequencies = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    phases = torch.outer(torch.arange(length, device=device), frequencies)
    values = torch.polar(torch.ones_like(phases), phases)
    return torch.stack((values.real, values.imag), dim=-1).to(torch.bfloat16)


def _apply_rope(x: Tensor, values: Tensor) -> Tensor:
    shaped = x.float().reshape(*x.shape[:-1], -1, 2)
    values = values.view(1, shaped.shape[1], 1, shaped.shape[3], 2)
    output = torch.stack(
        (
            shaped[..., 0] * values[..., 0] - shaped[..., 1] * values[..., 1],
            shaped[..., 1] * values[..., 0] + shaped[..., 0] * values[..., 1],
        ),
        dim=-1,
    )
    return output.flatten(3).to(x.dtype)


class ArkttsCodecRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return output.to(x.dtype) * self.weight


class ArkttsCodecLayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-2) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class ArkttsCodecTransformerConfig:
    """Plain option holder for the codec's transformer blocks.

    Deliberately not a dataclass / msgspec Struct: it mirrors a reference
    dataclass whose only job is to carry constructor arguments, and it is
    instantiated from Python literals inside this module only.
    """

    def __init__(
        self,
        n_layer: int,
        n_head: int,
        dim: int,
        intermediate_size: int,
        n_local_heads: int = -1,
        head_dim: int = 64,
        rope_base: float = _CODEC_ROPE_BASE,
        norm_eps: float = 1e-5,
        channels_first: bool = True,
    ) -> None:
        self.n_layer = int(n_layer)
        self.n_head = int(n_head)
        self.dim = int(dim)
        self.intermediate_size = int(intermediate_size)
        self.n_local_heads = int(n_head) if n_local_heads == -1 else int(n_local_heads)
        self.head_dim = int(head_dim)
        self.rope_base = float(rope_base)
        self.norm_eps = float(norm_eps)
        self.channels_first = bool(channels_first)


class ArkttsCodecAttention(nn.Module):
    def __init__(self, config: ArkttsCodecTransformerConfig) -> None:
        super().__init__()
        total = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        self.wqkv = nn.Linear(config.dim, total, bias=False)
        self.wo = nn.Linear(config.head_dim * config.n_head, config.dim, bias=False)
        self.n_head = config.n_head
        self.n_local_heads = config.n_local_heads
        self.head_dim = config.head_dim

    def forward(self, x: Tensor, rope_values: Tensor, mask: Tensor) -> Tensor:
        batch, length, _ = x.shape
        query_size = self.n_head * self.head_dim
        kv_size = self.n_local_heads * self.head_dim
        query, key, value = self.wqkv(x).split((query_size, kv_size, kv_size), dim=-1)
        query = query.view(batch, length, self.n_head, self.head_dim)
        key = key.view(batch, length, self.n_local_heads, self.head_dim)
        value = value.view(batch, length, self.n_local_heads, self.head_dim)
        query = _apply_rope(query, rope_values).transpose(1, 2)
        key = _apply_rope(key, rope_values).transpose(1, 2)
        value = value.transpose(1, 2)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            enable_gqa=self.n_local_heads != self.n_head,
        )
        output = output.transpose(1, 2).contiguous().view(batch, length, query_size)
        return self.wo(output)


class ArkttsCodecFeedForward(nn.Module):
    def __init__(self, config: ArkttsCodecTransformerConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ArkttsCodecTransformerBlock(nn.Module):
    def __init__(self, config: ArkttsCodecTransformerConfig) -> None:
        super().__init__()
        self.attention = ArkttsCodecAttention(config)
        self.feed_forward = ArkttsCodecFeedForward(config)
        self.ffn_norm = ArkttsCodecRMSNorm(config.dim, config.norm_eps)
        self.attention_norm = ArkttsCodecRMSNorm(config.dim, config.norm_eps)
        self.attention_layer_scale = ArkttsCodecLayerScale(config.dim)
        self.ffn_layer_scale = ArkttsCodecLayerScale(config.dim)

    def forward(self, x: Tensor, rope_values: Tensor, mask: Tensor) -> Tensor:
        hidden = x + self.attention_layer_scale(self.attention(self.attention_norm(x), rope_values, mask))
        return hidden + self.ffn_layer_scale(self.feed_forward(self.ffn_norm(hidden)))


class ArkttsCodecWindowTransformer(nn.Module):
    """Causal transformer with a sliding attention window."""

    def __init__(
        self,
        config: ArkttsCodecTransformerConfig,
        input_dim: int,
        window_size: int | None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ArkttsCodecTransformerBlock(config) for _ in range(config.n_layer)])
        self.norm = ArkttsCodecRMSNorm(config.dim, config.norm_eps)
        self.window_size = window_size
        self.channels_first = config.channels_first
        self.input_proj = nn.Linear(input_dim, config.dim) if input_dim != config.dim else nn.Identity()
        self.output_proj = nn.Linear(config.dim, input_dim) if input_dim != config.dim else nn.Identity()
        self.look_ahead_conv = nn.Identity()
        self.head_dim = config.head_dim
        self.rope_base = config.rope_base
        self._mask_cache: dict[tuple[int, str], Tensor] = {}
        self._rope_cache: dict[tuple[int, str], Tensor] = {}

    def _mask(self, length: int, device: torch.device) -> Tensor:
        key = (length, str(device))
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        row = torch.arange(length, device=device)[:, None]
        column = torch.arange(length, device=device)[None, :]
        mask = column <= row
        if self.window_size is not None:
            mask &= column >= (row - self.window_size + 1).clamp_min(0)
        mask = mask[None, None]
        self._mask_cache[key] = mask
        return mask

    def _rope(self, length: int, device: torch.device) -> Tensor:
        key = (length, str(device))
        cached = self._rope_cache.get(key)
        if cached is not None:
            return cached
        values = _rope_table(length, self.head_dim, self.rope_base, device)
        self._rope_cache[key] = values
        return values

    def forward(self, x: Tensor) -> Tensor:
        if self.channels_first:
            x = x.transpose(1, 2)
        x = self.look_ahead_conv(self.input_proj(x))
        length = int(x.shape[1])
        mask = self._mask(length, x.device)
        rope_values = self._rope(length, x.device)
        for layer in self.layers:
            x = layer(x, rope_values, mask)
        x = self.output_proj(self.norm(x))
        return x.transpose(1, 2) if self.channels_first else x


def _extra_padding(x: Tensor, kernel_size: int, stride: int, padding_total: int = 0) -> int:
    length = x.shape[-1]
    frames = (length - kernel_size + padding_total) / stride + 1
    ideal = (math.ceil(frames) - 1) * stride + kernel_size - padding_total
    return ideal - length


class ArkttsCausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, groups=groups)
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.padding = self.kernel_size - self.stride

    def forward(self, x: Tensor) -> Tensor:
        right = _extra_padding(x, self.kernel_size, self.stride, self.padding)
        return self.conv(F.pad(x, (self.padding, right))).contiguous()

    def apply_weight_norm(self) -> ArkttsCausalConv1d:
        self.conv = weight_norm(self.conv)
        return self


class ArkttsCausalConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation)
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        crop = self.kernel_size - self.stride
        return x[..., : x.shape[-1] - crop].contiguous() if crop else x.contiguous()

    def apply_weight_norm(self) -> ArkttsCausalConvTranspose1d:
        self.conv = weight_norm(self.conv)
        return self


def _causal_wn_conv(*args: Any, **kwargs: Any) -> ArkttsCausalConv1d:
    return ArkttsCausalConv1d(*args, **kwargs).apply_weight_norm()


def _causal_wn_transpose(*args: Any, **kwargs: Any) -> ArkttsCausalConvTranspose1d:
    return ArkttsCausalConvTranspose1d(*args, **kwargs).apply_weight_norm()


class ArkttsSnake1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        x = x.reshape(shape[0], shape[1], -1)
        x = x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).pow(2)
        return x.reshape(shape)


class ArkttsResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ArkttsSnake1d(dim),
            _causal_wn_conv(dim, dim, kernel_size=7, dilation=dilation),
            ArkttsSnake1d(dim),
            _causal_wn_conv(dim, dim, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        output = self.block(x)
        difference = x.shape[-1] - output.shape[-1]
        if difference > 0:
            x = x[..., :-difference]
        return x + output


class ArkttsEncoderBlock(nn.Module):
    def __init__(self, dim: int, stride: int, transformer_layers: int) -> None:
        super().__init__()
        modules: list[nn.Module] = [
            ArkttsResidualUnit(dim // 2, 1),
            ArkttsResidualUnit(dim // 2, 3),
            ArkttsResidualUnit(dim // 2, 9),
            ArkttsSnake1d(dim // 2),
            _causal_wn_conv(dim // 2, dim, kernel_size=2 * stride, stride=stride),
        ]
        if transformer_layers:
            config = ArkttsCodecTransformerConfig(
                n_layer=transformer_layers,
                n_head=dim // 64,
                dim=dim,
                intermediate_size=dim * 3,
            )
            modules.append(ArkttsCodecWindowTransformer(config, dim, window_size=512))
        else:
            modules.append(nn.Identity())
        self.block = nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ArkttsEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        dim = 64
        modules: list[nn.Module] = [_causal_wn_conv(1, dim, kernel_size=7)]
        for stride, transformer_layers in zip((2, 4, 8, 8), (0, 0, 0, 4), strict=True):
            dim *= 2
            modules.append(ArkttsEncoderBlock(dim, stride, transformer_layers))
        modules.extend((ArkttsSnake1d(dim), _causal_wn_conv(dim, 1024, kernel_size=3)))
        self.block = nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ArkttsDecoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ArkttsSnake1d(input_dim),
            _causal_wn_transpose(input_dim, output_dim, kernel_size=2 * stride, stride=stride),
            ArkttsResidualUnit(output_dim, 1),
            ArkttsResidualUnit(output_dim, 3),
            ArkttsResidualUnit(output_dim, 9),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ArkttsDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = 1536
        modules: list[nn.Module] = [_causal_wn_conv(1024, channels, kernel_size=7)]
        output_dim = channels
        for index, stride in enumerate((8, 8, 4, 2)):
            input_dim = channels // (2**index)
            output_dim = channels // (2 ** (index + 1))
            modules.append(ArkttsDecoderBlock(input_dim, output_dim, stride))
        modules.extend(
            (
                ArkttsSnake1d(output_dim),
                _causal_wn_conv(output_dim, 1, kernel_size=7),
                nn.Tanh(),
            )
        )
        self.model = nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class ArkttsVectorQuantizer(nn.Module):
    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int) -> None:
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.codebook_dim = int(codebook_dim)
        self.in_proj = legacy_weight_norm(nn.Conv1d(input_dim, codebook_dim, kernel_size=1))
        self.out_proj = legacy_weight_norm(nn.Conv1d(codebook_dim, input_dim, kernel_size=1))
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def decode_code(self, indices: Tensor) -> Tensor:
        return F.embedding(indices, self.codebook.weight).transpose(1, 2)

    def decode_latents(self, latents: Tensor) -> tuple[Tensor, Tensor]:
        batch, _, length = latents.shape
        flattened = F.normalize(latents.transpose(1, 2).reshape(batch * length, -1))
        codebook = F.normalize(self.codebook.weight)
        distances = (
            flattened.pow(2).sum(1, keepdim=True)
            - 2 * flattened @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = (-distances).argmax(1).view(batch, length)
        return self.decode_code(indices), indices

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        projected = self.in_proj(z)
        quantized, indices = self.decode_latents(projected)
        return self.out_proj(quantized), indices


class ArkttsResidualQuantizer(nn.Module):
    def __init__(self, input_dim: int, n_codebooks: int, codebook_size: int, codebook_dim: int) -> None:
        super().__init__()
        self.n_codebooks = int(n_codebooks)
        self.codebook_size = int(codebook_size)
        self.quantizers = nn.ModuleList(
            [ArkttsVectorQuantizer(input_dim, codebook_size, codebook_dim) for _ in range(n_codebooks)]
        )

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        quantized_sum: Tensor | float = 0.0
        residual = z
        codes: list[Tensor] = []
        for quantizer in self.quantizers:
            quantized, indices = quantizer(residual)
            quantized_sum = quantized_sum + quantized
            residual = residual - quantized
            codes.append(indices)
        return quantized_sum, torch.stack(codes, dim=1)

    def from_codes(self, codes: Tensor) -> Tensor:
        output: Tensor | float = 0.0
        for index in range(codes.shape[1]):
            projected = self.quantizers[index].decode_code(codes[:, index])
            output = output + self.quantizers[index].out_proj(projected)
        return output


class ArkttsConvNeXtBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dwconv = ArkttsCausalConv1d(dim, dim, kernel_size=7, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.dwconv(x).permute(0, 2, 1)
        x = self.pwconv2(self.act(self.pwconv1(self.norm(x))))
        x = (self.gamma * x).permute(0, 2, 1)
        return residual + x


class ArkttsDownsampleQuantizer(nn.Module):
    """1 semantic codebook (4096) + 9 residual codebooks (1024), 4x downsampled."""

    def __init__(
        self,
        post_n_layer: int = 8,
        post_n_head: int = 16,
        post_n_local_heads: int = 8,
        post_intermediate_size: int = 1216,
    ) -> None:
        super().__init__()
        self.semantic_quantizer = ArkttsResidualQuantizer(1024, 1, 4096, 8)
        self.quantizer = ArkttsResidualQuantizer(1024, 9, 1024, 8)
        self.downsample = nn.Sequential(
            nn.Sequential(ArkttsCausalConv1d(1024, 1024, kernel_size=2, stride=2), ArkttsConvNeXtBlock(1024)),
            nn.Sequential(ArkttsCausalConv1d(1024, 1024, kernel_size=2, stride=2), ArkttsConvNeXtBlock(1024)),
        )
        self.upsample = nn.Sequential(
            nn.Sequential(ArkttsCausalConvTranspose1d(1024, 1024, kernel_size=2, stride=2), ArkttsConvNeXtBlock(1024)),
            nn.Sequential(ArkttsCausalConvTranspose1d(1024, 1024, kernel_size=2, stride=2), ArkttsConvNeXtBlock(1024)),
        )
        self.pre_module = ArkttsCodecWindowTransformer(
            ArkttsCodecTransformerConfig(n_layer=8, n_head=16, dim=1024, intermediate_size=3072),
            1024,
            window_size=128,
        )
        self.post_module = ArkttsCodecWindowTransformer(
            ArkttsCodecTransformerConfig(
                n_layer=post_n_layer,
                n_head=post_n_head,
                n_local_heads=post_n_local_heads,
                dim=1024,
                intermediate_size=post_intermediate_size,
            ),
            1024,
            window_size=128,
        )
        self.semantic_predictor_module = nn.Identity()

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        original_length = z.shape[-1]
        z = self.pre_module(self.downsample(z))
        semantic, semantic_codes = self.semantic_quantizer(z)
        residual, residual_codes = self.quantizer(z - semantic)
        z = self.upsample(self.post_module(semantic + residual))
        difference = original_length - z.shape[-1]
        if difference > 0:
            z = F.pad(z, (difference, 0))
        elif difference < 0:
            z = z[..., -difference:]
        return z, torch.cat((semantic_codes, residual_codes), dim=1)

    def decode(self, indices: Tensor) -> Tensor:
        # Cloned because the clamps below are in-place and this is the caller's
        # codes buffer; once per decode chunk, not per AR step.
        indices = indices.clone()
        indices[:, 0].clamp_(0, self.semantic_quantizer.codebook_size - 1)
        indices[:, 1:].clamp_(0, self.quantizer.codebook_size - 1)
        semantic = self.semantic_quantizer.from_codes(indices[:, :1])
        residual = self.quantizer.from_codes(indices[:, 1:])
        return self.upsample(self.post_module(semantic + residual))


class ArkttsCodec(nn.Module):
    """Audio8 TTS codec: waveform <-> 10 codebooks at ~21.5 frames/s."""

    sample_rate = 44100
    frame_length = 2048

    def __init__(
        self,
        post_n_layer: int = 8,
        post_n_head: int = 16,
        post_n_local_heads: int = 8,
        post_intermediate_size: int = 1216,
    ) -> None:
        super().__init__()
        self.encoder = ArkttsEncoder()
        self.quantizer = ArkttsDownsampleQuantizer(
            post_n_layer=post_n_layer,
            post_n_head=post_n_head,
            post_n_local_heads=post_n_local_heads,
            post_intermediate_size=post_intermediate_size,
        )
        self.decoder = ArkttsDecoder()

    @torch.inference_mode()
    def encode(self, audio: Tensor, audio_lengths: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Encode ``[B, 1, samples]`` (or ``[B, samples]``) to ``[B, 10, frames]``."""
        if audio.ndim == 2:
            audio = audio[:, None]
        if audio.ndim != 3 or audio.shape[1] != 1:
            raise ValueError(f"audio must have shape [B, 1, samples], got {tuple(audio.shape)}")
        original_length = int(audio.shape[-1])
        right = math.ceil(original_length / self.frame_length) * self.frame_length - original_length
        if right:
            audio = F.pad(audio, (0, right))
        if audio_lengths is None:
            audio_lengths = torch.full((audio.shape[0],), original_length, device=audio.device, dtype=torch.long)
        _, codes = self.quantizer(self.encoder(audio))
        code_lengths = torch.ceil(audio_lengths.float() / self.frame_length).long()
        max_codes = int(codes.shape[-1])
        code_lengths = code_lengths.clamp_max(max_codes)
        frame_index = torch.arange(max_codes, device=codes.device)
        valid = frame_index[None, None, :] < code_lengths[:, None, None]
        return torch.where(valid, codes, torch.full_like(codes, -1)), code_lengths

    @torch.inference_mode()
    def decode(self, codes: Tensor) -> Tensor:
        """Decode ``[B, 10, frames]`` to ``[B, 1, samples]``."""
        return self.decoder(self.quantizer.decode(codes.long()))


def build_arktts_codec(
    *,
    post_n_layer: int = 8,
    post_n_head: int = 16,
    post_n_local_heads: int = 8,
    post_intermediate_size: int = 1216,
) -> ArkttsCodec:
    """Construct the codec with uninitialised weights, on CPU, in eval mode."""
    codec = ArkttsCodec(
        post_n_layer=post_n_layer,
        post_n_head=post_n_head,
        post_n_local_heads=post_n_local_heads,
        post_intermediate_size=post_intermediate_size,
    )
    codec.eval()
    return codec


__all__ = ["ArkttsCodec", "build_arktts_codec"]
