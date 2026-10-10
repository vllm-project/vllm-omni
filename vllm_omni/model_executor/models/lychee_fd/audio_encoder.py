# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Native Lychee Whisper-style audio encoder and LLM adaptor."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from .audio_cudnn import LycheeAudioConv1d
from .audio_math import LycheeAudioGELU, released_gelu
from .configuration_lychee import LycheeAudioEncoderConfig


class _DTypeStableLayerNorm(nn.LayerNorm):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            inputs,
            self.normalized_shape,
            self.weight.to(inputs.dtype) if self.weight is not None else None,
            self.bias.to(inputs.dtype) if self.bias is not None else None,
            self.eps,
        )


class _DTypeStableLinear(nn.Linear):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(
            inputs,
            self.weight.to(inputs.dtype),
            None if self.bias is None else self.bias.to(inputs.dtype),
        )


class _DTypeStableConv1d(nn.Conv1d):
    def _conv_forward(
        self,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        return super()._conv_forward(
            inputs,
            weight.to(inputs.dtype),
            None if bias is None else bias.to(inputs.dtype),
        )


def make_non_pad_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    if lengths.ndim != 1:
        raise ValueError(f"Expected one length per batch row, got {tuple(lengths.shape)}")
    positions = torch.arange(max_len, dtype=torch.int64, device=lengths.device)
    return positions.unsqueeze(0) < lengths.to(torch.int64).unsqueeze(1)


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if mask.dtype != torch.bool:
        raise TypeError(f"Expected bool mask, got {mask.dtype}")
    if dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise TypeError(f"Unsupported attention dtype: {dtype}")
    return (1.0 - mask.to(dtype)) * -1.0e10


class LycheeAudioAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.query = _DTypeStableLinear(hidden_size, hidden_size)
        self.key = _DTypeStableLinear(hidden_size, hidden_size, bias=False)
        self.value = _DTypeStableLinear(hidden_size, hidden_size)
        self.out = _DTypeStableLinear(hidden_size, hidden_size)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        query = self.query(inputs)
        key = self.key(inputs)
        value = self.value(inputs)
        _, time, hidden_size = query.shape
        scale = (hidden_size // self.num_heads) ** -0.25
        query = query.view(*query.shape[:2], self.num_heads, -1).permute(0, 2, 1, 3) * scale
        key = key.view(*key.shape[:2], self.num_heads, -1).permute(0, 2, 3, 1) * scale
        value = value.view(*value.shape[:2], self.num_heads, -1).permute(0, 2, 1, 3)
        scores = query @ key
        if mask is not None:
            scores = scores + mask
        probabilities = F.softmax(scores.float(), dim=-1).to(query.dtype)
        attended = (probabilities @ value).permute(0, 2, 1, 3).reshape(inputs.shape[0], time, hidden_size)
        return self.out(attended)


class LycheeAudioResidualBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.attn = LycheeAudioAttention(hidden_size, num_heads)
        self.attn_ln = _DTypeStableLayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            _DTypeStableLinear(hidden_size, hidden_size * 4),
            LycheeAudioGELU(),
            _DTypeStableLinear(hidden_size * 4, hidden_size),
        )
        self.mlp_ln = _DTypeStableLayerNorm(hidden_size)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        inputs = inputs + self.attn(self.attn_ln(inputs.contiguous()), mask=mask)
        return inputs + self.mlp(self.mlp_ln(inputs.contiguous()))


class LycheeAudioEncoder(nn.Module):
    def __init__(self, config: LycheeAudioEncoderConfig) -> None:
        super().__init__()
        self.conv1 = _DTypeStableConv1d(config.n_mels, config.n_audio_state, kernel_size=3, padding=1)
        self.conv2 = _DTypeStableConv1d(
            config.n_audio_state,
            config.n_audio_state,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.positional_embedding = nn.Embedding(config.n_audio_ctx, config.n_audio_state)
        self.positional_embedding.requires_grad_(False)
        self.blocks: Iterable[LycheeAudioResidualBlock] = nn.ModuleList(
            [LycheeAudioResidualBlock(config.n_audio_state, config.n_audio_head) for _ in range(config.n_audio_layer)]
        )
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.after_norm = _DTypeStableLayerNorm(config.n_audio_state)

    def forward(self, features: torch.Tensor, feature_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError(f"Expected audio features [batch, mel, time], got {tuple(features.shape)}")
        if feature_lengths.shape != (features.shape[0],):
            raise ValueError("feature_lengths must contain one value per batch row")

        input_time = int(features.shape[-1])
        hidden = released_gelu(self.conv1(features))
        hidden = released_gelu(self.conv2(hidden)).permute(0, 2, 1)
        if hidden.shape[1] > self.positional_embedding.num_embeddings:
            raise ValueError(
                f"Audio sequence has {hidden.shape[1]} positions, exceeding configured context "
                f"{self.positional_embedding.num_embeddings}"
            )

        mask = make_non_pad_mask(feature_lengths, input_time).unsqueeze(1)
        mask = mask[:, :, (input_time + 1) % 2 :: 2]
        if mask.shape[-1] != hidden.shape[1]:
            raise RuntimeError(f"Audio mask/hidden length mismatch: {mask.shape[-1]} != {hidden.shape[1]}")
        attention_bias = mask_to_bias(mask, hidden.dtype).unsqueeze(1)

        positions = self.positional_embedding.weight[: hidden.shape[1]].to(hidden.dtype)
        hidden = hidden + positions
        for block in self.blocks:
            hidden = block(hidden, attention_bias)

        hidden = self.avg_pooler(hidden.permute(0, 2, 1)).permute(0, 2, 1)
        output_lengths = (feature_lengths + 1) // 2 // 2
        return self.after_norm(hidden.contiguous()), output_lengths


class LycheeAudioAdaptor(nn.Module):
    def __init__(self, config: LycheeAudioEncoderConfig) -> None:
        super().__init__()
        self.stride = config.adapter_stride
        if self.stride != -1:
            self.conv = LycheeAudioConv1d(
                config.n_audio_state,
                config.n_audio_state,
                config.kernel_size,
                config.adapter_stride,
                padding=1,
            )
        self.linear1 = _DTypeStableLinear(config.n_audio_state, 2_048)
        self.relu = nn.ReLU()
        self.linear2 = _DTypeStableLinear(2_048, config.llm_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.stride != -1:
            hidden = released_gelu(self.conv(hidden.permute(0, 2, 1))).permute(0, 2, 1)
        return self.linear2(self.relu(self.linear1(hidden)))

    def output_lengths(self, encoder_lengths: torch.Tensor) -> torch.Tensor:
        if self.stride == -1:
            return encoder_lengths
        return (encoder_lengths + 2 - self.conv.kernel_size[0]) // self.stride + 1


__all__ = [
    "LycheeAudioAdaptor",
    "LycheeAudioAttention",
    "LycheeAudioEncoder",
    "LycheeAudioResidualBlock",
    "make_non_pad_mask",
    "mask_to_bias",
]
