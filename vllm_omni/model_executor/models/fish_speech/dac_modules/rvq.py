# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adopted from the fish-speech 0.1.0 PyPI release (Apache-2.0)
# https://pypi.org/project/fish-speech/0.1.0/
# Copyright (c) Fish Audio
"""Downsampled residual vector quantizer (1 semantic + N residual codebooks)."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_omni.model_executor.models.fish_speech.dac_modules.conv import (
    CausalConvNet,
    CausalTransConvNet,
    ConvNeXtBlock,
)
from vllm_omni.model_executor.models.fish_speech.dac_modules.layers import (
    ResidualVectorQuantize,
)


@dataclass
class VQResult:
    z: torch.Tensor
    codes: torch.Tensor
    latents: torch.Tensor
    codebook_loss: torch.Tensor
    commitment_loss: torch.Tensor
    semantic_distill_z: torch.Tensor | None = None


class DownsampleResidualVectorQuantize(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        n_codebooks: int = 9,
        codebook_dim: int = 8,
        quantizer_dropout: float = 0.5,
        codebook_size: int = 1024,
        semantic_codebook_size: int = 4096,
        downsample_factor: tuple[int] = (2, 2),
        downsample_dims: tuple[int] | None = None,
        pre_module: nn.Module | None = None,
        post_module: nn.Module | None = None,
        semantic_predictor_module: nn.Module | None = None,
    ):
        super().__init__()

        if downsample_dims is None:
            downsample_dims = [input_dim for _ in range(len(downsample_factor))]

        all_dims = (input_dim,) + tuple(downsample_dims)

        self.semantic_quantizer = ResidualVectorQuantize(
            input_dim=input_dim,
            n_codebooks=1,
            codebook_size=semantic_codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=0.0,
        )

        self.quantizer = ResidualVectorQuantize(
            input_dim=input_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )

        self.downsample_factor = downsample_factor
        self.downsample_dims = downsample_dims

        convnet_type = CausalConvNet
        transconvnet_type = CausalTransConvNet

        self.downsample = nn.Sequential(
            *[
                nn.Sequential(
                    convnet_type(
                        all_dims[idx],
                        all_dims[idx + 1],
                        kernel_size=factor,
                        stride=factor,
                    ),
                    ConvNeXtBlock(dim=all_dims[idx + 1]),
                )
                for idx, factor in enumerate(downsample_factor)
            ]
        )

        self.upsample = nn.Sequential(
            *[
                nn.Sequential(
                    transconvnet_type(
                        all_dims[idx + 1],
                        all_dims[idx],
                        kernel_size=factor,
                        stride=factor,
                    ),
                    ConvNeXtBlock(dim=all_dims[idx]),
                )
                for idx, factor in reversed(list(enumerate(downsample_factor)))
            ]
        )
        self.apply(self._init_weights)
        # leave for transformer, LSTM or Mamba or something else
        self.pre_module = pre_module if pre_module is not None else nn.Identity()
        self.post_module = post_module if post_module is not None else nn.Identity()
        self.semantic_predictor_module = (
            semantic_predictor_module if semantic_predictor_module is not None else nn.Identity()
        )

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, z, n_quantizers: int = None, semantic_len: torch.Tensor = None, **kwargs):
        # z: (B, D, T)
        original_shape = z.shape
        if semantic_len is None:
            semantic_len = torch.LongTensor([z.shape[-1]])
        z = self.downsample(z)
        z = self.pre_module(z)  # B, T, D
        (
            semantic_z,
            semantic_codes,
            semantic_latents,
            semantic_commitment_loss,
            semantic_codebook_loss,
        ) = self.semantic_quantizer(z)
        residual_z = z - semantic_z
        residual_z, codes, latents, commitment_loss, codebook_loss = self.quantizer(
            residual_z, n_quantizers=n_quantizers
        )
        z = semantic_z + residual_z
        commitment_loss = commitment_loss + semantic_commitment_loss
        codebook_loss = codebook_loss + semantic_codebook_loss
        codes = torch.cat([semantic_codes, codes], dim=1)
        latents = torch.cat([semantic_latents, latents], dim=1)
        z = self.post_module(z)
        z = self.upsample(z)
        # z: (B, D, T)

        # Pad or crop z to match original shape
        diff = original_shape[-1] - z.shape[-1]
        right = 0
        left = abs(diff) - right

        if diff > 0:
            z = F.pad(z, (left, right))
        elif diff < 0:
            z = z[..., left:]

        results = VQResult(
            z=z,
            codes=codes,
            latents=latents,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
        )

        return results

    def decode(self, indices: torch.Tensor):
        new_indices = torch.zeros_like(indices)
        new_indices[:, 0] = torch.clamp(indices[:, 0], max=self.semantic_quantizer.codebook_size - 1)
        new_indices[:, 1:] = torch.clamp(indices[:, 1:], max=self.quantizer.codebook_size - 1)

        z_q_semantic = self.semantic_quantizer.from_codes(new_indices[:, :1])[0]
        z_q_residual = self.quantizer.from_codes(new_indices[:, 1:])[0]
        z_q = z_q_semantic + z_q_residual
        z_q = self.post_module(z_q)
        z_q = self.upsample(z_q)
        return z_q
