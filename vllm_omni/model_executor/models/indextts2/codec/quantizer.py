# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright contributors to the Amphion project
"""Decode-only factorized residual vector quantizer.

Adapted from the Amphion codec implementation vendored by IndexTTS 2.5.
Training-only paths are intentionally omitted.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def _weight_norm_conv1d(*args, **kwargs) -> nn.Conv1d:
    return weight_norm(nn.Conv1d(*args, **kwargs))


class FactorizedVectorQuantizeDecoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_dim = codebook_dim
        if input_dim != codebook_dim:
            # Keep upstream module names and weight-normalized parameter layout
            # so codec.pth keys load without conversion.
            self.in_project = _weight_norm_conv1d(
                input_dim,
                codebook_dim,
                kernel_size=1,
            )
            self.out_project = _weight_norm_conv1d(
                codebook_dim,
                input_dim,
                kernel_size=1,
            )
        else:
            self.in_project = nn.Identity()
            self.out_project = nn.Identity()
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def vq2emb(self, codes: torch.Tensor) -> torch.Tensor:
        embedding = F.embedding(codes, self.codebook.weight).transpose(1, 2)
        return self.out_project(embedding)


class ResidualVQDecoder(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        num_quantizers: int,
        codebook_size: int,
        codebook_dim: int,
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.quantizers = nn.ModuleList(
            [
                FactorizedVectorQuantizeDecoder(
                    input_dim=input_dim,
                    codebook_size=codebook_size,
                    codebook_dim=codebook_dim,
                )
                for _ in range(num_quantizers)
            ]
        )

    def vq2emb(
        self,
        codes: torch.Tensor,
        n_quantizers: int | None = None,
    ) -> torch.Tensor:
        count = self.num_quantizers if n_quantizers is None else n_quantizers
        quantized: torch.Tensor | None = None
        for index, quantizer in enumerate(self.quantizers):
            if index >= count:
                break
            value = quantizer.vq2emb(codes[index])
            quantized = value if quantized is None else quantized + value
        if quantized is None:
            raise ValueError("EnhancedCodec requires at least one quantizer")
        return quantized
