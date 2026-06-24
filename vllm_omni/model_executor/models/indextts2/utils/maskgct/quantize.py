# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Merged from factorized_vector_quantize.py and residual_vq.py.
# Only FactorizedVectorQuantize ("fvq") quantizer type is retained
# as it is the only one used by RepCodec in IndexTTS2.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


class FactorizedVectorQuantize(nn.Module):
    def __init__(
        self,
        input_dim,
        codebook_size,
        codebook_dim,
        commitment=0.005,
        codebook_loss_weight=1.0,
        use_l2_normalize=True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment = commitment
        self.codebook_loss_weight = codebook_loss_weight
        self.use_l2_normalize = use_l2_normalize

        if self.input_dim != self.codebook_dim:
            self.in_project = WNConv1d(self.input_dim, self.codebook_dim, kernel_size=1)
            self.out_project = WNConv1d(self.codebook_dim, self.input_dim, kernel_size=1)
        else:
            self.in_project = nn.Identity()
            self.out_project = nn.Identity()

        self.codebook = nn.Embedding(self.codebook_size, self.codebook_dim)

    def forward(self, z):
        z_e = self.in_project(z)
        z_q, indices = self.decode_latents(z_e)

        if self.training:
            commit_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2]) * self.commitment
            codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2]) * self.codebook_loss_weight
        else:
            commit_loss = torch.zeros(z.shape[0], device=z.device)
            codebook_loss = torch.zeros(z.shape[0], device=z.device)

        z_q = z_e + (z_q - z_e).detach()
        z_q = self.out_project(z_q)

        return z_q, commit_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight

        if self.use_l2_normalize:
            encodings = F.normalize(encodings)
            codebook = F.normalize(codebook)

        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices

    def vq2emb(self, vq, out_proj=True):
        emb = self.decode_code(vq)
        if out_proj:
            emb = self.out_project(emb)
        return emb


class ResidualVQ(nn.Module):
    """Residual Vector Quantization (SoundStream).

    Only supports FactorizedVectorQuantize ("fvq") quantizer type.
    """

    def __init__(
        self,
        input_dim: int = 256,
        num_quantizers: int = 8,
        codebook_size: int = 1024,
        codebook_dim: int = 256,
        quantizer_type: str = "fvq",
        quantizer_dropout: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer_dropout = quantizer_dropout

        self.quantizers = nn.ModuleList(
            [
                FactorizedVectorQuantize(
                    input_dim=input_dim,
                    codebook_size=codebook_size,
                    codebook_dim=codebook_dim,
                    **kwargs,
                )
                for _ in range(num_quantizers)
            ]
        )

    def forward(self, z, n_quantizers: int = None):
        quantized_out = 0.0
        residual = z

        all_commit_losses = []
        all_codebook_losses = []
        all_indices = []
        all_quantized = []

        if n_quantizers is None:
            n_quantizers = self.num_quantizers

        if self.training:
            n_q = torch.ones((z.shape[0],)) * self.num_quantizers + 1
            dropout = torch.randint(1, self.num_quantizers + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_q[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_q.to(z.device)

        for i, quantizer in enumerate(self.quantizers):
            if not self.training and i >= n_quantizers:
                break

            z_q_i, commit_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(residual)

            mask = torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            quantized_out = quantized_out + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            commit_loss_i = (commit_loss_i * mask).mean()
            codebook_loss_i = (codebook_loss_i * mask).mean()

            all_commit_losses.append(commit_loss_i)
            all_codebook_losses.append(codebook_loss_i)
            all_indices.append(indices_i)
            all_quantized.append(z_q_i)

        (
            all_commit_losses,
            all_codebook_losses,
            all_indices,
            all_quantized,
        ) = map(
            torch.stack,
            (
                all_commit_losses,
                all_codebook_losses,
                all_indices,
                all_quantized,
            ),
        )

        return (
            quantized_out,
            all_indices,
            all_commit_losses,
            all_codebook_losses,
            all_quantized,
        )

    def vq2emb(self, vq, n_quantizers=None):
        quantized_out = 0.0
        if n_quantizers is None:
            n_quantizers = self.num_quantizers
        for idx, quantizer in enumerate(self.quantizers):
            if idx >= n_quantizers:
                break
            quantized_out += quantizer.vq2emb(vq[idx])
        return quantized_out
