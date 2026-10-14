# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torch.nn import functional as F

from vllm_omni.model_executor.models.indextts2.utils.maskgct.quantize import ResidualVQ
from vllm_omni.model_executor.models.indextts2.utils.maskgct.vocos import VocosBackbone


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class RepCodec(nn.Module):
    def __init__(
        self,
        codebook_size=8192,
        hidden_size=1024,
        codebook_dim=8,
        vocos_dim=384,
        vocos_intermediate_dim=2048,
        vocos_num_layers=12,
        num_quantizers=1,
        downsample_scale=1,
        cfg=None,
    ):
        super().__init__()
        if cfg is not None:
            _g = lambda attr, default: getattr(cfg, attr, default)  # noqa: E731
            codebook_size = _g("codebook_size", codebook_size)
            codebook_dim = _g("codebook_dim", codebook_dim)
            hidden_size = _g("hidden_size", hidden_size)
            vocos_dim = _g("vocos_dim", vocos_dim)
            vocos_intermediate_dim = _g("vocos_intermediate_dim", vocos_intermediate_dim)
            vocos_num_layers = _g("vocos_num_layers", vocos_num_layers)
            num_quantizers = _g("num_quantizers", num_quantizers)
            downsample_scale = _g("downsample_scale", downsample_scale)

        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.hidden_size = hidden_size
        self.vocos_dim = vocos_dim
        self.vocos_intermediate_dim = vocos_intermediate_dim
        self.vocos_num_layers = vocos_num_layers
        self.num_quantizers = num_quantizers
        self.downsample_scale = downsample_scale

        if self.downsample_scale is not None and self.downsample_scale > 1:
            self.down = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, stride=2, padding=1)
            self.up = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=3, stride=1, padding=1)

        self.encoder = nn.Sequential(
            VocosBackbone(
                input_channels=self.hidden_size,
                dim=self.vocos_dim,
                intermediate_dim=self.vocos_intermediate_dim,
                num_layers=self.vocos_num_layers,
                adanorm_num_embeddings=None,
            ),
            nn.Linear(self.vocos_dim, self.hidden_size),
        )
        self.decoder = nn.Sequential(
            VocosBackbone(
                input_channels=self.hidden_size,
                dim=self.vocos_dim,
                intermediate_dim=self.vocos_intermediate_dim,
                num_layers=self.vocos_num_layers,
                adanorm_num_embeddings=None,
            ),
            nn.Linear(self.vocos_dim, self.hidden_size),
        )

        self.quantizer = ResidualVQ(
            input_dim=hidden_size,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_type="fvq",
            quantizer_dropout=0.0,
            commitment=0.15,
            codebook_loss_weight=1.0,
            use_l2_normalize=True,
        )

        self.reset_parameters()

    def quantize(self, x):
        if self.downsample_scale is not None and self.downsample_scale > 1:
            x = x.transpose(1, 2)
            x = self.down(x)
            x = F.gelu(x)
            x = x.transpose(1, 2)

        x = self.encoder(x.transpose(1, 2)).transpose(1, 2)

        (
            quantized_out,
            all_indices,
            all_commit_losses,
            all_codebook_losses,
            _,
        ) = self.quantizer(x)

        if all_indices.shape[0] == 1:
            return all_indices.squeeze(0), quantized_out.transpose(1, 2)
        return all_indices, quantized_out.transpose(1, 2)

    def reset_parameters(self):
        self.apply(init_weights)
