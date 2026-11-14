# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""S3Gen's flow module: the parameters ``chatterbox_s3gen.flow_mels`` runs."""

import torch
import torch.nn as nn


class CausalMaskedDiffWithXvec(torch.nn.Module):
    """S3Gen's flow: token embedding, encoder, speaker projection and CFM decoder.

    A container for the parameters; ``chatterbox_s3gen.flow_mels`` runs them.

    Args:
        encoder: The token encoder, ``UpsampleConformerEncoder``.
        decoder: The flow-matching solver, ``CausalConditionalCFM``.
        input_size: Token embedding width.
        output_size: Mel bins.
        spk_embed_dim: Width of the CAMPPlus x-vector.
        vocab_size: Speech tokens.
        token_mel_ratio: Mel frames per token.
        pre_lookahead_len: Tokens the encoder reads ahead.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        vocab_size: int = 6561,
        token_mel_ratio: int = 2,
        pre_lookahead_len: int = 3,
    ):
        super().__init__()
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.token_mel_ratio = token_mel_ratio
        self.pre_lookahead_len = pre_lookahead_len
