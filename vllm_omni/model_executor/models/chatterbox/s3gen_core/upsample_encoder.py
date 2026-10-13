# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
#               2022 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
#               2024 Alibaba Inc (Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Encoder definition."""

import torch
from torch import nn
from torch.nn import functional as F

from vllm_omni.model_executor.models.chatterbox.s3gen_core.attention import RelPositionMultiHeadedAttention
from vllm_omni.model_executor.models.chatterbox.s3gen_core.embedding import EspnetRelPositionalEncoding
from vllm_omni.model_executor.models.chatterbox.s3gen_core.encoder_layer import ConformerEncoderLayer
from vllm_omni.model_executor.models.chatterbox.s3gen_core.positionwise_feed_forward import PositionwiseFeedForward
from vllm_omni.model_executor.models.chatterbox.s3gen_core.subsampling import LinearNoSubsampling


class Upsample1D(nn.Module):
    """A 1D upsampling layer with an optional convolution.

    Args:
        channels (int): Number of channels in the inputs.
        out_channels (int): Number of channels in the outputs.
        stride (int): Upsampling factor. Defaults to 2.
    """

    def __init__(self, channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.stride = stride
        # In this mode, first repeat interpolate, than conv with stride=1
        self.conv = nn.Conv1d(self.channels, self.out_channels, stride * 2 + 1, stride=1, padding=0)

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor):
        outputs = F.interpolate(inputs, scale_factor=float(self.stride), mode="nearest")
        outputs = F.pad(outputs, (self.stride * 2, 0), value=0.0)
        outputs = self.conv(outputs)
        return outputs, input_lengths * self.stride


class PreLookaheadLayer(nn.Module):
    def __init__(self, channels: int, pre_lookahead_len: int = 1):
        super().__init__()
        self.channels = channels
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=pre_lookahead_len + 1,
            stride=1,
            padding=0,
        )
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=0,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: (batch_size, seq_len, channels)
        """
        outputs = inputs.transpose(1, 2).contiguous()
        # look ahead
        outputs = F.pad(outputs, (0, self.pre_lookahead_len), mode="constant", value=0.0)
        outputs = F.leaky_relu(self.conv1(outputs))
        # outputs
        outputs = F.pad(outputs, (2, 0), mode="constant", value=0.0)
        outputs = self.conv2(outputs)
        outputs = outputs.transpose(1, 2).contiguous()

        # residual connection
        outputs = outputs + inputs
        return outputs


class UpsampleConformerEncoder(torch.nn.Module):
    """S3Gen's token encoder: six Conformer layers, a 2x upsampling, four more.

    Built as upstream's checkpoint selects it: a linear input layer, ESPnet
    relative positions, relative-position self-attention, SiLU feed-forward
    layers, pre-norm, no convolution module and no macaron feed-forward.

    Args:
        input_size (int): Input dimension.
        output_size (int): Attention dimension.
        attention_heads (int): Number of attention heads.
        linear_units (int): Hidden units of the position-wise feed-forward layers.
        num_blocks (int): Number of Conformer layers before the upsampling.
        dropout_rate (float): Dropout rate.
        positional_dropout_rate (float): Dropout rate after the positional encoding.
        attention_dropout_rate (float): Dropout rate in attention.
    """

    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 512,
        attention_heads: int = 8,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
    ):
        super().__init__()
        self._output_size = output_size

        self.embed = LinearNoSubsampling(
            input_size,
            output_size,
            dropout_rate,
            EspnetRelPositionalEncoding(output_size, positional_dropout_rate),
        )

        self.after_norm = torch.nn.LayerNorm(output_size, eps=1e-5)
        activation = torch.nn.SiLU()
        # self-attention module definition
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
        )
        # feed-forward module definition
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
        )
        self.pre_lookahead_layer = PreLookaheadLayer(channels=512, pre_lookahead_len=3)
        self.encoders = torch.nn.ModuleList(
            [
                ConformerEncoderLayer(
                    output_size,
                    RelPositionMultiHeadedAttention(*encoder_selfattn_layer_args),
                    PositionwiseFeedForward(*positionwise_layer_args),
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_blocks)
            ]
        )
        self.up_layer = Upsample1D(channels=512, out_channels=512, stride=2)
        self.up_embed = LinearNoSubsampling(
            input_size,
            output_size,
            dropout_rate,
            EspnetRelPositionalEncoding(output_size, positional_dropout_rate),
        )
        self.up_encoders = torch.nn.ModuleList(
            [
                ConformerEncoderLayer(
                    output_size,
                    RelPositionMultiHeadedAttention(*encoder_selfattn_layer_args),
                    PositionwiseFeedForward(*positionwise_layer_args),
                    dropout_rate=dropout_rate,
                )
                for _ in range(4)
            ]
        )

    def output_size(self) -> int:
        return self._output_size

    def forward_layers(
        self, xs: torch.Tensor, chunk_masks: torch.Tensor, pos_emb: torch.Tensor, mask_pad: torch.Tensor
    ) -> torch.Tensor:
        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        return xs

    def forward_up_layers(
        self, xs: torch.Tensor, chunk_masks: torch.Tensor, pos_emb: torch.Tensor, mask_pad: torch.Tensor
    ) -> torch.Tensor:
        for layer in self.up_encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        return xs
