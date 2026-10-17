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
"""S3Gen's token encoder: an ESPnet/WeNet Conformer with a 2x upsampling in the middle."""

import math

import torch
from torch import nn

from vllm_omni.model_executor.models.chatterbox.s3gen_core.attention import EspnetRelPositionMultiHeadedAttention
from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.layers import PreLookaheadLayer
from vllm_omni.model_executor.models.cosyvoice3.utils import make_pad_mask
from vllm_omni.model_executor.models.indextts2.gpt.conformer_encoder import (
    ConformerEncoderLayer,
    PositionwiseFeedForward,
)


class EspnetRelPositionalEncoding(torch.nn.Module):
    """ESPnet's relative positional encoding: one row per relative distance, left and right.

    See Transformer-XL (https://arxiv.org/abs/1901.02860), Appendix B, and
    https://github.com/espnet/espnet/pull/2816.

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Input length the table is first built for; it grows on demand.
    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: torch.Tensor):
        """Rebuild the table when ``x`` is longer than it covers, or move it to ``x``'s dtype and device."""
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # Suppose `i` means to the position of query vector and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://arxiv.org/abs/1901.02860
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Scale the input and take the positions it spans.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Positional embedding (1, 2 * time - 1, `*`), every
                relative distance from ``time - 1`` to ``-(time - 1)``.
        """
        self.extend_pe(x)
        x = x * self.xscale
        size = x.size(1)
        pos_emb = self.pe[:, self.pe.size(1) // 2 - size + 1 : self.pe.size(1) // 2 + size]
        return self.dropout(x), self.dropout(pos_emb)


class LinearNoSubsampling(torch.nn.Module):
    """Project the input and add relative positions, keeping every frame.

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc_class (torch.nn.Module): The positional encoding applied after the projection.
    """

    def __init__(self, idim: int, odim: int, dropout_rate: float, pos_enc_class: torch.nn.Module):
        super().__init__()
        self.out = torch.nn.Sequential(
            torch.nn.Linear(idim, odim),
            torch.nn.LayerNorm(odim, eps=1e-5),
            torch.nn.Dropout(dropout_rate),
        )
        self.pos_enc = pos_enc_class

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project ``x`` and take its positions.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: linear input tensor (#batch, time', odim),
                where time' = time .
            torch.Tensor: positional embedding (1, 2 * time' - 1, odim).
            torch.Tensor: linear input mask (#batch, 1, time'),
                where time' = time .
        """
        x = self.out(x)
        x, pos_emb = self.pos_enc(x)
        return x, pos_emb, x_mask


class WenetConformerEncoderLayer(ConformerEncoderLayer):
    """IndexTTS2's pre-norm Conformer layer with WeNet's LayerNorm epsilon.

    The S3Gen checkpoint was trained with ``eps=1e-12`` in both of the
    layer's norms; IndexTTS2's layer uses PyTorch's default ``1e-5``.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module.
        feed_forward (torch.nn.Module): Feed-forward module.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, size: int, self_attn: torch.nn.Module, feed_forward: torch.nn.Module, dropout_rate: float):
        super().__init__(size, self_attn, feed_forward, dropout_rate=dropout_rate)
        self.norm_ff.eps = self.norm_mha.eps = 1e-12


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
        # Nearest-neighbour repeat, then a stride-1 convolution.
        self.conv = nn.Conv1d(self.channels, self.out_channels, stride * 2 + 1, stride=1, padding=0)

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Upsample shape (B, C, T) to (B, out_channels, stride * T), causally, with the lengths."""
        outputs = nn.functional.interpolate(inputs, scale_factor=float(self.stride), mode="nearest")
        outputs = nn.functional.pad(outputs, (self.stride * 2, 0), value=0.0)
        outputs = self.conv(outputs)
        return outputs, input_lengths * self.stride


class UpsampleConformerEncoder(torch.nn.Module):
    """S3Gen's token encoder: six Conformer layers, a 2x upsampling, four more.

    Built as upstream's checkpoint selects it: a linear input layer, ESPnet
    relative positions, relative-position self-attention, SiLU feed-forward
    layers, pre-norm, no convolution module and no macaron feed-forward.

    Args:
        pre_lookahead_len (int): Tokens the look-ahead convolution reads past each token.
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
        pre_lookahead_len: int,
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
        self.pre_lookahead_layer = PreLookaheadLayer(
            in_channels=output_size, channels=output_size, pre_lookahead_len=pre_lookahead_len
        )
        self.encoders = torch.nn.ModuleList(
            [
                WenetConformerEncoderLayer(
                    output_size,
                    EspnetRelPositionMultiHeadedAttention(*encoder_selfattn_layer_args),
                    PositionwiseFeedForward(*positionwise_layer_args),
                    dropout_rate,
                )
                for _ in range(num_blocks)
            ]
        )
        self.up_layer = Upsample1D(channels=output_size, out_channels=output_size, stride=2)
        self.up_embed = LinearNoSubsampling(
            input_size,
            output_size,
            dropout_rate,
            EspnetRelPositionalEncoding(output_size, positional_dropout_rate),
        )
        self.up_encoders = torch.nn.ModuleList(
            [
                WenetConformerEncoderLayer(
                    output_size,
                    EspnetRelPositionMultiHeadedAttention(*encoder_selfattn_layer_args),
                    PositionwiseFeedForward(*positionwise_layer_args),
                    dropout_rate,
                )
                for _ in range(4)
            ]
        )

    def output_size(self) -> int:
        """The encoder's output width, which the flow's projection reads."""
        return self._output_size

    def forward(self, xs: torch.Tensor, xs_lens: torch.Tensor) -> torch.Tensor:
        """Upstream's ``forward`` with the padding zeroed.

        Upstream masks the token embeddings but not the encoder's own input
        projection, whose bias and LayerNorm turn a padded row's zeros into a
        constant. The pre-lookahead layer then convolves three positions ahead
        of every frame, so a short row's last tokens would mix that constant in
        and the row would not decode as it does alone. Zeroing after ``embed``
        changes nothing when there is nothing to pad.

        Nothing else of upstream's ``forward`` applies: the encoder is built
        without dynamic chunking, CNN module or CMVN, so its chunk masks are the
        padding masks.

        Args:
            xs (torch.Tensor): Shape (B, T, 512), the token embeddings; padded positions are zeroed after ``embed``.
            xs_lens (torch.Tensor): Shape (B,), each row's token count.

        Returns:
            torch.Tensor: Shape (B, 2T, 512).
        """
        masks = ~make_pad_mask(xs_lens, xs.shape[1]).unsqueeze(1)
        xs, pos_emb, masks = self.embed(xs, masks)
        xs = self.pre_lookahead_layer(xs * masks.transpose(1, 2).to(xs))
        for layer in self.encoders:
            xs, masks, _, _ = layer(xs, masks, pos_emb, masks)
        xs, xs_lens = self.up_layer(xs.transpose(1, 2).contiguous(), xs_lens)
        xs = xs.transpose(1, 2).contiguous()
        masks = ~make_pad_mask(xs_lens, xs.shape[1]).unsqueeze(1)
        xs, pos_emb, masks = self.up_embed(xs, masks)
        for layer in self.up_encoders:
            xs, masks, _, _ = layer(xs, masks, pos_emb, masks)
        return self.after_norm(xs)
