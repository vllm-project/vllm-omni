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
import torch._dynamo
from torch import nn
from torch.nn import functional as F

from vllm_omni.model_executor.models.lychee_fd.token2wav_modules.cosyvoice2.transformer.encoder_layer import (
    ConformerEncoderLayer,
)
from vllm_omni.model_executor.models.lychee_fd.token2wav_modules.cosyvoice2.utils.class_utils import (
    COSYVOICE_ACTIVATION_CLASSES,
    COSYVOICE_ATTENTION_CLASSES,
    COSYVOICE_EMB_CLASSES,
    COSYVOICE_SUBSAMPLE_CLASSES,
)
from vllm_omni.model_executor.models.lychee_fd.token2wav_modules.cosyvoice2.utils.mask import (
    make_pad_mask,
)

from .positionwise_feed_forward import (
    PositionwiseFeedForward,
)

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 128


class Upsample1D(nn.Module):
    """A 1D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
    """

    def __init__(self, channels: int, out_channels: int, stride: int = 2, scale_factor: float = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.stride = stride
        # In this mode, first repeat interpolate, than conv with stride=1
        self.conv = nn.Conv1d(self.channels, self.out_channels, stride * 2 + 1, stride=1, padding=0)
        self.scale_factor = float(self.stride) if scale_factor is None else float(scale_factor)

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor):
        outputs = F.interpolate(inputs, scale_factor=self.scale_factor, mode="nearest")
        outputs = F.pad(outputs, (self.stride * 2, 0), value=0.0)
        outputs = self.conv(outputs)
        return outputs, input_lengths * self.stride

    def forward_chunk(
        self, inputs: torch.Tensor, input_lengths: torch.Tensor, cache: torch.Tensor = torch.zeros((0, 0, 0))
    ):
        """
        Args:
            inputs(torch.Tensor): shape (b, c, t)
            input_length(torch.Tensor): shape (b), can be None
            cache(torch.Tensor): shape (b, c, cache_t), where cache_t = stride * 2
        """
        outputs = F.interpolate(inputs, scale_factor=self.scale_factor, mode="nearest")

        if cache is None:
            cache = inputs.new_zeros(inputs.shape[0], inputs.shape[1], self.stride * 2)
        outputs = torch.cat([cache, outputs], dim=2)
        new_cache = outputs[..., -self.stride * 2 :]
        outputs = self.conv(outputs)

        if input_lengths is not None:
            input_lengths = input_lengths * self.stride
        return outputs, input_lengths, new_cache


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

    def forward_chunk(self, inputs: torch.Tensor, cache: torch.Tensor = None):
        """
        Args:
            inputs(torch.Tensor): shape (b, t, c)
            cache(torch.Tensor): shape (b, c, cache_t=2), c = channels
        """
        outputs = inputs.transpose(1, 2).contiguous()
        outputs = F.leaky_relu(self.conv1(outputs))
        # the length of outputs is input length - pre_lookahead_len
        if cache is None:
            cache = outputs.new_zeros(outputs.shape[0], outputs.shape[1], 2)
        # NOTE
        new_cache = outputs[..., -2:]
        outputs = torch.cat([cache, outputs], dim=2)
        outputs = self.conv2(outputs)
        outputs = outputs.transpose(1, 2).contiguous()
        # residual connection
        outputs = outputs + inputs[:, : -self.pre_lookahead_len]
        return outputs, new_cache


"""Customize each sample's chunk attention mask
"""


class UpsampleConformerEncoderV2(torch.nn.Module):
    def __init__(
        self,
        # input & output
        input_size: int,
        output_size: int = 256,
        input_layer: str = "linear",
        pre_lookahead_len: int = 3,
        # size
        num_blocks: int = 6,
        num_up_blocks: int = 4,
        # upsampling
        up_stride: int = 2,
        up_scale_factor: float = 2,
        # attention
        attention_heads: int = 4,
        pos_enc_layer_type: str = "rel_pos_espnet",
        selfattention_layer_type: str = "rel_selfattn",
        key_bias: bool = True,
        # mlp
        linear_units: int = 2048,
        # dropouts
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        # other
        normalize_before: bool = True,
        activation_type: str = "swish",
        **kwargs,
    ):
        super().__init__()
        self._output_size = output_size
        self.embed = COSYVOICE_SUBSAMPLE_CLASSES[input_layer](
            input_size,
            output_size,
            dropout_rate,
            COSYVOICE_EMB_CLASSES[pos_enc_layer_type](output_size, positional_dropout_rate),
        )

        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(output_size, eps=1e-5)
        activation = COSYVOICE_ACTIVATION_CLASSES[activation_type]()
        # self-attention module definition
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
            key_bias,
        )
        # feed-forward module definition
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
        )
        self.pre_lookahead_layer = PreLookaheadLayer(channels=output_size, pre_lookahead_len=pre_lookahead_len)
        self.encoders = torch.nn.ModuleList(
            [
                ConformerEncoderLayer(
                    output_size,
                    COSYVOICE_ATTENTION_CLASSES[selfattention_layer_type](*encoder_selfattn_layer_args),
                    PositionwiseFeedForward(*positionwise_layer_args),
                    None,
                    None,
                    dropout_rate,
                    normalize_before,
                )
                for _ in range(num_blocks)
            ]
        )
        self.up_layer = Upsample1D(
            channels=output_size, out_channels=output_size, stride=up_stride, scale_factor=up_scale_factor
        )
        self.up_embed = COSYVOICE_SUBSAMPLE_CLASSES[input_layer](
            input_size,
            output_size,
            dropout_rate,
            COSYVOICE_EMB_CLASSES[pos_enc_layer_type](output_size, positional_dropout_rate),
        )
        self.up_encoders = torch.nn.ModuleList(
            [
                ConformerEncoderLayer(
                    output_size,
                    COSYVOICE_ATTENTION_CLASSES[selfattention_layer_type](*encoder_selfattn_layer_args),
                    PositionwiseFeedForward(*positionwise_layer_args),
                    None,
                    None,
                    dropout_rate,
                    normalize_before,
                )
                for _ in range(num_up_blocks)
            ]
        )

        self.enable_cuda_graph = False
        self.use_cuda_graph = False
        self.graph_encoder = {}
        self.graph_up_encoder = {}
        self.inference_buffers_encoder = {}
        self.inference_buffers_up_encoder = {}
        self.max_static_time = 1500

    # FIXME(sfy) revert hard-coded bfloat16
    # this method is skipped in CausalMaskedDiffWithXvec.scatter_cuda_graph
    def scatter_cuda_graph(self, enable_cuda_graph: bool):
        self.enable_cuda_graph = enable_cuda_graph
        if self.enable_cuda_graph:
            self._init_cuda_graph()

    def _init_cuda_graph(self):
        """初始化 CUDA Graph"""

        for capture_length in range(100, 1500, 10):
            static_x = torch.zeros((1, capture_length, 512), dtype=torch.float32, device=torch.device("cuda"))
            static_mask = torch.ones((1, 1, capture_length), dtype=torch.bool, device=torch.device("cuda"))
            static_pos_emb = torch.zeros(
                (1, 2 * capture_length - 1, 512), dtype=torch.float32, device=torch.device("cuda")
            )

            static_inputs = [
                static_x,
                static_mask,
                static_pos_emb,
            ]

            self._forward_impl_encoder(
                static_inputs[0],
                static_inputs[1],
                static_inputs[2],
            )
            graph = torch.cuda.CUDAGraph()
            with torch.no_grad():
                with torch.cuda.graph(graph):
                    static_out_x = self._forward_impl_encoder(static_inputs[0], static_inputs[1], static_inputs[2])
            self.graph_encoder[capture_length] = graph
            static_outputs = [
                static_out_x,
            ]
            self.inference_buffers_encoder[capture_length] = {
                "static_inputs": static_inputs,
                "static_outputs": static_outputs,
            }

        for capture_length in range(100, 1500, 10):
            static_x = torch.zeros((1, capture_length, 512), dtype=torch.float32, device=torch.device("cuda"))
            static_mask = torch.ones((1, 1, capture_length), dtype=torch.bool, device=torch.device("cuda"))
            static_pos_emb = torch.zeros(
                (1, 2 * capture_length - 1, 512), dtype=torch.float32, device=torch.device("cuda")
            )

            static_inputs = [
                static_x,
                static_mask,
                static_pos_emb,
            ]

            self._forward_impl_up_encoder(
                static_inputs[0],
                static_inputs[1],
                static_inputs[2],
            )
            graph = torch.cuda.CUDAGraph()
            with torch.no_grad():
                with torch.cuda.graph(graph):
                    static_out_x = self._forward_impl_up_encoder(static_inputs[0], static_inputs[1], static_inputs[2])
            self.graph_up_encoder[capture_length] = graph
            static_outputs = [
                static_out_x,
            ]
            self.inference_buffers_up_encoder[capture_length] = {
                "static_inputs": static_inputs,
                "static_outputs": static_outputs,
            }

        self.use_cuda_graph = True
        print("CUDA Graph initialized successfully for encoder and up_encoder")

    # @torch.compile(dynamic=True,backend="eager")
    def _forward_impl_encoder(self, x: torch.Tensor, mask: torch.Tensor, pos_emb: torch.Tensor):
        for layer in self.encoders:
            x, _, _, _ = layer(x, mask, pos_emb)
        return x

    # @torch.compile(dynamic=True,backend="eager")
    def _forward_impl_up_encoder(self, x: torch.Tensor, mask: torch.Tensor, pos_emb: torch.Tensor):
        for layer in self.up_encoders:
            x, _, _, _ = layer(x, mask, pos_emb)
        return x

    def output_size(self) -> int:
        return self._output_size

    # @torch.compile(dynamic=True,backend="eager")
    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # (sfy) chunk training strategy should not be open-sourced
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        xs, pos_emb, masks = self.embed(xs, masks)

        # lookahead
        xs = self.pre_lookahead_layer(xs)
        # conformer block
        if self.enable_cuda_graph and xs.shape[1] in self.graph_encoder:
            self.inference_buffers_encoder[xs.shape[1]]["static_inputs"][0].copy_(xs)
            self.inference_buffers_encoder[xs.shape[1]]["static_inputs"][1].copy_(masks)
            self.inference_buffers_encoder[xs.shape[1]]["static_inputs"][2].copy_(pos_emb)
            self.graph_encoder[xs.shape[1]].replay()
            xs = self.inference_buffers_encoder[xs.shape[1]]["static_outputs"][0]
        else:
            xs = self._forward_impl_encoder(xs, masks, pos_emb)
        # upsample
        xs = xs.transpose(1, 2).contiguous()
        xs, xs_lens = self.up_layer(xs, xs_lens)
        xs = xs.transpose(1, 2).contiguous()

        # 2nd conformer block
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        xs, pos_emb, masks = self.up_embed(xs, masks)
        if self.enable_cuda_graph and xs.shape[1] in self.graph_up_encoder:
            self.inference_buffers_up_encoder[xs.shape[1]]["static_inputs"][0].copy_(xs)
            self.inference_buffers_up_encoder[xs.shape[1]]["static_inputs"][1].copy_(masks)
            self.inference_buffers_up_encoder[xs.shape[1]]["static_inputs"][2].copy_(pos_emb)
            self.graph_up_encoder[xs.shape[1]].replay()
            xs = self.inference_buffers_up_encoder[xs.shape[1]]["static_outputs"][0]
        else:
            xs = self._forward_impl_up_encoder(xs, masks, pos_emb)
        # post norm
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks

    # 禁用 torch.compile，避免 Dynamo 转换失败时的 WON'T CONVERT 警告及潜在异常
    # @torch.compile(dynamic=True,backend="eager")
    def forward_chunk(
        self,
        xs: torch.Tensor,
        last_chunk: bool = False,
        cnn_cache: torch.Tensor = None,
        att_cache: torch.Tensor = None,
    ):
        """
        Args:
            xs: shape (b, dt, c)
            last_chunk: bool. If last chunk, will pad input with lookaheads
            att_cache: shape (depth1+depth2, b, nh, 2*t1, c).
            cnn_cache: shape (b, c, t1+t2). Where t1=2 (pre_lookahead_layer), t2=4 (up_layer)
        """
        if att_cache is not None:
            assert att_cache.shape[3] % 2 == 0, att_cache.shape
        if cnn_cache is not None:
            assert cnn_cache.shape[2] == 2 + self.up_layer.stride * 2, cnn_cache.shape

        # unpack caches
        offset1 = att_cache.shape[3] // 2 if att_cache is not None else 0
        att_cache1 = (
            att_cache[: len(self.encoders), :, :, :offset1] if att_cache is not None else [None] * len(self.encoders)
        )
        att_cache2 = att_cache[len(self.encoders) :] if att_cache is not None else [None] * len(self.encoders)
        cnn_cache1 = cnn_cache[:, :, :2] if cnn_cache is not None else None
        cnn_cache2 = cnn_cache[:, :, 2:] if cnn_cache is not None else None
        xs, _, _ = self.embed(xs, None)
        if last_chunk:
            xs = F.pad(xs, (0, 0, 0, self.pre_lookahead_layer.pre_lookahead_len))

        # this_cnn_cache: shape (b=1, c=512, t=2)
        xs, new_cnn_cache1 = self.pre_lookahead_layer.forward_chunk(xs, cache=cnn_cache1)

        # remake pos_emb, offset param is ignored by position_encoding
        pos_emb = self.embed.position_encoding(offset=None, size=offset1 + xs.shape[1])

        # first conformer
        chunk_masks = torch.zeros((0, 0, 0))
        new_att_cache1 = []

        for idx, layer in enumerate(self.encoders):
            # this_att_cache: shape (b, nh, t, c * 2)
            xs, _, this_new_att_cache1, _ = layer(xs, chunk_masks, pos_emb, att_cache=att_cache1[idx])
            new_att_cache1.append(this_new_att_cache1)
        new_att_cache1 = torch.stack(new_att_cache1, dim=0)

        # upsample + conformer encoder, xs: (b, t, c) -> (b, c, t)
        xs = xs.transpose(1, 2).contiguous()
        # this_cnn_cache: shape (b=1, c=512, t=2*2)
        xs, _, new_cnn_cache2 = self.up_layer.forward_chunk(xs, None, cache=cnn_cache2)
        xs = xs.transpose(1, 2).contiguous()

        # at this time, xs are doubled in length
        xs, _, _ = self.up_embed(xs, None)

        # remake pos_emb
        pos_emb = self.embed.position_encoding(offset=None, size=offset1 * self.up_layer.stride + xs.shape[1])

        # second conformer
        chunk_masks = torch.zeros((0, 0, 0), dtype=torch.bfloat16)
        new_att_cache2 = []

        for idx, layer in enumerate(self.up_encoders):
            xs, _, this_new_att_cache2, _ = layer(xs, chunk_masks, pos_emb, att_cache=att_cache2[idx])
            new_att_cache2.append(this_new_att_cache2)
        new_att_cache2 = torch.stack(new_att_cache2, dim=0)

        if self.normalize_before:
            xs = self.after_norm(xs)

        # pack new cache
        new_att_cache = torch.cat([new_att_cache1.repeat(1, 1, 1, 2, 1), new_att_cache2], dim=0)
        new_cnn_cache = torch.cat([new_cnn_cache1, new_cnn_cache2], dim=2)

        return xs, new_cnn_cache, new_att_cache
