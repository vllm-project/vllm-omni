# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from nemo.collections.asr.models.wav2vec.wav2vec_config import Wav2VecConvExtractorMode, Wav2VecTransformerConfig
from nemo.collections.asr.parts.layer_norm import LayerNorm
from nemo.collections.asr.parts.multihead_attention import MultiheadAttention


class TransposeLast(torch.nn.Module):
    """
    Transposes last dimension. Useful for adding to a sequential block.
    """

    def forward(self, x):
        return x.transpose(-2, -1)


class SamePad(torch.nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.remove = kernel_size % 2 == 0

    def forward(self, x):
        if self.remove:
            x = x[:, :, :-1]
        return x


class ConvFeatureEncoder(nn.Module):
    """
        Converts input raw audio into features for downstream transformer model.
        Uses 1D convolutional blocks with GeLU activation.
    """

    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        mode: Wav2VecConvExtractorMode = Wav2VecConvExtractorMode.default,
        conv_bias: bool = False,
    ):
        super().__init__()

        def block(
            n_in, n_out, k, stride, is_layer_norm=False, is_group_norm=False, conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (is_layer_norm and is_group_norm) is False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Sequential(TransposeLast(), nn.LayerNorm(dim, elementwise_affine=True), TransposeLast()),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(make_conv(), nn.GroupNorm(dim, dim, affine=True), nn.GELU(),)
            else:
                return nn.Sequential(make_conv(), nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode is Wav2VecConvExtractorMode.layer_norm,
                    is_group_norm=mode is Wav2VecConvExtractorMode.default and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):
        # BxT -> BxCxT
        x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)
        return x

    def get_subsampled_lens(self, lens):
        for m in self.conv_layers:
            conv = m[0]
            lens = (lens + 2 * conv.padding[0] - conv.dilation[0] * (conv.kernel_size[0] - 1) - 1) // conv.stride[0] + 1
        return lens


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        conv_cfg = args.conv

        self.dropout = args.dropout
        self.embedding_dim = args.encoder.embedding_dim

        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=conv_cfg.conv_pos,
            padding=conv_cfg.conv_pos // 2,
            groups=conv_cfg.conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (conv_cfg.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(conv_cfg.conv_pos), nn.GELU())
        self.pos_conv_layer_drop = conv_cfg.layer_drop

        encoder_cfg = args.encoder
        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=encoder_cfg.ffn_embedding_dim,
                    num_attention_heads=encoder_cfg.num_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=encoder_cfg.attention_dropout,
                    activation_dropout=encoder_cfg.activation_dropout,
                    activation_fn=encoder_cfg.activation_fn.value,
                    layer_norm_first=encoder_cfg.layer_norm_first,
                )
                for _ in range(encoder_cfg.encoder_layers)
            ]
        )

        self.layer_norm_first = encoder_cfg.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = encoder_cfg.encoder_layerdrop

        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None):
        x = self.extract_features(x, padding_mask)

        if self.layer_norm_first:
            x = self.layer_norm(x)

        return x

    def extract_features(self, x, padding_mask=None):

        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)

        if self.pos_conv_layer_drop > 0 and self.training and np.random.random() < self.pos_conv_layer_drop:
            pass
        else:
            x_conv = self.pos_conv(x.transpose(1, 2))
            x_conv = x_conv.transpose(1, 2)
            x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # layer_results = []
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability >= self.layerdrop):
                x, z = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
                # layer_results.append(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, attn


def index_put(tensor, indices, value):
    tensor[indices] = value
    return tensor


def get_activation_fn(activation: str):
    """ Returns the activation function corresponding to `activation` """
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


class Wav2VecTransformerEncoder(nn.Module):
    def __init__(self, cfg: Wav2VecTransformerConfig):
        super().__init__()

        conv_cfg = cfg.conv

        self.dropout = cfg.dropout
        self.embedding_dim = cfg.encoder.embedding_dim
        self.layer_norm_first = cfg.encoder.layer_norm_first
        assert not self.layer_norm_first, 'nn.TransformerEncoderLayer do not support layer_norm_first'

        # positional convolutional embeddings
        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=conv_cfg.conv_pos,
            padding=conv_cfg.conv_pos // 2,
            groups=conv_cfg.conv_pos_groups,
        )

        self.feature_dropout = nn.Dropout(self.dropout)

        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (conv_cfg.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(conv_cfg.conv_pos), nn.GELU())

        encoder_cfg = cfg.encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=encoder_cfg.num_attention_heads,
                dim_feedforward=encoder_cfg.ffn_embedding_dim,
                dropout=self.dropout,
                activation=encoder_cfg.activation_fn.value,
            ),
            num_layers=encoder_cfg.encoder_layers,
        )
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None):
        x = self.extract_features(x, padding_mask)

        if self.layer_norm_first:
            x = self.layer_norm(x)

        return x

    def extract_features(self, x, padding_mask=None):

        if padding_mask is not None:
            x[padding_mask] = 0

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x += x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = self.feature_dropout(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bias will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    def normal_(data):
        # Skip meta tensors during lazy/meta initialization; real weights are loaded later.
        if getattr(data, "is_meta", False):
            return
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(
            data.cpu().normal_(mean=0.0, std=0.02).to(data.device)
        )

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)
    if isinstance(module, nn.TransformerEncoderLayer):
        normal_(module.self_attn.in_proj_weight.data)
