# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn

from vllm_omni.model_executor.models.indextts2.s2mel.modules.encodec import SConv1d

from . import commons


class WN(torch.nn.Module):
    def __init__(
        self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0, causal=False
    ):
        super().__init__()
        conv1d_type = SConv1d
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            self.cond_layer = conv1d_type(gin_channels, 2 * hidden_channels * n_layers, 1, norm="weight_norm")

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = conv1d_type(
                hidden_channels,
                2 * hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
                norm="weight_norm",
                causal=causal,
            )
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = conv1d_type(hidden_channels, res_skip_channels, 1, norm="weight_norm", causal=causal)
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, g=None, **kwargs):
        output = torch.zeros_like(x)

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = commons.fused_add_tanh_sigmoid_multiply(x_in, g_l, self.hidden_channels)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, : self.hidden_channels, :]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        if self.gin_channels != 0:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        for conv_layer in self.in_layers:
            torch.nn.utils.remove_weight_norm(conv_layer)
        for conv_layer in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(conv_layer)
