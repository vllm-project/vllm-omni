# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn
from torch.nn import functional as F

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

        # The vectorized ragged path rebuilds logical right-reflect values in
        # place and therefore requires each row to be strictly longer than the
        # widest right padding. The decoder keeps shorter rows on the original
        # exact-length singleton path, preserving SConv1d's existing behavior.
        self.ragged_min_length = 1 + max(
            ((layer.conv.conv.kernel_size[0] - 1) * layer.conv.conv.dilation[0] + 1 - layer.conv.conv.stride[0]) // 2
            for layer in self.in_layers
        )

    @staticmethod
    def _logical_reflect_in_layer(
        in_layer: SConv1d,
        x: torch.Tensor,
        logical_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Run a WaveNet input convolution at each row's logical boundary.

        ``SConv1d`` reflects at the physical tensor boundary. In a padded
        request batch, shorter rows end before that boundary, so their right
        reflection values must be rebuilt before the convolution. This path is
        intentionally WaveNet-specific: its input layers are non-causal,
        reflect-padded, stride-one convolutions. The decoder keeps rows that
        are not strictly longer than the right reflection width on the
        original singleton path.
        """
        conv = in_layer.conv.conv
        stride = conv.stride[0]
        dilation = conv.dilation[0]
        effective_kernel = (conv.kernel_size[0] - 1) * dilation + 1
        if in_layer.causal or in_layer.pad_mode != "reflect" or stride != 1:
            raise RuntimeError(
                "Ragged WaveNet batching requires non-causal reflect-padded stride-one input convolutions"
            )

        padding_total = effective_kernel - stride
        padding_right = padding_total // 2
        padding_left = padding_total - padding_right
        padded = F.pad(x, (padding_left, padding_right), mode="reflect")

        if padding_right:
            offsets = torch.arange(
                padding_right,
                device=x.device,
                dtype=torch.long,
            ).unsqueeze(0)
            lengths = logical_lengths.to(device=x.device, dtype=torch.long).reshape(-1, 1)
            source_positions = padding_left + lengths - 2 - offsets
            target_positions = padding_left + lengths + offsets
            source_indices = source_positions.unsqueeze(1).expand(-1, x.shape[1], -1)
            target_indices = target_positions.unsqueeze(1).expand(-1, x.shape[1], -1)
            reflected = torch.gather(padded, dim=2, index=source_indices)
            padded.scatter_(dim=2, index=target_indices, src=reflected)

        return in_layer.conv(padded)

    def forward(self, x, x_mask, g=None, **kwargs):
        output = torch.zeros_like(x)

        logical_lengths = None
        if getattr(self, "_assume_full_mask", None) is False:
            # Under CFG, x_mask has already been repeated from B to 2B in the
            # same [conditional rows, unconditional rows] order as x.
            logical_lengths = x_mask[:, 0, :].sum(dim=-1)

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            if logical_lengths is None:
                x_in = self.in_layers[i](x)
            else:
                x_in = self._logical_reflect_in_layer(
                    self.in_layers[i],
                    x,
                    logical_lengths,
                )
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
