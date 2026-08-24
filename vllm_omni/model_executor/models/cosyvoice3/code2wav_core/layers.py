# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Neural network layers for CosyVoice3 Code2Wav."""

import torch
import torch.nn as nn
from torch.nn import functional as F


class PreLookaheadLayer(nn.Module):
    """Pre-lookahead convolutional layer for causal processing."""

    def __init__(self, in_channels: int, channels: int, pre_lookahead_len: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(
            in_channels,
            channels,
            kernel_size=pre_lookahead_len + 1,
            stride=1,
            padding=0,
        )
        self.conv2 = nn.Conv1d(
            channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=0,
        )

    def forward(self, inputs: torch.Tensor, context: torch.Tensor = torch.zeros(0, 0, 0)) -> torch.Tensor:
        """
        Args:
            inputs: (batch_size, seq_len, channels)
            context: Optional context tensor for streaming
        """
        outputs = inputs.transpose(1, 2).contiguous()
        context = context.transpose(1, 2).contiguous()
        # look ahead
        if context.size(2) == 0:
            outputs = F.pad(outputs, (0, self.pre_lookahead_len), mode="constant", value=0.0)
        else:
            assert self.training is False, "you have passed context, make sure that you are running inference mode"
            assert context.size(2) == self.pre_lookahead_len
            outputs = F.pad(
                torch.concat([outputs, context], dim=2),
                (0, self.pre_lookahead_len - context.size(2)),
                mode="constant",
                value=0.0,
            )
        outputs = F.leaky_relu(self.conv1(outputs))
        # outputs
        outputs = F.pad(outputs, (self.conv2.kernel_size[0] - 1, 0), mode="constant", value=0.0)
        outputs = self.conv2(outputs)
        outputs = outputs.transpose(1, 2).contiguous()

        # residual connection
        outputs = outputs + inputs
        return outputs
