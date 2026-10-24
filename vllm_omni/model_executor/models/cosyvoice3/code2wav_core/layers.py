# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Neural network layers for CosyVoice3 Code2Wav."""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class PreLookaheadCache:
    """Stable token-rate conditioning; independent of DiT block completion."""

    tokens: int
    hidden: torch.Tensor
    conv1_tail: torch.Tensor
    next_conv1_tail: torch.Tensor | None = None


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

    def forward_incremental(
        self, inputs: torch.Tensor, cache: PreLookaheadCache | None = None
    ) -> tuple[torch.Tensor, PreLookaheadCache]:
        """Consume new embeddings plus real right context, returning full conditioning.

        The caller skips ``cache.tokens`` embeddings. The pending window still
        includes the three (configured) lookahead tokens from the prior call.
        Cache every stable output, including incomplete attention blocks.
        Finalization retains the original zero-padded ``forward`` path.
        """
        count = inputs.shape[1] - self.pre_lookahead_len
        if count == 0 and cache is not None:
            return cache.hidden, cache
        first = F.leaky_relu(self.conv1(inputs.transpose(1, 2).contiguous()))
        margin = self.conv2.kernel_size[0] - 1
        history = first.new_zeros(first.shape[0], first.shape[1], margin) if cache is None else cache.conv1_tail
        joined = torch.cat((history, first), dim=2)
        hidden = self.conv2(joined).transpose(1, 2).contiguous() + inputs[:, :count]
        start = 0 if cache is None else cache.tokens
        if cache is not None:
            hidden = torch.cat((cache.hidden, hidden), dim=1)
        return hidden, PreLookaheadCache(start + count, hidden, joined[:, :, -margin:].clone())

    def forward_incremental_packed(
        self, inputs: list[torch.Tensor], caches: list[PreLookaheadCache | None]
    ) -> tuple[list[torch.Tensor], list[PreLookaheadCache]]:
        """Fuse both convolutions across requests, with double-buffered history."""
        if inputs[0].device.type != "cuda":
            values = [self.forward_incremental(value, cache) for value, cache in zip(inputs, caches)]
            return [value for value, _ in values], [cache for _, cache in values]
        from .prelookahead_kernel import packed_prelookahead

        active, old_tails, new_tails = [], [], []
        hidden, next_caches = {}, {}
        for row, (value, cache) in enumerate(zip(inputs, caches)):
            if value.shape[1] == self.pre_lookahead_len and cache is not None:
                hidden[row], next_caches[row] = cache.hidden, cache
                continue
            old = value.new_zeros(1, self.channels, 2) if cache is None else cache.conv1_tail
            new = torch.empty_like(old) if cache is None or cache.next_conv1_tail is None else cache.next_conv1_tail
            active.append(row)
            old_tails.append(old)
            new_tails.append(new)
        if active:
            values = packed_prelookahead(self, [inputs[row] for row in active], old_tails, new_tails)
            for i, (row, value) in enumerate(zip(active, values)):
                cache = caches[row]
                hidden[row] = value.clone() if cache is None else torch.cat((cache.hidden, value), dim=1)
                start = 0 if cache is None else cache.tokens
                next_caches[row] = PreLookaheadCache(start + value.shape[1], hidden[row], new_tails[i], old_tails[i])
        return [hidden[row] for row in range(len(inputs))], [next_caches[row] for row in range(len(inputs))]

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
