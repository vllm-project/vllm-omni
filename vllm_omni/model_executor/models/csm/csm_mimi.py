# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Incremental Mimi decoding: each call decodes only new frames, using state kept from earlier ones.

Mimi's decoder is causal, so decoding a sequence in pieces gives the same
waveform as decoding it at once, provided three kinds of state carry over:
the decoder transformer's keys and values, the left context of every causal
convolution, and the overlapping tail of every transposed convolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from transformers.models.mimi.modeling_mimi import (
    MimiConv1d,
    MimiConv1dPaddingCache,
    MimiConvTranspose1d,
    MimiResnetBlock,
)


@dataclass
class MimiDecoderState:
    """One request's decoder state between calls."""

    kv: Any = None
    padding: MimiConv1dPaddingCache | None = None
    tails: dict[int, torch.Tensor] = field(default_factory=dict)


class CsmMimiStreamingDecoder:
    """Decodes Mimi codes frame by frame with the modules of a transformers ``MimiModel``."""

    def __init__(self, codec: Any) -> None:
        # transformers 5.21 moved the decoder half into submodules of ``codec.decoder``.
        nested = hasattr(codec.encoder, "encoder_seanet")
        decoder = codec.decoder
        self.quantizer = codec.quantizer
        self.upsample = decoder.upsample if nested else codec.upsample
        self.transformer = decoder.decoder_transformer if nested else codec.decoder_transformer
        self.seanet = decoder.decoder_seanet if nested else decoder

        self._convs = [module for module in self.seanet.modules() if isinstance(module, MimiConv1d)]
        for index, conv in enumerate(self._convs):
            if not conv.causal:
                raise NotImplementedError("incremental Mimi decoding needs causal convolutions")
            # The padding cache finds each convolution's slot through this index.
            conv.layer_idx = index
        transposed = [module for module in self.seanet.modules() if isinstance(module, MimiConvTranspose1d)]
        for layer in [*transposed, *([self.upsample] if self.upsample is not None else [])]:
            if not layer.causal or layer.padding_left != 0:
                raise NotImplementedError("incremental Mimi decoding needs fully right-trimmed transposed convolutions")

    def _padding_cache(self) -> MimiConv1dPaddingCache:
        return MimiConv1dPaddingCache(
            num_layers=len(self._convs),
            per_layer_padding=[int(conv.padding_total) for conv in self._convs],
            per_layer_padding_mode=[conv.pad_mode for conv in self._convs],
            per_layer_in_channels=[conv.in_channels for conv in self._convs],
        )

    @staticmethod
    def _transposed(
        layer: MimiConvTranspose1d, hidden: torch.Tensor, state: MimiDecoderState, key: int
    ) -> torch.Tensor:
        """Overlap-add: emit the samples no later input can change, keep the rest for the next call."""
        output = layer.conv(hidden)
        overlap = layer.padding_right
        if overlap == 0:
            return output
        tail = state.tails.get(key)
        if tail is not None:
            output[..., :overlap] += tail
        keep = output.shape[-1] - overlap
        tail = output[..., keep:]
        if layer.conv.bias is not None:
            # The next call's output adds the bias to these positions again.
            tail = tail - layer.conv.bias[None, :, None]
        state.tails[key] = tail.clone()
        return output[..., :keep]

    def decode(self, codes: torch.Tensor, state: MimiDecoderState) -> torch.Tensor:
        """Decode new frames ``codes`` (``[1, num_codebooks, frames]``) to their samples (``[samples]``)."""
        if state.padding is None:
            state.padding = self._padding_cache()
        hidden = self.quantizer.decode(codes)
        if self.upsample is not None:
            hidden = self._transposed(self.upsample, hidden, state, -1)
        outputs = self.transformer(hidden.transpose(1, 2), past_key_values=state.kv, use_cache=True)
        state.kv = outputs.past_key_values
        hidden = outputs.last_hidden_state.transpose(1, 2)
        for index, layer in enumerate(self.seanet.layers):
            if isinstance(layer, MimiConvTranspose1d):
                hidden = self._transposed(layer, hidden, state, index)
            elif isinstance(layer, (MimiConv1d, MimiResnetBlock)):
                hidden = layer(hidden, padding_cache=state.padding)
            else:
                hidden = layer(hidden)
        return hidden[0, 0]
