# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Streaming conditioning advances independently of complete DiT blocks."""

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.layers import PreLookaheadLayer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("lengths", [(25, 50, 65, 100), (1, 2, 3, 7, 26)])
@torch.inference_mode()
def test_incremental_prelookahead_matches_full_and_preserves_tail_padding(lengths):
    torch.manual_seed(42)
    embedding = nn.Embedding(128, 8)
    layer = PreLookaheadLayer(8, 16, pre_lookahead_len=3).eval()
    tokens = torch.randint(0, 128, (1, lengths[-1] + 3))
    cache = None
    for length in lengths:
        full_input = embedding(tokens[:, : length + 3])
        expected = layer(full_input[:, :-3], context=full_input[:, -3:])
        start = 0 if cache is None else cache.tokens
        new_input = embedding(tokens[:, start : length + 3])
        actual, cache = layer.forward_incremental(new_input, cache)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        assert cache.tokens == length
    # Final keeps the original full forward with right zero padding. All prior
    # stable positions have real lookahead already and must remain unchanged.
    final = layer(embedding(tokens))
    torch.testing.assert_close(final[:, : cache.tokens], cache.hidden, rtol=1e-5, atol=1e-6)


@torch.inference_mode()
def test_packed_prelookahead_interface_keeps_ragged_request_caches_separate():
    torch.manual_seed(19)
    layer = PreLookaheadLayer(8, 16, 3).eval()
    streams = [torch.randn(1, 53, 8) for _ in range(2)]
    caches = [None, None]
    for lengths in ((25, 17), (50, 40)):
        pending = [
            value[:, 0 if cache is None else cache.tokens : length + 3]
            for value, cache, length in zip(streams, caches, lengths)
        ]
        actual, caches = layer.forward_incremental_packed(pending, caches)
        for value, length, hidden, cache in zip(streams, lengths, actual, caches):
            window = value[:, : length + 3]
            expected = layer(window[:, :-3], context=window[:, -3:])
            torch.testing.assert_close(hidden, expected, rtol=1e-5, atol=1e-6)
            assert cache.tokens == length
