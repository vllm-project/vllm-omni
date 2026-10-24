# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ragged fused pre-lookahead parity and per-request history isolation."""

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.layers import PreLookaheadLayer
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="requires CUDA")]


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_fused_ragged_prelookahead_matches_full_and_swaps_cache_buffers(dtype):
    torch.manual_seed(42)
    layer = PreLookaheadLayer(80, 1024, 3).cuda().to(dtype).eval()
    inputs = [torch.randn(1, 68, 80, device="cuda", dtype=dtype) for _ in range(3)]
    caches = [None] * 3
    for lengths in ((25, 17, 1), (50, 40, 2), (65, 65, 26)):
        previous = caches
        pending = [
            value[:, 0 if cache is None else cache.tokens : length + 3]
            for value, cache, length in zip(inputs, caches, lengths)
        ]
        actual, caches = layer.forward_incremental_packed(pending, caches)
        for row, (value, length, cache) in enumerate(zip(inputs, lengths, caches)):
            window = value[:, : length + 3]
            expected = layer(window[:, :-3], context=window[:, -3:])
            torch.testing.assert_close(actual[row], expected, rtol=3e-2, atol=3e-2)
            first = F.leaky_relu(layer.conv1(window.transpose(1, 2).contiguous()))
            tail = F.pad(first, (max(0, 2 - length), 0))[:, :, -2:]
            torch.testing.assert_close(cache.conv1_tail, tail, rtol=3e-2, atol=3e-2)
            assert cache.tokens == length
            assert cache.conv1_tail.data_ptr() != cache.next_conv1_tail.data_ptr()
            if previous[row] is not None:
                assert cache.conv1_tail.data_ptr() == previous[row].next_conv1_tail.data_ptr()
