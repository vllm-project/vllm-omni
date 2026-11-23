# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.attention.parallel.rocm_h3_head_overlap import _pack_torch, _restore_torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_head_group_packets_preserve_head_order_for_strided_inputs():
    rows, heads, width = 3, 56, 128
    # An interleaved V plane, a misaligned Q plane, and a non-unit K channel stride.
    v = torch.randn(rows, 3, heads, width, dtype=torch.bfloat16)[:, 2]
    q = torch.randn(v.numel() + 1, dtype=v.dtype)[1:].view_as(v)
    k = torch.randn(rows, heads, 2 * width, dtype=v.dtype)[:, :, ::2]
    first, second = _pack_torch(q, k, v)
    for packet, start, stop in ((first, 0, 4), (second, 4, 14)):
        assert packet.is_contiguous()
        for destination in range(4):
            for slot, plane in enumerate((q, k, v)):
                expected = plane[:, destination * 14 + start : destination * 14 + stop]
                assert torch.equal(packet[destination, :, slot], expected)

    for slot, plane in enumerate((q, k, v)):
        restored = _restore_torch(first[:, :, slot].contiguous(), second[:, :, slot].contiguous())
        assert restored.is_contiguous()
        assert torch.equal(restored, plane)
