# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Causal/GQA correctness and graph replay for the frame-local KV kernel."""

import pytest
import torch

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="requires CUDA"),
]


@pytest.mark.parametrize("batch", [1, 3, 64])
@pytest.mark.parametrize("heads,kv_heads,dim", [(16, 8, 128), (8, 8, 64)])
@pytest.mark.parametrize("positions", [[0, 1], [2], [8], [15]])
def test_short_attention_causal_gqa_and_graph(batch, heads, kv_heads, dim, positions):
    from vllm_omni.model_executor.models.qwen3_tts.short_kv_attention import short_kv_attention

    with torch.inference_mode():
        torch.manual_seed(42)
        pos = torch.tensor(positions, device="cuda")
        q = torch.randn(batch, len(positions), heads, dim, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        cache = torch.randn(2, batch + 3, kv_heads, 17, dim, device="cuda", dtype=torch.bfloat16)
        k, v = cache[0, :batch], cache[1, :batch]
        mask = torch.arange(17, device="cuda")[None, :] <= pos[:, None]
        kr = k.float().repeat_interleave(heads // kv_heads, dim=1)
        vr = v.float().repeat_interleave(heads // kv_heads, dim=1)
        scores = (q.float() @ kr.transpose(-1, -2)) * dim**-0.5
        expected = scores.masked_fill(~mask, -torch.inf).softmax(-1) @ vr
        actual = short_kv_attention(q, k, v, pos, dim**-0.5)
        torch.testing.assert_close(actual.float(), expected, atol=0.016, rtol=0.008)

        # Unwritten cache slots must not affect a new frame, even if poisoned.
        k[:, :, max(positions) + 1 :] = torch.nan
        v[:, :, max(positions) + 1 :] = torch.nan
        poisoned = short_kv_attention(q, k, v, pos, dim**-0.5)
        torch.testing.assert_close(poisoned, actual, atol=0, rtol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = short_kv_attention(q, k, v, pos, dim**-0.5)
        graph.replay()
        torch.testing.assert_close(captured, actual, atol=0, rtol=0)
