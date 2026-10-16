# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_marks

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
    pytest.mark.tts,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.mark.parametrize("batch,heads,dim", [(1, 4, 64), (16, 32, 80), (128, 32, 80), (3, 8, 128)])
def test_all_depth_positions_with_strided_cache_and_dirty_future(batch, heads, dim):
    from vllm_omni.model_executor.models.moss_tts.local_short_attention import local_short_attention

    torch.manual_seed(43)
    query = torch.randn(batch, heads, 1, dim, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(batch, heads, 12, dim, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    for position in range(12):
        # Non-contiguous prefixes retain the stride of the full 12-slot cache.
        k, v = key[:, :, : position + 1], value[:, :, : position + 1]
        before_k, before_v = key.clone(), value.clone()
        expected = F.scaled_dot_product_attention(query, k, v)
        actual = local_short_attention(query, k, v)
        assert torch.equal(key, before_k) and torch.equal(value, before_v)
        if position == 0:
            assert torch.equal(actual, v)
        torch.testing.assert_close(actual, expected, atol=0.015625, rtol=0.01)
        # Future cache contents must never affect an earlier position.
        key[:, :, position + 1 :] += 100
        value[:, :, position + 1 :] -= 100
        assert torch.equal(local_short_attention(query, k, v), actual)
        key.copy_(before_k)
        value.copy_(before_v)


def test_compile_and_graph_replay_reads_fresh_inputs():
    from vllm_omni.model_executor.models.moss_tts.local_short_attention import local_short_attention

    query = torch.randn(16, 32, 1, 96, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(16, 32, 12, 96, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    compiled = torch.compile(local_short_attention, fullgraph=True)
    for _ in range(3):
        compiled(query, key, value)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = compiled(query, key, value)
    for _ in range(3):
        query.normal_()
        key.normal_()
        value.normal_()
        graph.replay()
        torch.testing.assert_close(result, F.scaled_dot_product_attention(query, key, value), atol=0.015625, rtol=0.01)


def test_depth_block_cache_overwrite_and_compiled_positions(monkeypatch):
    from transformers import GPT2Config

    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_local_depth import (
        MossTTSLocalDepthTransformer,
    )

    monkeypatch.setenv("VLLM_OMNI_MOSS_LOCAL_SHORT_ATTN", "1")
    model = MossTTSLocalDepthTransformer(GPT2Config(n_embd=320, n_head=4, n_inner=640)).cuda().bfloat16().eval()
    attn = model.h[0].attn
    attn.prepare_rope_cache(12, torch.device("cuda"), torch.bfloat16)
    cache = tuple(torch.empty(4, 4, 12, 80, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    forward = torch.compile(model._forward_prefix, fullgraph=True, dynamic=True)
    with torch.inference_mode():
        for _ in range(2):
            inputs = torch.randn(4, 12, 320, device="cuda", dtype=torch.bfloat16)
            # The second frame overwrites the previous frame's K/V before use.
            for position in range(12):
                expected = model._forward_prefix(inputs[:, : position + 1])[:, -1:]
                actual = forward(inputs[:, position : position + 1], cache, position)
                torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.02)
