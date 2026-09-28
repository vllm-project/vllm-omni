# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from vllm_omni.model_executor.models.moss_tts.codec_attention import selective_slot
from vllm_omni.model_executor.models.moss_tts.slot_attention import slot_ring_attention


@pytest.mark.cuda
@pytest.mark.parametrize("execution", ["eager", "aot_graph", "compiled"])
def test_first_and_steady_chunks_share_opaque_dispatch(execution):
    torch.manual_seed(741)
    device = "cuda"
    heads, dim, capacity, states = 2, 64, 400, 20
    cache = torch.randn(2, states, heads, capacity, dim, device=device, dtype=torch.bfloat16)
    end = torch.zeros(states, device=device, dtype=torch.long)
    refcache, refend = cache.clone(), end.clone()
    fn = selective_slot
    if execution == "compiled":
        fn = torch.compile(fn, fullgraph=True, dynamic=True)
    for step, (batch, frames) in enumerate([(16, 480), (16, 32), (8, 480), (1, 32), (8, 32), (16, 480)]):
        projected = torch.randn(batch, frames, 3, heads, dim, device=device, dtype=cache.dtype)
        q, k, v = projected.permute(2, 0, 3, 1, 4).unbind(0)
        slots = torch.randperm(states, device=device)[:batch]
        lengths = torch.full((batch,), frames, device=device, dtype=torch.int32)
        lengths[-1] = 0
        rope = refend[slots].clone()
        args = (q, k, v, cache, end, slots, lengths, capacity, rope)
        if execution == "aot_graph" and step == 0:
            # Replay exactly the same FX graph without Dynamo shape guards.
            fn = make_fx(selective_slot, tracing_mode="symbolic")(*args)
            ops = [str(n.target) for n in fn.graph.nodes if n.op == "call_function"]
            assert any("moss_codec_selective_attention" in op for op in ops)
            assert not any("moss_codec_direct_fa3" in op for op in ops)
        actual = fn(*args)
        expected = slot_ring_attention(q, k, v, refcache, refend, slots, lengths, capacity, rope)
        torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.025)
        torch.testing.assert_close(cache, refcache, rtol=0, atol=0)
        torch.testing.assert_close(end, refend, rtol=0, atol=0)
        assert actual.stride() == (frames * heads * dim, dim, heads * dim, 1)


pytestmark = [pytest.mark.core_model, pytest.mark.cuda]
