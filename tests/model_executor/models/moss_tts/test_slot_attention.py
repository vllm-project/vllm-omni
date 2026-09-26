# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import copy

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 import (
    MossAudioTokenizerMultiheadAttention,
    MossAudioTokenizerRotaryEmbedding,
    RingKVCache,
    StreamingExecutionContext,
)
from vllm_omni.model_executor.models.moss_tts.slot_attention import slot_ring_attention

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]


def test_dense_ring_retains_last_tokens_when_chunk_exceeds_capacity():
    torch.manual_seed(19)
    capacity, frames = 400, 480
    state = RingKVCache(3, 2, 64, capacity, device=torch.device("cuda"), dtype=torch.float32)
    slots = torch.tensor([1, 0], device="cuda")
    context = StreamingExecutionContext(state_slot_ids=slots, valid_rows=torch.ones(2, device="cuda", dtype=torch.bool))
    for start in [0, frames]:
        k = torch.randn(2, 2, frames, 64, device="cuda")
        v = torch.randn_like(k)
        state.complete(k, v, execution_context=context)
        retained = torch.arange(frames - capacity, frames, device="cuda")
        destinations = (start + retained) % capacity
        torch.testing.assert_close(state.cache[0, slots][:, :, destinations], k[:, :, retained], rtol=0, atol=0)
        torch.testing.assert_close(state.cache[1, slots][:, :, destinations], v[:, :, retained], rtol=0, atol=0)
        torch.testing.assert_close(state.end_offset[slots], torch.full_like(slots, start + frames), rtol=0, atol=0)


def _ring_reference(q, k, v, cache, end, slots, lengths, context):
    """Independent chronological ring oracle with FP32 attention arithmetic."""
    capacity = cache.shape[-2]
    output = torch.zeros_like(q)
    for row, (slot, length) in enumerate(zip(slots.tolist(), lengths.tolist())):
        if length == 0:
            continue
        start = int(end[slot])
        for token in range(max(0, length - capacity), length):
            cache[0, slot, :, (start + token) % capacity] = k[row, :, token]
            cache[1, slot, :, (start + token) % capacity] = v[row, :, token]
        stop = start + length
        physical_positions = torch.full((capacity,), -1, device=q.device, dtype=torch.long)
        available = torch.arange(max(0, stop - capacity), stop, device=q.device)
        physical_positions[available % capacity] = available
        queries = torch.arange(start, stop, device=q.device)
        distance = queries[:, None] - physical_positions[None]
        mask = (physical_positions[None] >= 0) & (distance >= 0)
        if context > 0:
            mask &= distance < context
        output[row, :, :length] = F.scaled_dot_product_attention(
            q[row, :, :length].float(), cache[0, slot].float(), cache[1, slot].float(), mask[None]
        ).to(q.dtype)
        end[slot] = stop
    return output


@pytest.mark.parametrize("frames,capacity,head_dim", [(1, 125, 64), (15, 125, 64), (480, 400, 64), (33, 65, 32)])
@pytest.mark.parametrize("execution", ["eager", "graph", "compiled"])
def test_slot_attention_preserves_ring_state_and_padding(frames, capacity, head_dim, execution):
    torch.manual_seed(29)
    batch, heads, state_capacity = 3, 2, 5
    projected = torch.randn(batch, frames, 3, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    q, k, v = projected.permute(2, 0, 3, 1, 4).unbind(0)
    cache = torch.randn(2, state_capacity, heads, capacity, head_dim, device="cuda", dtype=q.dtype)
    # Strided metadata is allowed by the kernel contract.
    end = torch.zeros(state_capacity, 2, device="cuda", dtype=torch.long)[:, 0]
    slots = torch.arange(batch * 2, device="cuda")[::2]
    lengths = torch.full((batch, 2), frames, device="cuda", dtype=torch.int32)[:, 0]
    context = capacity
    attention = slot_ring_attention
    if execution == "compiled":
        # Functionalization must preserve mutations of the persistent ring
        # and its offsets between calls, including slot reuse.
        attention = torch.compile(slot_ring_attention, backend="inductor", fullgraph=True)
    if execution == "graph":
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                slot_ring_attention(q, k, v, cache, end, slots, lengths, context)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = slot_ring_attention(q, k, v, cache, end, slots, lengths, context)
    end.zero_()
    expected_cache, expected_end = cache.clone(), end.clone()
    for step in range(8):
        slots.copy_(torch.roll(torch.arange(state_capacity, device="cuda"), step)[:batch])
        lengths.copy_(torch.tensor([frames, max(1, frames - 3), 0], device="cuda", dtype=torch.int32))
        projected.normal_()
        if step == 4:
            end[1:3].zero_()
            expected_end[1:3].zero_()
        expected = _ring_reference(q, k, v, expected_cache, expected_end, slots, lengths, context)
        if execution == "graph":
            graph.replay()
        else:
            actual = attention(q, k, v, cache, end, slots, lengths, context)
        torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)
        torch.testing.assert_close(end, expected_end, rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.025)
        padding = torch.arange(frames, device="cuda")[None] >= lengths[:, None]
        assert actual.permute(0, 2, 1, 3)[padding].count_nonzero() == 0


@pytest.mark.parametrize("frames,context", [(1, 9), (3, 9), (15, 125), (480, 400)])
def test_slot_attention_integrates_with_current_mha(frames, context):
    torch.manual_seed(9)
    reference = MossAudioTokenizerMultiheadAttention(
        128,
        2,
        causal=True,
        context=context,
        rope=MossAudioTokenizerRotaryEmbedding(),
        device="cuda",
        dtype=torch.bfloat16,
    )
    candidate = copy.deepcopy(reference)
    candidate._slot_attention = slot_ring_attention
    for model in (reference, candidate):
        model._streaming_state = model._init_streaming_state(4)
    for step in range(8):
        slots = torch.tensor([1, 0, 3] if step % 2 else [0, 1, 3], device="cuda")
        valid = torch.tensor([True, True, False], device="cuda")
        execution_context = StreamingExecutionContext(state_slot_ids=slots, valid_rows=valid)
        x = torch.randn(3, frames, 128, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            expected = reference(x, x, x, execution_context=execution_context)
            actual = candidate(x, x, x, execution_context=execution_context)
        torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.02)
        cs, rs = candidate._streaming_state, reference._streaming_state
        torch.testing.assert_close(cs.offset, rs.offset, rtol=0, atol=0)
        torch.testing.assert_close(cs.kv_cache.end_offset, rs.kv_cache.end_offset, rtol=0, atol=0)
        torch.testing.assert_close(cs.kv_cache.cache[:, :3], rs.kv_cache.cache[:, :3], rtol=0, atol=0)
        if step == 4:
            for model in (reference, candidate):
                model._streaming_state.reset_slots(torch.tensor([0], device="cuda"))
