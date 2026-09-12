# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import copy

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 import (
    MossAudioTokenizerMultiheadAttention,
    StreamingExecutionContext,
)
from vllm_omni.model_executor.models.moss_tts.streaming_attention import masked_attention

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]


@pytest.mark.parametrize(
    "batch,heads,queries,keys", [(2, 20, 1, 125), (4, 20, 15, 125), (1, 12, 480, 400), (4, 12, 32, 250)]
)
def test_masked_attention_matches_sdpa_with_strides_and_invalid_rows(batch, heads, queries, keys):
    torch.manual_seed(7)
    q = torch.randn(batch, queries, 3, heads, 64, device="cuda", dtype=torch.bfloat16)[:, :, 0].permute(0, 2, 1, 3)
    k = torch.randn(batch, heads, keys, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    positions = torch.arange(keys, device="cuda").roll(17)
    query_positions = torch.arange(queries, device="cuda") + keys - queries
    mask = (positions[None, :] <= query_positions[:, None])[None, None].expand(batch, 1, -1, -1).clone()
    mask[0, :, 0] = False  # Cover an all-masked row alongside valid rows.
    actual = masked_attention(q, k, v, mask)
    expected = F.scaled_dot_product_attention(q, k, v, mask)
    assert bool(actual.isfinite().all())
    torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.01)
    assert actual[0, :, 0].count_nonzero() == 0


def test_streaming_attention_preserves_ring_wrap_reordering_and_slot_reset():
    torch.manual_seed(9)
    reference = MossAudioTokenizerMultiheadAttention(
        128, 2, causal=True, context=9, device="cuda", dtype=torch.bfloat16
    )
    candidate = copy.deepcopy(reference)
    candidate._streaming_attention = masked_attention
    for model in (reference, candidate):
        model._streaming_state = model._init_streaming_state(4)
    for step in range(8):
        slots = torch.tensor([1, 0, 3] if step % 2 else [0, 1, 3], device="cuda")
        valid = torch.tensor([True, True, False], device="cuda")
        context = StreamingExecutionContext(state_slot_ids=slots, valid_rows=valid)
        x = torch.randn(3, 3, 128, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            expected = reference(x, x, x, execution_context=context)
            actual = candidate(x, x, x, execution_context=context)
        torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.02)
        torch.testing.assert_close(candidate._streaming_state.offset, reference._streaming_state.offset, rtol=0, atol=0)
        if step == 4:
            for model in (reference, candidate):
                model._streaming_state.reset_slots(torch.tensor([0], device="cuda"))


def test_output_layout_compile():
    from vllm_omni.model_executor.models.moss_tts import streaming_attention as attention

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    q = torch.randn(1, 12, 15, 64, device="cuda", dtype=torch.bfloat16)
    k, v = torch.randn(2, 1, 12, 125, 64, device="cuda", dtype=q.dtype)
    mask = torch.ones(1, 1, 15, 125, device="cuda", dtype=torch.bool)
    out = attention.masked_attention(q, k, v, mask)
    if attention.OUTPUT_BTHD:
        flattened = out.transpose(1, 2).reshape(1, 15, 768)
        assert flattened.data_ptr() == out.data_ptr()
        assert flattened.is_contiguous()
    compiled = torch.compile(attention.masked_attention, fullgraph=True)
    value = compiled(q, k, v, mask)
    assert value.stride() == out.stride()
    torch.testing.assert_close(value, out, rtol=0, atol=0)
