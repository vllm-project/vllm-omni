# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the always-on explicit codec glue ops (rope unpack + packed mask)."""

import pytest
import torch

from vllm_omni.model_executor.models.moss_tts import audio_tokenizer_v2 as atv2
from vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 import (
    MossAudioTokenizerTransformer,
    StreamingExecutionContext,
    apply_rope,
)

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.tts,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@torch.inference_mode()
@pytest.mark.parametrize("batch", [1, 4, 8, 16])
@pytest.mark.parametrize("frames", [1, 15, 30])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_rope_unpack_qkv_matches_apply_rope(batch, frames, dtype):
    from vllm_omni.model_executor.models.moss_tts.codec_fused_ops import codec_rope_unpack_qkv

    torch.manual_seed(batch * 1000 + frames)
    heads, dim = 20, 64
    projected = torch.randn(batch, frames, 3, heads, dim, device="cuda", dtype=dtype)
    offset = torch.randint(0, 2**31 - 1, (batch,), device="cuda", dtype=torch.long)
    q, k, v = codec_rope_unpack_qkv(projected, offset, 10_000.0)

    unpacked = projected.permute(2, 0, 3, 1, 4)
    q_ref, k_ref = apply_rope(unpacked[0], unpacked[1], offset, 10_000.0)
    v_ref = unpacked[2]

    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    # libdevice trig + disabled FP fusion make the rotation bitwise identical.
    torch.testing.assert_close(q, q_ref, atol=0, rtol=0)
    torch.testing.assert_close(k, k_ref, atol=0, rtol=0)
    torch.testing.assert_close(v, v_ref, atol=0, rtol=0)


@torch.inference_mode()
def test_rope_unpack_qkv_nonstandard_dims():
    from vllm_omni.model_executor.models.moss_tts.codec_fused_ops import codec_rope_unpack_qkv

    torch.manual_seed(7)
    for heads, dim in [(20, 64), (12, 64), (4, 96), (2, 128)]:
        projected = torch.randn(3, 11, 3, heads, dim, device="cuda", dtype=torch.bfloat16)
        offset = torch.randint(0, 100_000, (3,), device="cuda", dtype=torch.long)
        q, k, v = codec_rope_unpack_qkv(projected, offset, 10_000.0)
        unpacked = projected.permute(2, 0, 3, 1, 4)
        q_ref, k_ref = apply_rope(unpacked[0], unpacked[1], offset, 10_000.0)
        torch.testing.assert_close(q, q_ref, atol=0, rtol=0)
        torch.testing.assert_close(k, k_ref, atol=0, rtol=0)
        torch.testing.assert_close(v, unpacked[2], atol=0, rtol=0)


@torch.inference_mode()
@pytest.mark.parametrize("capacity", [125, 250, 400])
@pytest.mark.parametrize("frames", [1, 15, 30])
@pytest.mark.parametrize("context", [None, 100])
def test_packed_causal_mask_integer_exact(capacity, frames, context):
    from vllm_omni.model_executor.models.moss_tts.codec_fused_ops import codec_causal_mask

    torch.manual_seed(capacity * 100 + frames)
    batch = 5
    offset_end = torch.randint(0, 20_000, (batch,), device="cuda", dtype=torch.long)
    query_offset = torch.randint(0, 20_000, (batch,), device="cuda", dtype=torch.long)
    valid_rows = torch.rand(batch, device="cuda") < 0.6
    valid_rows[0] = True
    context_value = 0 if context is None else min(context, capacity)

    actual = codec_causal_mask(offset_end, query_offset, valid_rows, frames, capacity, context_value)

    # Reference: RingKVCache.complete positions chain + forward mask math.
    cache_indexes = torch.arange(capacity, device="cuda", dtype=torch.long)
    next_offset = torch.where(valid_rows, offset_end + frames, offset_end)
    last_offset = offset_end.view(-1, 1) + frames - 1
    end_index = last_offset % capacity
    delta_index = cache_indexes - end_index
    pos_k = torch.where(
        delta_index <= 0,
        last_offset + delta_index,
        last_offset + delta_index - capacity,
    )
    invalid = cache_indexes >= next_offset.view(-1, 1)
    pos_k = torch.where(invalid, torch.full_like(pos_k, -1), pos_k)[:, None]
    pos_q = query_offset.view(-1, 1, 1) + torch.arange(frames, device="cuda", dtype=torch.long).view(-1, 1)
    delta = pos_q - pos_k
    expected = (pos_k >= 0) & (delta >= 0)
    if context is not None:
        expected = expected & (delta < context_value)
    expected = expected[:, None]

    assert actual.shape == (batch, 1, frames, capacity)
    assert actual.dtype == torch.bool
    assert torch.equal(actual, expected)


def _build_tiny_transformer(seed=11):
    torch.manual_seed(seed)
    return MossAudioTokenizerTransformer(
        d_model=128,
        num_heads=2,
        num_layers=3,
        dim_feedforward=256,
        causal=True,
        context=125,
        positional_embedding="rope",
        max_period=10_000,
        layer_scale=1e-4,
        norm="layer_norm",
        device="cuda",
        dtype=torch.bfloat16,
    )


@torch.inference_mode()
def _run_streaming_steps(transformer, ctx, slot_pool, chunks, use_kv_pack):
    with transformer.streaming(slot_pool):
        for module in transformer.modules():
            if isinstance(module, atv2.MossAudioTokenizerMultiheadAttention):
                kv_cache = module._streaming_state.kv_cache
                if kv_cache is not None:
                    kv_cache._use_kv_pack = use_kv_pack
        outputs: list[torch.Tensor] = []
        for frames in chunks:
            torch.manual_seed(frames * 91 + len(outputs))
            x = torch.randn(len(ctx.state_slot_ids), frames, 128, device="cuda", dtype=torch.bfloat16)
            outputs.append(transformer(x, execution_context=ctx))
        state = transformer._streaming_state
        attn_state = transformer.layers[0].self_attn._streaming_state
        final = (
            state.offsets.clone(),
            attn_state.kv_cache.end_offset.clone(),
            attn_state.offset.clone(),
        )
    return outputs, final


@torch.inference_mode()
@pytest.mark.parametrize("chunks", [[15, 1, 30], [1, 15]])
def test_transformer_stack_ops_equivalence(chunks):
    """The always-on explicit glue ops (+ pack_ring_kv) must match the unfused stack."""
    slots = torch.tensor([1, 4, 6], device="cuda")
    valid = torch.tensor([True, False, True], device="cuda")
    ctx = StreamingExecutionContext(slots, valid)

    reference = _build_tiny_transformer()
    ref_outputs, ref_final = _run_streaming_steps(reference, ctx, 8, chunks, use_kv_pack=False)

    for use_kv_pack in (False, True):
        candidate = _build_tiny_transformer()
        out_outputs, out_final = _run_streaming_steps(candidate, ctx, 8, chunks, use_kv_pack=use_kv_pack)
        for ref, out in zip(ref_outputs, out_outputs):
            torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0)
        for ref_state, out_state in zip(ref_final, out_final):
            torch.testing.assert_close(out_state, ref_state, atol=0, rtol=0)
