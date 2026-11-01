# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""NPU parity tests for MOSS-TTS codec attention, ring-cache, and RoPE.

Verifies that the NPU-gated fast paths in ``audio_tokenizer_v2.py`` produce
results equivalent to the reference (eager / SDPA) implementations:

1. **Attention masks** — ``npu_fusion_attention`` with inverted bool mask
   matches ``F.scaled_dot_product_attention`` for both masked and unmasked
   (``attn_bias=None``) cases.  Includes streaming-mode coverage with
   ``capacity >> T`` and an already-rotated ring-buffer mask.
2. **Ring-cache wraparound** — ``RingKVCache`` positions are correct when
   ``end_offset + T`` exceeds ``capacity`` (ring buffer wraparound).
3. **RoPE layout changes** — the NPU path (interleaved→neox conversion +
   ``npu_rotary_mul`` + back-conversion skip) matches the eager GPT-J
   interleaved rotation, and the skip is safe because attention QK^T is
   layout-independent.

Platform-independent tests (ring position arithmetic, mask convention,
arange cache identity) run on CPU in CI.  NPU-specific parity tests are
marked ``@npu_only`` and require a real NPU device.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 import (
    MossAudioTokenizerMultiheadAttention,
    MossAudioTokenizerRotaryEmbedding,
    RingKVCache,
    StreamingExecutionContext,
    apply_rope,
)

pytestmark = [pytest.mark.core_model, pytest.mark.tts]


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return bool(hasattr(torch, "npu") and torch.npu.is_available())


npu_only = pytest.mark.skipif(not _npu_available(), reason="NPU device or torch_npu not available.")
npu_device = pytest.param("npu", marks=npu_only)


# =============================================================================
# 1a. Platform-independent: ring-cache position arithmetic (runs on CPU in CI)
# =============================================================================


@pytest.mark.cpu
def test_ring_kv_cache_positions_after_wraparound():
    """RingKVCache positions are correct when end_offset + T > capacity.

    After writing enough frames to wrap around the ring buffer, positions
    should reflect the modular arithmetic: entries older than the window
    are marked -1, and valid entries are in sequential order.
    """
    dev = torch.device("cpu")
    dtype = torch.bfloat16
    B, H, D = 2, 4, 64
    capacity = 8
    T = 5

    cache = RingKVCache(B, H, D, capacity, respect_exec_mask=True, device=dev, dtype=dtype)

    k1 = torch.randn(B, H, T, D, device=dev, dtype=dtype)
    v1 = torch.randn_like(k1)
    cache.complete(k1, v1, exec_mask=torch.ones(B, dtype=torch.bool, device=dev))
    assert cache.end_offset.tolist() == [T, T]

    k2 = torch.randn(B, H, T, D, device=dev, dtype=dtype)
    v2 = torch.randn_like(k2)
    result2 = cache.complete(k2, v2, exec_mask=torch.ones(B, dtype=torch.bool, device=dev))

    assert cache.end_offset.tolist() == [2 * T, 2 * T]
    positions = result2.positions
    assert positions.shape == (B, capacity)
    assert (positions >= 0).all()
    for b in range(B):
        batch_positions = positions[b]
        assert batch_positions.unique().numel() == capacity
        assert batch_positions.min().item() == 2 * T - capacity
        assert batch_positions.max().item() == 2 * T - 1


@pytest.mark.cpu
def test_ring_kv_cache_arange_uses_object_lifetime_storage():
    """_get_arange_t returns a slice of _arange_capacity (same storage).

    This verifies the fix for the CUDA-graph dangling-pointer issue: the
    returned tensor shares storage with _arange_capacity (allocated in
    __init__, lives for the object's lifetime) so captured CUDA graphs
    never read freed memory.
    """
    dev = torch.device("cpu")
    cache = RingKVCache(2, 4, 64, 10, device=dev, dtype=torch.bfloat16)

    assert cache._arange_capacity is not None
    assert cache._arange_capacity.shape == (10,)
    assert cache._arange_capacity.tolist() == list(range(10))

    # Same T should return a slice of _arange_capacity (same storage).
    t1 = cache._get_arange_t(5, dev, torch.long)
    assert t1 is not cache._arange_capacity  # slice, not the same tensor
    assert t1.data_ptr() >= cache._arange_capacity.data_ptr()
    assert t1.data_ptr() < cache._arange_capacity.data_ptr() + cache._arange_capacity.numel() * 8
    assert t1.tolist() == [0, 1, 2, 3, 4]

    # Different T within capacity also returns a slice (same underlying storage).
    t2 = cache._get_arange_t(8, dev, torch.long)
    assert t2.tolist() == [0, 1, 2, 3, 4, 5, 6, 7]

    # T > capacity falls back to torch.arange (separate allocation).
    t3 = cache._get_arange_t(15, dev, torch.long)
    assert t3.tolist() == list(range(15))


@pytest.mark.cpu
def test_ring_kv_cache_wraparound_with_execution_context():
    """RingKVCache wraparound with StreamingExecutionContext (dynamic slots).

    Verifies that positions after wraparound are correct, not just end_offset.
    """
    dev = torch.device("cpu")
    dtype = torch.bfloat16
    B, H, D = 3, 2, 32
    capacity = 6
    T = 4

    cache = RingKVCache(B, H, D, capacity, respect_exec_mask=True, device=dev, dtype=dtype)

    slots = torch.tensor([0, 1, 2], device=dev, dtype=torch.long)
    valid = torch.tensor([True, True, False], device=dev, dtype=torch.bool)
    ctx = StreamingExecutionContext(state_slot_ids=slots, valid_rows=valid)

    k = torch.randn(B, H, T, D, device=dev, dtype=dtype)
    v = torch.randn_like(k)
    result = cache.complete(k, v, execution_context=ctx)

    positions = result.positions
    assert positions.shape == (B, capacity)

    # Slot 2 (valid=False) should have all positions = -1
    assert (positions[2] == -1).all()

    # Slots 0 and 1 (valid=True): T=4 positions are valid, rest are -1
    assert (positions[0] >= 0).sum().item() == T
    assert (positions[1] >= 0).sum().item() == T
    # Valid positions should be [0, 1, 2, 3]
    valid_pos = positions[0][positions[0] >= 0]
    torch.testing.assert_close(valid_pos, torch.arange(T, dtype=torch.long))
    # Slot 2: all invalid (valid=False, offset stays 0, all cache_indexes >= 0)
    assert (positions[2] == -1).all()

    # Second step: offset goes from 4 to 8, wrapping around capacity=6
    k2 = torch.randn(B, H, T, D, device=dev, dtype=dtype)
    v2 = torch.randn_like(k2)
    result2 = cache.complete(k2, v2, execution_context=ctx)
    positions2 = result2.positions

    # Slot 2 still invalid
    assert (positions2[2] == -1).all()
    # Slots 0 and 1: offset=8 > capacity=6, so all 6 slots valid
    assert (positions2[0] >= 0).all()
    assert (positions2[1] >= 0).all()
    assert positions2[0].unique().numel() == capacity
    # Expected range: [2*4 - 6, 2*4 - 1] = [2, 7]
    assert positions2[0].min().item() == 2 * T - capacity
    assert positions2[0].max().item() == 2 * T - 1


@pytest.mark.cpu
def test_ring_kv_cache_mask_convention_sdpa():
    """Verify the mask convention: True=attend in SDPA, True=masked in NPU.

    This test runs on CPU and verifies that the ~attn_bias inversion used
    in the NPU branch produces the correct logical complement.
    """
    q_len, kv_len = 5, 10
    pos_q = torch.arange(q_len).view(-1, 1)
    pos_k = torch.arange(kv_len).view(1, -1)
    attn_bias = pos_k <= pos_q + kv_len - q_len  # True = attend

    # NPU convention: True = masked (inverted)
    atten_mask = ~attn_bias

    # The inversion should flip all values
    assert (atten_mask != attn_bias).all()
    # attend positions should be False in atten_mask
    assert not atten_mask[0, 0].item()  # pos_k=0 <= pos_q=0+5: attend -> False in mask
    # masked positions should be True in atten_mask
    assert atten_mask[0, 9].item()  # pos_k=9 > pos_q=0+5: masked -> True in mask


# =============================================================================
# 1c. Platform-independent: attention arange cache retains all sizes (P1)
# =============================================================================


@pytest.mark.cpu
def test_attention_arange_cache_retains_all_sizes():
    """_arange_t_cache must retain tensors for all (device, T) pairs, not just
    the most recent one.

    This is the P1 regression test from amy-why-3459: if changing T replaces
    the cached tensor, a previously captured CUDA graph that baked in the old
    pointer would read freed memory.  The fix uses a dict keyed by
    (device, T) so no allocation is ever released.

    On CPU we verify that both tensors survive in the cache after both sizes
    are requested.
    """
    dev = torch.device("cpu")
    dtype = torch.bfloat16
    embed_dim, num_heads = 128, 2

    attn = MossAudioTokenizerMultiheadAttention(
        embed_dim, num_heads, causal=True, context=30, device=dev, dtype=dtype
    ).eval()

    # Run with T=5 (populates cache for (cpu, 5))
    x5 = torch.randn(1, 5, embed_dim, device=dev, dtype=dtype)
    with torch.inference_mode():
        attn(x5, x5, x5)
    assert (dev, 5) in attn._arange_t_cache
    t5_first = attn._arange_t_cache[(dev, 5)]

    # Run with T=3 (populates cache for (cpu, 3) — must NOT free (cpu, 5))
    x3 = torch.randn(1, 3, embed_dim, device=dev, dtype=dtype)
    with torch.inference_mode():
        attn(x3, x3, x3)
    assert (dev, 3) in attn._arange_t_cache

    # The T=5 tensor must still be the same object (not freed and reallocated)
    assert (dev, 5) in attn._arange_t_cache
    assert attn._arange_t_cache[(dev, 5)] is t5_first, (
        "T=5 cache entry was replaced — CUDA graph holding the old pointer would read freed memory after T changed"
    )

    # Run T=5 again — should reuse the same cached tensor
    with torch.inference_mode():
        attn(x5, x5, x5)
    assert attn._arange_t_cache[(dev, 5)] is t5_first


# =============================================================================
# 1b. Platform-independent: RoPE freqs cache (runs on CPU in CI)
# =============================================================================


@pytest.mark.cpu
def test_rope_freqs_cache_matches_reference():
    """MossAudioTokenizerRotaryEmbedding._get_freqs caches and matches reference."""
    dev = torch.device("cpu")
    rope = MossAudioTokenizerRotaryEmbedding(max_period=10000.0)

    D = 64
    freqs = rope._get_freqs(D, dev)

    ds = torch.arange(D // 2, device=dev, dtype=torch.float32)
    expected = torch.exp(ds * (-math.log(10000.0) * 2 / D))

    torch.testing.assert_close(freqs, expected, atol=0, rtol=0)

    # Second call should return the exact same tensor object (cached).
    freqs2 = rope._get_freqs(D, dev)
    assert freqs is freqs2


# =============================================================================
# 2. NPU-only: Attention mask parity (npu_fusion_attention vs SDPA)
# =============================================================================


@npu_only
@pytest.mark.parametrize("batch,heads,q_len,kv_len", [(2, 4, 8, 16), (1, 8, 15, 15), (4, 2, 1, 30)])
def test_npu_fusion_attention_with_mask_matches_sdpa(batch, heads, q_len, kv_len):
    """npu_fusion_attention (inverted bool mask) == SDPA (bool mask)."""
    import torch_npu

    torch.manual_seed(42)
    dtype = torch.bfloat16
    dev = torch.device("npu")
    q = torch.randn(batch, heads, q_len, 64, device=dev, dtype=dtype)
    k = torch.randn(batch, heads, kv_len, 64, device=dev, dtype=dtype)
    v = torch.randn_like(k)

    pos_q = torch.arange(q_len, device=dev).view(-1, 1)
    pos_k = torch.arange(kv_len, device=dev).view(1, -1)
    attn_bias = (pos_k <= pos_q + kv_len - q_len).unsqueeze(0).unsqueeze(0).expand(batch, 1, q_len, kv_len).clone()

    atten_mask = ~attn_bias
    scale = 1.0 / 64**0.5
    x_npu = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        head_num=heads,
        input_layout="BNSD",
        atten_mask=atten_mask,
        scale=scale,
        keep_prob=1.0,
        pre_tockens=kv_len,
        next_tockens=k.shape[-2],
        sparse_mode=0,
    )[0]
    ref = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)
    torch.testing.assert_close(x_npu, ref, atol=0, rtol=0)


@npu_only
@pytest.mark.parametrize("batch,heads,q_len", [(2, 4, 16), (1, 8, 30)])
def test_npu_fusion_attention_no_mask_matches_sdpa(batch, heads, q_len):
    """npu_fusion_attention (atten_mask=None) == SDPA (attn_bias=None)."""
    import torch_npu

    torch.manual_seed(99)
    dtype = torch.bfloat16
    dev = torch.device("npu")
    q = torch.randn(batch, heads, q_len, 64, device=dev, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    scale = 1.0 / 64**0.5
    x_npu = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        head_num=heads,
        input_layout="BNSD",
        atten_mask=None,
        scale=scale,
        keep_prob=1.0,
        pre_tockens=q_len,
        next_tockens=q_len,
        sparse_mode=0,
    )[0]
    ref = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    torch.testing.assert_close(x_npu, ref, atol=0, rtol=0)


@npu_only
def test_attention_forward_npu_branch_matches_sdpa_fallback():
    """Full attention forward: NPU merged branch produces same result as SDPA."""
    torch.manual_seed(123)
    dev = torch.device("npu")
    dtype = torch.bfloat16
    embed_dim, num_heads = 128, 2

    attn = MossAudioTokenizerMultiheadAttention(
        embed_dim, num_heads, causal=True, context=9, device=dev, dtype=dtype
    ).eval()

    B, T = 3, 5
    x = torch.randn(B, T, embed_dim, device=dev, dtype=dtype)

    with torch.inference_mode():
        result_npu = attn(x, x, x)
        attn_cpu = MossAudioTokenizerMultiheadAttention(
            embed_dim, num_heads, causal=True, context=9, device="cpu", dtype=dtype
        ).eval()
        attn_cpu.load_state_dict(attn.state_dict())
        x_cpu = x.cpu()
        ref = attn_cpu(x_cpu, x_cpu, x_cpu).to(dev)
        torch.testing.assert_close(result_npu, ref, atol=1e-2, rtol=1e-2)


@npu_only
def test_attention_streaming_mode_with_rotated_mask():
    """NPU attention with streaming mode (capacity >> T, rotated ring-buffer mask).

    Initializes streaming state so the RingKVCache fills and wraps around.
    Asserts that offsets advance through wraparound and compares each step
    against an SDPA reference with equivalent state and weights.
    """
    dev = torch.device("npu")
    dtype = torch.bfloat16
    embed_dim, num_heads = 128, 2
    capacity = 30
    T = 5

    # NPU attention (uses npu_fusion_attention)
    attn_npu = MossAudioTokenizerMultiheadAttention(
        embed_dim, num_heads, causal=True, context=capacity, device=dev, dtype=dtype
    ).eval()

    # CPU reference (uses SDPA)
    attn_cpu = MossAudioTokenizerMultiheadAttention(
        embed_dim, num_heads, causal=True, context=capacity, device="cpu", dtype=dtype
    ).eval()
    attn_cpu.load_state_dict(attn_npu.state_dict())

    B = 2
    # Initialize streaming state for both
    attn_npu._streaming_state = attn_npu._init_streaming_state(B)
    attn_cpu._streaming_state = attn_cpu._init_streaming_state(B)

    x_npu = torch.randn(B, T, embed_dim, device=dev, dtype=dtype)
    x_cpu = x_npu.cpu()

    with torch.inference_mode():
        for step in range(8):
            result_npu = attn_npu(x_npu, x_npu, x_npu)
            result_cpu = attn_cpu(x_cpu, x_cpu, x_cpu)

            # Assert offsets advance
            state_npu = attn_npu._streaming_state
            state_cpu = attn_cpu._streaming_state
            expected_offset = (step + 1) * T
            assert state_npu.offset[0].item() == expected_offset, (
                f"step {step}: NPU offset={state_npu.offset[0].item()}, expected={expected_offset}"
            )
            assert state_cpu.offset[0].item() == expected_offset, (
                f"step {step}: CPU offset={state_cpu.offset[0].item()}, expected={expected_offset}"
            )

            # After step 6 (offset=35 > capacity=30): wraparound occurred
            if step >= 6:
                assert state_npu.offset[0].item() > capacity, (
                    f"step {step}: offset should exceed capacity={capacity} for wraparound"
                )

            # Compare NPU vs CPU (SDPA) outputs
            torch.testing.assert_close(result_npu.cpu(), result_cpu, atol=1e-2, rtol=1e-2)

    assert not torch.any(torch.isnan(result_npu))


# =============================================================================
# 3. NPU-only: RoPE layout parity (neox path vs eager interleaved)
# =============================================================================


@npu_only
def test_rope_npu_neox_path_matches_eager_interleaved():
    """NPU RoPE path (neox conversion + npu_rotary_mul + back-conv skip)
    matches the eager GPT-J interleaved rotation via QK^T parity.
    """
    dev = torch.device("npu")
    dtype = torch.bfloat16
    B, H, T, D = 2, 4, 8, 64

    torch.manual_seed(77)
    q = torch.randn(B, H, T, D, device=dev, dtype=dtype)
    k = torch.randn_like(q)
    offset = torch.tensor([3, 5], device=dev, dtype=torch.long)

    rope = MossAudioTokenizerRotaryEmbedding(max_period=10000.0)
    freqs = rope._get_freqs(D, dev)
    q_npu, k_npu = apply_rope(q, k, offset, freqs_cache=freqs)

    q_cpu, k_cpu, offset_cpu = q.cpu(), k.cpu(), offset.cpu()
    q_ref, k_ref = apply_rope(q_cpu, k_cpu, offset_cpu)

    # Convert eager result to neox for comparison.
    dims = q_ref.shape[:-1]
    q_ref_neox = q_ref.view(*dims, D // 2, 2).transpose(-1, -2).reshape(*dims, D)
    k_ref_neox = k_ref.view(*dims, D // 2, 2).transpose(-1, -2).reshape(*dims, D)

    torch.testing.assert_close(q_npu.cpu(), q_ref_neox, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(k_npu.cpu(), k_ref_neox, atol=1e-2, rtol=1e-2)

    # Verify QK^T parity (layout independence).
    qkt_npu = q_npu.float() @ k_npu.float().transpose(-1, -2)
    qkt_ref = q_ref.float() @ k_ref.float().transpose(-1, -2)
    torch.testing.assert_close(qkt_npu.cpu(), qkt_ref, atol=1e-3, rtol=1e-3)


@pytest.mark.cpu
def test_rope_back_conversion_skip_is_layout_independent():
    """Verify that skipping the neox→interleaved back-conversion is safe.

    Attention QK^T = sum_d(Q[d] * K[d]) is invariant to element ordering
    within the D dimension. This test constructs Q/K with both layouts and
    confirms the dot product matches exactly in fp32.
    """
    dev = torch.device("cpu")
    dtype = torch.float32
    B, H, T, D = 1, 2, 4, 16

    torch.manual_seed(11)
    q_interleaved = torch.randn(B, H, T, D, device=dev, dtype=dtype)

    dims = q_interleaved.shape[:-1]
    q_neox = q_interleaved.view(*dims, D // 2, 2).transpose(-1, -2).reshape(*dims, D).contiguous()
    k_interleaved = torch.randn_like(q_interleaved)
    k_neox = k_interleaved.view(*dims, D // 2, 2).transpose(-1, -2).reshape(*dims, D).contiguous()

    qkt_interleaved = q_interleaved @ k_interleaved.transpose(-1, -2)
    qkt_neox = q_neox @ k_neox.transpose(-1, -2)

    torch.testing.assert_close(qkt_neox, qkt_interleaved, atol=1e-5, rtol=1e-5)
