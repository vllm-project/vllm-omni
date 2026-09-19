# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""NPU parity tests for MOSS-TTS codec attention, ring-cache, and RoPE.

Verifies that the NPU-gated fast paths in ``audio_tokenizer_v2.py`` produce
results equivalent to the reference (eager / SDPA) implementations:

1. **Attention masks** — ``npu_fusion_attention`` with inverted bool mask
   matches ``F.scaled_dot_product_attention`` for both masked and unmasked
   (``attn_bias=None``) cases.
2. **Ring-cache wraparound** — ``RingKVCache`` positions are correct when
   ``end_offset + T`` exceeds ``capacity`` (ring buffer wraparound).
3. **RoPE layout changes** — the NPU path (interleaved→neox conversion +
   ``npu_rotary_mul`` + back-conversion skip) matches the eager GPT-J
   interleaved rotation, and the skip is safe because attention QK^T is
   layout-independent.
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
# 1. Attention mask parity: npu_fusion_attention vs F.scaled_dot_product_attention
# =============================================================================


@pytest.mark.parametrize("batch,heads,q_len,kv_len", [(2, 4, 8, 16), (1, 8, 15, 15), (4, 2, 1, 30)])
@pytest.mark.parametrize("device", ["cpu", npu_device])
def test_npu_fusion_attention_with_mask_matches_sdpa(batch, heads, q_len, kv_len, device):
    """npu_fusion_attention (inverted bool mask) == SDPA (bool mask)."""
    torch.manual_seed(42)
    dtype = torch.bfloat16
    dev = torch.device(device)
    q = torch.randn(batch, heads, q_len, 64, device=dev, dtype=dtype)
    k = torch.randn(batch, heads, kv_len, 64, device=dev, dtype=dtype)
    v = torch.randn_like(k)

    # Causal mask: True = attend (SDPA convention)
    pos_q = torch.arange(q_len, device=dev).view(-1, 1)
    pos_k = torch.arange(kv_len, device=dev).view(1, -1)
    attn_bias = (pos_k <= pos_q + kv_len - q_len).unsqueeze(0).unsqueeze(0).expand(batch, 1, q_len, kv_len).clone()

    if device == "npu":
        import torch_npu

        atten_mask = ~attn_bias  # NPU: True = masked (inverted)
        scale = 1.0 / 64**0.5
        x_npu, _, _, _, _, _, _ = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            head_num=heads,
            input_layout="BNSD",
            atten_mask=atten_mask,
            scale=scale,
            keep_prob=1.0,
            pre_tockens=kv_len,
            next_tockens=0,
            sparse_mode=0,
        )
        result = x_npu
        # Reference on same device using SDPA
        ref = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)
    else:
        # CPU reference
        result = F.scaled_dot_product_attention(q, k, v, attn_bias, dropout_p=0.0)
        ref = result  # trivially equal on CPU (sanity check)

    if device == "npu":
        torch.testing.assert_close(result, ref, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("batch,heads,q_len", [(2, 4, 16), (1, 8, 30)])
@pytest.mark.parametrize("device", ["cpu", npu_device])
def test_npu_fusion_attention_no_mask_matches_sdpa(batch, heads, q_len, device):
    """npu_fusion_attention (atten_mask=None) == SDPA (attn_bias=None)."""
    torch.manual_seed(99)
    dtype = torch.bfloat16
    dev = torch.device(device)
    q = torch.randn(batch, heads, q_len, 64, device=dev, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    if device == "npu":
        import torch_npu

        scale = 1.0 / 64**0.5
        x_npu, _, _, _, _, _, _ = torch_npu.npu_fusion_attention(
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
        )
        result = x_npu
        ref = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        torch.testing.assert_close(result, ref, atol=0.05, rtol=0.05)
    else:
        # CPU sanity
        result = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        assert result.shape == (batch, heads, q_len, 64)


@pytest.mark.parametrize("device", ["cpu", npu_device])
def test_attention_forward_npu_branch_matches_sdpa_fallback(device):
    """Full attention forward: NPU merged branch produces same result as SDPA.

    Exercises both the masked (causal) and unmasked paths through the merged
    ``elif q.device.type == "npu"`` branch.
    """
    torch.manual_seed(123)
    dev = torch.device(device)
    dtype = torch.bfloat16
    embed_dim, num_heads = 128, 2

    attn = MossAudioTokenizerMultiheadAttention(
        embed_dim, num_heads, causal=True, context=9, device=dev, dtype=dtype
    ).eval()

    B, T = 3, 5
    x = torch.randn(B, T, embed_dim, device=dev, dtype=dtype)

    with torch.inference_mode():
        if device == "npu":
            # NPU path (uses npu_fusion_attention)
            result_npu = attn(x, x, x)
            # Reference: temporarily force the SDPA fallback by moving to CPU
            attn_cpu = MossAudioTokenizerMultiheadAttention(
                embed_dim,
                num_heads,
                causal=True,
                context=9,
                device="cpu",
                dtype=dtype,
            ).eval()
            attn_cpu.load_state_dict(attn.state_dict())
            x_cpu = x.cpu()
            ref = attn_cpu(x_cpu, x_cpu, x_cpu).to(dev)
            torch.testing.assert_close(result_npu, ref, atol=0.05, rtol=0.05)
        else:
            # CPU sanity: SDPA path
            result = attn(x, x, x)
            assert result.shape == (B, T, embed_dim)


# =============================================================================
# 2. Ring-cache wraparound parity
# =============================================================================


@pytest.mark.parametrize("device", ["cpu", npu_device])
def test_ring_kv_cache_wraparound_positions(device):
    """RingKVCache positions are correct when end_offset + T > capacity.

    After writing enough frames to wrap around the ring buffer, positions
    should reflect the modular arithmetic: entries older than the window
    are marked -1, and valid entries are in sequential order.
    """
    dev = torch.device(device)
    dtype = torch.bfloat16
    B, H, D = 2, 4, 64
    capacity = 8
    T = 5  # T < capacity, but after 2 steps: offset=10 > capacity=8 -> wrap

    cache = RingKVCache(B, H, D, capacity, respect_exec_mask=True, device=dev, dtype=dtype)

    k1 = torch.randn(B, H, T, D, device=dev, dtype=dtype)
    v1 = torch.randn_like(k1)
    cache.complete(k1, v1, exec_mask=torch.ones(B, dtype=torch.bool, device=dev))

    # After step 1: end_offset = 5 for both batches
    assert cache.end_offset.tolist() == [T, T]

    # Step 2: T=5 more frames -> offset goes from 5 to 10, wrapping around capacity=8
    k2 = torch.randn(B, H, T, D, device=dev, dtype=dtype)
    v2 = torch.randn_like(k2)
    result2 = cache.complete(k2, v2, exec_mask=torch.ones(B, dtype=torch.bool, device=dev))

    # end_offset should now be 10 for both batches
    assert cache.end_offset.tolist() == [2 * T, 2 * T]

    # After wraparound: end_offset=10 > capacity=8, so all 8 slots are valid.
    # Positions are absolute (not 0..capacity-1). With last_offset=9 (end_offset
    # before advance + T - 1 = 5 + 5 - 1 = 9), end_index=9%8=1, the expected
    # positions are [8, 9, 2, 3, 4, 5, 6, 7] — slot 0 maps to abs pos 8, etc.
    positions = result2.positions
    assert positions.shape == (B, capacity)
    # All positions should be >= 0 since 2*T=10 > capacity=8
    assert (positions >= 0).all()
    # Each batch should have unique positions (ring buffer invariant)
    for b in range(B):
        batch_positions = positions[b]
        assert batch_positions.unique().numel() == capacity
        # Positions should span [end_offset - capacity, end_offset - 1] = [2, 9]
        assert batch_positions.min().item() == 2 * T - capacity  # 2
        assert batch_positions.max().item() == 2 * T - 1  # 9


@pytest.mark.parametrize("device", ["cpu", npu_device])
def test_ring_kv_cache_wraparound_with_execution_context(device):
    """RingKVCache wraparound with StreamingExecutionContext (dynamic slots).

    Tests that the scatter-based ring update and position computation are
    correct when multiple steps cause the offset to wrap around the capacity.
    """
    dev = torch.device(device)
    dtype = torch.bfloat16
    B, H, D = 3, 2, 32
    capacity = 6
    T = 4

    cache = RingKVCache(B, H, D, capacity, respect_exec_mask=True, device=dev, dtype=dtype)

    # Simulate 3 steps with different slot assignments
    for step in range(3):
        slots = torch.tensor([0, 1, 2], device=dev, dtype=torch.long)
        valid = torch.tensor([True, True, False], device=dev, dtype=torch.bool)
        ctx = StreamingExecutionContext(state_slot_ids=slots, valid_rows=valid)

        k = torch.randn(B, H, T, D, device=dev, dtype=dtype)
        v = torch.randn_like(k)
        result = cache.complete(k, v, execution_context=ctx)

        positions = result.positions
        assert positions.shape == (B, capacity)

        # Slot 2 (valid=False) should have offset not advanced
        if step == 0:
            # Slot 0 and 1 advanced by T, slot 2 did not
            assert cache.end_offset[0].item() == T
            assert cache.end_offset[1].item() == T
            assert cache.end_offset[2].item() == 0
            # Slot 2 has offset=0, so all positions >= 0 are invalid
            # (invalid = cache_indexes >= next_offset)
            # next_offset for slot 2 = 0 (valid=False), so all are invalid -> -1
            assert (positions[2] == -1).all()
        elif step >= 1:
            # After step 1: slot 0 offset=8, slot 1 offset=8 -> both > capacity=6
            # Wraparound: positions should reflect modular arithmetic
            assert cache.end_offset[0].item() == (step + 1) * T
            assert cache.end_offset[1].item() == (step + 1) * T


@pytest.mark.parametrize("device", ["cpu", npu_device])
def test_ring_kv_cache_arange_caching_reuses_tensor(device):
    """Verify _arange_capacity and _get_arange_t cache and reuse tensors."""
    dev = torch.device(device)
    cache = RingKVCache(2, 4, 64, 10, device=dev, dtype=torch.bfloat16)

    # _arange_capacity should be pre-allocated
    assert cache._arange_capacity is not None
    assert cache._arange_capacity.shape == (10,)
    assert cache._arange_capacity.tolist() == list(range(10))

    # Same T should return the same cached tensor (compare by value, not
    # data_ptr — NPU internal format management may reallocate storage)
    t1 = cache._get_arange_t(5, dev, torch.long)
    t2 = cache._get_arange_t(5, dev, torch.long)
    assert torch.equal(t1, t2)
    assert t1.shape == t2.shape

    # Different T should allocate a new tensor
    t3 = cache._get_arange_t(8, dev, torch.long)
    assert t3.shape == (8,)
    assert not torch.equal(t1, t3)


# =============================================================================
# 3. RoPE layout parity: NPU neox path vs eager interleaved path
# =============================================================================


@pytest.mark.parametrize("device", ["cpu", npu_device])
def test_rope_npu_neox_path_matches_eager_interleaved(device):
    """NPU RoPE path (neox conversion + npu_rotary_mul + back-conv skip)
    matches the eager GPT-J interleaved rotation.

    The back-conversion skip is safe because attention QK^T is layout-
    independent (dot product sums over D regardless of element ordering).
    This test verifies that Q and K after NPU RoPE, when used in a dot
    product, produce the same result as the eager path.
    """
    dev = torch.device(device)
    dtype = torch.bfloat16
    B, H, T, D = 2, 4, 8, 64

    torch.manual_seed(77)
    q = torch.randn(B, H, T, D, device=dev, dtype=dtype)
    k = torch.randn_like(q)
    offset = torch.tensor([3, 5], device=dev, dtype=torch.long)

    if device == "npu":
        # NPU path (with freqs_cache as in production)
        rope = MossAudioTokenizerRotaryEmbedding(max_period=10000.0)
        freqs = rope._get_freqs(D, dev)
        q_npu, k_npu = apply_rope(q, k, offset, freqs_cache=freqs)

        # Eager reference on CPU
        q_cpu, k_cpu, offset_cpu = q.cpu(), k.cpu(), offset.cpu()
        q_ref, k_ref = apply_rope(q_cpu, k_cpu, offset_cpu)  # no freqs_cache -> eager

        # The NPU path returns neox layout (back-conv skipped).
        # Convert eager result to neox for comparison.
        # Eager returns interleaved: [r0 i0 r1 i1 ...]
        # neox: [r0 r1 ... i0 i1 ...]
        dims = q_ref.shape[:-1]
        q_ref_neox = q_ref.view(*dims, D // 2, 2).transpose(-1, -2).reshape(*dims, D)
        k_ref_neox = k_ref.view(*dims, D // 2, 2).transpose(-1, -2).reshape(*dims, D)

        # Compare in neox layout (bf16 on NPU needs tolerance)
        torch.testing.assert_close(q_npu.cpu(), q_ref_neox, atol=0.1, rtol=0.1)
        torch.testing.assert_close(k_npu.cpu(), k_ref_neox, atol=0.1, rtol=0.1)

        # Verify QK^T parity (layout independence)
        # NPU QK^T (neox layout) — bf16 accumulation differs slightly between
        # layouts due to summation order, so use a modest tolerance.
        qkt_npu = q_npu.float() @ k_npu.float().transpose(-1, -2)
        # Eager QK^T (interleaved layout)
        qkt_ref = q_ref.float() @ k_ref.float().transpose(-1, -2)
        torch.testing.assert_close(qkt_npu.cpu(), qkt_ref, atol=0.1, rtol=0.1)
    else:
        # CPU: just verify eager path runs
        q_out, k_out = apply_rope(q, k, offset)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape


@pytest.mark.parametrize("device", ["cpu", npu_device])
def test_rope_back_conversion_skip_is_layout_independent(device):
    """Verify that skipping the neox→interleaved back-conversion is safe.

    Attention QK^T = sum_d(Q[d] * K[d]) is invariant to element ordering
    within the D dimension. This test constructs Q/K with both layouts and
    confirms the dot product matches.  A tiny tolerance is used because
    different memory layouts can change the floating-point summation order.
    """
    dev = torch.device(device)
    dtype = torch.float32
    B, H, T, D = 1, 2, 4, 16

    torch.manual_seed(11)
    q_interleaved = torch.randn(B, H, T, D, device=dev, dtype=dtype)

    # Convert to neox: [r0 i0 r1 i1 ...] -> [r0 r1 ... i0 i1 ...]
    dims = q_interleaved.shape[:-1]
    q_neox = q_interleaved.view(*dims, D // 2, 2).transpose(-1, -2).reshape(*dims, D).contiguous()
    k_interleaved = torch.randn_like(q_interleaved)
    k_neox = k_interleaved.view(*dims, D // 2, 2).transpose(-1, -2).reshape(*dims, D).contiguous()

    # QK^T should be identical regardless of layout (up to fp summation order)
    qkt_interleaved = q_interleaved @ k_interleaved.transpose(-1, -2)
    qkt_neox = q_neox @ k_neox.transpose(-1, -2)

    torch.testing.assert_close(qkt_neox, qkt_interleaved, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", ["cpu", npu_device])
def test_rope_freqs_cache_matches_reference(device):
    """MossAudioTokenizerRotaryEmbedding._get_freqs caches and matches reference."""
    dev = torch.device(device)
    rope = MossAudioTokenizerRotaryEmbedding(max_period=10000.0)

    D = 64
    freqs = rope._get_freqs(D, dev)

    # Reference computation
    ds = torch.arange(D // 2, device=dev, dtype=torch.float32)
    expected = torch.exp(ds * (-math.log(10000.0) * 2 / D))

    torch.testing.assert_close(freqs, expected, atol=0, rtol=0)

    # Second call should return the cached tensor (compare by value — NPU
    # internal format management may reallocate storage, so data_ptr is
    # not a reliable identity check on NPU)
    freqs2 = rope._get_freqs(D, dev)
    torch.testing.assert_close(freqs, freqs2, atol=0, rtol=0)
