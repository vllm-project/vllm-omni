# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Test script for FlashAttention backend with padding handling.

This script tests two main scenarios:
1. Case 1: Comparing padded vs unpadded inputs for batch_size=1
2. Case 2: Comparing FlashAttention and SDPA backends for batch_size=2 with padding
"""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl
from vllm_omni.diffusion.attention.backends.utils import fa  # noqa: E402
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

is_gpu = current_omni_platform.is_cuda_alike() or current_omni_platform.is_xpu()
HAS_FLASH_ATTN = fa.HAS_FLASH_ATTN
flash_attn_func = fa.flash_attn_func  # noqa: N813


def create_attention_mask(batch_size: int, seq_len: int, valid_len: int, device: torch.device) -> torch.Tensor:
    """
    Create attention mask where first valid_len tokens are valid (1) and rest are padding (0).

    Args:
        batch_size: Batch size
        seq_len: Total sequence length (including padding)
        valid_len: Number of valid (non-padded) tokens

    Returns:
        Attention mask of shape (batch_size, seq_len)
    """
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    mask[:, :valid_len] = True
    return mask


def pad_tensor(tensor: torch.Tensor, target_seq_len: int, pad_value: float = 0.0) -> torch.Tensor:
    """
    Pad tensor along sequence dimension (dim=1).

    Args:
        tensor: Input tensor of shape (batch_size, seq_len, num_heads, head_dim)
        target_seq_len: Target sequence length after padding
        pad_value: Value to use for padding

    Returns:
        Padded tensor of shape (batch_size, target_seq_len, num_heads, head_dim)
    """
    batch_size, seq_len, num_heads, head_dim = tensor.shape
    if target_seq_len <= seq_len:
        return tensor

    padding = torch.full(
        (batch_size, target_seq_len - seq_len, num_heads, head_dim), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    return torch.cat([tensor, padding], dim=1)


@pytest.mark.skipif(not is_gpu, reason="FlashAttention requires CUDA or XPU")
def test_padding_equivalence():
    """
    Case 1: Test that padded and unpadded inputs produce similar outputs.

    - Input A: batch_size=1, hidden_states (1, 48), encoder_hidden_states (1, 16)
      Concatenated length: 64, NO attention_mask
    - Input B: Same data but padded: hidden_states (1, 58), encoder_hidden_states (1, 26)
      Concatenated length: 84, WITH attention_mask

    Expected: Output A and Output B should be very close.
    """
    device = torch.device(current_omni_platform.device_type)
    dtype = torch.bfloat16

    # Configuration
    batch_size = 1
    hidden_seq_len = 48
    encoder_seq_len = 16
    pad_length = 10
    num_heads = 8
    head_dim = 64

    # Initialize FlashAttention
    fa_impl = FlashAttentionImpl(
        num_heads=num_heads, head_size=head_dim, softmax_scale=1.0 / (head_dim**0.5), causal=False
    )

    # Create base tensors with random values (same for both A and B)
    torch.manual_seed(42)
    hidden_states_base = torch.randn(batch_size, hidden_seq_len, num_heads, head_dim, device=device, dtype=dtype)
    encoder_hidden_states_base = torch.randn(
        batch_size, encoder_seq_len, num_heads, head_dim, device=device, dtype=dtype
    )

    # ========== Input A: Unpadded, no attention mask ==========
    query_a = torch.cat([hidden_states_base, encoder_hidden_states_base], dim=1)
    key_a = query_a.clone()
    value_a = query_a.clone()

    attn_metadata_a = AttentionMetadata(attn_mask=None)

    output_a = fa_impl.forward(query=query_a, key=key_a, value=value_a, attn_metadata=attn_metadata_a)

    # ========== Input B: Padded with attention mask ==========
    hidden_states_padded = pad_tensor(hidden_states_base, hidden_seq_len + pad_length)
    encoder_hidden_states_padded = pad_tensor(encoder_hidden_states_base, encoder_seq_len + pad_length)

    query_b = torch.cat([hidden_states_padded, encoder_hidden_states_padded], dim=1)
    key_b = query_b.clone()
    value_b = query_b.clone()

    # Create attention mask
    attn_mask_b = torch.cat(
        [
            create_attention_mask(batch_size, hidden_seq_len + pad_length, hidden_seq_len, device),
            create_attention_mask(batch_size, encoder_seq_len + pad_length, encoder_seq_len, device),
        ],
        dim=1,
    )

    attn_metadata_b = AttentionMetadata(attn_mask=attn_mask_b)

    output_b = fa_impl.forward(query=query_b, key=key_b, value=value_b, attn_metadata=attn_metadata_b)

    # Extract non-padded portion from output_b
    output_b_unpadded = torch.cat(
        [
            output_b[:, :hidden_seq_len, :, :],
            output_b[:, hidden_seq_len + pad_length : hidden_seq_len + pad_length + encoder_seq_len, :, :],
        ],
        dim=1,
    )

    # Compare outputs
    max_diff = torch.max(torch.abs(output_a - output_b_unpadded)).item()
    mean_diff = torch.mean(torch.abs(output_a - output_b_unpadded)).item()

    print("\n=== Case 1: Padding Equivalence Test ===")
    print(f"Output A shape: {output_a.shape}")
    print(f"Output B shape: {output_b.shape}")
    print(f"Output B unpadded shape: {output_b_unpadded.shape}")
    print(f"Max absolute difference: {max_diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")

    # Assert that outputs are close
    # Using higher tolerance for bfloat16
    assert max_diff < 0.1, f"Max difference {max_diff} exceeds threshold 0.1"
    assert mean_diff < 0.01, f"Mean difference {mean_diff} exceeds threshold 0.01"

    print("✓ Case 1 PASSED: Padded and unpadded outputs are very close!")


@pytest.mark.skipif(not is_gpu, reason="FlashAttention requires CUDA or XPU")
def test_fa_vs_sdpa():
    """
    Case 2: Compare FlashAttention and SDPA backends with padding.

    - batch_size=2
    - hidden_states: (2, 48) padded to (2, 58)
    - encoder_hidden_states: (2, 16) padded to (2, 26)
    - Concatenated length: 84
    - Compare FA and SDPA outputs

    Expected: FA and SDPA outputs should be very close.
    """
    device = torch.device(current_omni_platform.device_type)
    dtype = torch.bfloat16

    # Configuration
    batch_size = 2
    hidden_seq_len = 48
    encoder_seq_len = 16
    pad_length = 10
    num_heads = 8
    head_dim = 64

    # Initialize both backends
    fa_impl = FlashAttentionImpl(
        num_heads=num_heads, head_size=head_dim, softmax_scale=1.0 / (head_dim**0.5), causal=False
    )

    sdpa_impl = SDPAImpl(num_heads=num_heads, head_size=head_dim, softmax_scale=1.0 / (head_dim**0.5), causal=False)

    # Create base tensors
    torch.manual_seed(123)
    hidden_states_base = torch.randn(batch_size, hidden_seq_len, num_heads, head_dim, device=device, dtype=dtype)
    encoder_hidden_states_base = torch.randn(
        batch_size, encoder_seq_len, num_heads, head_dim, device=device, dtype=dtype
    )

    # Pad tensors
    hidden_states_padded = pad_tensor(hidden_states_base, hidden_seq_len + pad_length)
    encoder_hidden_states_padded = pad_tensor(encoder_hidden_states_base, encoder_seq_len + pad_length)

    # Concatenate
    query = torch.cat([hidden_states_padded, encoder_hidden_states_padded], dim=1)
    key = query.clone()
    value = query.clone()

    # Create attention mask
    attn_mask = torch.cat(
        [
            create_attention_mask(batch_size, hidden_seq_len + pad_length, hidden_seq_len, device),
            create_attention_mask(batch_size, encoder_seq_len + pad_length, encoder_seq_len, device),
        ],
        dim=1,
    )

    attn_metadata = AttentionMetadata(attn_mask=attn_mask)

    # Run FlashAttention
    output_fa = fa_impl.forward(query=query.clone(), key=key.clone(), value=value.clone(), attn_metadata=attn_metadata)

    # Run SDPA
    # SDPA expects 4D attention mask: (batch_size, 1, seq_len, seq_len) or (batch_size, seq_len)
    # For causal=False, we need to convert 2D mask to 4D
    if attn_mask is not None:
        # Expand mask for SDPA: (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
        attn_mask_4d = attn_mask.unsqueeze(1).unsqueeze(2)
        # Convert bool to float: True -> 0.0, False -> -inf
        attn_mask_float = torch.zeros_like(attn_mask_4d, dtype=dtype)
        attn_mask_float.masked_fill_(~attn_mask_4d, float("-inf"))
        attn_metadata_sdpa = AttentionMetadata(attn_mask=attn_mask_float)
    else:
        attn_metadata_sdpa = AttentionMetadata(attn_mask=None)

    output_sdpa = sdpa_impl.forward(
        query=query.clone(), key=key.clone(), value=value.clone(), attn_metadata=attn_metadata_sdpa
    )

    # Compare outputs (only compare valid regions)
    output_fa_valid = torch.cat(
        [
            output_fa[:, :hidden_seq_len, :, :],
            output_fa[:, hidden_seq_len + pad_length : hidden_seq_len + pad_length + encoder_seq_len, :, :],
        ],
        dim=1,
    )
    output_sdpa_valid = torch.cat(
        [
            output_sdpa[:, :hidden_seq_len, :, :],
            output_sdpa[:, hidden_seq_len + pad_length : hidden_seq_len + pad_length + encoder_seq_len, :, :],
        ],
        dim=1,
    )

    max_diff = torch.max(torch.abs(output_fa_valid - output_sdpa_valid)).item()
    mean_diff = torch.mean(torch.abs(output_fa_valid - output_sdpa_valid)).item()

    print("\n=== Case 2: FA vs SDPA Comparison ===")
    print(f"Batch size: {batch_size}")
    print(f"FA output shape: {output_fa.shape}")
    print(f"SDPA output shape: {output_sdpa.shape}")
    print(f"Max absolute difference (valid region): {max_diff:.6f}")
    print(f"Mean absolute difference (valid region): {mean_diff:.6f}")

    # Assert that outputs are close
    # Using higher tolerance for bfloat16 and different implementations
    assert max_diff < 0.01, f"Max difference {max_diff} exceeds threshold 0.01"
    assert mean_diff < 0.001, f"Mean difference {mean_diff} exceeds threshold 0.001"

    print("✓ Case 2 PASSED: FA and SDPA outputs are very close!")


@pytest.mark.skipif(not is_gpu, reason="FlashAttention requires CUDA or XPU")
def test_flash_attn_func_preferred_over_varlen():
    """Test flash_attn_func availability and basic forward call."""
    if not HAS_FLASH_ATTN:
        pytest.skip("No Flash Attention available")

    if flash_attn_func is None:
        pytest.skip("flash_attn_func not available, will use varlen fallback")

    device = torch.device(current_omni_platform.device_type)
    dtype = torch.bfloat16

    num_heads, head_dim = 8, 64
    fa_impl = FlashAttentionImpl(
        num_heads=num_heads, head_size=head_dim, softmax_scale=1.0 / (head_dim**0.5), causal=False
    )

    torch.manual_seed(42)
    q = torch.randn(1, 32, num_heads, head_dim, device=device, dtype=dtype)
    k = q.clone()
    v = q.clone()

    attn_metadata = AttentionMetadata(attn_mask=None)
    output = fa_impl.forward(q, k, v, attn_metadata)

    assert output.shape == q.shape
    assert not torch.isnan(output).any()
    print("✓ flash_attn_func forward works correctly!")


def test_piecewise_flash_attn_uses_varlen_fallback(monkeypatch):
    calls = []

    def fake_varlen_func(q, k, v, **kwargs):
        assert q.shape[1] == 4
        assert k.shape[1] == 2
        assert v.shape[1] == 2
        calls.append(kwargs)
        return q

    monkeypatch.setattr(fa, "flash_attn_func", None)
    monkeypatch.setattr(fa, "flash_attn_varlen_func", fake_varlen_func)
    monkeypatch.setattr(fa, "HAS_FLASH_ATTN", True)

    impl = FlashAttentionImpl(num_heads=4, num_kv_heads=2, head_size=4, softmax_scale=0.5, causal=False)
    query = torch.randn(1, 3, 4, 4)
    key = torch.randn(1, 3, 2, 4)
    value = torch.randn(1, 3, 2, 4)
    metadata = AttentionMetadata(full_attn_spans=[[(0, 3)]])

    output = impl.forward_cuda(query, key, value, metadata)

    assert output.shape == query.shape
    assert len(calls) == 1
    assert calls[0]["max_seqlen_q"] == 3
    assert calls[0]["max_seqlen_k"] == 3
    assert calls[0]["causal"] is False
    assert calls[0]["softmax_scale"] == 0.5


def test_packed_varlen_metadata_bypasses_mask_unpadding(monkeypatch):
    calls = []

    def fake_varlen_func(q, k, v, **kwargs):
        calls.append((q.shape, k.shape, v.shape, kwargs))
        return q

    monkeypatch.setattr(fa, "HAS_FLASH_ATTN", True)
    monkeypatch.setattr(fa, "flash_attn_varlen_func", fake_varlen_func)

    impl = FlashAttentionImpl(num_heads=2, head_size=4, softmax_scale=0.5, causal=False)
    query = torch.randn(1, 8, 2, 4)
    cu_seqlens = torch.tensor([0, 6, 8], dtype=torch.int32)
    metadata = AttentionMetadata(
        attn_mask=torch.tensor([[True] * 6 + [False] * 2]),
        extra={
            "cu_seqlens_q": cu_seqlens,
            "cu_seqlens_k": cu_seqlens,
            "max_seqlen_q": 6,
            "max_seqlen_k": 6,
        },
    )

    output = impl.forward_cuda(query, query, query, metadata)

    assert torch.equal(output, query)
    assert len(calls) == 1
    assert calls[0][0] == torch.Size([8, 2, 4])
    assert calls[0][3]["cu_seqlens_q"] is cu_seqlens
    assert calls[0][3]["max_seqlen_q"] == 6


def test_packed_varlen_metadata_must_be_complete(monkeypatch):
    monkeypatch.setattr(fa, "HAS_FLASH_ATTN", True)
    impl = FlashAttentionImpl(num_heads=2, head_size=4, softmax_scale=0.5, causal=False)
    query = torch.randn(1, 8, 2, 4)
    metadata = AttentionMetadata(extra={"cu_seqlens_q": torch.tensor([0, 8], dtype=torch.int32)})

    with pytest.raises(ValueError, match="Incomplete packed FlashAttention metadata"):
        impl.forward_cuda(query, query, query, metadata)


# ---------------------------------------------------------------------------
# NPU mask-free packed attention (PR #5891).
#
# These tests cover the backend-side NPU branches that ``forward_fa_npu`` adds:
#   - boundary resolution in ``_resolve_packed_seq_npu`` (pure Python, no device
#     sync, no MindIE-SD import);
#   - env dispatch in ``forward_fa_npu`` (which op a packed forward takes);
#   - the laser input pre-scaling math in ``_forward_prefix_kv_slice_npu``.
#
# They run on CPU without a real NPU by injecting a fake ``mindiesd`` module
# into ``sys.modules`` (same pattern as ``test_paged_attention`` /
# ``benchmarks/conftest``). No production code is exercised for real kernels.


def _npu_impl(causal: bool = False) -> FlashAttentionImpl:
    return FlashAttentionImpl(num_heads=2, head_size=4, softmax_scale=0.5, causal=causal)


def _fake_mindiesd(monkeypatch, *, attention_forward=None, attention_forward_varlen=None):
    """Install a stub ``mindiesd`` so local ``from mindiesd import ...`` works.

    Restored automatically by monkeypatch. Defaults are no-op callables so the
    top-level import in ``forward_fa_npu`` succeeds even when the test mocks the
    instance methods that would otherwise call them.
    """
    fake = SimpleNamespace(
        attention_forward=attention_forward or Mock(return_value=torch.zeros(1)),
        attention_forward_varlen=attention_forward_varlen or Mock(return_value=torch.zeros(1)),
    )
    monkeypatch.setitem(sys.modules, "mindiesd", fake)
    return fake


# --- Test group A: boundary resolution (_resolve_packed_seq_npu) -------------


def test_resolve_packed_seq_accepts_real_plus_pad_contract():
    impl = _npu_impl()
    q = torch.randn(1, 8, 2, 4)
    cu = torch.tensor([0, 5, 8], dtype=torch.int32)
    extra = {"cu_seqlens_q": cu, "cu_seqlens_k": cu, "max_seqlen_q": 5, "max_seqlen_k": 5}

    assert impl._resolve_packed_seq_npu(q, q, extra) == ([5, 8], [5, 8])


def test_resolve_packed_seq_single_document_no_padding():
    impl = _npu_impl()
    q = torch.randn(1, 8, 2, 4)
    cu = torch.tensor([0, 8], dtype=torch.int32)
    extra = {"cu_seqlens_q": cu, "cu_seqlens_k": cu, "max_seqlen_q": 8, "max_seqlen_k": 8}

    assert impl._resolve_packed_seq_npu(q, q, extra) == ([8], [8])


@pytest.mark.parametrize(
    "case",
    [
        "causal",
        "missing_key",
        "batch_ne_1",
        "cu_too_long",
        "cu_qk_mismatch",
        "max_seqlen_not_int",
        "used_gt_total",
        "used_zero",
        "pad_longer_than_real",
    ],
)
def test_resolve_packed_seq_rejects_malformed_metadata(case):
    """Any metadata outside the exact [real, pad] contract returns None so the
    caller falls back to the masked path."""
    cu3 = torch.tensor([0, 5, 8], dtype=torch.int32)
    cu2 = torch.tensor([0, 8], dtype=torch.int32)
    cu4 = torch.tensor([0, 3, 5, 8], dtype=torch.int32)
    cu_short = torch.tensor([0, 3, 8], dtype=torch.int32)
    base = {"cu_seqlens_q": cu3, "cu_seqlens_k": cu3, "max_seqlen_q": 5, "max_seqlen_k": 5}
    q = torch.randn(1, 8, 2, 4)

    impl = _npu_impl(causal=(case == "causal"))
    if case == "missing_key":
        extra = {"cu_seqlens_q": cu3, "max_seqlen_q": 5}  # 3 of 4 keys absent
    elif case == "batch_ne_1":
        q = torch.randn(2, 8, 2, 4)
        extra = base
    elif case == "cu_too_long":
        extra = {**base, "cu_seqlens_q": cu4, "cu_seqlens_k": cu4}
    elif case == "cu_qk_mismatch":
        extra = {**base, "cu_seqlens_q": cu3, "cu_seqlens_k": cu2}
    elif case == "max_seqlen_not_int":
        extra = {**base, "max_seqlen_q": 5.0}
    elif case == "used_gt_total":
        extra = {**base, "max_seqlen_q": 9}  # 9 > total length 8
    elif case == "used_zero":
        extra = {**base, "max_seqlen_q": 0, "max_seqlen_k": 0}
    elif case == "pad_longer_than_real":
        # real=3, pad=5: the real document must be the longer one.
        extra = {
            "cu_seqlens_q": cu_short,
            "cu_seqlens_k": cu_short,
            "max_seqlen_q": 3,
            "max_seqlen_k": 3,
        }
    else:
        extra = base

    assert impl._resolve_packed_seq_npu(q, q, extra) is None


# --- Test group B: env dispatch in forward_fa_npu (current behavior) --------


def _npu_impl_with_mocked_paths() -> tuple[FlashAttentionImpl, torch.Tensor]:
    """FlashAttentionImpl whose two packed paths are Mocks returning a sentinel.

    Lets a test assert which branch ``forward_fa_npu`` dispatched to without
    running any real kernel.
    """
    impl = _npu_impl()
    sentinel = torch.tensor([1.0])
    impl._forward_varlen_packed_npu = Mock(return_value=sentinel)
    impl._forward_prefix_kv_slice_npu = Mock(return_value=sentinel)
    return impl, sentinel


def test_npu_env_unset_uses_varlen(monkeypatch):
    monkeypatch.delenv("MINDIE_SD_FA_TYPE", raising=False)
    _fake_mindiesd(monkeypatch)
    impl, _ = _npu_impl_with_mocked_paths()
    metadata = AttentionMetadata(extra={"npu_attn_varlen": True})

    impl.forward_fa_npu(torch.randn(1, 8, 2, 4), *[torch.randn(1, 8, 2, 4)] * 2, metadata)

    impl._forward_varlen_packed_npu.assert_called_once()
    impl._forward_prefix_kv_slice_npu.assert_not_called()


def test_npu_env_laser_uses_slice(monkeypatch):
    monkeypatch.setenv("MINDIE_SD_FA_TYPE", "ascend_laser_attention")
    _fake_mindiesd(monkeypatch)
    impl, _ = _npu_impl_with_mocked_paths()
    metadata = AttentionMetadata(extra={"npu_attn_varlen": True})

    impl.forward_fa_npu(torch.randn(1, 8, 2, 4), *[torch.randn(1, 8, 2, 4)] * 2, metadata)

    impl._forward_prefix_kv_slice_npu.assert_called_once()
    impl._forward_varlen_packed_npu.assert_not_called()


@pytest.mark.parametrize("fa_type", ["prompt_flash_attn", "fused_attn_score", "ascend_laser_attenton", "garbage"])
def test_npu_env_non_laser_value_falls_back_to_varlen(monkeypatch, fa_type):
    """Locks in the CURRENT dispatch behavior (PR #5891).

    Any value other than ``ascend_laser_attention`` — including the valid
    ``prompt_flash_attn``/``fused_attn_score`` and typos — is silently routed to
    the TND varlen op, which cannot honor ``MINDIE_SD_FA_TYPE``. This is review
    comment #1 on the PR and is intentionally captured here as a regression
    net: update this test when the dispatch starts honoring/rejecting these.
    """
    monkeypatch.setenv("MINDIE_SD_FA_TYPE", fa_type)
    _fake_mindiesd(monkeypatch)
    impl, _ = _npu_impl_with_mocked_paths()
    metadata = AttentionMetadata(extra={"npu_attn_varlen": True})

    impl.forward_fa_npu(torch.randn(1, 8, 2, 4), *[torch.randn(1, 8, 2, 4)] * 2, metadata)

    impl._forward_varlen_packed_npu.assert_called_once()
    impl._forward_prefix_kv_slice_npu.assert_not_called()


def test_npu_varlen_opt_in_unset_takes_mask_path(monkeypatch):
    """Models that do not set ``npu_attn_varlen`` (e.g. Wan/Cosmos) keep using
    the existing masked attention_forward path; the new branches are skipped."""
    fake_forward = Mock(return_value=torch.zeros(1, 8, 2, 4))
    _fake_mindiesd(monkeypatch, attention_forward=fake_forward)
    impl, _ = _npu_impl_with_mocked_paths()
    metadata = AttentionMetadata(
        attn_mask=torch.tensor([[True, True, True, True, True, False, False, False]]),
        extra={},  # no npu_attn_varlen opt-in
    )
    q = torch.randn(1, 8, 2, 4)

    out = impl.forward_fa_npu(q, q, q, metadata)

    impl._forward_varlen_packed_npu.assert_not_called()
    impl._forward_prefix_kv_slice_npu.assert_not_called()
    fake_forward.assert_called_once()
    assert out is fake_forward.return_value


# --- Test group C: laser input pre-scaling in _forward_prefix_kv_slice_npu --


def test_prefix_kv_slice_applies_laser_input_scaling(monkeypatch):
    monkeypatch.setenv("MINDIE_SD_FA_TYPE", "ascend_laser_attention")
    captured: dict = {}

    def fake_attention_forward(query, key, value, **kwargs):
        captured["query"] = query
        captured["key"] = key
        captured["value"] = value
        captured.update(kwargs)
        return torch.ones_like(query)  # deterministic stand-in kernel output

    _fake_mindiesd(monkeypatch, attention_forward=fake_attention_forward)
    impl = _npu_impl()  # softmax_scale == 0.5
    q = torch.randn(1, 8, 2, 4)
    cu = torch.tensor([0, 5, 8], dtype=torch.int32)
    extra = {
        "cu_seqlens_q": cu,
        "cu_seqlens_k": cu,
        "max_seqlen_q": 5,
        "max_seqlen_k": 5,
        "laser_input_scale": 256.0,
    }

    out = impl._forward_prefix_kv_slice_npu(q, q, q, extra)

    # K/V sliced to the valid prefix (used_k == 5); q keeps full length.
    assert captured["key"].shape == (1, 5, 2, 4)
    assert captured["value"].shape == (1, 5, 2, 4)
    assert captured["query"].shape == (1, 8, 2, 4)
    assert captured["attn_mask"] is None
    # Pre-divide q/k/v by the factor and compensate the kernel scale by its
    # square (exact for power-of-two); output scaled back by the factor.
    torch.testing.assert_close(captured["query"], q / 256.0)
    torch.testing.assert_close(captured["key"], q[:, :5] / 256.0)
    assert captured["scale"] == pytest.approx(0.5 * 256.0**2)
    torch.testing.assert_close(out, torch.ones(1, 8, 2, 4) * 256.0)


def test_prefix_kv_slice_no_scaling_without_factor(monkeypatch):
    monkeypatch.setenv("MINDIE_SD_FA_TYPE", "ascend_laser_attention")
    captured: dict = {}

    def fake_attention_forward(query, key, value, **kwargs):
        captured["query"] = query
        captured.update(kwargs)
        return torch.ones_like(query)

    _fake_mindiesd(monkeypatch, attention_forward=fake_attention_forward)
    impl = _npu_impl()
    q = torch.randn(1, 8, 2, 4)
    cu = torch.tensor([0, 5, 8], dtype=torch.int32)
    extra = {
        "cu_seqlens_q": cu,
        "cu_seqlens_k": cu,
        "max_seqlen_q": 5,
        "max_seqlen_k": 5,
        # no laser_input_scale → no pre-scaling
    }

    out = impl._forward_prefix_kv_slice_npu(q, q, q, extra)

    torch.testing.assert_close(captured["query"], q)
    assert captured["scale"] == pytest.approx(0.5)
    torch.testing.assert_close(out, torch.ones(1, 8, 2, 4))


if __name__ == "__main__":
    print("Running FlashAttention Padding Tests...")
    print("=" * 60)

    # Try to run tests
    if is_gpu:
        try:
            print("\n[Running Case 1: Padding Equivalence for FA]")
            test_padding_equivalence()
        except Exception as e:
            print(f"✗ Case 1 failed: {e}")
            import traceback

            traceback.print_exc()

        try:
            print("\n[Running Case 2: FA vs SDPA]")
            test_fa_vs_sdpa()
        except Exception as e:
            print(f"✗ Case 2 failed: {e}")
            import traceback

            traceback.print_exc()
    else:
        raise RuntimeError("CUDA is not available")
    print("\n" + "=" * 60)
    print("Test suite completed!")
