# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

import vllm_omni.diffusion.attention.backends.sdpa as sdpa_backend
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl
from vllm_omni.diffusion.attention.backends.utils.attn_runtime_selector import can_sdpa_use_fused_gqa

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeNPUTensor:
    device = torch.device("npu")
    is_cuda = False


def test_sdpa_native_gqa_is_available_on_npu():
    query = key = value = _FakeNPUTensor()

    assert can_sdpa_use_fused_gqa(query, key, value, None, False)


def test_sdpa_rejects_invalid_gqa_head_ratio_before_native_dispatch(monkeypatch):
    monkeypatch.setattr(sdpa_backend, "can_sdpa_use_fused_gqa", lambda *args: True)

    impl = SDPAImpl(num_heads=3, num_kv_heads=2, head_size=8, softmax_scale=0.5)
    with pytest.raises(ValueError, match="q_heads=3 and kv_heads=2"):
        impl.forward_npu(
            torch.randn(1, 3, 3, 8),
            torch.randn(1, 3, 2, 8),
            torch.randn(1, 3, 2, 8),
        )


def test_sdpa_expands_kv_when_native_gqa_kernel_is_unavailable(monkeypatch):
    calls = []

    def fake_sdpa(query, key, value, **kwargs):
        calls.append((query.shape, key.shape, value.shape, kwargs))
        return query

    monkeypatch.setattr(sdpa_backend, "can_sdpa_use_fused_gqa", lambda *args: False)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)

    impl = SDPAImpl(num_heads=4, num_kv_heads=2, head_size=8, softmax_scale=0.5)
    output = impl.forward_cuda(
        torch.randn(1, 3, 4, 8),
        torch.randn(1, 3, 2, 8),
        torch.randn(1, 3, 2, 8),
    )

    query_shape, key_shape, value_shape, kwargs = calls[0]
    assert query_shape == key_shape == value_shape == (1, 4, 3, 8)
    assert kwargs["enable_gqa"] is False
    assert output.shape == (1, 3, 4, 8)


def test_sdpa_keeps_compressed_kv_when_native_gqa_kernel_is_available(monkeypatch):
    calls = []

    def fake_sdpa(query, key, value, **kwargs):
        calls.append((query.shape, key.shape, value.shape, kwargs))
        return query

    monkeypatch.setattr(sdpa_backend, "can_sdpa_use_fused_gqa", lambda *args: True)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)

    impl = SDPAImpl(num_heads=4, num_kv_heads=2, head_size=8, softmax_scale=0.5)
    output = impl.forward_cuda(
        torch.randn(1, 3, 4, 8),
        torch.randn(1, 3, 2, 8),
        torch.randn(1, 3, 2, 8),
    )

    query_shape, key_shape, value_shape, kwargs = calls[0]
    assert query_shape == (1, 4, 3, 8)
    assert key_shape == value_shape == (1, 2, 3, 8)
    assert kwargs["enable_gqa"] is True
    assert output.shape == (1, 3, 4, 8)


def test_sdpa_native_gqa_matches_explicit_kv_expansion(monkeypatch):
    torch.manual_seed(0)
    query = torch.randn(2, 5, 4, 8)
    key = torch.randn(2, 5, 2, 8)
    value = torch.randn(2, 5, 2, 8)
    metadata = AttentionMetadata(
        attn_mask=torch.tensor([[True, True, True, False, False], [True, True, True, True, False]])
    )
    impl = SDPAImpl(num_heads=4, num_kv_heads=2, head_size=8, softmax_scale=0.5)

    monkeypatch.setattr(sdpa_backend, "can_sdpa_use_fused_gqa", lambda *args: True)
    native = impl.forward_cuda(query, key, value, metadata)

    monkeypatch.setattr(sdpa_backend, "can_sdpa_use_fused_gqa", lambda *args: False)
    expanded = impl.forward_cuda(query, key, value, metadata)

    torch.testing.assert_close(native, expanded)


def test_sdpa_npu_native_gqa_uses_full_qk_mask(monkeypatch):
    calls = []

    def fake_sdpa(query, key, value, **kwargs):
        calls.append((query.shape, key.shape, value.shape, kwargs))
        return query

    monkeypatch.setattr(sdpa_backend, "can_sdpa_use_fused_gqa", lambda *args: True)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)

    impl = SDPAImpl(num_heads=4, num_kv_heads=2, head_size=8, softmax_scale=0.5)
    metadata = AttentionMetadata(attn_mask=torch.tensor([[True, True, True, False, False]]))
    output = impl.forward_npu(
        torch.randn(1, 3, 4, 8),
        torch.randn(1, 5, 2, 8),
        torch.randn(1, 5, 2, 8),
        metadata,
    )

    query_shape, key_shape, value_shape, kwargs = calls[0]
    assert query_shape == (1, 4, 3, 8)
    assert key_shape == value_shape == (1, 2, 5, 8)
    assert kwargs["enable_gqa"] is True
    assert kwargs["attn_mask"].shape == (1, 1, 3, 5)
    assert kwargs["attn_mask"].is_contiguous()
    assert torch.equal(
        kwargs["attn_mask"],
        metadata.attn_mask[:, None, None, :].expand(1, 1, 3, 5),
    )
    assert output.shape == (1, 3, 4, 8)
