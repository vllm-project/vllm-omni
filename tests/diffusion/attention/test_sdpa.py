# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm_omni.diffusion.attention.backends.sdpa as sdpa_backend
import vllm_omni.diffusion.attention.backends.utils.attn_runtime_selector as attn_runtime_selector
from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


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


def test_sdpa_npu_uses_native_causal_gqa(monkeypatch):
    calls = []

    def fake_sdpa(query, key, value, **kwargs):
        calls.append((query.shape, key.shape, value.shape, kwargs))
        return query

    capability_calls = []

    def native_gqa_available(*args):
        capability_calls.append(args)
        return True

    monkeypatch.setattr(sdpa_backend, "can_sdpa_use_fused_gqa", native_gqa_available)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)

    impl = SDPAImpl(num_heads=4, num_kv_heads=2, head_size=8, softmax_scale=0.5, causal=True)
    output = impl.forward_npu(
        torch.randn(1, 3, 4, 8),
        torch.randn(1, 3, 2, 8),
        torch.randn(1, 3, 2, 8),
    )

    query_shape, key_shape, value_shape, kwargs = calls[0]
    assert query_shape == (1, 4, 3, 8)
    assert key_shape == value_shape == (1, 2, 3, 8)
    assert kwargs["enable_gqa"] is True
    assert kwargs["is_causal"] is True
    assert kwargs["scale"] == 0.5
    assert output.shape == (1, 3, 4, 8)
    assert len(capability_calls) == 1


def test_sdpa_npu_rejects_invalid_gqa_head_ratio():
    impl = SDPAImpl(num_heads=3, num_kv_heads=2, head_size=8, softmax_scale=0.5)

    with pytest.raises(ValueError, match="query heads to be a multiple"):
        impl.forward_npu(
            torch.randn(1, 3, 3, 8),
            torch.randn(1, 3, 2, 8),
            torch.randn(1, 3, 2, 8),
        )


class _FakePlatform:
    def __init__(self, name: str):
        self.name = name

    def is_npu(self):
        return self.name == "npu"


@pytest.mark.parametrize(
    ("platform_name", "expected"),
    [("npu", True), ("xpu", False), ("musa", False), ("unspecified", False)],
)
def test_sdpa_native_gqa_capability_is_platform_aware(monkeypatch, platform_name, expected):
    monkeypatch.setattr(
        attn_runtime_selector,
        "current_omni_platform",
        _FakePlatform(platform_name),
    )
    query = torch.randn(1, 4, 3, 8)
    key = value = torch.randn(1, 2, 3, 8)

    assert attn_runtime_selector.can_sdpa_use_fused_gqa(query, key, value, None, True) is expected
