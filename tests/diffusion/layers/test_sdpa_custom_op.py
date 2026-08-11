# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.diffusion.layers.custom_op import CustomOp
from vllm_omni.diffusion.layers.sdpa import ScaledDotProductAttention

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_sdpa_inherits_custom_op_and_native_expands_gqa_kv(monkeypatch):
    calls = []

    def fake_sdpa(query, key, value, **kwargs):
        calls.append((query.shape, key.shape, value.shape, kwargs))
        return query

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)
    op = ScaledDotProductAttention(causal=True)
    output = op.forward_native(
        torch.randn(1, 4, 3, 8),
        torch.randn(1, 2, 3, 8),
        torch.randn(1, 2, 3, 8),
    )

    query_shape, key_shape, value_shape, kwargs = calls[0]
    assert isinstance(op, CustomOp)
    assert query_shape == key_shape == value_shape == (1, 4, 3, 8)
    assert "enable_gqa" not in kwargs
    assert kwargs["is_causal"] is True
    assert output.shape == (1, 4, 3, 8)


def test_sdpa_npu_keeps_compressed_kv_for_native_gqa(monkeypatch):
    calls = []

    def fake_sdpa(query, key, value, **kwargs):
        calls.append((query.shape, key.shape, value.shape, kwargs))
        return query

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)
    op = ScaledDotProductAttention(causal=True)
    output = op.forward_npu(
        torch.randn(1, 4, 3, 8),
        torch.randn(1, 2, 3, 8),
        torch.randn(1, 2, 3, 8),
    )

    query_shape, key_shape, value_shape, kwargs = calls[0]
    assert query_shape == (1, 4, 3, 8)
    assert key_shape == value_shape == (1, 2, 3, 8)
    assert kwargs["enable_gqa"] is True
    assert kwargs["is_causal"] is True
    assert output.shape == (1, 4, 3, 8)


def test_sdpa_npu_disables_gqa_for_equal_head_counts(monkeypatch):
    calls = []

    def fake_sdpa(query, key, value, **kwargs):
        calls.append((query.shape, key.shape, value.shape, kwargs))
        return query

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)
    op = ScaledDotProductAttention(causal=True)
    output = op.forward_npu(
        torch.randn(1, 4, 3, 8),
        torch.randn(1, 4, 3, 8),
        torch.randn(1, 4, 3, 8),
    )

    query_shape, key_shape, value_shape, kwargs = calls[0]
    assert query_shape == key_shape == value_shape == (1, 4, 3, 8)
    assert kwargs["enable_gqa"] is False
    assert output.shape == (1, 4, 3, 8)


def test_sdpa_rejects_invalid_gqa_head_ratio():
    op = ScaledDotProductAttention()
    query = torch.randn(1, 3, 3, 8)
    key = torch.randn(1, 2, 3, 8)
    value = torch.randn(1, 2, 3, 8)

    with pytest.raises(ValueError, match="query heads to be a multiple"):
        op.forward_npu(query, key, value)


@pytest.mark.parametrize("forward_name", ["forward_native", "forward_npu"])
def test_sdpa_rejects_mismatched_kv_head_counts(forward_name):
    op = ScaledDotProductAttention()
    query = torch.randn(1, 4, 3, 8)
    key = torch.randn(1, 2, 3, 8)
    value = torch.randn(1, 1, 3, 8)

    with pytest.raises(ValueError, match="key and value to have the same number of heads"):
        getattr(op, forward_name)(query, key, value)
