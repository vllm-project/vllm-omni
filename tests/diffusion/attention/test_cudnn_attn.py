# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext

import pytest
import torch
from torch.nn.attention import SDPBackend

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.cudnn_attn import CuDNNAttentionImpl

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


@pytest.mark.parametrize(
    ("query_sequence_length", "key_sequence_length", "expected_backend"),
    [
        (1, 1, SDPBackend.MATH),
        (8, 1, SDPBackend.MATH),
        (1, 8, SDPBackend.MATH),
        (8, 8, SDPBackend.CUDNN_ATTENTION),
    ],
)
def test_cudnn_routes_single_token_attention_to_math(
    monkeypatch, query_sequence_length, key_sequence_length, expected_backend
):
    selected_backends = []

    def fake_sdpa_kernel(backends):
        selected_backends.append(backends)
        return nullcontext()

    monkeypatch.setattr(
        "vllm_omni.diffusion.attention.backends.cudnn_attn.sdpa_kernel",
        fake_sdpa_kernel,
    )
    monkeypatch.setattr(
        torch.nn.functional,
        "scaled_dot_product_attention",
        lambda query, _key, _value, **_kwargs: query,
    )
    impl = CuDNNAttentionImpl(
        num_heads=2,
        head_size=4,
        softmax_scale=0.5,
    )
    query = torch.randn(1, query_sequence_length, 2, 4)
    key = torch.randn(1, key_sequence_length, 2, 4)
    value = torch.randn_like(key)

    output = impl.forward_cuda(query, key, value)

    assert output.shape == query.shape
    assert selected_backends == [[expected_backend]]


def test_cudnn_slices_valid_kv_prefix_without_padding_mask(monkeypatch):
    observed = {}

    def fake_sdpa(query, key, value, **kwargs):
        observed.update(query=query, key=key, value=value, kwargs=kwargs)
        return query

    monkeypatch.setattr(
        "vllm_omni.diffusion.attention.backends.cudnn_attn.sdpa_kernel",
        lambda _backends: nullcontext(),
    )
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)
    impl = CuDNNAttentionImpl(
        num_heads=2,
        head_size=4,
        softmax_scale=0.5,
    )
    query = torch.randn(1, 8, 2, 4)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    output = impl.forward_cuda(
        query,
        key,
        value,
        AttentionMetadata(extra={"valid_kv_length": 5}),
    )

    assert output.shape == query.shape
    assert observed["query"].shape == (1, 2, 8, 4)
    assert observed["key"].shape == (1, 2, 5, 4)
    assert observed["value"].shape == (1, 2, 5, 4)
    assert observed["kwargs"]["attn_mask"] is None


def test_cudnn_rejects_invalid_valid_kv_length():
    impl = CuDNNAttentionImpl(
        num_heads=2,
        head_size=4,
        softmax_scale=0.5,
    )
    query = torch.randn(1, 8, 2, 4)

    with pytest.raises(ValueError, match="valid_kv_length"):
        impl.forward_cuda(
            query,
            query,
            query,
            AttentionMetadata(extra={"valid_kv_length": 9}),
        )
