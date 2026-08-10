# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.cudnn_attn import CuDNNAttentionImpl

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


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
