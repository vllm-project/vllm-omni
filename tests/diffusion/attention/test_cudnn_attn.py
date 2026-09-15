# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import contextmanager, nullcontext
from typing import Any

import pytest
import torch
from torch.nn.attention import SDPBackend

import vllm_omni.diffusion.attention.backends.cudnn_attn as cudnn_backend
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.cudnn_attn import CuDNNAttentionBackend, CuDNNAttentionImpl

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_cudnn_backend_uses_math_for_kv_seq_len_one(monkeypatch):
    """Automatic CUDNN_ATTN (platform default) may use MATH for singleton K/V."""
    selected_backends = []

    @contextmanager
    def fake_sdpa_kernel(backends):
        selected_backends.append(tuple(backends))
        yield

    def fake_sdpa(query, key, value, **kwargs):
        return query

    monkeypatch.setattr(cudnn_backend, "sdpa_kernel", fake_sdpa_kernel)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)

    impl = CuDNNAttentionImpl(num_heads=2, head_size=8, softmax_scale=0.5)
    query = torch.randn(1, 2, 2, 8)
    singleton_kv = torch.randn(1, 1, 2, 8)

    output = impl.forward_cuda(query, singleton_kv, singleton_kv)

    assert output.shape == query.shape
    assert selected_backends == [(SDPBackend.MATH,)]


def test_explicit_cudnn_rejects_kv_seq_len_one():
    impl = CuDNNAttentionImpl(
        num_heads=2,
        head_size=8,
        softmax_scale=0.5,
        backend_explicit=True,
    )
    query = torch.randn(1, 2, 2, 8)
    singleton_kv = torch.randn(1, 1, 2, 8)

    with pytest.raises(ValueError, match="explicitly selected.*sequence length 1"):
        impl.forward_cuda(query, singleton_kv, singleton_kv)


def test_cudnn_backend_pins_cudnn_only_when_kv_seq_len_gt_one(monkeypatch):
    selected_backends = []

    @contextmanager
    def fake_sdpa_kernel(backends):
        selected_backends.append(tuple(backends))
        yield

    def reject_shape(*args, **kwargs):
        raise RuntimeError("No available kernel. Aborting execution.")

    monkeypatch.setattr(cudnn_backend, "sdpa_kernel", fake_sdpa_kernel)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", reject_shape)

    impl = CuDNNAttentionImpl(num_heads=2, head_size=8, softmax_scale=0.5)
    tensors = torch.randn(1, 2, 2, 8)

    with pytest.raises(RuntimeError, match="No available kernel"):
        impl.forward_cuda(tensors, tensors, tensors)

    assert selected_backends == [(SDPBackend.CUDNN_ATTENTION,)]


def test_cudnn_slices_valid_kv_prefix_without_padding_mask(monkeypatch):
    observed: dict[str, Any] = {}

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


@pytest.mark.parametrize("head_size", [8, 64, 128, 256])
def test_cudnn_backend_accepts_blackwell_fmha_head_sizes(head_size):
    assert CuDNNAttentionBackend.supports_head_size(head_size)
    assert head_size in CuDNNAttentionBackend.get_supported_head_sizes()


@pytest.mark.parametrize("head_size", [0, 7, 12, 320])
def test_cudnn_backend_rejects_incompatible_head_sizes(head_size):
    assert not CuDNNAttentionBackend.supports_head_size(head_size)
    assert head_size not in CuDNNAttentionBackend.get_supported_head_sizes()
