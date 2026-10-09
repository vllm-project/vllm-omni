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

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.cpu
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


@pytest.mark.cpu
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


@pytest.mark.cpu
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


@pytest.mark.cpu
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


@pytest.mark.cpu
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
@pytest.mark.cpu
def test_cudnn_backend_accepts_blackwell_fmha_head_sizes(head_size):
    assert CuDNNAttentionBackend.supports_head_size(head_size)
    assert head_size in CuDNNAttentionBackend.get_supported_head_sizes()


@pytest.mark.parametrize("head_size", [0, 7, 12, 320])
@pytest.mark.cpu
def test_cudnn_backend_rejects_incompatible_head_sizes(head_size):
    assert not CuDNNAttentionBackend.supports_head_size(head_size)
    assert head_size not in CuDNNAttentionBackend.get_supported_head_sizes()


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("kv_heads", [1, 2])
@pytest.mark.parametrize("query_offset", [0, 4])
@pytest.mark.cpu
def test_piecewise_matches_dense_block_causal_mask(monkeypatch, batch_size, kv_heads, query_offset):
    monkeypatch.setattr(cudnn_backend, "sdpa_kernel", lambda _: nullcontext())
    torch.manual_seed(42)
    query = torch.randn(batch_size, 12, 2, 8)
    key = torch.randn(batch_size, 12, kv_heads, 8)
    value = torch.randn_like(key)
    spans = [(2, 5), (7, 10)]
    positions = torch.arange(12)
    mask = positions[:, None] >= positions[None, :]
    for start, end in spans:
        mask[start:end, start:end] = True
    expected = torch.nn.functional.scaled_dot_product_attention(
        query[:, query_offset:].transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=mask[None, None, query_offset:],
        scale=0.5,
        enable_gqa=kv_heads == 1,
    ).transpose(1, 2)
    impl = CuDNNAttentionImpl(num_heads=2, head_size=8, softmax_scale=0.5)
    actual = impl.forward_cuda(
        query[:, query_offset:], key, value, AttentionMetadata(full_attn_spans=[spans] * batch_size)
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.cpu
def test_piecewise_preserves_discontinuous_query_ranges(monkeypatch):
    from vllm_omni.diffusion.attention.backends.abstract import QueryRange

    monkeypatch.setattr(cudnn_backend, "sdpa_kernel", lambda _: nullcontext())
    query, key, value = [torch.randn(2, 12, 2, 8) for _ in range(3)]
    impl = CuDNNAttentionImpl(num_heads=2, head_size=8, softmax_scale=0.5)
    metadata = AttentionMetadata(full_attn_spans=[[(2, 5), (7, 12)]] * 2)
    expected = impl.forward_cuda(query, key, value, metadata)
    metadata.query_ranges = (QueryRange(0, 3, 0), QueryRange(3, 6, 8))
    actual = impl.forward_cuda(torch.cat([query[:, :3], query[:, 8:11]], dim=1), key, value, metadata)
    torch.testing.assert_close(actual, torch.cat([expected[:, :3], expected[:, 8:11]], dim=1))


@pytest.mark.cpu
def test_piecewise_rejects_padding_mask(monkeypatch):
    impl = CuDNNAttentionImpl(num_heads=2, head_size=8, softmax_scale=0.5)
    query = torch.randn(1, 8, 2, 8)
    metadata = AttentionMetadata(full_attn_spans=[[(2, 8)]], attn_mask=torch.ones(1, 8, dtype=torch.bool))
    with pytest.raises(ValueError, match="padding mask"):
        impl.forward_cuda(query, query, query, metadata)


@pytest.mark.cpu
def test_dense_mask_takes_precedence_over_spans(monkeypatch):
    monkeypatch.setattr(cudnn_backend, "sdpa_kernel", lambda _: nullcontext())
    query, key, value = [torch.randn(2, 8, 2, 8) for _ in range(3)]
    mask = torch.ones(2, 1, 8, 8, dtype=torch.bool).tril()
    mask[1, :, :, 3] = False
    impl = CuDNNAttentionImpl(num_heads=2, head_size=8, softmax_scale=0.5)
    expected = impl.forward_cuda(query, key, value, AttentionMetadata(attn_mask=mask))
    actual = impl.forward_cuda(query, key, value, AttentionMetadata(attn_mask=mask, full_attn_spans=[[(2, 8)]] * 2))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.gpu
@pytest.mark.parametrize("batch_size", [1, 2])
def test_cudnn_piecewise_matches_dense_on_device(batch_size):
    generator = torch.Generator(device="cuda").manual_seed(42)
    query, key, value = [
        torch.randn(batch_size, 96, 2, 128, generator=generator, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    ]
    spans = [(8, 40), (48, 96)]
    mask = torch.ones(96, 96, device="cuda", dtype=torch.bool).tril()
    for start, end in spans:
        mask[start:end, start:end] = True
    impl = CuDNNAttentionImpl(num_heads=2, head_size=128, softmax_scale=128**-0.5)
    actual = impl.forward_cuda(query, key, value, AttentionMetadata(full_attn_spans=[spans] * batch_size))
    expected = impl.forward_cuda(query, key, value, AttentionMetadata(attn_mask=mask[None, None]))
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.004)
