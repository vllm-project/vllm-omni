# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Piecewise block-causal attention through the SDPA backend (CPU).

Platforms that serve SDPA as the default backend (the Ascend 310P among them)
have no fused masked kernel, so a dense block-causal mask turns every
attention call into a matmul/softmax loop, while the unmasked target span of
the same plan reaches a flash kernel. These tests pin the contract the NPU
platform relies on:

  * ``full_attn_spans`` are consumed by ``SDPAImpl.forward_npu`` instead of a
    dense mask;
  * the segmented result equals the dense block-causal mask it replaces.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import vllm_omni.diffusion.attention.backends.sdpa as sdpa_module
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

DEVICE = torch.device("cpu")


def _impl(heads: int = 2, head_dim: int = 4, kv_heads: int | None = None) -> SDPAImpl:
    return SDPAImpl(
        num_heads=heads,
        head_size=head_dim,
        softmax_scale=head_dim**-0.5,
        causal=False,
        num_kv_heads=kv_heads,
    )


def _block_causal_mask(block_start: int, seq_len: int) -> torch.Tensor:
    """``(q_idx >= kv_idx) or kv inside the full-attention block``, SDPA keep convention."""
    q_idx = torch.arange(seq_len).view(-1, 1)
    kv_idx = torch.arange(seq_len).view(1, -1)
    keep = q_idx >= kv_idx
    keep[block_start:, block_start:] = True
    return keep.view(1, 1, seq_len, seq_len)


def _dense_reference(query, key, value, keep_mask, scale):
    return F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=keep_mask,
        scale=scale,
    ).transpose(1, 2)


@pytest.mark.parametrize("kv_heads", [None, 1])
def test_forward_npu_piecewise_matches_dense_block_causal_mask(kv_heads):
    """The segment plan reproduces the dense block-causal mask on NPU SDPA."""
    torch.manual_seed(0)
    heads, head_dim = 4, 8
    prefix_len, block_len = 5, 4
    seq_len = prefix_len + block_len
    kv = kv_heads or heads
    impl = _impl(heads=heads, head_dim=head_dim, kv_heads=kv_heads)

    query = torch.randn(1, seq_len, heads, head_dim, device=DEVICE)
    key = torch.randn(1, seq_len, kv, head_dim, device=DEVICE)
    value = torch.randn(1, seq_len, kv, head_dim, device=DEVICE)

    metadata = AttentionMetadata(full_attn_spans=[[(prefix_len, seq_len)]])
    segmented = impl.forward_npu(query, key, value, metadata)
    dense = _dense_reference(query, key, value, _block_causal_mask(prefix_len, seq_len), impl.softmax_scale)

    torch.testing.assert_close(segmented, dense, rtol=1e-5, atol=1e-6)


def test_forward_npu_routes_spans_to_piecewise(monkeypatch):
    """``forward_npu`` hands the spans and query ranges to ``piecewise_attn``."""
    impl = _impl()
    spans = [[(2, 5)]]
    ranges = ((0, 5, 0),)
    metadata = AttentionMetadata(full_attn_spans=spans, query_ranges=ranges)
    query = torch.randn(1, 5, 2, 4, device=DEVICE)
    seen: dict[str, object] = {}

    def fake_piecewise(query, key, value, full_attn_spans, softmax_scale, attn_func, query_ranges=None):
        seen.update(
            spans=full_attn_spans,
            scale=softmax_scale,
            attn_func=attn_func,
            query_ranges=query_ranges,
        )
        return torch.zeros_like(query)

    monkeypatch.setattr(sdpa_module, "piecewise_attn", fake_piecewise)
    output = impl.forward_npu(query, query, query, metadata)

    assert output.shape == query.shape
    assert seen["spans"] is spans
    assert seen["scale"] == impl.softmax_scale
    assert seen["query_ranges"] is ranges
    assert seen["attn_func"] is SDPAImpl._piecewise_segment_attention


def test_forward_npu_without_spans_does_not_segment(monkeypatch):
    """A mask-only (or metadata-free) call keeps the plain SDPA path."""
    impl = _impl()
    query = torch.randn(1, 4, 2, 4, device=DEVICE)
    metadata = AttentionMetadata(attn_mask=torch.ones(1, 4, dtype=torch.bool))

    def unexpected_piecewise(*args, **kwargs):
        pytest.fail("piecewise attention requires full_attn_spans")

    monkeypatch.setattr(sdpa_module, "piecewise_attn", unexpected_piecewise)
    output = impl.forward_npu(query, query, query, metadata)
    assert output.shape == query.shape


def _block_causal_mask_from_spans(spans: list[tuple[int, int]], seq_len: int) -> torch.Tensor:
    """``(q_idx >= kv_idx) or same full-attention block``, SDPA keep convention."""
    block = torch.full((seq_len,), -1, dtype=torch.long)
    for block_id, (start, end) in enumerate(spans):
        block[start:end] = block_id
    q_idx = torch.arange(seq_len).view(-1, 1)
    kv_idx = torch.arange(seq_len).view(1, -1)
    keep = (q_idx >= kv_idx) | ((block.view(-1, 1) == block.view(1, -1)) & (block.view(-1, 1) >= 0))
    return keep.view(1, 1, seq_len, seq_len)


def test_forward_npu_piecewise_with_condition_block_in_prefix():
    """A full-attention block inside the prefix leaves a rectangular causal segment.

    ``piecewise_attn`` defines a causal segment with FlashAttention's
    bottom-right diagonal; PyTorch's ``is_causal`` is top-left, and the two
    disagree exactly when the segment is rectangular. Text to image has a
    single trailing image block, so its only causal segment is the square text
    prefix; image-conditioned generation has condition-image blocks inside the
    prefix and hits this case.
    """
    torch.manual_seed(1)
    heads, head_dim = 2, 8
    seq_len = 24
    spans = [(2, 6), (12, seq_len)]
    impl = _impl(heads=heads, head_dim=head_dim)

    query, key, value = (torch.randn(1, seq_len, heads, head_dim, device=DEVICE) for _ in range(3))
    metadata = AttentionMetadata(full_attn_spans=[list(spans)])
    segmented = impl.forward_npu(query, key, value, metadata)
    dense = _dense_reference(
        query, key, value, _block_causal_mask_from_spans(spans, seq_len), impl.softmax_scale
    )

    torch.testing.assert_close(segmented, dense, rtol=1e-5, atol=1e-6)
