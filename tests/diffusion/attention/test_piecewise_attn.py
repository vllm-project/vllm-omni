# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end test for ``piecewise_attn`` (CPU).

Verify that running attention in segments (causal outside full-attn spans,
bidirectional inside full-attn spans) matches running a single full SDPA call
with the equivalent 2D attention mask.

Covers:
  * batch size = 1 and batch size > 1 (homogeneous CFG-like batch)
  * query length == key length   (full prefill)
  * query length <  key length   (decode-like tail slice)
  * various full-attn-span layouts (none / start / middle / end / multi)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.attention.backends.utils.piecewise_attn import (
    piecewise_attn,
)

DEVICE = torch.device("cpu")


def _sdpa_attn_func(q, k, v, causal, softmax_scale):
    q_ = q.transpose(1, 2)
    k_ = k.transpose(1, 2)
    v_ = v.transpose(1, 2)
    attn_mask = None
    if causal:
        Sq, Sk = q_.shape[-2], k_.shape[-2]
        i = torch.arange(Sq, device=q.device).unsqueeze(1)
        j = torch.arange(Sk, device=q.device).unsqueeze(0)
        attn_mask = j <= (i + (Sk - Sq))
    out = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=attn_mask, scale=softmax_scale)
    return out.transpose(1, 2).contiguous()


def _full_reference(query, key, value, global_spans, q_start, q_end, softmax_scale):
    """Build a full 2D mask with global spans and compute reference output."""
    Sk = key.shape[1]
    mask = torch.tril(torch.ones(Sk, Sk, dtype=torch.bool, device=key.device))
    for a, e in global_spans:
        mask[a:e, :e] = True
    mask_q = mask[q_start:q_end, :]
    q_ = query.transpose(1, 2)
    k_ = key.transpose(1, 2)
    v_ = value.transpose(1, 2)
    out = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=mask_q, scale=softmax_scale)
    return out.transpose(1, 2).contiguous()


SPAN_CASES = [
    pytest.param([], id="no-spans"),
    pytest.param([(0, 10)], id="span-at-start"),
    pytest.param([(10, 30), (54, 64)], id="multi-spans"),
]

Q_RANGE_CASES = [
    pytest.param((0, 64), id="q_eq_k"),  # Sq == Sk (prefill)
    pytest.param((53, 64), id="q_lt_k"),  # Sq < Sk (decode-like)
]

BATCH_CASES = [
    pytest.param(1, id="B1"),
    pytest.param(2, id="B2"),
]


@pytest.mark.parametrize("global_spans", SPAN_CASES)
@pytest.mark.parametrize("q_range", Q_RANGE_CASES)
@pytest.mark.parametrize("batch_size", BATCH_CASES)
def test_piecewise_matches_full(global_spans, q_range, batch_size):
    torch.manual_seed(0)
    H, D, Sk = 2, 16, 64
    q_start, q_end = q_range
    Sq = q_end - q_start

    key = torch.randn(batch_size, Sk, H, D, device=DEVICE)
    value = torch.randn(batch_size, Sk, H, D, device=DEVICE)
    query = torch.randn(batch_size, Sq, H, D, device=DEVICE)

    full_attn_spans = [list(global_spans) for _ in range(batch_size)]
    softmax_scale = 1.0 / (D**0.5)

    got = piecewise_attn(
        query,
        key,
        value,
        full_attn_spans=full_attn_spans,
        softmax_scale=softmax_scale,
        attn_func=_sdpa_attn_func,
    )
    expected = _full_reference(query, key, value, global_spans, q_start, q_end, softmax_scale)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_piecewise_span_fully_before_qstart():
    """Spans entirely before query region produce pure causal attention."""
    torch.manual_seed(0)
    B, H, D, Sk = 1, 2, 16, 30
    q_start, q_end = 15, 30
    Sq = q_end - q_start

    key = torch.randn(B, Sk, H, D, device=DEVICE)
    value = torch.randn(B, Sk, H, D, device=DEVICE)
    query = torch.randn(B, Sq, H, D, device=DEVICE)

    global_spans = [(5, 10)]
    full_attn_spans = [list(global_spans) for _ in range(B)]
    softmax_scale = 1.0 / (D**0.5)

    got = piecewise_attn(
        query,
        key,
        value,
        full_attn_spans=full_attn_spans,
        softmax_scale=softmax_scale,
        attn_func=_sdpa_attn_func,
    )
    expected = _full_reference(query, key, value, global_spans, q_start, q_end, softmax_scale)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)
