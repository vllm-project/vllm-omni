# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
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

import vllm_omni.diffusion.attention.backends.utils.piecewise_attn as piecewise_module
from vllm_omni.diffusion.attention.backends.abstract import QueryRange
from vllm_omni.diffusion.attention.backends.utils.piecewise_attn import (
    Segment,
    build_paged_piecewise_plan,
    build_segments,
    piecewise_attn,
    run_paged_piecewise_plan,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


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


def test_build_segments_full_span_retains_global_kv_end():
    segments = build_segments([(2, 7)], query_offset=4, query_len=2)

    assert segments == [Segment(q_start=4, q_end=6, kv_end=7, mode="full")]


def test_paged_piecewise_runner_uses_packed_indices_and_kv_endpoints():
    full_attn_spans = [[(2, 5)], [(4, 7)]]
    plan = build_paged_piecewise_plan(
        full_attn_spans,
        query_offsets=[0, 2],
        query_lens=[6, 6],
        seq_lens=[6, 8],
        device=DEVICE,
    )
    query = torch.arange(12 * 2 * 4, dtype=torch.float32).reshape(12, 2, 4)
    key = query + 100
    value = query + 200
    calls: list[object] = []

    output_buffer = torch.empty_like(query)

    def run_segment(segment_query, segment_key, segment_value, segment_batch, segment_output_buffer):
        assert segment_output_buffer is None
        calls.append(
            (
                [segment.mode for segment in segment_batch.row_segments],
                [segment.kv_end for segment in segment_batch.row_segments],
                segment_batch.query_indices.tolist(),
            )
        )
        torch.testing.assert_close(segment_key, segment_query + 100)
        torch.testing.assert_close(segment_value, segment_query + 200)
        return segment_query

    output = run_paged_piecewise_plan(
        query,
        key,
        value,
        plan,
        plan.segments,
        run_segment,
        output_buffer=output_buffer,
    )

    assert output is output_buffer
    torch.testing.assert_close(output, query)
    assert all(segment.query_range is None for segment in plan.segments)
    assert not plan.segments_cover_query_contiguously
    assert calls == [
        (["causal", "causal"], [2, 4], [0, 1, 6, 7]),
        (["full", "full"], [5, 7], [2, 3, 4, 8, 9, 10]),
        (["causal", "causal"], [6, 8], [5, 11]),
    ]


def test_paged_piecewise_runner_preserves_homogeneous_batch_without_scatter(monkeypatch: pytest.MonkeyPatch):
    plan = build_paged_piecewise_plan(
        [[(2, 5)], [(2, 5)]],
        query_offsets=[0, 0],
        query_lens=[6, 6],
        seq_lens=[6, 6],
        device=DEVICE,
    )
    assert plan.homogeneous_batch_shape == (2, 6)
    query = torch.arange(12 * 2 * 4, dtype=torch.float32).reshape(12, 2, 4)
    calls: list[torch.Tensor] = []

    def fail_index_copy(*_args, **_kwargs):
        raise AssertionError("homogeneous piecewise path must not use index_copy_")

    monkeypatch.setattr(torch.Tensor, "index_copy_", fail_index_copy)

    def run_segment(segment_query, segment_key, segment_value, _metadata, segment_output_buffer):
        assert segment_key is None
        assert segment_value is None
        assert segment_output_buffer is None
        calls.append(segment_query.clone())
        return segment_query + 1

    output_buffer = torch.empty_like(query)
    output = run_paged_piecewise_plan(
        query,
        None,
        None,
        plan,
        plan.segments,
        run_segment,
        output_buffer=output_buffer,
        use_homogeneous_batch=True,
    )

    assert output is output_buffer
    torch.testing.assert_close(output, query + 1)
    assert [call.shape[0] for call in calls] == [4, 6, 2]
    torch.testing.assert_close(calls[0], query[[0, 1, 6, 7]])
    torch.testing.assert_close(calls[1], query[[2, 3, 4, 8, 9, 10]])
    torch.testing.assert_close(calls[2], query[[5, 11]])


def test_paged_piecewise_runner_keeps_indexed_fallback_for_heterogeneous_layout():
    plan = build_paged_piecewise_plan(
        [[(2, 5)], [(1, 4)]],
        query_offsets=[0, 0],
        query_lens=[6, 6],
        seq_lens=[6, 6],
        device=DEVICE,
    )
    assert plan.homogeneous_batch_shape is None
    query = torch.arange(12 * 2 * 4, dtype=torch.float32).reshape(12, 2, 4)
    output = run_paged_piecewise_plan(
        query,
        None,
        None,
        plan,
        plan.segments,
        lambda segment_query, *_args: segment_query,
        use_homogeneous_batch=True,
    )
    torch.testing.assert_close(output, query)


def test_paged_piecewise_runner_slices_contiguous_segments():
    plan = build_paged_piecewise_plan(
        [[(2, 5)]],
        query_offsets=[0],
        query_lens=[6],
        seq_lens=[6],
        device=DEVICE,
    )
    query = torch.arange(6 * 2 * 4, dtype=torch.float32).reshape(6, 2, 4)
    key = query + 100
    value = query + 200
    expected_ranges = [(0, 2), (2, 5), (5, 6)]
    calls: list[tuple[int, int]] = []

    def run_segment(segment_query, segment_key, segment_value, _segment_batch, segment_output_buffer):
        assert segment_output_buffer is None
        expected_range = expected_ranges[len(calls)]
        start, end = expected_range
        calls.append(expected_range)
        assert segment_query.untyped_storage().data_ptr() == query.untyped_storage().data_ptr()
        assert segment_key.untyped_storage().data_ptr() == key.untyped_storage().data_ptr()
        assert segment_value.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
        torch.testing.assert_close(segment_query, query[start:end])
        return segment_query

    output = run_paged_piecewise_plan(query, key, value, plan, plan.segments, run_segment)

    assert calls == expected_ranges
    assert [segment.query_range for segment in plan.segments] == expected_ranges
    assert plan.segments_cover_query_contiguously
    torch.testing.assert_close(output, query)


def test_paged_piecewise_runner_writes_contiguous_segments_into_output_buffer():
    plan = build_paged_piecewise_plan(
        [[(2, 5)]],
        query_offsets=[0],
        query_lens=[6],
        seq_lens=[6],
        device=DEVICE,
    )
    query = torch.arange(6 * 2 * 4, dtype=torch.float32).reshape(6, 2, 4)
    output_buffer = torch.empty(6, 2, 3)
    targets = []

    def run_segment(segment_query, _key, _value, _metadata, segment_output_buffer):
        assert segment_output_buffer is not None
        assert segment_output_buffer.untyped_storage().data_ptr() == output_buffer.untyped_storage().data_ptr()
        targets.append(segment_output_buffer)
        return segment_output_buffer.copy_(segment_query[..., :3])

    output = run_paged_piecewise_plan(
        query,
        query,
        query,
        plan,
        plan.segments,
        run_segment,
        output_buffer=output_buffer,
    )

    assert output is output_buffer
    assert [target.shape[0] for target in targets] == [2, 3, 1]
    torch.testing.assert_close(output, query[..., :3])


def test_paged_piecewise_runner_returns_single_full_segment_directly():
    plan = build_paged_piecewise_plan(
        [[]],
        query_offsets=[0],
        query_lens=[6],
        seq_lens=[6],
        device=DEVICE,
    )
    query = torch.randn(6, 2, 4)
    runner_output = query + 1

    output = run_paged_piecewise_plan(
        query,
        query,
        query,
        plan,
        plan.segments,
        lambda *_args: runner_output,
    )

    assert output is runner_output


def test_paged_piecewise_runner_uses_runner_output_shape():
    plan = build_paged_piecewise_plan(
        [[(2, 5)]],
        query_offsets=[0],
        query_lens=[6],
        seq_lens=[6],
        device=DEVICE,
    )
    query = torch.arange(6 * 2 * 4, dtype=torch.float32).reshape(6, 2, 4)

    output = run_paged_piecewise_plan(
        query,
        query,
        query,
        plan,
        plan.segments,
        lambda segment_query, *_args: segment_query[..., :3],
    )

    assert output.shape == (6, 2, 3)
    torch.testing.assert_close(output, query[..., :3])


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


def test_piecewise_noncontiguous_query_ranges_match_full_mask():
    torch.manual_seed(0)
    B, H, D, Sk = 1, 2, 8, 8
    key = torch.randn(B, Sk, H, D, device=DEVICE)
    value = torch.randn(B, Sk, H, D, device=DEVICE)
    query = torch.randn(B, 4, H, D, device=DEVICE)
    spans = [(2, 7)]
    query_ranges = (
        QueryRange(0, 1, 0),
        QueryRange(1, 4, 4),
    )
    softmax_scale = D**-0.5

    got = piecewise_attn(
        query,
        key,
        value,
        full_attn_spans=[spans],
        softmax_scale=softmax_scale,
        attn_func=_sdpa_attn_func,
        query_ranges=query_ranges,
    )
    expected = torch.cat(
        [
            _full_reference(query[:, :1], key, value, spans, 0, 1, softmax_scale),
            _full_reference(query[:, 1:], key, value, spans, 4, 7, softmax_scale),
        ],
        dim=1,
    )
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_piecewise_rejects_incomplete_query_ranges():
    query = torch.zeros(1, 3, 1, 2)
    key = torch.zeros(1, 3, 1, 2)
    value = torch.zeros_like(key)

    with pytest.raises(ValueError, match="cover local query contiguously"):
        piecewise_attn(
            query,
            key,
            value,
            full_attn_spans=[[]],
            softmax_scale=1.0,
            attn_func=_sdpa_attn_func,
            query_ranges=(QueryRange(1, 3, 1),),
        )


def test_piecewise_combines_noncontiguous_query_ranges_with_one_cat(monkeypatch):
    query = torch.randn(1, 4, 2, 8)
    key = torch.randn(1, 8, 2, 8)
    value = torch.randn_like(key)
    cat_calls = 0
    original_cat = torch.cat

    def counted_cat(tensors, dim=0):
        nonlocal cat_calls
        cat_calls += 1
        return original_cat(tensors, dim=dim)

    monkeypatch.setattr(piecewise_module.torch, "cat", counted_cat)

    output = piecewise_attn(
        query,
        key,
        value,
        full_attn_spans=[[(0, 8)]],
        softmax_scale=1.0,
        attn_func=lambda q, k, v, **kwargs: q,
        query_ranges=(QueryRange(0, 1, 0), QueryRange(1, 4, 4)),
    )

    assert cat_calls == 1
    torch.testing.assert_close(output, query)
