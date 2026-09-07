# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Characterization tests for ``vllm_omni.worker.payload_span``.

These pin the CURRENT behaviour of the thinker-decode span helpers so the later
worker refactors that touch span
plumbing have a regression net. Pure CPU — no model, no CUDA.
"""

import pytest
import torch

from vllm_omni.worker.payload_span import (
    get_tensor_span,
    get_tensor_span_row,
    merge_tensor_spans,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _rows(n: int, *, start_val: int = 0, hidden: int = 1) -> torch.Tensor:
    """A ``[n, hidden]`` tensor whose first column counts up from ``start_val``.

    ``[n, hidden]`` mirrors the connector async-chunk shape (tokens x hidden).
    """
    base = torch.arange(start_val, start_val + n, dtype=torch.float32).unsqueeze(1)
    return base.repeat(1, hidden)


# --------------------------------------------------------------------------- #
# get_tensor_span                                                             #
# --------------------------------------------------------------------------- #
def test_get_tensor_span_valid_returns_triplet():
    tensor = _rows(3, hidden=4)
    payload = {"emb": tensor, "start": 5, "end": 8}

    span = get_tensor_span(payload, tensor_key="emb", start_key="start", end_key="end")

    assert span is not None
    got_tensor, start, end = span
    assert got_tensor is tensor
    assert (start, end) == (5, 8)


def test_get_tensor_span_non_tensor_returns_none():
    payload = {"emb": None, "start": 0, "end": 0}
    assert get_tensor_span(payload, tensor_key="emb", start_key="start", end_key="end") is None


def test_get_tensor_span_non_int_bounds_returns_none():
    tensor = _rows(2)
    # start is a float, not an int -> rejected.
    payload = {"emb": tensor, "start": 0.0, "end": 2}
    assert get_tensor_span(payload, tensor_key="emb", start_key="start", end_key="end") is None


def test_get_tensor_span_negative_start_returns_none():
    tensor = _rows(1)
    payload = {"emb": tensor, "start": -1, "end": 0}
    assert get_tensor_span(payload, tensor_key="emb", start_key="start", end_key="end") is None


def test_get_tensor_span_end_before_start_returns_none():
    tensor = _rows(2)
    payload = {"emb": tensor, "start": 5, "end": 3}
    assert get_tensor_span(payload, tensor_key="emb", start_key="start", end_key="end") is None


def test_get_tensor_span_length_mismatch_returns_none():
    tensor = _rows(3)  # tensor.shape[0] == 3
    payload = {"emb": tensor, "start": 0, "end": 2}  # end - start == 2
    assert get_tensor_span(payload, tensor_key="emb", start_key="start", end_key="end") is None


# --------------------------------------------------------------------------- #
# merge_tensor_spans                                                          #
# --------------------------------------------------------------------------- #
def test_merge_none_operand_returns_none():
    span = (_rows(2), 0, 2)
    assert merge_tensor_spans(None, span) is None
    assert merge_tensor_spans(span, None) is None


def test_merge_adjacent_spans_concatenates():
    existing = (_rows(3, start_val=0), 0, 3)
    incoming = (_rows(2, start_val=3), 3, 5)

    merged = merge_tensor_spans(existing, incoming)

    assert merged is not None
    tensor, start, end = merged
    assert (start, end) == (0, 5)
    assert tensor.shape[0] == 5
    assert tensor[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_merge_overlapping_spans_trims_incoming():
    # existing covers [0, 3); incoming covers [2, 5) -> 1 row of overlap.
    existing = (_rows(3, start_val=0), 0, 3)
    incoming = (_rows(3, start_val=100), 2, 5)

    merged = merge_tensor_spans(existing, incoming)

    assert merged is not None
    tensor, start, end = merged
    assert (start, end) == (0, 5)
    # First 3 rows from existing, then incoming's tail (overlap=1 row trimmed).
    assert tensor[:, 0].tolist() == [0.0, 1.0, 2.0, 101.0, 102.0]


def test_merge_fully_overlapping_incoming_keeps_existing():
    # incoming [1, 3) is fully covered by existing [0, 3): overlap >= incoming len.
    existing = (_rows(3, start_val=0), 0, 3)
    incoming = (_rows(2, start_val=100), 1, 3)

    merged = merge_tensor_spans(existing, incoming)

    assert merged is not None
    tensor, start, end = merged
    assert (start, end) == (0, 3)
    assert tensor is existing[0]  # returned unchanged


def test_merge_non_contiguous_spans_returns_none():
    # gap between existing end (3) and incoming start (5).
    existing = (_rows(3, start_val=0), 0, 3)
    incoming = (_rows(2, start_val=100), 5, 7)

    assert merge_tensor_spans(existing, incoming) is None


def test_merge_casts_incoming_to_existing_dtype_and_device():
    existing = (_rows(2, start_val=0).to(torch.float32), 0, 2)
    incoming = (torch.arange(2, 4, dtype=torch.int64).unsqueeze(1), 2, 4)

    merged = merge_tensor_spans(existing, incoming)

    assert merged is not None
    tensor, _, _ = merged
    assert tensor.dtype == torch.float32
    assert tensor.device == existing[0].device


# --------------------------------------------------------------------------- #
# get_tensor_span_row                                                         #
# --------------------------------------------------------------------------- #
def test_get_tensor_span_row_in_range():
    span = (_rows(3, start_val=10), 5, 8)  # rows map to indices 5, 6, 7
    row = get_tensor_span_row(span, 6)
    assert row is not None
    assert row[0].item() == 11.0  # second row (start_val 10 + 1)


def test_get_tensor_span_row_none_span_returns_none():
    assert get_tensor_span_row(None, 0) is None


def test_get_tensor_span_row_below_start_returns_none():
    span = (_rows(3), 5, 8)
    assert get_tensor_span_row(span, 4) is None


def test_get_tensor_span_row_at_or_above_end_returns_none():
    span = (_rows(3), 5, 8)
    assert get_tensor_span_row(span, 8) is None
