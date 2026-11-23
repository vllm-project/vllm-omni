# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for MiniCPM-o 4.5 audio chunk attention masks."""

from __future__ import annotations

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    MiniCPMO45OmniLLMForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _reference_chunk_mask(
    size: int,
    chunk_size: int,
    num_left_chunks: int,
    num_lookhead: int,
) -> torch.Tensor:
    """Preserve the original row-by-row implementation as a test oracle."""
    mask = torch.zeros(size, size, dtype=torch.bool)
    for row in range(size):
        if num_left_chunks < 0:
            start = 0
        else:
            start = max(
                (row // chunk_size - num_left_chunks) * chunk_size,
                0,
            )
        end = min(
            (row // chunk_size + 1) * chunk_size + num_lookhead,
            size,
        )
        mask[row, start:end] = True
    return mask


@pytest.mark.parametrize(
    ("size", "chunk_size", "num_left_chunks", "num_lookhead"),
    [
        pytest.param(0, 50, -1, 0, id="empty"),
        pytest.param(1, 50, 0, 0, id="single-token"),
        pytest.param(49, 50, 1, 7, id="before-chunk-boundary"),
        pytest.param(50, 50, 1, 7, id="at-chunk-boundary"),
        pytest.param(51, 50, 1, 7, id="after-chunk-boundary"),
        pytest.param(1500, 50, 3, 7, id="long-audio"),
        # Cross context settings only on inputs long enough to exercise every
        # left window and lookahead, instead of repeating empty/full masks.
        *[
            (4 * chunk_size + 1, chunk_size, left_chunks, lookahead)
            for chunk_size in (1, 17, 50)
            for left_chunks in (-1, 0, 1, 3)
            for lookahead in (0, 1, 7)
        ],
    ],
)
def test_subsequent_chunk_mask_matches_row_by_row_reference(
    size: int,
    chunk_size: int,
    num_left_chunks: int,
    num_lookhead: int,
) -> None:
    actual = MiniCPMO45OmniLLMForConditionalGeneration.subsequent_chunk_mask(
        None,
        size=size,
        chunk_size=chunk_size,
        num_left_chunks=num_left_chunks,
        num_lookhead=num_lookhead,
    )
    expected = _reference_chunk_mask(
        size,
        chunk_size,
        num_left_chunks,
        num_lookhead,
    )

    assert actual.dtype == torch.bool
    assert actual.shape == (size, size)
    assert torch.equal(actual, expected)
