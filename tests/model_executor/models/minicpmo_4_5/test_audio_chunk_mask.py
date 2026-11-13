# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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


@pytest.mark.parametrize("size", [0, 1, 49, 50, 51, 1500])
@pytest.mark.parametrize("chunk_size", [1, 17, 50])
@pytest.mark.parametrize("num_left_chunks", [-1, 0, 1, 3])
@pytest.mark.parametrize("num_lookhead", [0, 1, 7])
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
