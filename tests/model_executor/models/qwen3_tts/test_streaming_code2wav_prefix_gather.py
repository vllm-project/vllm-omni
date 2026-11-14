# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.model_executor.models.qwen3_tts.streaming_code2wav import _gather_prefix_rows

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_prefix_gather_matches_per_row_cat():
    torch.manual_seed(0)
    for rows, frames in [([0, 1, 2], 5), (list(range(8)), 10), ([3, 1, 0], 7)]:
        codes = torch.randint(0, 100, (8, 4, 10))
        expected = torch.cat([codes[row : row + 1, :, :frames] for row in rows], dim=0)
        actual = _gather_prefix_rows(codes, rows, frames)
        assert torch.equal(expected, actual)

    # Empty groups are never produced by prefix_groups today, but the helper
    # should still return a well-shaped empty tensor rather than raise.
    empty = _gather_prefix_rows(codes, [], 5)
    assert empty.shape == (0, 4, 5)
