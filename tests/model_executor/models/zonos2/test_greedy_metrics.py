# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks that P2-09 cannot pass truncated prefixes or hide bad frames."""

import pytest
import torch

from tests.model_executor.models.zonos2.greedy_metrics import greedy_code_parity

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_exact_twelve_frames_pass_all_nine_codebooks():
    reference = torch.arange(108, dtype=torch.int32).reshape(12, 9)
    result = greedy_code_parity(reference, reference.long())
    assert result["prefix_matches"] == 108
    assert result["whole_frame_matches"] == 12
    assert result["l2_pass"]


def test_one_wrong_code_in_prefix_fails_even_with_high_total_agreement():
    reference = torch.zeros((1000, 9), dtype=torch.int64)
    actual = reference.clone()
    actual[11, 8] = 1
    result = greedy_code_parity(reference, actual)
    assert result["whole_frame_agreement"] == 0.999
    assert result["prefix_matches"] == 107
    assert not result["l2_pass"]
    assert result["first_difference_frame"] == 11


@pytest.mark.parametrize("frames", [0, 1, 11])
def test_incomplete_prefix_never_passes(frames):
    codes = torch.zeros((frames, 9), dtype=torch.int32)
    assert not greedy_code_parity(codes, codes)["l2_pass"]


def test_frame_metric_requires_all_nine_codes_and_uses_95_percent_threshold():
    reference = torch.zeros((100, 9), dtype=torch.int32)
    actual = reference.clone()
    actual[12:17, 0] = 1
    result = greedy_code_parity(reference, actual)
    assert result["code_agreement"] > 0.99
    assert result["whole_frame_agreement"] == 0.95
    assert result["l2_pass"]
    actual[17, 0] = 1
    assert not greedy_code_parity(reference, actual)["l2_pass"]


@pytest.mark.parametrize("extra_tail", [False, True])
def test_unmatched_tail_is_penalized_instead_of_silently_cropped(extra_tail):
    short = torch.zeros((100, 9), dtype=torch.int32)
    long = torch.zeros((110, 9), dtype=torch.int32)
    a, b = (short, long) if extra_tail else (long, short)
    result = greedy_code_parity(a, b)
    assert result["whole_frame_total"] == 110
    assert result["whole_frame_matches"] == 100
    assert result["unpaired_tail_frames"] == 10
    assert result["first_difference_frame"] == 100
    assert not result["l2_pass"]


@pytest.mark.parametrize(
    "bad",
    [
        torch.zeros((12, 8), dtype=torch.int32),
        torch.zeros((12, 9), dtype=torch.float32),
        torch.full((12, 9), 1026, dtype=torch.int32),
    ],
)
def test_malformed_or_out_of_vocabulary_codes_are_rejected(bad):
    with pytest.raises((ValueError, TypeError)):
        greedy_code_parity(torch.zeros((12, 9), dtype=torch.int32), bad)
