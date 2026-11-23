# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.models.physis.h3.positions import output_frames, phase_aligned_frames, video_positions

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("chunks,frames", [(3, 9), (5, 17), (48, 162)])
def test_whole_sequence_frame_contributions(chunks, frames):
    assert output_frames(chunks) == frames


@pytest.mark.parametrize(
    "chunk,expected",
    [
        (1, tuple(range(2))),
        (19, tuple(range(20))),
        (20, (0, *range(2, 21))),
        (24, (0, *range(1, 20))),
        (48, (0, *range(5, 24))),
    ],
)
def test_clean_sink_and_recent_history_preserve_clock_phase(chunk, expected):
    read = min(chunk, 19)
    effective = phase_aligned_frames(read, 1, chunk)
    assert effective == expected
    assert effective[0] == 0
    assert effective[-1] % 5 == chunk % 5


def test_video_positions_use_h3_temporal_clock():
    positions = video_positions((0, 1, 5, 6), 30, 52, (1, 2, 2), 7, "cpu")
    times = positions.reshape(4, 390, 3)[:, 0, 0]
    expected = torch.tensor((7, 7 + 5 / 3, 7 + 85 / 3, 7 + 90 / 3), dtype=torch.float64)
    torch.testing.assert_close(times, expected, atol=1e-13, rtol=0)
