# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for FrameSimilarityFilter (Phase 2 EVS)."""

from __future__ import annotations

import pytest

from tests.entrypoints.openai_api.conftest_video import (
    make_gradient_jpeg,
    make_jpeg,
)
from vllm_omni.entrypoints.openai.video_frame_filter import FrameSimilarityFilter


class TestFrameSimilarityFilter:
    def test_first_frame_always_retained(self):
        f = FrameSimilarityFilter(threshold=0.99)
        assert f.should_retain(make_jpeg()) is True

    def test_identical_frames_dropped(self):
        f = FrameSimilarityFilter(threshold=0.90)
        frame = make_jpeg(100, 100, 100)
        assert f.should_retain(frame) is True
        assert f.should_retain(frame) is False

    def test_very_different_frames_retained(self):
        f = FrameSimilarityFilter(threshold=0.95)
        assert f.should_retain(make_jpeg(255, 255, 255)) is True
        assert f.should_retain(make_jpeg(0, 0, 0)) is True

    def test_low_threshold_keeps_slightly_different(self):
        f = FrameSimilarityFilter(threshold=0.50)
        assert f.should_retain(make_jpeg(100, 100, 100)) is True
        assert f.should_retain(make_jpeg(110, 110, 110)) is True

    def test_random_frames_all_retained(self):
        f = FrameSimilarityFilter(threshold=0.95)
        for i in range(5):
            assert f.should_retain(make_gradient_jpeg(seed=i)) is True

    def test_reset_clears_state(self):
        f = FrameSimilarityFilter(threshold=0.90)
        frame = make_jpeg(128, 128, 128)
        assert f.should_retain(frame) is True
        assert f.should_retain(frame) is False
        f.reset()
        assert f.should_retain(frame) is True

    def test_stats_counting(self):
        f = FrameSimilarityFilter(threshold=0.90)
        frame = make_jpeg(50, 50, 50)
        f.should_retain(frame)  # retained
        f.should_retain(frame)  # dropped
        f.should_retain(frame)  # dropped
        stats = f.stats
        assert stats["retained_count"] == 1
        assert stats["dropped_count"] == 2
        assert stats["total_count"] == 3
        assert abs(stats["drop_rate"] - 2.0 / 3.0) < 1e-6

    def test_stats_empty(self):
        stats = FrameSimilarityFilter().stats
        assert stats["total_count"] == 0
        assert stats["drop_rate"] == 0.0

    def test_stats_reset(self):
        f = FrameSimilarityFilter(threshold=0.90)
        f.should_retain(make_jpeg())
        f.reset()
        assert f.stats["total_count"] == 0

    def test_invalid_threshold(self):
        with pytest.raises(ValueError, match="threshold"):
            FrameSimilarityFilter(threshold=1.5)
        with pytest.raises(ValueError, match="threshold"):
            FrameSimilarityFilter(threshold=-0.1)

    def test_invalid_thumbnail_size(self):
        with pytest.raises(ValueError, match="thumbnail_size"):
            FrameSimilarityFilter(thumbnail_size=0)

    def test_different_image_sizes_same_colour(self):
        """Filter should handle frames of varying resolutions."""
        f = FrameSimilarityFilter(threshold=0.90)
        small = make_jpeg(100, 100, 100, size=32)
        large = make_jpeg(100, 100, 100, size=256)
        assert f.should_retain(small) is True
        assert f.should_retain(large) is False
