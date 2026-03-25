# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the SLERP interpolation math in speaker_embedding_interpolation.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Import the slerp function from the example script.
_examples_dir = str(Path(__file__).parent.parent.parent / "examples" / "online_serving" / "qwen3_tts")
sys.path.insert(0, _examples_dir)
from speaker_embedding_interpolation import slerp  # noqa: E402

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestSlerp:
    def test_endpoints(self):
        """t=0 returns v0, t=1 returns v1."""
        v0 = np.random.randn(1024).astype(np.float64)
        v1 = np.random.randn(1024).astype(np.float64)
        np.testing.assert_allclose(slerp(v0, v1, 0.0), v0, atol=1e-6)
        np.testing.assert_allclose(slerp(v0, v1, 1.0), v1, atol=1e-6)

    def test_midpoint_unit_norm(self):
        """Midpoint of two unit vectors should also be approximately unit norm."""
        v0 = np.random.randn(1024)
        v0 /= np.linalg.norm(v0)
        v1 = np.random.randn(1024)
        v1 /= np.linalg.norm(v1)
        mid = slerp(v0, v1, 0.5)
        assert abs(np.linalg.norm(mid) - 1.0) < 0.05

    def test_parallel_vectors_fallback(self):
        """Parallel vectors (omega~0) fall back to lerp without error."""
        v0 = np.ones(1024)
        v1 = np.ones(1024) * 1.001  # nearly parallel
        result = slerp(v0, v1, 0.5)
        expected = 0.5 * v0 + 0.5 * v1
        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_antiparallel_vectors(self):
        """Antiparallel vectors (omega~pi) should not produce NaN."""
        v0 = np.ones(1024)
        v1 = -np.ones(1024)
        result = slerp(v0, v1, 0.5)
        assert not np.any(np.isnan(result))

    def test_output_shape_matches_input(self):
        """Output shape must match input shape."""
        v0 = np.random.randn(2048)
        v1 = np.random.randn(2048)
        assert slerp(v0, v1, 0.3).shape == (2048,)
