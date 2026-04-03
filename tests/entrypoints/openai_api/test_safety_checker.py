# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import PIL.Image
import pytest


def _make_image(size=(64, 64)):
    return PIL.Image.new("RGB", size, color="red")


def _make_checker_with_mock_pipeline(safety_checker_module, pipeline_output):
    """Create a SafetyChecker with a pre-loaded mock pipeline."""
    SafetyChecker = safety_checker_module.SafetyChecker
    checker = SafetyChecker()
    mock_pipe = MagicMock()
    mock_pipe.return_value = pipeline_output
    checker._pipeline = mock_pipe
    return checker, mock_pipe


@pytest.mark.core_model
@pytest.mark.cpu
class TestSafetyChecker:
    def test_safe_image(self, safety_checker_module):
        checker, _ = _make_checker_with_mock_pipeline(
            safety_checker_module,
            [{"label": "normal", "score": 0.9}, {"label": "nsfw", "score": 0.1}],
        )
        results = checker.check_images([_make_image()])
        assert len(results) == 1
        is_safe, score = results[0]
        assert is_safe is True
        assert score == pytest.approx(0.1)

    def test_unsafe_image(self, safety_checker_module):
        checker, _ = _make_checker_with_mock_pipeline(
            safety_checker_module,
            [{"label": "normal", "score": 0.2}, {"label": "nsfw", "score": 0.8}],
        )
        results = checker.check_images([_make_image()])
        assert len(results) == 1
        is_safe, score = results[0]
        assert is_safe is False
        assert score == pytest.approx(0.8)

    def test_boundary_score(self, safety_checker_module):
        """nsfw score == 0.5 -> is_safe=False (strict less-than)."""
        checker, _ = _make_checker_with_mock_pipeline(
            safety_checker_module,
            [{"label": "normal", "score": 0.5}, {"label": "nsfw", "score": 0.5}],
        )
        results = checker.check_images([_make_image()])
        is_safe, _ = results[0]
        assert is_safe is False

    def test_multiple_images(self, safety_checker_module):
        checker, _ = _make_checker_with_mock_pipeline(
            safety_checker_module,
            [
                [{"label": "normal", "score": 0.9}, {"label": "nsfw", "score": 0.1}],
                [{"label": "normal", "score": 0.2}, {"label": "nsfw", "score": 0.8}],
            ],
        )
        results = checker.check_images([_make_image(), _make_image()])
        assert len(results) == 2
        assert results[0][0] is True  # safe
        assert results[1][0] is False  # unsafe

    def test_check_images_reuses_pipeline(self, safety_checker_module):
        """Multiple check_images calls reuse the same pipeline instance."""
        SafetyChecker = safety_checker_module.SafetyChecker
        checker = SafetyChecker()
        assert checker._pipeline is None

        # First call: _pipeline is None, so _ensure_loaded will set it
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"label": "normal", "score": 0.9}, {"label": "nsfw", "score": 0.1}]

        # Replace _ensure_loaded to inject mock without real transformers import
        def fake_ensure():
            if checker._pipeline is not None:
                return
            checker._pipeline = mock_pipe

        checker._ensure_loaded = fake_ensure
        checker.check_images([_make_image()])
        checker.check_images([_make_image()])
        # Pipeline was set once; second call skipped (early return)
        assert checker._pipeline is mock_pipe
        assert mock_pipe.call_count == 2

    def test_empty_list(self, safety_checker_module):
        SafetyChecker = safety_checker_module.SafetyChecker
        checker = SafetyChecker()
        # Should not trigger model loading
        results = checker.check_images([])
        assert results == []
