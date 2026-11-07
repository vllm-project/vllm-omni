# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the NSFW SafetyChecker classifier wrapper."""

import pytest
from PIL import Image

from vllm_omni.entrypoints.openai.safety_checker import SafetyChecker

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _labels(nsfw_score: float) -> list[dict]:
    return [{"label": "normal", "score": 1 - nsfw_score}, {"label": "nsfw", "score": nsfw_score}]


def _checker_with_pipeline(fake_pipeline) -> SafetyChecker:
    checker = SafetyChecker()
    checker._pipeline = fake_pipeline
    return checker


@pytest.mark.parametrize(
    ("nsfw_score", "expected_safe"),
    [(0.1, True), (0.49, True), (0.5, False), (0.9, False)],
)
def test_check_images_threshold(nsfw_score, expected_safe):
    checker = _checker_with_pipeline(lambda images: [_labels(nsfw_score) for _ in images])

    [(is_safe, score)] = checker.check_images([Image.new("RGB", (8, 8))])

    assert is_safe is expected_safe
    assert score == pytest.approx(nsfw_score)


def test_check_images_scores_each_image():
    scores = iter([0.1, 0.8])
    checker = _checker_with_pipeline(lambda images: [_labels(next(scores)) for _ in images])

    results = checker.check_images([Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))])

    assert [is_safe for is_safe, _ in results] == [True, False]


def test_check_images_empty_does_not_load_model():
    assert SafetyChecker().check_images([]) == []
