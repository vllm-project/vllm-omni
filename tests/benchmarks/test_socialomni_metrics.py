# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from benchmarks.socialomni.metrics import (
    JudgeCompletenessError,
    bootstrap_mean_interval,
    compute_socialomni_level1_metrics,
    compute_socialomni_level2_metrics,
    validate_judge_scores,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _scores(a: int, b: int, c: int) -> dict[str, int]:
    return {"gpt-4o": a, "gemini-2.5-pro": b, "qwen3-omni": c}


@pytest.mark.parametrize("score", [True, 25.0, "25", [], None, 80])
def test_judge_scores_reject_invalid_types_and_buckets(score) -> None:
    scores = _scores(25, 50, 75)
    scores["gpt-4o"] = score
    with pytest.raises(JudgeCompletenessError):
        validate_judge_scores(scores, "invalid")


def test_level1_uses_fixed_denominator_and_visibility_strata() -> None:
    metrics = compute_socialomni_level1_metrics(
        [
            {
                "gold_answer": "A",
                "predicted_answer": "A",
                "visibility": "speaker_visible",
                "is_success": True,
            },
            {
                "gold_answer": "B",
                "predicted_answer": "",
                "visibility": "speaker_visible",
                "is_success": False,
            },
            {
                "gold_answer": "C",
                "predicted_answer": "A",
                "visibility": "visibility_mismatch",
                "is_success": True,
            },
            {
                "gold_answer": "D",
                "predicted_answer": "D",
                "visibility": "visibility_mismatch",
                "is_success": True,
            },
        ]
    )

    assert metrics["total_samples"] == 4
    assert metrics["accuracy"] == 0.5
    assert metrics["unparsable_or_failed"] == 1
    assert len(metrics["per_class"]) == 4
    assert metrics["strata"]["speaker_visible"]["accuracy"] == 0.5
    assert metrics["strata"]["visibility_mismatch"]["accuracy"] == 0.5
    assert metrics["visible_minus_mismatch_accuracy"] == 0


def test_level2_computes_paper_metrics() -> None:
    records = [
        {
            "sample_id": "yes-covered",
            "gold_when": "YES",
            "predicted_when": "YES",
            "when_success": True,
            "gold_response": "one",
            "gold_response_success": True,
            "gold_judge_scores": _scores(25, 50, 75),
        },
        {
            "sample_id": "yes-missed",
            "gold_when": "YES",
            "predicted_when": "NO",
            "when_success": True,
            "gold_response": "two",
            "gold_response_success": True,
            "gold_judge_scores": _scores(75, 75, 75),
        },
        {
            "sample_id": "false-positive",
            "gold_when": "NO",
            "predicted_when": "YES",
            "when_success": True,
        },
        {
            "sample_id": "unparsable",
            "gold_when": "NO",
            "predicted_when": "",
            "when_success": True,
        },
    ]

    metrics = compute_socialomni_level2_metrics(records, bootstrap_samples=100)

    assert metrics["when"]["accuracy"] == 0.25
    assert metrics["when"]["unparsable_or_failed"] == 1
    assert metrics["quality"]["qgold"] == 62.5
    assert metrics["quality"]["qens"] == 50
    assert metrics["quality"]["cov_plus"] == 0.5
    assert metrics["quality"]["qens_joint"] == 25


def test_level2_requires_all_three_judges_for_nonempty_response() -> None:
    record = {
        "sample_id": "incomplete",
        "gold_when": "YES",
        "predicted_when": "YES",
        "when_success": True,
        "gold_response": "hello",
        "gold_response_success": True,
        "gold_judge_scores": {"gpt-4o": 75, "qwen3-omni": 75},
    }
    with pytest.raises(JudgeCompletenessError):
        compute_socialomni_level2_metrics([record])


def test_failed_response_scores_zero_without_judges() -> None:
    metrics = compute_socialomni_level2_metrics(
        [
            {
                "sample_id": "empty",
                "gold_when": "YES",
                "predicted_when": "YES",
                "when_success": True,
                "gold_response": "",
                "gold_response_success": False,
                "gold_judge_scores": {},
            }
        ],
        bootstrap_samples=10,
    )
    assert metrics["quality"]["qgold"] == 0
    assert metrics["quality"]["cov_plus"] == 0
    assert metrics["quality"]["qens"] is None


def test_bootstrap_interval_is_deterministic() -> None:
    first = bootstrap_mean_interval([0, 25, 100], seed=7, samples=100)
    second = bootstrap_mean_interval([0, 25, 100], seed=7, samples=100)
    assert first == second
    assert first is not None and 0 <= first["low"] <= first["high"] <= 100
