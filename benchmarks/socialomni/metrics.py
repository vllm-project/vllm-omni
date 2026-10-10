# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Metrics for the SocialOmni paper protocol."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

SOCIALOMNI_JUDGE_NAMES = ("gpt-4o", "gemini-2.5-pro", "qwen3-omni")
SOCIALOMNI_SCORE_BUCKETS = frozenset({0, 25, 50, 75, 100})


class JudgeCompletenessError(ValueError):
    """A response has missing or invalid judge scores."""


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def wilson_interval(successes: int, total: int) -> dict[str, float]:
    """Return a two-sided Wilson 95% confidence interval."""
    if not 0 <= successes <= total:
        raise ValueError("Wilson interval requires 0 <= successes <= total")
    if total == 0:
        return {"low": 0.0, "high": 0.0}
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    return {"low": max(0.0, center - radius), "high": min(1.0, center + radius)}


def _classification(gold: Sequence[int], predicted: Sequence[int | None], labels: Sequence[int]) -> dict[str, Any]:
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted lengths differ")
    correct = sum(expected == actual for expected, actual in zip(gold, predicted))
    per_class: dict[str, dict[str, Any]] = {}
    f1s: list[float] = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        f1 = _ratio(2 * precision * recall, precision + recall)
        f1s.append(f1)
        per_class[str(label)] = {
            "support": tp + fn,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "precision_ci95": wilson_interval(tp, tp + fp),
            "recall": recall,
            "recall_ci95": wilson_interval(tp, tp + fn),
            "f1": f1,
        }
    return {
        "total_samples": len(gold),
        "correct": correct,
        "accuracy": _ratio(correct, len(gold)),
        "accuracy_ci95": wilson_interval(correct, len(gold)),
        "macro_f1": _ratio(sum(f1s), len(labels)),
        "per_class": per_class,
        "unparsable_or_failed": sum(value is None for value in predicted),
    }


def _choice(value: object) -> int | None:
    text = str(value or "").strip().upper()
    return ord(text) - ord("A") if text in {"A", "B", "C", "D"} else None


def compute_socialomni_level1_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute fixed-denominator Level 1 metrics and visibility strata."""
    rows = list(records)
    gold = [_choice(row["gold_answer"]) for row in rows]
    if any(value is None for value in gold):
        raise ValueError("invalid Level 1 gold answer")
    predicted = [(_choice(row.get("predicted_answer", "")) if row.get("is_success", True) else None) for row in rows]
    result = _classification(gold, predicted, (0, 1, 2, 3))  # type: ignore[arg-type]
    strata: dict[str, Any] = {}
    for name in ("speaker_visible", "visibility_mismatch"):
        indices = [i for i, row in enumerate(rows) if row["visibility"] == name]
        current = _classification(
            [gold[i] for i in indices],  # type: ignore[list-item]
            [predicted[i] for i in indices],
            (0, 1, 2, 3),
        )
        strata[name] = {key: current[key] for key in ("total_samples", "correct", "accuracy", "accuracy_ci95")}
    result["strata"] = strata
    result["visible_minus_mismatch_accuracy"] = (
        strata["speaker_visible"]["accuracy"] - strata["visibility_mismatch"]["accuracy"]
    )
    return result


def _when(value: object) -> int | None:
    text = str(value or "").strip().upper()
    return 1 if text == "YES" else 0 if text == "NO" else None


def validate_judge_scores(value: object, sample_id: str) -> dict[str, int]:
    """Require one integer score in the allowed buckets from each fixed judge."""
    if not isinstance(value, Mapping) or set(value) != set(SOCIALOMNI_JUDGE_NAMES):
        raise JudgeCompletenessError(f"sample {sample_id!r} requires all judges {SOCIALOMNI_JUDGE_NAMES}")
    scores: dict[str, int] = {}
    for judge in SOCIALOMNI_JUDGE_NAMES:
        score = value[judge]
        if type(score) is not int or score not in SOCIALOMNI_SCORE_BUCKETS:
            raise JudgeCompletenessError(f"sample {sample_id!r} has invalid {judge!r} score {score!r}")
        scores[judge] = int(score)
    return scores


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def bootstrap_mean_interval(values: Sequence[float], *, seed: int, samples: int) -> dict[str, float] | None:
    """Return a deterministic percentile-bootstrap 95% interval for a mean."""
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if not values:
        return None
    generator = random.Random(seed)
    size = len(values)
    means = [sum(values[generator.randrange(size)] for _ in range(size)) / size for _ in range(samples)]
    return {"low": _quantile(means, 0.025), "high": _quantile(means, 0.975)}


def compute_socialomni_level2_metrics(
    records: Iterable[Mapping[str, Any]],
    *,
    bootstrap_seed: int = 20260902,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    """Compute turn-entry and complete three-judge response metrics."""
    rows = list(records)
    when_metrics, gold, predicted = _level2_when_metrics(rows)

    gold_scores: list[float] = []
    conditional_scores: list[float] = []
    joint_scores: list[float] = []
    positive_total = 0
    for index, row in enumerate(rows):
        if gold[index] != 1:
            continue
        positive_total += 1
        response = row.get("gold_response", "")
        has_response = bool(row.get("gold_response_success", False) and isinstance(response, str) and response.strip())
        score = 0.0
        if has_response:
            scores = validate_judge_scores(
                row.get("gold_judge_scores", {}),
                str(row.get("sample_id", index)),
            )
            score = sum(scores.values()) / len(scores)
        gold_scores.append(score)
        covered = predicted[index] == 1 and has_response
        if covered:
            conditional_scores.append(score)
        joint_scores.append(score if covered else 0.0)

    covered = len(conditional_scores)
    qgold = _ratio(sum(gold_scores), len(gold_scores))
    qens = _ratio(sum(conditional_scores), covered) if conditional_scores else None
    cov_plus = _ratio(covered, positive_total)
    quality = {
        "gold_positive_samples": positive_total,
        "covered_positive_samples": covered,
        "qgold": qgold,
        "qens": qens,
        "cov_plus": cov_plus,
        "qens_joint": cov_plus * qens if qens is not None else 0.0,
        "cov_plus_ci95": wilson_interval(covered, positive_total),
        "qgold_ci95": bootstrap_mean_interval(gold_scores, seed=bootstrap_seed, samples=bootstrap_samples),
        "qens_ci95": bootstrap_mean_interval(conditional_scores, seed=bootstrap_seed + 1, samples=bootstrap_samples),
        "qens_joint_ci95": bootstrap_mean_interval(joint_scores, seed=bootstrap_seed + 2, samples=bootstrap_samples),
    }
    return {
        "judge_names": list(SOCIALOMNI_JUDGE_NAMES),
        "bootstrap": {"seed": bootstrap_seed, "samples": bootstrap_samples},
        "when": when_metrics,
        "quality": quality,
    }


def _level2_when_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[int | None], list[int | None]]:
    gold = [_when(row["gold_when"]) for row in rows]
    if any(value is None for value in gold):
        raise ValueError("invalid Level 2 gold decision")
    predicted = [(_when(row.get("predicted_when", "")) if row.get("when_success", True) else None) for row in rows]
    classification = _classification(gold, predicted, (0, 1))  # type: ignore[arg-type]
    positive = classification["per_class"]["1"]
    negative = classification["per_class"]["0"]
    when_metrics = {
        "total_samples": classification["total_samples"],
        "correct": classification["correct"],
        "accuracy": classification["accuracy"],
        "accuracy_ci95": classification["accuracy_ci95"],
        "macro_f1": classification["macro_f1"],
        "unparsable_or_failed": classification["unparsable_or_failed"],
        "positive_precision": positive["precision"],
        "positive_precision_ci95": positive["precision_ci95"],
        "positive_recall": positive["recall"],
        "positive_recall_ci95": positive["recall_ci95"],
        "positive_f1": positive["f1"],
        "confusion": {
            "true_positive": positive["true_positive"],
            "false_positive": positive["false_positive"],
            "false_negative": positive["false_negative"],
            "true_negative": negative["true_positive"],
        },
    }
    return when_metrics, gold, predicted


def compute_socialomni_when_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute Level 2 classification metrics without requiring judge output."""
    metrics, _, _ = _level2_when_metrics(list(records))
    return metrics
