# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Omni-DuplexEval accuracy contract shared by the DFX runner and CPU tests."""

import json
import math
from pathlib import Path

from vllm_omni.benchmarks.duplex.omni_duplex_eval_metrics import PROTOCOL_PIN
from vllm_omni.benchmarks.duplex.omni_duplex_eval_runner import MIX_QUESTION


def select_duplex_eval_baseline(accuracy_config: dict, hardware: object) -> dict:
    """Return this hardware's score floors, or ``{}`` to record scores without gating.

    Nested accuracy-config baselines (``{"H100": {...}, "A3": {...}}``) enforce only
    when the runtime label has a non-empty metric dict. An omitted label or an
    empty ``{}`` entry records scores without a floor. A nested map requires a
    non-empty hardware label: missing or blank labels fail rather than silently
    disabling calibrated gates. Flat metric dicts still apply to every hardware
    label. Mixing hardware maps with flat metric keys is rejected.
    """
    baseline = accuracy_config.get("baseline") or {}
    if not isinstance(baseline, dict) or not baseline:
        return {}
    nested_flags = [isinstance(value, dict) for value in baseline.values()]
    if any(nested_flags) and not all(nested_flags):
        raise ValueError("duplex-eval baseline mixes hardware maps and flat metrics")
    if all(nested_flags):
        if not isinstance(hardware, str) or not hardware:
            raise ValueError("nested duplex-eval baseline requires a runtime hardware label")
        selected = baseline.get(hardware)
        if not isinstance(selected, dict) or not selected:
            return {}
        return {str(key): value for key, value in selected.items() if value is not None}
    return {str(key): value for key, value in baseline.items() if value is not None}


def load_accuracy_config(path: str, params: dict) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("dataset_revision"):
        raise NotImplementedError("dataset_revision pinning is not yet supported")
    assert config["protocol_pin"] == PROTOCOL_PIN, "config protocol_pin does not match source of truth"
    assert params.get("duplex_eval_evaluate") is True, "accuracy guard requires duplex_eval_evaluate"
    assert params["dataset_path"] == config["dataset"], "accuracy dataset differs from CI config"
    expected_ids = {sid for ids in config["sample_ids_per_split"].values() for sid in ids}
    assert set(params["duplex_eval_ids"]) == expected_ids, "benchmark sample IDs differ from CI config"
    assert params["num_prompts"] == [len(expected_ids)], "benchmark sample count differs from CI config"
    assert params.get("duplex_eval_mix", MIX_QUESTION) == config["mix"], "benchmark mix differs from CI config"
    return config


def assert_accuracy_result(benchmark: dict, config: dict, *, enforce_score_floors: bool = True) -> None:
    """Validate the saved summary and selected scores, optionally enforcing calibrated floors."""
    expected_files = {
        Path(split) / f"{sid}.json" for split, ids in config["sample_ids_per_split"].items() for sid in ids
    }
    expected = len(expected_files)
    assert benchmark["completed"] == expected, f"expected {expected} completed requests"
    assert benchmark["failed"] == 0, "failed benchmark requests"
    duplex = benchmark["omni_duplex_eval"]
    assert duplex["published"] == expected, f"expected {expected} published samples"
    assert not duplex["artifact_errors"], f"artifact errors: {duplex['artifact_errors']}"
    summary = duplex["accuracy"]
    assert summary["status"] == "completed", f"evaluation did not complete: {summary}"
    assert summary["total"] == expected, f"expected {expected} total samples"
    assert summary["evaluated"] == expected, f"expected {expected} evaluated samples"
    assert summary["failed"] == 0, "failed evaluations"
    assert summary["skipped"] == 0, "skipped evaluations"
    assert not summary["errors"], f"evaluation errors: {summary['errors']}"
    assert config["protocol_pin"] == PROTOCOL_PIN, "config protocol_pin does not match source of truth"
    assert summary["protocol_pin"] == PROTOCOL_PIN, "summary protocol_pin changed"
    assert summary["samples"] == expected, f"expected {expected} samples"

    score_root = Path(summary["score_root"])
    score_files = [p for p in score_root.rglob("*.json") if not p.name.endswith("_summary.json")]
    assert len(score_files) == expected, f"expected {expected} score files, found {len(score_files)}"
    assert {p.relative_to(score_root) for p in score_files} == expected_files, "score sample IDs differ from CI config"
    for path in score_files:
        row = json.loads(path.read_text(encoding="utf-8"))
        assert row.get("protocol_pin") == PROTOCOL_PIN, f"protocol_pin mismatch in score file: {path}"
        mix = row.get("response_meta", {}).get("mix", MIX_QUESTION)
        assert mix == config["mix"], f"mix {mix!r} in score file differs from CI config: {path}"

    rtd = summary.get("rtd", {})
    pr = summary.get("pr", {})
    assert rtd.get("by_task", {}).keys() >= {
        "RTD_world_knowledge",
        "RTD_counting",
        "RTD_fine_grained_movement",
        "RTD_interaction_relation",
        "RTD_OCR",
        "RTD_Omni",
    }, "RTD by_task keys mismatch"
    assert pr.get("by_task", {}).keys() >= {
        "correction",
        "proactive_reminder",
        "post_event_reminder",
    }, "PR by_task keys mismatch"
    scores = (
        ("RTD content score", rtd.get("mean_content_score"), "min_rtd_mean_content_score", 3.0),
        ("RTD temporal score", rtd.get("mean_avg_temporal_score"), "min_rtd_mean_temporal_score", 3.0),
        ("PR success rate", pr.get("mean_all_success"), "min_pr_mean_all_success", 1.0),
    )
    for label, value, _minimum_key, maximum in scores:
        assert (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and 0 <= value <= maximum
        ), f"{label} {value!r} outside [0, {maximum}]"

    if not enforce_score_floors:
        return
    baseline = config["baseline"]
    for label, value, minimum_key, maximum in scores:
        minimum = baseline[minimum_key]
        assert 0 < minimum <= maximum, f"invalid {label} threshold: {minimum}"
        assert value >= minimum, f"{label} {value!r} outside [{minimum}, {maximum}]"
