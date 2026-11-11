# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real raw metrics and independent numerical/semantic accuracy boundaries."""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

import pytest

from benchmarks.lingbot_world.regression import (
    RegressionError,
    check_binary_activity,
    check_quality,
    performance_metrics,
    read_json,
    verify_environment,
)
from vllm_omni.benchmarks.realtime_video.workload import ChunkRecord, compute_metrics

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def document(interval=0.5, num_chunks=40):
    records = [ChunkRecord(i, (i + 1) * interval, interval, 1024, 9 if i == 0 else 12) for i in range(num_chunks)]
    config = {"target_fps": 12, "warmup_chunks": 6, "chunk_slo_ms": 1000, "playback_buffer_chunks": 1}
    metrics = compute_metrics(records, fps=12, warmup_chunks=6, slo_ms=1000, ttfc_s=interval)
    return {"config": config, "sessions": [{"chunks": [r.to_dict() for r in records], "metrics": metrics}]}


@pytest.mark.parametrize("change", ["missing", "reordered", "frames", "nonfinite", "interval", "wall"])
def test_invalid_population_cannot_pass(change):
    data = document()
    chunks = data["sessions"][0]["chunks"]
    if change == "missing":
        chunks.pop()
    elif change == "reordered":
        chunks[1], chunks[2] = chunks[2], chunks[1]
    elif change == "frames":
        chunks[0]["num_frames"] = 12
    elif change == "nonfinite":
        chunks[7]["arrival_s"] = float("nan")
    elif change == "interval":
        chunks[7]["inter_arrival_ms"] = 0.01
    else:
        data["sessions"][0]["metrics"]["wall_seconds"] = 1
    with pytest.raises(RegressionError):
        performance_metrics(data, num_chunks=40, min_steady=30)


def test_raw_chunk_times_override_forged_aggregate():
    data = document(interval=0.8)
    data["sessions"][0]["metrics"]["interval_steady"]["mean_ms"] = 1
    assert performance_metrics(data, num_chunks=40, min_steady=30)["mean_ms"] == pytest.approx(800)


def test_quality_regression_cannot_hide_behind_improved_aesthetics():
    reference = [{"imaging": 70, "aesthetic": 0.5}] * 3
    result = check_quality(reference, {"imaging": 60, "aesthetic": 0.9}, tolerances={"imaging": 1, "aesthetic": 0.01})
    assert not result["passed"]
    assert result["dimensions"]["aesthetic"]["passed"]
    assert not result["dimensions"]["imaging"]["passed"]


def test_quality_requires_repeatability_and_complete_dimensions():
    with pytest.raises(RegressionError, match="Unstable reference"):
        check_quality([{"imaging": v} for v in (70, 65, 72)], {"imaging": 71}, tolerances={"imaging": 1})
    with pytest.raises(RegressionError, match="exact dimension"):
        check_quality([{"imaging": 70}] * 3, {}, tolerances={"imaging": 1})


def test_nonfinite_baseline_json_is_rejected(tmp_path: Path):
    path = tmp_path / "baseline.json"
    path.write_text('{"mean": NaN}')
    with pytest.raises(RegressionError, match="Nonfinite"):
        read_json(path)


def test_evaluator_dependency_lock_is_checked_in_selected_interpreter(tmp_path: Path):
    path = tmp_path / "requirements.txt"
    version = importlib.metadata.version("pytest")
    path.write_text(f"pytest=={version}\n")
    assert verify_environment(Path(sys.executable), path) == {"pytest": version}
    path.write_text("pytest==0.0.0\n")
    with pytest.raises(RegressionError, match="environment differs"):
        verify_environment(Path(sys.executable), path)


def test_unpinned_evaluator_dependencies_are_rejected(tmp_path: Path):
    path = tmp_path / "requirements.txt"
    path.write_text("pytest>=8\n")
    with pytest.raises(RegressionError, match="not exactly pinned"):
        verify_environment(Path(sys.executable), path)


@pytest.mark.parametrize(
    "reference,candidate,passed",
    [
        ([0, 1, 0], [0, 1, 0], True),
        ([0, 1, 0], [1, 0, 0], True),
        ([0, 0, 0], [0, 1, 0], True),
        ([0, 0, 0], [1, 1, 0], False),
        ([1, 1, 1], [1, 0, 0], False),
        ([0, 1, 0], [1, 1, 1], False),
        ([0, 0, 0], [1, 1, 1], False),
        ([1, 1, 1], [0, 0, 0], False),
    ],
)
def test_binary_activity_keeps_one_trial_budget_and_rejects_larger_changes(reference, candidate, passed):
    # Actual H100 build 17431 repeats are [0, 1, 0] in both roles.
    result = check_binary_activity(reference, candidate, tolerance=0.34)
    assert result["passed"] is passed
    assert result["allowed_count_change"] == 1
    assert result["reference"] == reference and result["candidate"] == candidate
    strict = check_binary_activity(reference, candidate, tolerance=0)
    assert strict["passed"] is (sum(reference) == sum(candidate))


@pytest.mark.parametrize("invalid", [[], [0, 1], [0, 1, 0, 1], [0, 0.5, 1], [0, float("nan"), 1], [0, True, 1]])
def test_binary_activity_rejects_invalid_native_classifications_or_missing_starts(invalid):
    for reference, candidate in ((invalid, [0, 0, 0]), ([0, 0, 0], invalid)):
        with pytest.raises(RegressionError):
            check_binary_activity(reference, candidate, tolerance=0.34)


@pytest.mark.parametrize("tolerance", [-1, 1, float("inf"), float("nan"), True])
def test_binary_activity_budget_cannot_disable_the_gate(tolerance):
    with pytest.raises(RegressionError):
        check_binary_activity([0, 0, 0], [1, 1, 1], tolerance=tolerance)


def test_reference_quality_spread_does_not_widen_candidate_budget():
    # H100-generated full rollout scores, obtained with the pinned VBench
    # evaluator. These diagnose calibration; target-runner acceptance is separate.
    reference = [{"background": value} for value in (0.876767968, 0.898090779, 0.894397383)]
    candidate = {"background": 0.872326250}
    with pytest.raises(RegressionError, match="Unstable reference"):
        check_quality(reference, candidate, tolerances={"background": 0.01})
    options = {"tolerances": {"background": 0.01}, "max_reference_spreads": {"background": 0.03}}
    report = check_quality(reference, candidate, **options)
    assert report["passed"]
    dimension = report["dimensions"]["background"]
    assert dimension["minimum"] == pytest.approx(0.866767968)
    assert dimension["candidate_tolerance"] == 0.01
    assert dimension["max_reference_spread"] == 0.03
    assert not check_quality(reference, {"background": 0.86}, **options)["passed"]
    with pytest.raises(RegressionError, match="Unstable reference"):
        check_quality([*reference[:2], {"background": 0.93}], candidate, **options)


@pytest.mark.parametrize("limits", [{}, {"wrong": 0.03}, {"imaging": float("nan")}, {"imaging": -1}])
def test_invalid_quality_reference_spread_limits_fail_closed(limits):
    with pytest.raises(RegressionError):
        check_quality(
            [{"imaging": 0.8}] * 3, {"imaging": 0.8}, tolerances={"imaging": 0.01}, max_reference_spreads=limits
        )
