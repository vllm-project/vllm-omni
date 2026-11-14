# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real raw metrics and independent numerical/semantic accuracy boundaries."""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

import pytest

from vllm_omni.benchmarks.realtime_video.eval import read_json, verify_environment
from vllm_omni.benchmarks.realtime_video.metrics import RegressionError, performance_metrics
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
