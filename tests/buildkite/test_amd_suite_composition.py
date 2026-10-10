# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

REPO_ROOT = Path(__file__).resolve().parents[2]
COMBINER_PATH = REPO_ROOT / ".buildkite/amd/scripts/combine_test_suites.py"
SPEC = importlib.util.spec_from_file_location("combine_test_suites", COMBINER_PATH)
assert SPEC is not None and SPEC.loader is not None
COMBINER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMBINER)

AMD_READY_PIPELINE = REPO_ROOT / ".buildkite/amd/test-amd-ready.yml"
AMD_MERGE_PIPELINE = REPO_ROOT / ".buildkite/amd/test-amd-merge.yml"


def _flatten_steps(pipeline: dict) -> list[dict]:
    return [step for entry in pipeline["steps"] for step in entry.get("steps", [entry])]


def _physical_job_count(steps: list[dict]) -> int:
    return sum(step.get("parallelism", 1) for step in steps)


def test_combiner_deduplicates_only_structurally_identical_steps(tmp_path, monkeypatch):
    ready = {
        "env": {"SHARED": "value"},
        "steps": [
            {"label": "identical", "command": "same"},
            {"label": "same name", "command": "core"},
            {"label": "ready only", "command": "ready"},
        ],
    }
    merge = {
        "env": {"SHARED": "value"},
        "steps": [
            {"label": "identical", "command": "same"},
            {"label": "same name", "command": "advanced"},
            {"label": "merge only", "command": "merge"},
        ],
    }
    (tmp_path / "ready.yml").write_text(yaml.safe_dump(ready), encoding="utf-8")
    (tmp_path / "merge.yml").write_text(yaml.safe_dump(merge), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    combined = COMBINER.combine_test_suites(("READY:ready.yml", "MERGE:merge.yml"))

    assert combined["env"] == {"SHARED": "value"}
    assert [group["group"] for group in combined["steps"]] == ["READY", "MERGE"]
    assert [step["label"] for step in combined["steps"][0]["steps"]] == [
        "identical",
        "same name",
        "ready only",
    ]
    assert [step["label"] for step in combined["steps"][1]["steps"]] == [
        "same name",
        "merge only",
    ]


def test_combiner_rejects_conflicting_env(tmp_path, monkeypatch):
    ready = {"env": {"SHARED": "ready"}, "steps": []}
    merge = {"env": {"SHARED": "merge"}, "steps": []}
    (tmp_path / "ready.yml").write_text(yaml.safe_dump(ready), encoding="utf-8")
    (tmp_path / "merge.yml").write_text(yaml.safe_dump(merge), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="Conflicting environment value for SHARED"):
        COMBINER.combine_test_suites(("READY:ready.yml", "MERGE:merge.yml"))


def test_ready_and_merge_composition_removes_redundant_jobs():
    ready = yaml.safe_load(AMD_READY_PIPELINE.read_text(encoding="utf-8"))
    merge = yaml.safe_load(AMD_MERGE_PIPELINE.read_text(encoding="utf-8"))
    raw_steps = _flatten_steps(ready) + _flatten_steps(merge)

    combined = COMBINER.combine_test_suites(
        (
            f"READY_TESTS:{AMD_READY_PIPELINE}",
            f"MERGE_TESTS:{AMD_MERGE_PIPELINE}",
        )
    )
    combined_steps = _flatten_steps(combined)

    # Pin the live ready + merge composition: 56 raw physical jobs minus the
    # 13 jobs from 8 shared step definitions. Update both counts when either
    # suite adds or removes jobs.
    assert _physical_job_count(raw_steps) == 56
    assert _physical_job_count(combined_steps) == 43
    assert all(step not in combined_steps[:index] for index, step in enumerate(combined_steps))

    labels = [step["label"] for step in combined_steps]
    assert len(labels) == len(set(labels))
    assert "Diffusion · Wan22 Core Test" in labels
    assert "Diffusion · Wan22 Advanced Test" in labels
    assert "Engine Test · Ready" in labels
    assert "Engine Test · Merge" in labels
