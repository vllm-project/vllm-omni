# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Load live CI jobs and exercise their production source-dependency filters."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".buildkite/common/scripts"))
sys.path.insert(0, str(ROOT / ".buildkite/amd/scripts"))

from filter_test_suite import filter_amd_nightly_suite  # noqa: E402
from upload_pipeline import filter_pipeline_by_source_dependencies  # noqa: E402

PIPELINES = {
    "cuda": Path(".buildkite/cuda/test-nightly.yml"),
    "npu": Path(".buildkite/npu/test-npu-nightly.yml"),
    "rocm": Path(".buildkite/amd/test-amd-nightly.yml"),
}


def leaf_steps(doc):
    """Yield runnable steps, recursively excluding pipeline groups."""
    for step in doc.get("steps", []):
        if isinstance(step, dict):
            if "steps" in step:
                yield from leaf_steps(step)
            else:
                yield step


def dependency_keys(step):
    """Normalize a step's registry-key metadata to a list."""
    value = step.get("source_file_dependencies", [])
    return [value] if isinstance(value, str) else value


def evaluation_families(doc):
    """Discover accuracy/performance consumers without pinning job counts."""
    families: dict[str, set[str]] = {}
    for step in leaf_steps(doc):
        for key in dependency_keys(step):
            if key.endswith(("_accuracy", "_perf")) or key == "omni_duplex_eval_ci":
                families.setdefault(key, set()).add(step["label"])
    return families


@lru_cache
def load_pipeline(path: Path):
    """Read one repository pipeline; callers pass it to non-mutating filters."""
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def selected_labels(path, doc, changes):
    """Select jobs through the platform's real filter and verify metadata stripping."""
    if path == PIPELINES["rocm"] and changes is not None:
        filtered = filter_amd_nightly_suite(doc, changes, pipeline_path=path)
    else:
        filtered = filter_pipeline_by_source_dependencies(doc, changes, pipeline_path=path)
    steps = list(leaf_steps(filtered))
    assert all("source_file_dependencies" not in step for step in steps)
    return {step["label"] for step in steps}


@pytest.fixture(name="nightly_pipeline", scope="module", params=PIPELINES)
def nightly_pipeline_fixture(request):
    """Provide the same live platform inventory to all routing test modules."""
    platform = request.param
    path = PIPELINES[platform]
    return platform, path, load_pipeline(path)
