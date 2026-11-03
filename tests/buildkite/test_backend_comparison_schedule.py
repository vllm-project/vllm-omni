# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise PR-owned comparison routing with local stubs; never submit jobs."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / ".buildkite/cuda/pipeline.yml"
COMMIT = "a" * 40


@pytest.fixture
def comparison_runner(tmp_path: Path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    calls = tmp_path / "calls"
    stub = bindir / "buildkite-agent"
    stub.write_text('#!/bin/bash\nprintf "%s %s\\n" "${0##*/}" "$*" >> "$TEST_CALLS"\n')
    stub.chmod(0o755)
    git = bindir / "git"
    git.write_text('#!/bin/bash\nprintf "%s\\n" "$TEST_CHECKOUT"\n')
    git.chmod(0o755)
    entry = yaml.safe_load(ENTRY.read_text())
    command = next(step for step in entry["steps"] if step["key"] == "upload-backend-comparison")["commands"][0]
    # The existing initial upload unescapes $$ before this command executes.
    command = command.replace("$$", "$")
    env = {
        "PATH": f"{bindir}:/usr/bin:/bin",
        "TEST_CALLS": str(calls),
        "TEST_CHECKOUT": COMMIT,
        "BACKEND_COMPARISON_ONLY": "1",
        "BUILDKITE_SOURCE": "schedule",
        "BUILDKITE_BRANCH": "refs/pull/1994/head",
        "BUILDKITE_PIPELINE_SLUG": "vllm-omni",
        "BUILDKITE_REPO": "https://github.com/vllm-project/vllm-omni.git",
        "BUILDKITE_COMMIT": COMMIT,
    }

    def run(overrides: dict[str, str]):
        result = subprocess.run(
            ["/bin/bash", "-c", command], cwd=tmp_path, env={**env, **overrides}, capture_output=True, text=True
        )
        return result, calls.read_text().splitlines() if calls.exists() else []

    return run


def test_schedule_uploads_only_sglang(comparison_runner):
    result, calls = comparison_runner({})
    assert result.returncode == 0, result.stderr
    assert calls == ["buildkite-agent pipeline upload .buildkite/cuda/test-sglang-omni.yml"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"BACKEND_COMPARISON_ONLY": ""},
        {"BACKEND_COMPARISON_ONLY": "0"},
        {"BUILDKITE_SOURCE": "webhook"},
        {"BUILDKITE_SOURCE": "ui"},
        {"BUILDKITE_SOURCE": "api"},
        {"BUILDKITE_BRANCH": "main"},
        {"BUILDKITE_BRANCH": "congw729:ci/separate-image"},
        {"BUILDKITE_PIPELINE_SLUG": "vllm-omni-npu-ci"},
        {"BUILDKITE_REPO": "https://github.com/congw729/vllm-omni.git"},
        {"BUILDKITE_COMMIT": "HEAD"},
        {"TEST_CHECKOUT": "b" * 40},
        {"NIGHTLY": "1"},
        {"WEEKLY": "1"},
        {"NON_CRITICAL": "1"},
        {"MIRROR_HW": "B200"},
    ],
)
def test_invalid_comparison_stops_without_uploading_any_pipeline(comparison_runner, overrides):
    result, calls = comparison_runner(overrides)
    assert result.returncode != 0
    assert calls == []


def test_existing_label_selected_comparison_still_uploads(comparison_runner):
    result, calls = comparison_runner(
        {"BACKEND_COMPARISON_ONLY": "", "BUILDKITE_BRANCH": "congw729:ci/separate-image", "BUILDKITE_SOURCE": "webhook"}
    )
    assert result.returncode == 0, result.stderr
    assert calls == ["buildkite-agent pipeline upload .buildkite/cuda/test-sglang-omni.yml"]


def test_shared_bootstrap_reaches_the_pr_owned_cuda_entry():
    # The installed init selects v2 only when this file is present. Recheck
    # this contract when rebasing: a new generator needs its own routing.
    assert not (ROOT / ".buildkite/.pipeline_gen_v2").exists()
    assert ENTRY.is_file()


@pytest.mark.parametrize("branch", ["main", "refs/pull/1994/head", "congw729:ci/separate-image"])
@pytest.mark.parametrize("flag", [None, "", "0", "1"])
@pytest.mark.parametrize("comparison_label", [False, True])
def test_only_one_entry_route_is_selected(branch: str, flag: str | None, comparison_label: bool):
    selected = []
    for step in yaml.safe_load(ENTRY.read_text())["steps"]:
        # Evaluate the boolean subset used by these actual YAML expressions;
        # values are fixed test inputs, and no builtins are available.
        expression = step["if"].replace('build.env("BACKEND_COMPARISON_ONLY")', repr(flag))
        expression = expression.replace(
            'build.pull_request.labels includes "backend-comparison-test"', repr(comparison_label)
        )
        expression = expression.replace("build.branch", repr(branch)).replace("null", "None")
        expression = expression.replace("&&", "and").replace("||", "or").replace("!(", "not (")
        if eval(expression, {"__builtins__": {}}):
            selected.append(step["key"])
    if flag or branch == "refs/pull/1994/head" or (branch != "main" and comparison_label):
        assert selected == ["upload-backend-comparison"]
    else:
        assert selected == ["upload-ci-pipeline"]


@pytest.mark.parametrize("checkout_matches", [False, True])
def test_child_jobs_verify_the_uploader_commit_before_work(tmp_path: Path, checkout_matches: bool):
    pipeline = yaml.safe_load((ROOT / ".buildkite/cuda/test-sglang-omni.yml").read_text())
    assert pipeline["env"] == {"BACKEND_COMPARISON_COMMIT": "${BUILDKITE_COMMIT}"}
    stub = tmp_path / "git"
    stub.write_text('#!/bin/bash\nprintf "%s\\n" "$TEST_CHECKOUT"\n')
    stub.chmod(0o755)
    for step in pipeline["steps"]:
        result = subprocess.run(
            ["/bin/bash", "-c", step["commands"][0].replace("$$", "$")],
            env={
                "PATH": f"{tmp_path}{os.pathsep}/usr/bin:/bin",
                "TEST_CHECKOUT": COMMIT if checkout_matches else "b" * 40,
                "BACKEND_COMPARISON_COMMIT": COMMIT,
            },
            capture_output=True,
            text=True,
        )
        assert (result.returncode == 0) == checkout_matches
