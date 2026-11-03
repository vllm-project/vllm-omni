# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Execute the installed bootstrap with local command stubs; never submit jobs."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = ROOT / ".buildkite/cuda/bootstrap-backend-comparison.yml"
COMMIT = "a" * 40


@pytest.fixture
def bootstrap_runner(tmp_path: Path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    calls = tmp_path / "calls"
    for name in ("buildkite-agent", "pip", "pipeline-generator"):
        stub = bindir / name
        stub.write_text('#!/bin/bash\nprintf "%s %s\\n" "${0##*/}" "$*" >> "$TEST_CALLS"\n')
        stub.chmod(0o755)
    git = bindir / "git"
    git.write_text('#!/bin/bash\nprintf "%s\\n" "$TEST_CHECKOUT"\n')
    git.chmod(0o755)
    command = yaml.safe_load(BOOTSTRAP.read_text())["steps"][0]["commands"][0]
    # Buildkite's Settings editor unescapes $$ before the shell executes it.
    command = command.replace("$$", "$").replace(
        "/var/lib/buildkite-agent/.local/bin/pipeline-generator", "pipeline-generator"
    )
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

    def run(overrides: dict[str, str], *, generator_v2: bool = False):
        if generator_v2:
            (tmp_path / ".buildkite").mkdir()
            (tmp_path / ".buildkite/.pipeline_gen_v2").touch()
        result = subprocess.run(
            ["/bin/bash", "-c", command], cwd=tmp_path, env={**env, **overrides}, capture_output=True, text=True
        )
        return result, calls.read_text().splitlines() if calls.exists() else []

    return run


@pytest.mark.parametrize("generator_v2", [False, True])
def test_schedule_uploads_only_sglang_before_either_regular_generator(bootstrap_runner, generator_v2):
    result, calls = bootstrap_runner({}, generator_v2=generator_v2)
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
def test_invalid_comparison_stops_without_uploading_any_pipeline(bootstrap_runner, overrides):
    result, calls = bootstrap_runner(overrides, generator_v2=True)
    assert result.returncode != 0
    assert calls == []


def test_existing_nightly_still_uses_regular_entry(bootstrap_runner):
    result, calls = bootstrap_runner({"BACKEND_COMPARISON_ONLY": "", "BUILDKITE_BRANCH": "main", "NIGHTLY": "1"})
    assert result.returncode == 0, result.stderr
    assert calls == ["buildkite-agent pipeline upload .buildkite/cuda/pipeline.yml"]


def test_existing_v2_generator_route_is_preserved(bootstrap_runner):
    result, calls = bootstrap_runner({"BACKEND_COMPARISON_ONLY": "", "BUILDKITE_BRANCH": "main"}, generator_v2=True)
    assert result.returncode == 0, result.stderr
    assert [call.split()[0] for call in calls] == ["pip", "pipeline-generator", "buildkite-agent", "buildkite-agent"]
    assert calls[-1] == "buildkite-agent pipeline upload .buildkite/pipeline.yaml"


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
