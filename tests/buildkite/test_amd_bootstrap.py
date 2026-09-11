# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

REPO_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_PATH = REPO_ROOT / ".buildkite/amd/scripts/bootstrap-amd-omni.sh"


def _run_bootstrap(
    tmp_path: Path,
    *,
    debug_test_yaml: str | None = None,
    curl_status: int = 0,
    jq_status: int = 0,
    labels: tuple[str, ...] = (),
    pull_request: bool = True,
) -> subprocess.CompletedProcess[str]:
    for relative_path in (
        ".buildkite/common/scripts/resolve_skip_ci.sh",
        ".buildkite/amd/scripts/select_test_suites.py",
    ):
        destination = tmp_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative_path, destination)

    # Stop after real suite selection, before rendering or uploading a pipeline.
    (tmp_path / ".buildkite/common/scripts/skip_ci.py").write_text("print('test-skip')\n", encoding="utf-8")
    env = {
        "PATH": os.environ["PATH"],
        "BUILDKITE_PULL_REQUEST": "123" if pull_request else "false",
        "BUILDKITE_BRANCH": "feature" if pull_request else "main",
        "TEST_PYTHON": sys.executable,
        "TEST_CURL_STATUS": str(curl_status),
        "TEST_JQ_STATUS": str(jq_status),
        "TEST_PR_LABELS": "\n".join(labels),
    }
    if debug_test_yaml is not None:
        env["DEBUG_TEST_YAML"] = debug_test_yaml

    # Source the unmodified bootstrap with subprocess-local command doubles.
    script = r"""
curl() {
    echo 'curl called' >&2
    return "$TEST_CURL_STATUS"
}
jq() {
    echo 'jq called' >&2
    printf '%s\n' "$TEST_PR_LABELS"
    return "$TEST_JQ_STATUS"
}
git() { :; }
python3() { "$TEST_PYTHON" "$@"; }
source "$1"
"""
    return subprocess.run(
        ["bash", "-c", script, "test-amd-bootstrap", str(BOOTSTRAP_PATH)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


@pytest.mark.parametrize(("curl_status", "jq_status"), [(22, 0), (0, 4)], ids=["request-failure", "parse-failure"])
@pytest.mark.parametrize(
    ("debug_test_yaml", "expected_levels"),
    [
        ("ready", ["l2"]),
        ("merge", ["l3"]),
        ("ready,merge", ["l2", "l3"]),
        (" MERGE, ready,", ["l3", "l2"]),
    ],
)
def test_debug_override_skips_label_lookup(tmp_path, debug_test_yaml, expected_levels, curl_status, jq_status):
    result = _run_bootstrap(
        tmp_path,
        debug_test_yaml=debug_test_yaml,
        curl_status=curl_status,
        jq_status=jq_status,
    )

    assert result.returncode == 0, result.stderr
    assert "curl called" not in result.stderr
    assert "jq called" not in result.stderr
    assert "legacy ready-suite fallback" not in result.stderr
    assert [
        line.split()[2] for line in result.stdout.splitlines() if line.startswith("Skipping AMD ")
    ] == expected_levels


@pytest.mark.parametrize(("curl_status", "jq_status"), [(22, 0), (0, 4)], ids=["request-failure", "parse-failure"])
@pytest.mark.parametrize("debug_test_yaml", [None, ""], ids=["unset", "empty"])
def test_label_lookup_failure_without_override(tmp_path, debug_test_yaml, curl_status, jq_status):
    result = _run_bootstrap(
        tmp_path,
        debug_test_yaml=debug_test_yaml,
        curl_status=curl_status,
        jq_status=jq_status,
    )

    assert result.returncode == 1
    assert "Could not read PR labels" in result.stderr
    assert "Skipping AMD " not in result.stdout


@pytest.mark.parametrize(
    ("labels", "expected_levels"),
    [
        (("ready",), ["l2"]),
        (("merge-test",), ["l3"]),
        (("ready", "merge-test"), ["l2", "l3"]),
        ((), ["l2"]),
    ],
)
def test_label_lookup_selects_tiers(tmp_path, labels, expected_levels):
    result = _run_bootstrap(tmp_path, labels=labels)

    assert result.returncode == 0, result.stderr
    assert "curl called" in result.stderr
    assert "jq called" in result.stderr
    assert [
        line.split()[2] for line in result.stdout.splitlines() if line.startswith("Skipping AMD ")
    ] == expected_levels


@pytest.mark.parametrize("debug_test_yaml", ["ready,ready", "nightly", ", ,", " \t"])
def test_invalid_debug_override_fails_without_label_lookup(tmp_path, debug_test_yaml):
    result = _run_bootstrap(tmp_path, debug_test_yaml=debug_test_yaml, curl_status=22)

    assert result.returncode == 1
    assert "DEBUG_TEST_YAML" in result.stderr
    assert "curl called" not in result.stderr
    assert "jq called" not in result.stderr
    assert "Skipping AMD " not in result.stdout


def test_main_does_not_lookup_labels(tmp_path):
    result = _run_bootstrap(tmp_path, pull_request=False, curl_status=22, jq_status=4)

    assert result.returncode == 0, result.stderr
    assert "curl called" not in result.stderr
    assert "jq called" not in result.stderr
    assert "Skipping AMD l3 suite" in result.stdout
