# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

REPO_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_PATH = REPO_ROOT / ".buildkite/amd/scripts/bootstrap-amd-omni.sh"
COMBINER_PATH = REPO_ROOT / ".buildkite/amd/scripts/combine_test_suites.py"
COMBINER_SPEC = importlib.util.spec_from_file_location(
    "combine_test_suites",
    COMBINER_PATH,
)
assert COMBINER_SPEC is not None and COMBINER_SPEC.loader is not None
COMBINER = importlib.util.module_from_spec(COMBINER_SPEC)
COMBINER_SPEC.loader.exec_module(COMBINER)


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


@pytest.mark.parametrize("debug_test_yaml", ["ready,ready", "nightly,nightly", ", ,", " \t"])
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


def _write_suite(path: Path, suite: dict) -> None:
    path.write_text(yaml.safe_dump(suite), encoding="utf-8")


def test_combiner_deduplicates_only_identical_cross_suite_steps(
    tmp_path,
    capsys,
):
    shared = {
        "label": "shared",
        "parallelism": 2,
        "commands": ["pytest tests/shared.py"],
    }
    ready_steps = [
        shared,
        {"label": "same label", "commands": ["pytest tests/ready.py"]},
        {"label": "ready only", "commands": ["pytest tests/ready_only.py"]},
    ]
    merge_steps = [
        shared,
        {"label": "same label", "commands": ["pytest tests/merge.py"]},
        {"label": "merge only", "commands": ["pytest tests/merge_only.py"]},
    ]
    ready_path = tmp_path / "ready.yml"
    merge_path = tmp_path / "merge.yml"
    _write_suite(
        ready_path,
        {"env": {"SHARED": "value"}, "steps": [{"group": "ready", "steps": ready_steps}]},
    )
    _write_suite(
        merge_path,
        {"env": {"SHARED": "value"}, "steps": [{"group": "merge", "steps": merge_steps}]},
    )

    combined = COMBINER.combine_test_suites(
        (
            f"READY_TESTS:{ready_path}",
            f"MERGE_TESTS:{merge_path}",
        )
    )

    assert combined["env"] == {"SHARED": "value"}
    assert combined["steps"] == [
        {"group": "READY_TESTS", "steps": ready_steps},
        {"group": "MERGE_TESTS", "steps": merge_steps[1:]},
    ]
    assert "Deduplicated 1 identical MERGE_TESTS step definitions (2 jobs)." in capsys.readouterr().err


@pytest.mark.parametrize(
    "difference",
    [
        {"commands": ["pytest tests/other.py"]},
        {"grade": "NonBlocking"},
        {"env": {"BACKEND": "other"}},
        {"timeout_in_minutes": 30},
        {"retry": {"automatic": [{"exit_status": 134, "limit": 1}]}},
        {"artifact_paths": ["artifacts/other/*.xml"]},
        {"agent_pool": "mi300_2"},
        {"mirror_hardwares": ["amdtentative"]},
        {"parallelism": 2},
        {"depends_on": "another-build"},
    ],
)
def test_combiner_retains_same_label_with_a_different_execution_contract(tmp_path, difference):
    ready = {"label": "shared", "commands": ["pytest tests/shared.py"], "grade": "Blocking"}
    merge = ready | difference
    ready_path = tmp_path / "ready.yml"
    merge_path = tmp_path / "merge.yml"
    _write_suite(ready_path, {"steps": [ready]})
    _write_suite(merge_path, {"steps": [merge]})

    combined = COMBINER.combine_test_suites((f"READY_TESTS:{ready_path}", f"MERGE_TESTS:{merge_path}"))

    assert combined["steps"] == [
        {"group": "READY_TESTS", "steps": [ready]},
        {"group": "MERGE_TESTS", "steps": [merge]},
    ]


def test_combiner_preserves_duplicates_within_one_suite(tmp_path):
    shared = {"label": "repeat", "commands": ["pytest tests/shared.py"]}
    ready_path = tmp_path / "ready.yml"
    merge_path = tmp_path / "merge.yml"
    _write_suite(ready_path, {"steps": [shared, shared]})
    _write_suite(merge_path, {"steps": [shared]})

    combined = COMBINER.combine_test_suites((f"READY_TESTS:{ready_path}", f"MERGE_TESTS:{merge_path}"))

    assert combined["steps"] == [{"group": "READY_TESTS", "steps": [shared, shared]}]


def test_combiner_accepts_every_selector_tier(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "select_test_suites", REPO_ROOT / ".buildkite/amd/scripts/select_test_suites.py"
    )
    assert spec is not None and spec.loader is not None
    selector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selector)
    specs = []
    expected_groups = []
    for suite_spec in selector.SUITE_SPECS.values():
        group, filename = suite_spec.split(":", 1)
        path = tmp_path / filename
        _write_suite(path, {"steps": [{"label": group, "commands": [f"pytest tests/{group}.py"]}]})
        specs.append(f"{group}:{path}")
        expected_groups.append(group)

    output_path = tmp_path / "combined.yml"
    result = subprocess.run(
        [sys.executable, str(COMBINER_PATH), str(output_path), *specs],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    assert not result.stderr
    combined = yaml.safe_load(output_path.read_text(encoding="utf-8"))

    assert [group["group"] for group in combined["steps"]] == expected_groups


def test_combiner_rejects_conflicting_suite_environment(tmp_path):
    ready_path = tmp_path / "ready.yml"
    merge_path = tmp_path / "merge.yml"
    step = {"label": "test", "commands": ["pytest tests/test.py"]}
    _write_suite(ready_path, {"env": {"SHARED": "ready"}, "steps": [step]})
    _write_suite(merge_path, {"env": {"SHARED": "merge"}, "steps": [step]})

    with pytest.raises(ValueError, match="Conflicting environment value for SHARED"):
        COMBINER.combine_test_suites(
            (
                f"READY_TESTS:{ready_path}",
                f"MERGE_TESTS:{merge_path}",
            )
        )


@pytest.mark.parametrize(
    "suite_specs",
    [
        (),
        ("UNKNOWN:suite.yml",),
        ("READY_TESTS",),
        ("READY_TESTS:",),
    ],
)
def test_combiner_rejects_unknown_or_malformed_suite_specs(suite_specs):
    with pytest.raises(ValueError):
        COMBINER.combine_test_suites(suite_specs)


def test_combiner_rejects_duplicate_suite_groups(tmp_path):
    suite_path = tmp_path / "ready.yml"
    _write_suite(
        suite_path,
        {"steps": [{"label": "test", "commands": ["pytest tests/test.py"]}]},
    )

    with pytest.raises(ValueError, match="Duplicate AMD test suite group"):
        COMBINER.combine_test_suites(
            (
                f"READY_TESTS:{suite_path}",
                f"READY_TESTS:{suite_path}",
            )
        )
