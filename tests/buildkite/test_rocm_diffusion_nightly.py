# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU fault injection for the nightly runner, not model qualification."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

RUNNER = Path(".buildkite/amd/scripts/run-rocm-diffusion-nightly.sh")


@pytest.mark.parametrize(
    ("overrides", "expected_status", "executes_tests"),
    [
        ({}, 0, True),
        ({"ROCM_DIFFUSION_RUN_LEVEL": "core_model"}, 0, True),
        ({"COLLECTED": "0"}, 1, False),
        ({"COLLECTED": "4"}, 1, False),
        ({"COLLECTION_STATUS": "2"}, 2, False),
        ({"JUNIT_TESTS": "0"}, 5, True),
        ({"JUNIT_SKIPPED": "1"}, 5, True),
        ({"JUNIT_FAILURES": "1"}, 5, True),
        ({"JUNIT_ERRORS": "1"}, 5, True),
        ({"JUNIT_TESTS": "7"}, 0, True),
        ({"JUNIT_CASES": "4"}, 5, True),
        ({"JUNIT_CHILD": "failure"}, 5, True),
        ({"JUNIT_CHILD": "error"}, 5, True),
        ({"JUNIT_CHILD": "skipped"}, 5, True),
        ({"MISSING_XML": "1"}, 1, True),
        ({"PYTEST_STATUS": "1"}, 1, True),
        ({"TIMEOUT_STATUS": "124"}, 124, False),
        ({"LEAK": "1"}, 1, True),
        ({"LEAK": "1", "PYTEST_STATUS": "2"}, 2, True),
        ({"PS_FAILURE": "1"}, 1, True),
    ],
    ids=[
        "pass",
        "core-model-run-level",
        "empty-collection",
        "missing-case",
        "collection-error",
        "no-results",
        "skip",
        "failure-in-xml",
        "error-in-xml",
        "passing-subtests",
        "missing-testcase",
        "failure-element",
        "error-element",
        "skip-element",
        "missing-xml",
        "pytest-failure",
        "timeout",
        "leaked-server",
        "preserve-pytest-error",
        "process-inspection-error",
    ],
)
def test_nightly_runner_fails_closed(
    tmp_path: Path, overrides: dict[str, str], expected_status: int, executes_tests: bool
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()

    def executable(name: str, body: str) -> None:
        path = fake_bin / name
        path.write_text("#!/bin/bash\nset -euo pipefail\n" + body, encoding="utf-8")
        path.chmod(0o755)

    executable("rocminfo", "exit 0\n")
    executable("git", "exit 1\n")
    executable("rocm-smi", "exit 0\n")
    executable("sleep", "exit 0\n")
    executable(
        "python3",
        'if [[ "$#" == 2 && "$1" == - && "$2" != *.xml ]]; then cat >/dev/null; exit 0; fi\n'
        'if [[ "${1:-}" == -m ]]; then exit 0; fi\n'
        'exec "$REAL_PYTHON" "$@"\n',
    )
    executable(
        "ps",
        'if [[ -f "$STATE/ps-called" ]]; then\n'
        '  if [[ "$PS_FAILURE" == 1 ]]; then exit 1; fi\n'
        '  if [[ "$LEAK" == 1 ]]; then echo "200 1 Mon Nov 2 10:00:01 2026 S python leaked-server"; fi\n'
        "fi\n"
        'echo "100 1 Mon Nov 2 10:00:00 2026 S bash baseline"\n'
        'touch "$STATE/ps-called"\n',
    )
    executable(
        "timeout",
        'shift 3\nif [[ "$1" == pytest && "$TIMEOUT_STATUS" != 0 ]]; then exit "$TIMEOUT_STATUS"; fi\nexec "$@"\n',
    )
    executable(
        "pytest",
        'printf "%s\\n" "$@" >> "$STATE/pytest-args"\n'
        'if [[ "$1" == --collect-only ]]; then\n'
        '  for ((i=0; i<COLLECTED; i++)); do echo "tests/fake.py::test_$i"; done\n'
        '  exit "$COLLECTION_STATUS"\n'
        "fi\n"
        'touch "$STATE/executed"\n'
        'for arg in "$@"; do\n'
        '  if [[ "$arg" == --junitxml=* && "$MISSING_XML" == 0 ]]; then\n'
        '    { printf \'<testsuite tests="%s" skipped="%s" failures="%s" errors="%s">\' \\\n'
        '      "$JUNIT_TESTS" "$JUNIT_SKIPPED" "$JUNIT_FAILURES" "$JUNIT_ERRORS"\n'
        "      for ((i=0; i<JUNIT_CASES; i++)); do\n"
        '        printf \'<testcase classname="tests.fake" name="test_%s">\' "$i"\n'
        '        if [[ -n "$JUNIT_CHILD" ]]; then printf "<%s/>" "$JUNIT_CHILD"; fi\n'
        '        printf "</testcase>"\n'
        "      done\n"
        '      printf "</testsuite>"; } > "${arg#*=}"\n'
        "  fi\n"
        "done\n"
        'exit "$PYTEST_STATUS"\n',
    )
    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "REAL_PYTHON": sys.executable,
        "BUILDKITE_BUILD_CHECKOUT_PATH": str(tmp_path),
        "BUILDKITE_COMMIT": "fixture-buildkite-commit",
        "STATE": str(tmp_path),
        "COLLECTED": "5",
        "COLLECTION_STATUS": "0",
        "JUNIT_TESTS": "5",
        "JUNIT_SKIPPED": "0",
        "JUNIT_FAILURES": "0",
        "JUNIT_ERRORS": "0",
        "JUNIT_CASES": "5",
        "JUNIT_CHILD": "",
        "MISSING_XML": "0",
        "PYTEST_STATUS": "0",
        "TIMEOUT_STATUS": "0",
        "LEAK": "0",
        "PS_FAILURE": "0",
        "ROCM_DIFFUSION_RUN_LEVEL": "full_model",
        **overrides,
    }
    result = subprocess.run(
        [
            "bash",
            str(RUNNER),
            "4",
            "5",
            "20m",
            "nightly-test",
            "tests/fake.py",
            "-m",
            "full_model and diffusion and rocm and MI325 and cards_3",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == expected_status, result.stdout + result.stderr
    assert (tmp_path / "executed").exists() == executes_tests
    args = (tmp_path / "pytest-args").read_text()
    assert f"--run-level\n{env['ROCM_DIFFUSION_RUN_LEVEL']}\n" in args
    assert "--shard-id\n0\n--num-shards\n1\n" in args
    inventory = (tmp_path / "artifacts/nightly-test/runtime-inventory.txt").read_text()
    assert "buildkite_commit=fixture-buildkite-commit" in inventory
    assert "checkout_head=" not in inventory
    assert (tmp_path / "artifacts/nightly-test/process-cleanup.txt").exists() != (overrides.get("PS_FAILURE") == "1")


@pytest.mark.parametrize("outcome", ["pass", "fail", "skip", "missing-case"])
def test_nightly_junit_checks_real_pytest_subtests(tmp_path: Path, outcome: str) -> None:
    """pytest 9 includes subtests in suite totals, but not testcase elements."""
    case = tmp_path / "test_subtests.py"
    action = {"fail": "assert index != 1", "skip": "pytest.skip()"}.get(outcome, "assert index >= 0")
    case.write_text(
        "import pytest\n"
        "def test_plain():\n    assert True\n"
        "def test_subtests(subtests):\n"
        "    for index in range(5):\n"
        "        with subtests.test(index=index):\n"
        f"            {action}\n",
        encoding="utf-8",
    )
    xml = tmp_path / "pytest.xml"
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--noconftest",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            f"--junitxml={xml}",
            str(case),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == (1 if outcome == "fail" else 0), run.stdout + run.stderr
    validator = "import sys\nimport xml.etree.ElementTree as ET\nroot = ET.parse"
    validator += RUNNER.read_text(encoding="utf-8").split("root = ET.parse", 1)[1].split("\nPY", 1)[0]
    checked = subprocess.run(
        [sys.executable, "-c", validator, str(xml), "3" if outcome == "missing-case" else "2"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert checked.returncode == (0 if outcome == "pass" else 5), checked.stdout + checked.stderr
