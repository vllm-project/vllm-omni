# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The shared CI guard must retain evidence without duplicating console logs."""

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("command_status", "report", "expected_status"),
    [
        (0, None, 1),
        (0, "<testsuite/>", 1),
        (0, '<testsuite><testcase name="skipped"><skipped/></testcase></testsuite>', 1),
        (0, '<testsuite><testcase name="completed"/></testsuite>', 0),
        (17, None, 17),
        (1, '<testsuite><testcase name="assertion"><failure>original error</failure></testcase></testsuite>', 1),
    ],
)
def test_shared_ci_guard_requires_a_complete_junit_and_preserves_failure(
    tmp_path: Path, command_status: int, report: str | None, expected_status: int
) -> None:
    output = tmp_path / "artifacts"
    policy = tmp_path / "policy.json"
    policy.write_bytes((ROOT / "tests/dfx/lingbot_world/quality.json").read_bytes())
    command = (
        "import sys\nfrom pathlib import Path\n"
        "print('test stdout', flush=True)\n"
        "print('test stderr', file=sys.stderr, flush=True)\n"
        f"report = {report!r}\n"
        "if report is not None: Path(sys.argv[1]).write_text(report)\n"
        f"sys.exit({command_status})\n"
    )
    result = subprocess.run(
        [
            "bash",
            str(ROOT / ".buildkite/common/scripts/run-pytest-ci.sh"),
            str(output),
            sys.executable,
            "-c",
            command,
            str(output / "pytest.xml"),
        ],
        cwd=tmp_path,
        env={**os.environ, "CI_POLICY_FILE": str(policy)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == expected_status, result.stdout + result.stderr
    assert "test stdout" in result.stdout and "test stderr" in result.stderr
    assert (output / "environment.json").is_file()
    assert (output / "policy.json").read_bytes() == policy.read_bytes()
    assert not (output / "run.log").exists()
    junit = ET.parse(output / "pytest.xml")
    if expected_status:
        assert not json.loads((output / "setup-failure.json").read_text())["passed"]
        assert list(junit.iter("failure"))
        if report and "original error" in report:
            assert "original error" in (output / "pytest.xml").read_text()
    else:
        assert not (output / "setup-failure.json").exists()
        assert not list(junit.iter("failure"))
