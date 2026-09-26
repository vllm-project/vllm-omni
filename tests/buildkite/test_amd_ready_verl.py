# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import importlib.util
from pathlib import Path
from shlex import split

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PIPELINE = Path(".buildkite/amd/test-amd-ready.yml")
EVIDENCE_SCRIPT = Path(".buildkite/amd/scripts/rocm_ci_evidence.py")
LABEL = "ROCm · VeRL-Omni E2E Coverage"


def _find_step() -> dict:
    pipeline = yaml.safe_load(PIPELINE.read_text(encoding="utf-8"))
    for item in pipeline["steps"]:
        for step in item.get("steps", []):
            if step.get("label") == LABEL:
                return step
    raise AssertionError(f"missing AMD pipeline step: {LABEL}")


def test_verl_job_matches_ready_contract() -> None:
    step = _find_step()
    commands = step["commands"]
    command_text = "\n".join(commands)
    pytest_commands = [command for command in commands if "pytest -s" in command]

    assert step["agent_pool"] == "mi300_1"
    assert step["grade"] == "NonBlocking"
    assert step["timeout_in_minutes"] == 30
    assert step["artifact_paths"] == ["artifacts/rocm-verl-omni-e2e/**/*"]
    assert len(pytest_commands) == 1
    assert "--collect-only" not in command_text
    assert "VLLM_CI_ALLOW_NO_TESTS" not in command_text
    assert 'python3 -m pip install --no-deps "ray==2.56.1" "omegaconf==2.3.0"' in commands
    for evidence in (
        "dependencies.txt",
        "environment.txt",
        "pytest.log",
        "pytest.xml",
        "pytest-summary.txt",
        "pytest-result.txt",
        "process-cleanup.txt",
    ):
        assert evidence in command_text


def test_verl_job_preserves_cuda_source_command_and_cleanup_headroom() -> None:
    step = _find_step()
    command = next(command for command in step["commands"] if "pytest -s" in command)
    argv = split(command)

    assert "tests/e2e/features/rlhf_test/test_verl_omni_e2e.py" in argv
    assert argv[:5] == ["timeout", "--signal=TERM", "--kill-after=1m", "20m", "pytest"]
    assert step["timeout_in_minutes"] > 20


def test_evidence_helper_fails_closed_and_reports_selection(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("rocm_ci_evidence", EVIDENCE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    xml = tmp_path / "pytest.xml"
    log = tmp_path / "pytest.log"
    output = tmp_path / "pytest-result.txt"
    xml.write_text('<testsuite tests="2" failures="0" errors="0" skipped="1"/>', encoding="utf-8")
    log.write_text("1 passed, 1 skipped, 7 deselected\n", encoding="utf-8")

    module.summarize_pytest([xml], log, output)

    assert output.read_text(encoding="utf-8") == (
        "collected=9 selected=2 passed=1 failed=0 skipped=1 deselected=7 errors=0 executed=1\n"
    )
