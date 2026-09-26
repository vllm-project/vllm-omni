# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PIPELINE = Path(".buildkite/amd/test-amd-ready.yml")
JOBS = {
    "ROCm · MiniCPM-o 4.5 Offline Coverage": (
        30,
        "20m",
        "artifacts/rocm-minicpmo-45-offline",
        "tests/e2e/offline_inference/test_minicpmo_4_5.py",
        "core_model",
    ),
    "ROCm · MiniCPM-o 4.5 Online Coverage": (
        30,
        "20m",
        "artifacts/rocm-minicpmo-45-online",
        "tests/e2e/online_serving/test_minicpmo_4_5.py",
        "core_model",
    ),
    "ROCm · MiniCPM-o 4.5 Duplex Coverage": (
        45,
        "35m",
        "artifacts/rocm-minicpmo-45-duplex",
        "tests/e2e/online_serving/test_minicpmo_4_5_duplex.py",
        "core_model and cuda",
    ),
}


def _find_step(label: str) -> dict:
    pipeline = yaml.safe_load(PIPELINE.read_text(encoding="utf-8"))
    for item in pipeline["steps"]:
        for step in item.get("steps", []):
            if step.get("label") == label:
                return step
    raise AssertionError(f"missing AMD pipeline step: {label}")


@pytest.mark.parametrize(("label", "contract"), JOBS.items())
def test_minicpmo_jobs_match_ready_contract(label: str, contract: tuple[int, str, str, str, str]) -> None:
    timeout, pytest_timeout, artifact_dir, test_path, markers = contract
    step = _find_step(label)
    commands = step["commands"]
    command_text = "\n".join(commands)
    pytest_commands = [command for command in commands if "pytest -s" in command]

    assert step["agent_pool"] == "mi300_1"
    assert step["grade"] == "NonBlocking"
    assert step["timeout_in_minutes"] == timeout
    assert step["artifact_paths"] == [f"{artifact_dir}/**/*"]
    assert len(pytest_commands) == 1
    assert f"--kill-after=1m {pytest_timeout}" in pytest_commands[0]
    assert test_path in pytest_commands[0]
    assert f"-m '{markers}'" in pytest_commands[0]
    assert "--run-level 'core_model'" in pytest_commands[0]
    assert "--collect-only" not in command_text
    assert "VLLM_CI_ALLOW_NO_TESTS" not in command_text
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


def test_duplex_job_runs_server_and_live_client_scopes_together() -> None:
    commands = "\n".join(_find_step("ROCm · MiniCPM-o 4.5 Duplex Coverage")["commands"])
    assert "tests/e2e/online_serving/test_minicpmo_4_5_duplex.py" in commands
    assert "tests/e2e/online_serving/test_duplex_client_live.py" in commands
