# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import subprocess
from pathlib import Path
from shlex import split

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PIPELINE = Path(".buildkite/amd/test-amd-ready.yml")
LABEL = "ROCm · VeRL-Omni E2E Coverage"


def _find_step() -> dict:
    pipeline = yaml.safe_load(PIPELINE.read_text(encoding="utf-8"))

    def walk(steps: list[dict]) -> dict | None:
        for step in steps:
            if step.get("label") == LABEL:
                return step
            if nested := walk(step.get("steps", [])):
                return nested
        return None

    step = walk(pipeline["steps"])
    assert step is not None, f"missing AMD pipeline step: {LABEL}"
    return step


def test_verl_job_is_nonblocking_single_gpu_coverage() -> None:
    step = _find_step()

    assert step["agent_pool"] == "mi300_1"
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["grade"] == "NonBlocking"
    assert step["timeout_in_minutes"] == 30
    assert step["artifact_paths"] == ["artifacts/rocm-verl-omni-e2e/**/*"]


def test_verl_job_runs_existing_e2e_once_and_fails_closed() -> None:
    commands = _find_step()["commands"]
    command_text = "\n".join(commands)
    pytest_commands = [command for command in commands if "pytest" in split(command)]

    assert len(pytest_commands) == 1
    argv = split(pytest_commands[0])
    assert argv[:4] == ["pytest", "-s", "-v", "-ra"]
    assert "tests/e2e/features/rlhf_test/test_verl_omni_e2e.py" in argv
    assert "timeout" not in argv
    assert "--collect-only" not in command_text
    assert "VLLM_CI_ALLOW_NO_TESTS" not in command_text
    assert "--junitxml=$$ROCM_VERL_ARTIFACT_DIR/pytest.xml" in argv
    assert "$$ROCM_VERL_ARTIFACT_DIR/pytest.log" in pytest_commands[0]


def test_verl_job_pins_dependencies_and_retains_cleanup_evidence() -> None:
    commands = _find_step()["commands"]
    command_text = "\n".join(commands)

    assert 'python3 -m pip install --no-deps "ray==2.56.1" "omegaconf==2.3.0"' in commands
    assert 'python3 -m pip show vllm ray omegaconf > "$$ROCM_VERL_ARTIFACT_DIR/dependencies.txt"' in commands
    assert "export DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA" in commands
    for evidence in (
        "dependencies.txt",
        "environment.txt",
        "pytest.log",
        "pytest.xml",
        "pytest-summary.txt",
        "ray-cleanup.txt",
        "process-cleanup.txt",
    ):
        assert evidence in command_text
    assert "status=PASS ray_tmpdir_removed=1" in command_text
    assert "status=PASS leaked_processes=0" in command_text


def test_ray_socket_directory_is_short_unique_and_cleaned() -> None:
    commands = _find_step()["commands"]
    setup = next(command for command in commands if command.startswith("export RAY_TMPDIR="))
    cleanup = next(command for command in commands if command.startswith("trap "))
    script = "\n".join([setup, cleanup, 'test -d "$RAY_TMPDIR"', 'printf "%s\\n" "$RAY_TMPDIR"']).replace("$$", "$")

    paths = []
    for _ in range(2):
        result = subprocess.run(["bash", "-euc", script], capture_output=True, text=True, check=True, timeout=5)
        path = Path(result.stdout.strip())
        assert path.parent == Path("/tmp")
        assert not path.exists()
        socket_path = path / "ray/session_2026-09-30_23-46-27_335763_235/sockets/plasma_store"
        assert len(str(socket_path).encode()) <= 107
        paths.append(path)

    assert paths[0] != paths[1]
