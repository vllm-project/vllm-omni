# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path
from shlex import split

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

AMD_READY_PIPELINE = Path(".buildkite/amd/test-amd-ready.yml")
R2_02_LABEL = "ROCm · Entrypoints GPU Coverage (R2-02)"
SINGLE_GPU_MARKERS = (
    "core_model and cuda and not (cards_2 or cards_3 or cards_4 or cards_5 or cards_6 or cards_7 or cards_8)"
)


def _find_step(label: str) -> dict:
    pipeline = yaml.safe_load(AMD_READY_PIPELINE.read_text(encoding="utf-8"))

    def walk(steps: list[dict]) -> dict | None:
        for step in steps:
            if step.get("label") == label:
                return step
            if nested := walk(step.get("steps", [])):
                return nested
        return None

    step = walk(pipeline.get("steps", []))
    assert step is not None, f"missing AMD pipeline step: {label}"
    return step


def test_r2_02_is_nonblocking_bounded_single_gpu_coverage() -> None:
    step = _find_step(R2_02_LABEL)

    assert step["agent_pool"] == "mi300_1"
    assert step["grade"] == "NonBlocking"
    assert step["timeout_in_minutes"] == 30
    assert step["env"]["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["artifact_paths"] == ["artifacts/rocm-r2-02/**/*"]


def test_r2_02_fails_closed_and_publishes_collection_results_and_cleanup() -> None:
    commands = _find_step(R2_02_LABEL)["commands"]
    collect_command = next(command for command in commands if "--collect-only" in command)
    run_command = next(command for command in commands if "--junitxml" in command)
    all_commands = "\n".join(commands)

    for command in (collect_command, run_command):
        argv = split(command)
        assert "tests/entrypoints/" in argv
        assert argv[argv.index("-m") + 1] == SINGLE_GPU_MARKERS
        assert argv[argv.index("--run-level") + 1] == "core_model"

    assert "VLLM_CI_ALLOW_NO_TESTS" not in all_commands
    assert "collected-nodeids.txt" in collect_command
    assert "timeout --signal=TERM --kill-after=1m 25m" in run_command
    assert "-ra" in split(run_command)
    assert "--durations=0" in split(run_command)
    assert "--junitxml=artifacts/rocm-r2-02/pytest.xml" in split(run_command)
    assert all_commands.count("ps -eo pid=,ppid=,lstart=,stat=,comm=") == 2
    assert "processes-before.txt" in all_commands
    assert "processes-after.txt" in all_commands
    assert "process-cleanup.txt" in all_commands
    assert "$$9" in all_commands
    assert "status=FAIL leaked_processes_detected=1" in all_commands
    assert "status=PASS leaked_processes=0" in all_commands
    assert "pytest-summary.txt" in all_commands
