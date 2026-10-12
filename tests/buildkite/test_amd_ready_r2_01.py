# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path
from shlex import split

import pytest
import yaml
from _pytest.mark.expression import Expression

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

AMD_READY_PIPELINE = Path(".buildkite/amd/test-amd-ready.yml")
R2_01_LABEL = "ROCm · Model Executor GPU Coverage (R2-01)"
ENGINE_LABEL = "Engine Test"
SINGLE_GPU_MARKERS = (
    "core_model and cuda and not nvidia_only and not "
    "(cards_2 or cards_3 or cards_4 or cards_5 or cards_6 or cards_7 or cards_8)"
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


def _assert_single_gpu_selection(command: str) -> None:
    argv = split(command)
    assert "tests/engine/" not in argv
    assert "tests/model_executor/" in argv
    assert "tests/worker/test_batched_omni_output.py" not in argv
    assert argv[argv.index("-m") + 1] == SINGLE_GPU_MARKERS
    assert "-k" not in argv
    assert argv[argv.index("--run-level") + 1] == "core_model"


@pytest.mark.parametrize(
    ("marks", "selected"),
    [
        ({"core_model", "cuda", "nvidia_only", "L4", "cards_1"}, False),
        ({"core_model", "cuda"}, True),
        ({"core_model", "cuda", "L4", "MI325", "cards_1"}, True),
        ({"core_model", "cuda", "cards_2"}, False),
    ],
)
def test_r2_01_marker_selection_contract(marks: set[str], selected: bool) -> None:
    expression = Expression.compile(SINGLE_GPU_MARKERS)
    assert expression.evaluate(marks.__contains__) is selected


def test_r2_01_is_nonblocking_single_gpu_coverage() -> None:
    step = _find_step(R2_01_LABEL)

    assert step["agent_pool"] == "mi300_1"
    assert step["grade"] == "NonBlocking"
    assert step["timeout_in_minutes"] == 20
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["artifact_paths"] == ["artifacts/rocm-r2-01/**/*"]


def test_dedicated_engine_job_preserves_r2_01_engine_coverage() -> None:
    commands = "\n".join(_find_step(ENGINE_LABEL)["commands"])
    assert "tests/engine/test_async_omni_engine_abort.py" in commands


def test_r2_01_runs_once_fails_closed_and_publishes_results() -> None:
    commands = _find_step(R2_01_LABEL)["commands"]
    pytest_commands = [command for command in commands if command.lstrip().startswith("pytest ")]
    assert len(pytest_commands) == 1
    run_command = pytest_commands[0]
    artifact_dir = "$$BUILDKITE_BUILD_CHECKOUT_PATH/artifacts/rocm-r2-01"

    _assert_single_gpu_selection(run_command)
    assert "--collect-only" not in "\n".join(commands)
    assert "VLLM_CI_ALLOW_NO_TESTS" not in "\n".join(commands)
    assert any(artifact_dir in command for command in commands)
    assert "-v" in split(run_command)
    assert "-ra" in split(run_command)
    assert "--durations=0" in split(run_command)
    assert "--junitxml=$$R2_01_ARTIFACT_DIR/pytest.xml" in split(run_command)
    assert "$$R2_01_ARTIFACT_DIR/pytest.log" in run_command
    assert any("$$R2_01_ARTIFACT_DIR/pytest-summary.txt" in command for command in commands)
