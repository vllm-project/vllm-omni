# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path
from shlex import split

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

AMD_MERGE_PIPELINE = Path(".buildkite/amd/test-amd-merge.yml")
AMD_NIGHTLY_PIPELINE = Path(".buildkite/amd/test-amd-nightly.yml")
AMD_READY_PIPELINE = Path(".buildkite/amd/test-amd-ready.yml")
CUDA_MERGE_PIPELINE = Path(".buildkite/cuda/test-merge.yml")
CUDA_READY_PIPELINE = Path(".buildkite/cuda/test-ready.yml")
PI05_LABEL = "Simple · Pi0.5 CPU Test"
PI05_TEST_PATH = "tests/diffusion/models/pi05/test_pi05_units.py"


def _pipeline_steps(pipeline_path: Path) -> list[dict]:
    pipeline = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
    flattened = []

    def walk(steps: list[dict]) -> None:
        for step in steps:
            flattened.append(step)
            walk(step.get("steps", []))

    walk(pipeline.get("steps", []))
    return flattened


def _find_step(label: str, pipeline_path: Path) -> dict:
    step = next((step for step in _pipeline_steps(pipeline_path) if step.get("label") == label), None)
    assert step is not None, f"missing pipeline step: {label}"
    return step


@pytest.mark.parametrize(
    "pipeline_path",
    [AMD_READY_PIPELINE, AMD_MERGE_PIPELINE],
    ids=["ready", "merge"],
)
def test_pi05_cpu_suite_is_excluded_from_amd_pr_gates(pipeline_path: Path) -> None:
    assert all(step.get("label") != PI05_LABEL for step in _pipeline_steps(pipeline_path))

    step = _find_step("Simple · Diffusion Test · Shard %N/%t", pipeline_path)
    pytest_command = next(command for command in step["commands"] if "pytest" in command)
    assert f"--ignore={PI05_TEST_PATH}" in split(pytest_command)


def test_pi05_cpu_suite_runs_nonblocking_in_amd_nightly() -> None:
    steps = [step for step in _pipeline_steps(AMD_NIGHTLY_PIPELINE) if step.get("label") == PI05_LABEL]
    assert len(steps) == 1

    step = steps[0]
    assert step["agent_pool"] == "mi300_1"
    assert step["grade"] == "NonBlocking"
    assert step["timeout_in_minutes"] == 60

    pytest_command = next(command for command in step["commands"] if "pytest" in command)
    argv = split(pytest_command)
    assert PI05_TEST_PATH in argv
    assert argv[argv.index("-m") + 1] == "core_model and cpu"


@pytest.mark.parametrize(
    "pipeline_path",
    [CUDA_READY_PIPELINE, CUDA_MERGE_PIPELINE],
    ids=["ready", "merge"],
)
def test_cuda_pr_gates_retain_pi05_cpu_coverage(pipeline_path: Path) -> None:
    step = _find_step("Simple · Diffusion Test", pipeline_path)
    pytest_command = next(command for command in step["commands"] if "pytest" in command)
    argv = split(pytest_command)

    assert "source_file_dependencies" not in step
    assert "tests/diffusion" in argv
    assert argv[argv.index("-m") + 1] == "core_model and cpu"
    assert f"--ignore={PI05_TEST_PATH}" not in argv
