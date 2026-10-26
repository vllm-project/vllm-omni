# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path
from shlex import split

import pytest
import yaml
from _pytest.mark.expression import Expression

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

AMD_READY_PIPELINE = Path(".buildkite/amd/test-amd-ready.yml")
R2_01_LABEL = "ROCm · Model Executor GPU Coverage (R2-01) · Shard %N/%t"
R2_01_SHARD_SUFFIX = "$${BUILDKITE_PARALLEL_JOB}-of-$${BUILDKITE_PARALLEL_JOB_COUNT}"
ENGINE_LABEL = "Engine Test"
SINGLE_GPU_MARKERS = (
    "core_model and rocm and MI325 and not nvidia_only and not "
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
        ({"core_model", "rocm", "nvidia_only", "MI325", "cards_1"}, False),
        ({"core_model", "rocm", "MI325"}, True),
        ({"core_model", "rocm", "MI325", "cards_1"}, True),
        ({"core_model", "rocm", "MI325", "cards_2"}, False),
        ({"core_model", "rocm", "cards_1"}, False),
        ({"core_model", "cuda", "L4", "cards_1"}, False),
    ],
)
def test_r2_01_marker_selection_contract(marks: set[str], selected: bool) -> None:
    expression = Expression.compile(SINGLE_GPU_MARKERS)
    assert expression.evaluate(marks.__contains__) is selected


def test_r2_01_is_nonblocking_single_gpu_coverage() -> None:
    step = _find_step(R2_01_LABEL)

    assert step["agent_pool"] == "mi300_1"
    assert step["parallelism"] == 2
    assert step["grade"] == "NonBlocking"
    assert step["timeout_in_minutes"] == 50
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["artifact_paths"] == ["artifacts/rocm-r2-01/**/*"]


def test_dedicated_engine_job_preserves_r2_01_engine_coverage() -> None:
    commands = "\n".join(_find_step(ENGINE_LABEL)["commands"])
    assert "tests/engine/test_async_omni_engine_abort.py" in commands


def test_r2_01_runs_once_fails_closed_and_publishes_results() -> None:
    commands = _find_step(R2_01_LABEL)["commands"]
    joined = "\n".join(commands)
    pytest_commands = [command for command in commands if "pytest " in command]
    assert len(pytest_commands) == 1
    run_command = pytest_commands[0]
    argv = split(run_command)
    artifact_dir = "$$BUILDKITE_BUILD_CHECKOUT_PATH/artifacts/rocm-r2-01"

    assert argv[:4] == [
        "timeout",
        "--signal=TERM",
        "--kill-after=2m",
        "45m",
    ]
    _assert_single_gpu_selection(run_command)
    assert "--num-shards=$$BUILDKITE_PARALLEL_JOB_COUNT" in argv
    assert "--shard-id=$$BUILDKITE_PARALLEL_JOB" in argv
    assert "--collect-only" not in joined
    assert "VLLM_CI_ALLOW_NO_TESTS" not in joined
    assert any(artifact_dir in command for command in commands)
    assert "-v" in argv
    assert "-ra" in argv
    assert "--durations=0" in argv
    assert f"--junitxml=$$R2_01_ARTIFACT_DIR/pytest-{R2_01_SHARD_SUFFIX}.xml" in argv
    assert f"$$R2_01_ARTIFACT_DIR/pytest-{R2_01_SHARD_SUFFIX}.log" in run_command
    assert any(f"$$R2_01_ARTIFACT_DIR/pytest-summary-{R2_01_SHARD_SUFFIX}.txt" in command for command in commands)
