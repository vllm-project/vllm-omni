# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path
from shlex import split

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PIPELINE = Path(".buildkite/amd/test-amd-ready.yml")
DISTRIBUTED_LABEL = "ROCm · Distributed Core GPU Coverage"
TINY_BASE_LABEL = "ROCm · Tiny Diffusion Base GPU Coverage"


def _find_step(label: str) -> dict:
    pipeline = yaml.safe_load(PIPELINE.read_text(encoding="utf-8"))

    def walk(steps: list[dict]) -> dict | None:
        for step in steps:
            if step.get("label") == label:
                return step
            if nested := walk(step.get("steps", [])):
                return nested
        return None

    step = walk(pipeline["steps"])
    assert step is not None, f"missing AMD pipeline step: {label}"
    return step


@pytest.mark.parametrize(
    ("label", "artifact_dir"),
    [
        (DISTRIBUTED_LABEL, "artifacts/rocm-distributed-core"),
        (TINY_BASE_LABEL, "artifacts/rocm-tiny-diffusion-base"),
    ],
)
def test_new_ready_jobs_follow_rocm_evidence_contract(label: str, artifact_dir: str) -> None:
    step = _find_step(label)
    commands = step["commands"]
    pytest_commands = [command for command in commands if "pytest " in command]

    assert step["agent_pool"] == "mi300_1"
    assert step["grade"] == "NonBlocking"
    assert step["timeout_in_minutes"] == 30
    assert step["artifact_paths"] == [f"{artifact_dir}/**/*"]
    assert len(pytest_commands) == 1
    assert "--collect-only" not in "\n".join(commands)
    assert "VLLM_CI_ALLOW_NO_TESTS" not in "\n".join(commands)
    assert "environment.txt" in "\n".join(commands)
    assert "torch.accelerator.device_count()" in "\n".join(commands)
    assert "expected {expected_gpus} ROCm GPU(s)" in "\n".join(commands)
    assert "process-cleanup.txt" in "\n".join(commands)
    assert "pytest-summary.txt" in "\n".join(commands)
    assert "--junitxml=" in pytest_commands[0]


def test_distributed_job_matches_current_cuda_scope() -> None:
    command = next(command for command in _find_step(DISTRIBUTED_LABEL)["commands"] if "pytest " in command)
    argv = split(command)
    assert "tests/distributed/" in argv
    assert argv[argv.index("-m") + 1] == "core_model and cuda and L4"
    assert argv[argv.index("--run-level") + 1] == "core_model"


def test_tiny_base_job_matches_current_cuda_scope() -> None:
    command = next(command for command in _find_step(TINY_BASE_LABEL)["commands"] if "pytest " in command)
    argv = split(command)
    assert "tests/model_tests/diffusion/" in argv
    assert argv[argv.index("-m") + 1] == "core_model and cuda"
    assert argv[argv.index("--run-level") + 1] == "core_model"
    assert argv[argv.index("-n") + 1] == "4"
