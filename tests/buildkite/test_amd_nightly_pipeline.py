# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

AMD_NIGHTLY_PIPELINE = Path(".buildkite/amd/test-amd-nightly.yml")
EXPECTED_LABELS = {
    "Qwen3-Omni Function Expansion",
    "Qwen3-Omni Accuracy",
    "Qwen3-Omni Documentation Examples",
    "Qwen3-Omni AITER-on Smoke",
}
ROCM_SELECTION = "full_model and rocm and MI325 and cards_2"


def _leaf_steps() -> list[dict]:
    pipeline = yaml.safe_load(AMD_NIGHTLY_PIPELINE.read_text(encoding="utf-8"))
    leaves: list[dict] = []

    def walk(steps: list[dict]) -> None:
        for step in steps:
            if "steps" in step:
                walk(step["steps"])
            else:
                leaves.append(step)

    walk(pipeline["steps"])
    return leaves


def test_initial_nightly_inventory_is_explicit_and_non_blocking() -> None:
    steps = _leaf_steps()

    assert {step["label"] for step in steps} == EXPECTED_LABELS
    assert all(step["grade"] == "NonBlocking" for step in steps)
    assert all(step["depends_on"] == "amd-build" for step in steps)
    assert all(step["agent_pool"] == "mi300_2" for step in steps)
    assert all(step["mirror_hardwares"] == ["amdproduction"] for step in steps)
    assert all(step["timeout_in_minutes"] > 0 for step in steps)


def test_nightly_commands_select_rocm_full_model_cases() -> None:
    steps = _leaf_steps()

    for step in steps:
        command = "\n".join(step["commands"])
        assert "pytest " in command
        assert f'-m "{ROCM_SELECTION}"' in command
        assert '--run-level "full_model"' in command

    aiter = next(step for step in steps if step["label"] == "Qwen3-Omni AITER-on Smoke")
    assert "VLLM_ROCM_USE_AITER=1" in "\n".join(aiter["commands"])


def test_accuracy_results_are_published_as_artifacts() -> None:
    accuracy = next(step for step in _leaf_steps() if step["label"] == "Qwen3-Omni Accuracy")

    assert accuracy["artifact_paths"] == ["tests/e2e/accuracy/qwen3_omni/results/qwen_omni_acc/*.json"]
