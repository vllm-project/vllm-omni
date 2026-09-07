# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path
from shlex import split

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

AMD_MERGE_PIPELINE = Path(".buildkite/amd/test-amd-merge.yml")


def _find_step(label: str) -> dict:
    pipeline = yaml.safe_load(AMD_MERGE_PIPELINE.read_text(encoding="utf-8"))

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


def test_qwen3_tts_base_preserves_advanced_model_arguments() -> None:
    step = _find_step("Qwen3-TTS Base E2E Test")
    commands = step["commands"]

    assert all("bash -c" not in command for command in commands)
    pytest_command = next(command for command in commands if "pytest" in command)
    argv = split(pytest_command)

    marker_index = argv.index("-m")
    run_level_index = argv.index("--run-level")
    assert argv[marker_index + 1] == "advanced_model and cuda"
    assert argv[run_level_index + 1] == "advanced_model"
