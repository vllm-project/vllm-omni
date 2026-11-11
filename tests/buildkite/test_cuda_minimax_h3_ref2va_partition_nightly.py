# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CUDA Nightly wiring for the MiniMax-H3 Ref2VA DP2+DLO partition boot."""

from pathlib import Path
from shlex import split

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CUDA_NIGHTLY_PIPELINE = Path(".buildkite/cuda/test-nightly.yml")
PARTITION_LABEL = ":full_moon: Diffusion · MiniMax-H3 Ref2VA DP2 DLO Partition · 2-GPU"
PARTITION_FILE = "tests/e2e/online_serving/minimax_h3/test_minimax_h3_ref2va_partition.py"
# The test module is marked advanced_model, not full_model. --run-level does
# not select tests, so a full_model marker expression would collect nothing.
PARTITION_MARKERS = "advanced_model and cuda and diffusion and H100 and cards_2"


def _iter_steps(steps: list | None):
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        yield step
        yield from _iter_steps(step.get("steps"))


def test_cuda_nightly_runs_minimax_h3_ref2va_dp2_partition() -> None:
    pipeline = yaml.safe_load(CUDA_NIGHTLY_PIPELINE.read_text(encoding="utf-8"))
    step = next(item for item in _iter_steps(pipeline["steps"]) if item.get("label") == PARTITION_LABEL)

    assert step["source_file_dependencies"] == "diffusion_minimax_h3_function"
    assert step["mirror_hardwares"] == "h100_2"
    assert step["timeout_in_minutes"] == 60

    pytest_command = next(command for command in step["commands"] if "pytest" in command)
    argv = split(pytest_command)
    assert argv[:2] == ["timeout", "35m"]
    assert PARTITION_FILE in argv
    assert argv[argv.index("-m") + 1] == PARTITION_MARKERS
    assert argv[argv.index("--run-level") + 1] == "full_model"
