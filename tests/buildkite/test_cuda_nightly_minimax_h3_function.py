# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CUDA Nightly Function Test wiring for the MiniMax-H3 Ref2VA partition boot."""

from pathlib import Path
from shlex import split

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CUDA_NIGHTLY_PIPELINE = Path(".buildkite/cuda/test-nightly.yml")
X2V_GROUP = ":card_index_dividers: Diffusion X2V Model Test"
FUNCTION_LABEL = ":full_moon: Diffusion X2V · MiniMax H3 Function Test"
MODULE = "tests/e2e/online_serving/minimax_h3/test_minimax_h3_dlo_dp2_fl2va_ref2va.py"


def test_cuda_nightly_minimax_h3_function_collects_ref2va_partition() -> None:
    pipeline = yaml.safe_load(CUDA_NIGHTLY_PIPELINE.read_text(encoding="utf-8"))
    group = next(step for step in pipeline["steps"] if step.get("group") == X2V_GROUP)
    matches = [step for step in group.get("steps", []) if step.get("label") == FUNCTION_LABEL]
    assert len(matches) == 1

    pytest_command = next(command for command in matches[0]["commands"] if "pytest" in command)
    argv = split(pytest_command)
    # The partition case is full_model. --run-level does not select tests, so
    # an advanced_model-only expression would silently deselect it.
    markers = argv[argv.index("-m") + 1]
    assert "full_model" in markers and "advanced_model" not in markers
    assert MODULE in argv
    assert "test_minimax_h3_ref2va_partition" in argv
    assert argv[argv.index("--run-level") + 1] == "full_model"
