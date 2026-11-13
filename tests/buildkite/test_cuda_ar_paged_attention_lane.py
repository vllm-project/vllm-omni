# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path
from shlex import split

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CUDA_MERGE_PIPELINE = Path(".buildkite/cuda/test-merge.yml")
DIFFUSION_GROUP = ":card_index_dividers: Diffusion Test"
AR_PAGED_ATTENTION_LABEL = "Diffusion · AR Paged Attention GPU Test"


def test_cuda_merge_keeps_ar_paged_attention_gpu_coverage() -> None:
    pipeline = yaml.safe_load(CUDA_MERGE_PIPELINE.read_text(encoding="utf-8"))
    group = next(step for step in pipeline["steps"] if step.get("group") == DIFFUSION_GROUP)
    lane = next(step for step in group["steps"] if step.get("label") == AR_PAGED_ATTENTION_LABEL)

    assert lane["timeout_in_minutes"] == 20
    assert lane["mirror_hardwares"] == "l4_1"
    assert "export VLLM_OMNI_AR_FA_REQUIRED=1" in lane["commands"]

    pytest_command = next(command for command in lane["commands"] if "pytest" in command)
    argv = split(pytest_command)
    assert argv[:2] == ["timeout", "15m"]
    assert "tests/diffusion/ar_diffusion/test_paged_attention.py" in argv
    assert argv[argv.index("-m") + 1] == "core_model and cuda and L4 and cards_1"
    assert argv[argv.index("--run-level") + 1] == "core_model"
