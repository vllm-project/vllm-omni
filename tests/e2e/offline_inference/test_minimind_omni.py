# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E offline inference tests for the dense MiniMind-Omni pipeline."""

from __future__ import annotations

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from pathlib import Path

import pytest

from tests.e2e.offline_inference.test_minimind_omni_moe import (
    assert_real_audio_if_enabled,
    assert_real_text_if_enabled,
    run_minimind_request,
    save_response_outputs,
)
from tests.helpers.mark import hardware_test
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

MODEL = os.environ.get("MINIMIND_OMNI_MODEL", "jingyaogong/minimind-3o")
_CI_DEPLOY = get_deploy_config_path("ci/minimind_omni.yaml")
_OUTPUT_DIR = Path(os.environ.get("MINIMIND_OMNI_OUTPUT_DIR", "/tmp/minimind_omni_outputs"))
_AUDIO_SAMPLE_RATE = int(os.environ.get("MINIMIND_OMNI_SAMPLE_RATE", "24000"))


def get_eager_config() -> str:
    return modify_stage_config(
        _CI_DEPLOY,
        updates={
            "stages": {
                0: {"enforce_eager": True},
                1: {"enforce_eager": True},
                2: {"enforce_eager": True},
            },
        },
    )


test_params = [
    (
        MODEL,
        get_eager_config(),
        {"stage_init_timeout": 600, "init_timeout": 1200},
    )
]


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_text_to_text(omni_runner, run_level) -> None:
    """Test dense MiniMind-Omni thinker text generation."""
    response = run_minimind_request(
        omni_runner,
        prompt="请简单介绍一下你自己",
        modalities=["text"],
    )
    assert response.success
    save_response_outputs(
        response,
        "text_to_text",
        output_dir=_OUTPUT_DIR,
        sample_rate=_AUDIO_SAMPLE_RATE,
        log_prefix="MiniMind-Omni",
    )
    assert_real_text_if_enabled(response, run_level)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_text_to_audio(omni_runner, run_level) -> None:
    """Test dense MiniMind-Omni thinker -> talker -> code2wav generation."""
    response = run_minimind_request(
        omni_runner,
        prompt="请简单介绍一下你自己",
        modalities=["audio"],
    )
    assert response.success
    save_response_outputs(
        response,
        "text_to_audio",
        output_dir=_OUTPUT_DIR,
        sample_rate=_AUDIO_SAMPLE_RATE,
        log_prefix="MiniMind-Omni",
    )
    assert_real_audio_if_enabled(response, run_level)
