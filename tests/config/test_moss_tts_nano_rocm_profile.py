# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pin MOSS-TTS-Nano's ROCm attention-metadata shim."""

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.stage_config import (
    _apply_platform_overrides,
    load_deploy_config,
    merge_pipeline_deploy,
)
from vllm_omni.model_executor.models.moss_tts_nano.pipeline import MOSS_TTS_NANO_PIPELINE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_rocm_supplies_the_nested_gpt2_head_dimension() -> None:
    deploy = _apply_platform_overrides(
        load_deploy_config(get_deploy_config_path("moss_tts_nano.yaml")),
        platform="rocm",
    )
    (stage,) = merge_pipeline_deploy(MOSS_TTS_NANO_PIPELINE, deploy)

    assert stage.yaml_engine_args["hf_overrides"]["head_dim"] == 64


def test_head_dimension_override_is_rocm_only() -> None:
    deploy = _apply_platform_overrides(
        load_deploy_config(get_deploy_config_path("moss_tts_nano.yaml")),
        platform="cuda",
    )

    assert "hf_overrides" not in deploy.stages[0].engine_extras
