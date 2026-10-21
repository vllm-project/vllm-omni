# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Resolve the Qwen3-Omni duplex Model Runner V2 deployment contract."""

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.omni_config import VllmOmniConfig
from vllm_omni.config.stage_config import (
    _apply_platform_overrides,
    load_deploy_config,
)
from vllm_omni.model_executor.models.qwen3_omni.pipeline import QWEN3_OMNI_PIPELINE
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DEPLOY = "qwen3_omni_duplex_mrv2.yaml"


def _resolve_cuda_stages(monkeypatch):
    # Resolve CUDA overrides even when this test runs on a non-CUDA host.
    monkeypatch.setattr(current_omni_platform, "device_name", "cuda")
    config = _apply_platform_overrides(load_deploy_config(get_deploy_config_path(_DEPLOY)), platform="cuda")
    return config, VllmOmniConfig.from_pipeline_config(QWEN3_OMNI_PIPELINE, user_deploy_config=config).stage_configs


def test_duplex_mrv2_cuda_profile(monkeypatch) -> None:
    config, stages = _resolve_cuda_stages(monkeypatch)
    assert config.session_mode == "duplex"
    assert [stage.model_config.use_v2_model_runner for stage in stages] == [True, True, True]
    assert [stage.model_config.session_mode for stage in stages] == ["duplex", "duplex", "duplex"]
    assert all(stage.connector_config.async_chunk for stage in stages)
    assert stages[0].model_config.limit_mm_per_prompt == {"audio": 4, "image": 8}
    connector = config.connectors["connector_of_shared_memory"]["extra"]
    assert connector["talker_first_audio"] is True


@pytest.mark.parametrize("platform", ["npu", "xpu", "rocm", "musa"])
def test_duplex_mrv2_profile_keeps_non_cuda_on_v1(monkeypatch, platform) -> None:
    monkeypatch.setattr(current_omni_platform, "device_name", platform)
    config = _apply_platform_overrides(load_deploy_config(get_deploy_config_path(_DEPLOY)), platform=platform)
    stages = VllmOmniConfig.from_pipeline_config(QWEN3_OMNI_PIPELINE, user_deploy_config=config).stage_configs
    assert config.session_mode == "duplex"
    assert [stage.model_config.use_v2_model_runner for stage in stages] == [False, False, False]
