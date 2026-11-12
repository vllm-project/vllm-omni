# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Resolve the Qwen3-Omni duplex deployment contract."""

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.stage_config import (
    _apply_platform_overrides,
    load_deploy_config,
    merge_pipeline_deploy,
)
from vllm_omni.model_executor.models.qwen3_omni.pipeline import QWEN3_OMNI_PIPELINE
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DEPLOY = "qwen3_omni_duplex.yaml"


def _resolve_cuda_stages(monkeypatch):
    # Resolve CUDA overrides even when this test runs on a non-CUDA host.
    monkeypatch.setattr(current_omni_platform, "device_name", "cuda")
    config = _apply_platform_overrides(load_deploy_config(get_deploy_config_path(_DEPLOY)), platform="cuda")
    return config, merge_pipeline_deploy(QWEN3_OMNI_PIPELINE, config)


def test_duplex_cuda_profile_uses_mrv2_with_resident_thinker(monkeypatch) -> None:
    config, stages = _resolve_cuda_stages(monkeypatch)
    assert config.session_mode == "duplex"
    assert [stage.yaml_engine_args["use_v2_model_runner"] for stage in stages] == [True, True, True]
    assert [stage.yaml_engine_args["session_mode"] for stage in stages] == ["duplex", "duplex", "duplex"]
    assert all(stage.yaml_engine_args["async_chunk"] for stage in stages)
    assert stages[0].yaml_engine_args["enable_prefix_caching"] is True
    assert stages[0].yaml_engine_args["limit_mm_per_prompt"] == {"audio": 4, "image": 8}
    connector = config.connectors["connector_of_shared_memory"]["extra"]
    assert connector["talker_first_audio"] is True


@pytest.mark.parametrize(("platform", "use_v2"), [("npu", True), ("xpu", False), ("rocm", False), ("musa", False)])
def test_duplex_profile_follows_default_runner(monkeypatch, platform, use_v2) -> None:
    monkeypatch.setattr(current_omni_platform, "device_name", platform)
    config = _apply_platform_overrides(load_deploy_config(get_deploy_config_path(_DEPLOY)), platform=platform)
    stages = merge_pipeline_deploy(QWEN3_OMNI_PIPELINE, config)
    assert config.session_mode == "duplex"
    assert [stage.yaml_engine_args["use_v2_model_runner"] for stage in stages] == [use_v2] * 3
