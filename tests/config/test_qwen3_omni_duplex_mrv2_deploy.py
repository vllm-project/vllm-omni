# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Resolve the Qwen3-Omni duplex Model Runner V2 deployment contract."""

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.omni_config import VllmOmniConfig
from vllm_omni.config.stage_config import (
    _apply_platform_overrides,
    load_deploy_config,
    merge_pipeline_deploy,
)
from vllm_omni.model_executor.models.qwen3_omni.pipeline import QWEN3_OMNI_PIPELINE
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DEPLOY = "qwen3_omni_duplex_mrv2.yaml"


def _resolve_stages(monkeypatch, platform: str):
    # Resolve platform overrides even when this test runs on a different host.
    monkeypatch.setattr(current_omni_platform, "device_name", platform)
    config = _apply_platform_overrides(load_deploy_config(get_deploy_config_path(_DEPLOY)), platform=platform)
    return config, merge_pipeline_deploy(QWEN3_OMNI_PIPELINE, config)


def test_duplex_mrv2_cuda_profile(monkeypatch) -> None:
    config, stages = _resolve_stages(monkeypatch, "cuda")
    assert config.session_mode == "duplex"
    assert [stage.yaml_engine_args["use_v2_model_runner"] for stage in stages] == [True, True, True]
    assert [stage.yaml_engine_args["session_mode"] for stage in stages] == ["duplex", "duplex", "duplex"]
    assert all(stage.yaml_engine_args["async_chunk"] for stage in stages)
    assert stages[0].yaml_engine_args["limit_mm_per_prompt"] == {"audio": 4, "image": 8}
    connector = config.connectors["connector_of_shared_memory"]["extra"]
    assert connector["talker_first_audio"] is True


def test_duplex_mrv2_rocm_profile_runs_v2(monkeypatch) -> None:
    config, stages = _resolve_stages(monkeypatch, "rocm")
    assert config.model_runner == "v2"
    assert config.session_mode == "duplex"
    assert [stage.yaml_engine_args["use_v2_model_runner"] for stage in stages] == [True, True, True]
    assert [stage.yaml_engine_args["session_mode"] for stage in stages] == ["duplex", "duplex", "duplex"]
    assert all(stage.yaml_engine_args["async_chunk"] for stage in stages)
    assert stages[0].yaml_engine_args["enable_prefix_caching"] is False
    assert stages[2].yaml_engine_args["enforce_eager"] is True

    structured = VllmOmniConfig.from_pipeline_config(QWEN3_OMNI_PIPELINE, user_deploy_config=config)
    assert [stage.model_config.use_v2_model_runner for stage in structured.stage_configs] == [True, True, True]
    assert structured.stage_by_id(0).cache_config.enable_prefix_caching is False
    assert structured.stage_by_id(2).model_config.enforce_eager is True


def test_duplex_mrv2_profile_runs_v2_on_xpu(monkeypatch) -> None:
    # XPU has no V1 fallback: vLLM 0.34 removed Model Runner V1.
    config, stages = _resolve_stages(monkeypatch, "xpu")
    assert config.model_runner == "v2"
    assert [stage.yaml_engine_args["use_v2_model_runner"] for stage in stages] == [True, True, True]


@pytest.mark.parametrize("platform", ["npu", "musa"])
def test_duplex_mrv2_profile_keeps_non_cuda_on_v1(monkeypatch, platform) -> None:
    monkeypatch.setattr(current_omni_platform, "device_name", platform)
    config = _apply_platform_overrides(load_deploy_config(get_deploy_config_path(_DEPLOY)), platform=platform)
    assert config.model_runner == "v1"
    with pytest.raises(ValueError, match="Model Runner V1 is not supported"):
        merge_pipeline_deploy(QWEN3_OMNI_PIPELINE, config)
