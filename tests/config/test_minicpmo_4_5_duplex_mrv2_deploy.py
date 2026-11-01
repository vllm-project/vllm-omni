# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Resolve the MiniCPM-o default and explicit V2 duplex deployment contracts."""

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.stage_config import (
    _apply_platform_overrides,
    load_deploy_config,
    merge_pipeline_deploy,
)
from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import MINICPMO_4_5_PIPELINE
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DEPLOY = "minicpmo_4_5_duplex_mrv2.yaml"


def _resolve_cuda_stages(monkeypatch, filename):
    # Resolve CUDA overrides even when this test runs on a non-CUDA host.
    monkeypatch.setattr(current_omni_platform, "device_name", "cuda")
    config = _apply_platform_overrides(load_deploy_config(get_deploy_config_path(filename)), platform="cuda")
    return config, merge_pipeline_deploy(MINICPMO_4_5_PIPELINE, config)


@pytest.mark.parametrize("filename,use_v2", [("minicpmo_4_5.yaml", True), (_DEPLOY, True)])
def test_duplex_cuda_runner_selection(monkeypatch, filename, use_v2) -> None:
    config, stages = _resolve_cuda_stages(monkeypatch, filename)
    assert config.session_mode == "duplex"
    # All three stages follow the default or explicit runner.
    assert [s.yaml_engine_args["use_v2_model_runner"] for s in stages] == [use_v2] * 3
    # Stage 0 duplex preprocessing stays on the synchronous path while the
    # downstream stages keep streaming chunk transfer.
    assert [s.yaml_engine_args["async_chunk"] for s in stages] == [False, True, True]
    # Preserve asynchronous AR scheduling, including the Talker.
    assert all(s.yaml_engine_args["async_scheduling"] for s in stages[:2])
    assert stages[2].yaml_engine_args.get("async_scheduling") is not False
    # Retain the base profile capacity and Talker KV budget.
    assert [s.yaml_engine_args["max_num_seqs"] for s in stages] == [32, 32, 32]
    assert stages[1].yaml_engine_args["kv_cache_memory_bytes"] == 8 * 1024**3


@pytest.mark.parametrize("filename", ["minicpmo_4_5.yaml", _DEPLOY])
@pytest.mark.parametrize("platform", ["npu", "xpu", "rocm", "musa"])
def test_duplex_profiles_keep_non_cuda_on_v1(monkeypatch, platform, filename) -> None:
    monkeypatch.setattr(current_omni_platform, "device_name", platform)
    config = _apply_platform_overrides(load_deploy_config(get_deploy_config_path(filename)), platform=platform)
    stages = merge_pipeline_deploy(MINICPMO_4_5_PIPELINE, config)
    assert [s.yaml_engine_args["use_v2_model_runner"] for s in stages] == [False, False, False]
