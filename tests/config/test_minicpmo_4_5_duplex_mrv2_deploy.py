# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Resolve the MiniCPM-o default duplex deployment's CUDA and non-CUDA runners."""

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

_DEPLOY = "minicpmo_4_5.yaml"


def _resolve_cuda_stages(monkeypatch):
    # Resolve CUDA overrides even when this test runs on a non-CUDA host.
    monkeypatch.setattr(current_omni_platform, "device_name", "cuda")
    config = _apply_platform_overrides(load_deploy_config(get_deploy_config_path(_DEPLOY)), platform="cuda")
    return config, merge_pipeline_deploy(MINICPMO_4_5_PIPELINE, config)


def test_duplex_mrv2_cuda_profile(monkeypatch) -> None:
    config, stages = _resolve_cuda_stages(monkeypatch)
    assert config.session_mode == "duplex"
    # Thinker and Code2Wav use MRv2. The Talker stays on V1 because its
    # multi-frame decode re-embeds the previous codec frame only there.
    assert [s.yaml_engine_args["use_v2_model_runner"] for s in stages] == [True, False, True]
    assert [s.yaml_engine_args["async_chunk"] for s in stages] == [True, True, True]
    assert stages[0].yaml_engine_args["async_scheduling"] is True
    # K-step books its span with the synchronous runner contract.
    assert stages[1].yaml_engine_args["speculative_config"]["num_speculative_tokens"] == 7
    assert stages[1].yaml_engine_args["async_scheduling"] is False
    assert stages[2].yaml_engine_args.get("async_scheduling") is not False
    # Retain the base profile capacity and Talker KV budget.
    assert [s.yaml_engine_args["max_num_seqs"] for s in stages] == [32, 32, 32]
    assert stages[1].yaml_engine_args["kv_cache_memory_bytes"] == 8 * 1024**3


@pytest.mark.parametrize("platform", ["npu", "xpu", "rocm", "musa"])
def test_duplex_mrv2_profile_keeps_non_cuda_on_v1(monkeypatch, platform) -> None:
    monkeypatch.setattr(current_omni_platform, "device_name", platform)
    config = _apply_platform_overrides(load_deploy_config(get_deploy_config_path(_DEPLOY)), platform=platform)
    stages = merge_pipeline_deploy(MINICPMO_4_5_PIPELINE, config)
    assert [s.yaml_engine_args["use_v2_model_runner"] for s in stages] == [False, False, False]


@pytest.mark.parametrize("profile", ["minicpmo_4_5.yaml", _DEPLOY, "minicpmo_4_5_turn.yaml"])
@pytest.mark.parametrize("platform,capacity", [("cuda", 32), ("npu", 16)])
def test_default_platform_capacity(monkeypatch, profile, platform, capacity) -> None:
    monkeypatch.setattr(current_omni_platform, "device_name", platform)
    config = _apply_platform_overrides(load_deploy_config(get_deploy_config_path(profile)), platform=platform)
    stages = merge_pipeline_deploy(MINICPMO_4_5_PIPELINE, config)
    assert [s.yaml_engine_args["max_num_seqs"] for s in stages] == [capacity] * 3
    assert config.duplex_session.max_sessions == capacity
    extra = config.connectors["connector_of_shared_memory"]["extra"]
    assert "micro_batch_size" not in extra
    assert "max_graph_batch" not in extra
    if platform == "npu":
        assert stages[2].yaml_engine_args["additional_config"]["cfm_graph_batch_buckets"][-1] == capacity
