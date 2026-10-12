# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Resolve the MiniCPM-o three-stage duplex MRv2 deployment contract."""

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.omni_config import VllmOmniConfig
from vllm_omni.config.stage_config import load_deploy_config
from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import MINICPMO_4_5_PIPELINE
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DEPLOY = "minicpmo_4_5_duplex_mrv2.yaml"


def _resolve_cuda_stages(monkeypatch):
    # Resolve CUDA overrides even when this test runs on a non-CUDA host.
    monkeypatch.setattr(current_omni_platform, "device_name", "cuda")
    config = load_deploy_config(get_deploy_config_path(_DEPLOY))
    return VllmOmniConfig.from_pipeline_config(MINICPMO_4_5_PIPELINE, user_deploy_config=config).stage_configs


def test_duplex_mrv2_cuda_profile(monkeypatch) -> None:
    stages = _resolve_cuda_stages(monkeypatch)
    assert all(stage.model_config.session_mode == "duplex" for stage in stages)
    # All three stages use MRv2; there is no hidden Talker fallback.
    assert [s.model_config.use_v2_model_runner for s in stages] == [True, True, True]
    # Stage 0 duplex preprocessing stays on the synchronous path while the
    # downstream stages keep streaming chunk transfer.
    assert [s.connector_config.async_chunk for s in stages] == [False, True, True]
    # Preserve asynchronous AR scheduling, including the Talker.
    assert all(s.scheduler_config.async_scheduling for s in stages[:2])
    assert stages[2].scheduler_config.async_scheduling is not False
    # Retain the base profile capacity and Talker KV budget.
    assert [s.scheduler_config.max_num_seqs for s in stages] == [16, 16, 16]
    assert stages[1].cache_config.kv_cache_memory_bytes == 4 * 1024**3


@pytest.mark.parametrize("platform", ["npu", "xpu", "rocm", "musa"])
def test_duplex_mrv2_profile_keeps_non_cuda_on_v1(monkeypatch, platform) -> None:
    monkeypatch.setattr(current_omni_platform, "device_name", platform)
    config = load_deploy_config(get_deploy_config_path(_DEPLOY))
    stages = VllmOmniConfig.from_pipeline_config(MINICPMO_4_5_PIPELINE, user_deploy_config=config).stage_configs
    assert [s.model_config.use_v2_model_runner for s in stages] == [False, False, False]
