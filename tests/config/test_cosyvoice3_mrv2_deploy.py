# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Resolve the opt-in CosyVoice3 MRv2 streaming deployment."""

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.stage_config import _apply_platform_overrides, load_deploy_config, merge_pipeline_deploy
from vllm_omni.model_executor.models.cosyvoice3.pipeline import COSYVOICE3_PIPELINE
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_cuda_mrv2_profile_batches_streaming_code2wav(monkeypatch):
    monkeypatch.setattr(current_omni_platform, "device_name", "cuda")
    config = _apply_platform_overrides(
        load_deploy_config(get_deploy_config_path("cosyvoice3_mrv2.yaml")), platform="cuda"
    )
    stages = merge_pipeline_deploy(COSYVOICE3_PIPELINE, config)
    assert config.session_mode == "turn"
    assert [stage.yaml_engine_args["use_v2_model_runner"] for stage in stages] == [True, True]
    assert all(stage.yaml_engine_args["supports_native_mrv2_data_plane"] for stage in stages)
    assert all(stage.yaml_engine_args["async_chunk"] for stage in stages)
    assert all(stage.yaml_engine_args["enable_chunked_prefill"] is False for stage in stages)
    assert config.stages[1].env["COSYVOICE3_BATCH_FLOW"] == "1"
    # Keep default deployment selection and concurrency independent of opt-in.
    assert load_deploy_config(get_deploy_config_path("cosyvoice3.yaml")).model_runner == "v1"
    assert [stage.yaml_engine_args["max_num_seqs"] for stage in stages] == [8, 8]


@pytest.mark.parametrize("platform", ["npu", "xpu", "rocm", "musa"])
def test_non_cuda_profile_uses_v1(monkeypatch, platform):
    monkeypatch.setattr(current_omni_platform, "device_name", platform)
    config = _apply_platform_overrides(
        load_deploy_config(get_deploy_config_path("cosyvoice3_mrv2.yaml")), platform=platform
    )
    assert all(
        not stage.yaml_engine_args["use_v2_model_runner"]
        for stage in merge_pipeline_deploy(COSYVOICE3_PIPELINE, config)
    )


def test_mrv2_profile_rejects_native_duplex_sessions(monkeypatch):
    monkeypatch.setattr(current_omni_platform, "device_name", "cuda")
    config = load_deploy_config(get_deploy_config_path("cosyvoice3_mrv2.yaml"))
    config.session_mode = "duplex"
    with pytest.raises(ValueError, match="turn.*only"):
        merge_pipeline_deploy(COSYVOICE3_PIPELINE, config)
