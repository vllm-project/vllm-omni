# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Resolve JoyAI's turn/duplex pipelines with and without the ASR stage."""

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.stage_config import (
    _apply_platform_overrides,
    load_deploy_config,
    merge_pipeline_deploy,
)
from vllm_omni.model_executor.models.joyai_vl_interaction.pipeline import (
    JOYAI_VL_INTERACTION_ASR_PIPELINE,
    JOYAI_VL_INTERACTION_PIPELINE,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("platform", ["cuda", "npu", "xpu", "rocm", "musa"])
@pytest.mark.parametrize("asr", [False, True], ids=["visual", "asr"])
@pytest.mark.parametrize("duplex", [False, True], ids=["turn", "duplex"])
def test_joyai_runner_and_transport_contract(monkeypatch, platform, asr, duplex):
    monkeypatch.setattr(current_omni_platform, "device_name", platform)
    name = "joyai_vl_interaction" + ("_asr" if asr else "") + ("_duplex" if duplex else "") + ".yaml"
    deploy = _apply_platform_overrides(load_deploy_config(get_deploy_config_path(name)), platform=platform)
    pipeline = JOYAI_VL_INTERACTION_ASR_PIPELINE if asr else JOYAI_VL_INTERACTION_PIPELINE
    stages = merge_pipeline_deploy(pipeline, deploy)
    assert len(stages) == (4 if asr else 3)
    for stage in stages:
        args = stage.yaml_engine_args
        assert args["use_v2_model_runner"] is (platform == "cuda")
        assert args["supports_native_mrv2_data_plane"] is True
        assert args["session_mode"] == ("duplex" if duplex else "turn")
        assert args["async_chunk"] is (duplex and not asr)
    assert stages[-2].yaml_engine_args["recompute_preemption"] == "fail"
    assert stages[-1].yaml_engine_args["requires_full_payload_input"] is True
    joyai = pipeline.stages[int(asr)]
    assert joyai.model_stage == "joyvl"
    assert stages[-2].custom_process_input_func.endswith(".joyai_action_to_tts")
    if asr:
        assert pipeline.stages[0].bypass_without_modalities == ("audio",)
        assert stages[1].custom_process_input_func.endswith(".asr_to_joyai")


@pytest.mark.parametrize("v1_stages", [(0,), (1,), (2,), (0, 1, 2)])
def test_joyai_stage_runner_override(monkeypatch, v1_stages):
    monkeypatch.setattr(current_omni_platform, "device_name", "cuda")
    deploy = _apply_platform_overrides(
        load_deploy_config(get_deploy_config_path("joyai_vl_interaction_duplex.yaml")), platform="cuda"
    )
    for stage_id in v1_stages:
        deploy.stages[stage_id].model_runner = "v1"
    stages = merge_pipeline_deploy(JOYAI_VL_INTERACTION_PIPELINE, deploy)
    assert [stage.yaml_engine_args["use_v2_model_runner"] for stage in stages] == [
        stage_id not in v1_stages for stage_id in range(3)
    ]
