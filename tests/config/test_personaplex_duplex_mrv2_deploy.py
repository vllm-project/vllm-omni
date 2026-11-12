# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The duplex MRv2 deploy profile resolves to the config keys the model relies on.

Full-duplex Model Runner V2 serving for PersonaPlex is selected by
``personaplex_duplex_mrv2.yaml``. Stage 0 (Talker) stays on V1 for lockstep
post-sample depformer execution and batched appends; Stage 1 (Code2Wav)
opts into MRv2 on CUDA while falling back to V1 on non-CUDA platforms.
"""

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.stage_config import (
    _apply_platform_overrides,
    load_deploy_config,
    merge_pipeline_deploy,
)
from vllm_omni.model_executor.models.personaplex.pipeline import PERSONAPLEX_PIPELINE
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DEPLOY = "personaplex_duplex_mrv2.yaml"


def _resolve_cuda_stages(monkeypatch):
    monkeypatch.setattr(current_omni_platform, "device_name", "cuda")
    config = _apply_platform_overrides(load_deploy_config(get_deploy_config_path(_DEPLOY)), platform="cuda")
    return config, merge_pipeline_deploy(PERSONAPLEX_PIPELINE, config)


def test_personaplex_duplex_mrv2_profile_requests_duplex_and_v2(monkeypatch) -> None:
    config, stages = _resolve_cuda_stages(monkeypatch)
    assert config.session_mode == "duplex"
    # Stage 0 (Talker) stays on V1 for lockstep post-sample depformer execution;
    # Stage 1 (Code2Wav) opts in to MRv2.
    assert [s.yaml_engine_args["use_v2_model_runner"] for s in stages] == [False, True]


@pytest.mark.parametrize("platform", ["npu", "xpu", "rocm", "musa"])
def test_personaplex_duplex_mrv2_profile_keeps_non_cuda_on_v1(monkeypatch, platform) -> None:
    monkeypatch.setattr(current_omni_platform, "device_name", platform)
    config = _apply_platform_overrides(load_deploy_config(get_deploy_config_path(_DEPLOY)), platform=platform)
    stages = merge_pipeline_deploy(PERSONAPLEX_PIPELINE, config)
    assert [s.yaml_engine_args["use_v2_model_runner"] for s in stages] == [False, False]


def test_personaplex_duplex_mrv2_profile_preserves_pipeline_attributes(monkeypatch) -> None:
    config, stages = _resolve_cuda_stages(monkeypatch)
    # Stage 0: synchronous scheduling for paced lockstep, 8 max seqs, perf switches
    assert stages[0].yaml_engine_args["async_scheduling"] is False
    assert stages[0].yaml_engine_args["max_num_seqs"] == 8
    overrides = stages[0].yaml_engine_args["hf_overrides"]
    assert overrides["mimi_cuda_graphs"] is True
    assert overrides["depformer_cuda_graphs"] is True

    # Stage 1: async scheduling for chunk streaming, shared Mimi decode with TF32
    assert stages[1].yaml_engine_args["async_scheduling"] is True
    assert stages[1].yaml_engine_args["max_num_seqs"] == 10
    assert stages[1].yaml_engine_args["hf_overrides"]["mimi_cuda_graphs"] is True
    assert stages[1].yaml_engine_args["hf_overrides"]["mimi_decode_tf32"] is True
