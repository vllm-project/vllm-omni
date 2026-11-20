# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PersonaPlex profiles still require a retired V1 Talker stage.

Keep the profile metadata checks and verify that full-workflow admission fails.
The V2 Code2Wav component alone does not qualify the complete model.
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
    return config


def test_personaplex_duplex_mrv2_profile_rejects_v1_talker(monkeypatch) -> None:
    config = _resolve_cuda_stages(monkeypatch)
    assert config.session_mode == "duplex"
    assert config.stages[0].model_runner == "v1"
    with pytest.raises(ValueError, match="Model Runner V1 is not supported"):
        merge_pipeline_deploy(PERSONAPLEX_PIPELINE, config)


@pytest.mark.parametrize("platform", ["npu", "xpu", "rocm", "musa"])
def test_personaplex_duplex_mrv2_profile_keeps_non_cuda_on_v1(monkeypatch, platform) -> None:
    monkeypatch.setattr(current_omni_platform, "device_name", platform)
    config = _apply_platform_overrides(load_deploy_config(get_deploy_config_path(_DEPLOY)), platform=platform)
    assert config.model_runner == "v1"
    with pytest.raises(ValueError, match="Model Runner V1 is not supported"):
        merge_pipeline_deploy(PERSONAPLEX_PIPELINE, config)


def test_personaplex_duplex_mrv2_profile_preserves_pipeline_attributes(monkeypatch) -> None:
    config = _resolve_cuda_stages(monkeypatch)
    # Stage 0: synchronous scheduling for paced lockstep, 8 max seqs, perf switches
    assert config.stages[0].async_scheduling is False
    assert config.stages[0].max_num_seqs == 8
    overrides = config.stages[0].engine_extras["hf_overrides"]
    assert overrides["mimi_cuda_graphs"] is True
    assert overrides["depformer_cuda_graphs"] is True

    # Stage 1: async scheduling for chunk streaming, shared Mimi decode with TF32
    assert config.stages[1].async_scheduling is True
    assert config.stages[1].max_num_seqs == 10
    assert config.stages[1].engine_extras["hf_overrides"]["mimi_cuda_graphs"] is True
    assert config.stages[1].engine_extras["hf_overrides"]["mimi_decode_tf32"] is True

    with pytest.raises(ValueError, match="Model Runner V1 is not supported"):
        merge_pipeline_deploy(PERSONAPLEX_PIPELINE, config)
