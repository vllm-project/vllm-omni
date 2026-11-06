# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU regression coverage for Qwen3-Omni Code2Wav deploy cleanup (#6176)."""

from copy import deepcopy
from pathlib import Path

import pytest

from vllm_omni.config.omni_config import VllmOmniConfig
from vllm_omni.config.stage_config import StageExecutionType, load_deploy_config, merge_pipeline_deploy
from vllm_omni.model_executor.models.qwen3_omni.pipeline import QWEN3_OMNI_PIPELINE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DEPLOY_PATH = Path(__file__).parents[2] / "vllm_omni" / "deploy" / "qwen3_omni_moe.yaml"


def _before_cleanup(deploy):
    """Reconstruct the old Stage-2 greedy filters in memory, not in YAML."""
    baseline = deepcopy(deploy)
    stage2 = next(stage for stage in baseline.stages if stage.stage_id == 2)
    assert stage2.default_sampling_params is not None
    stage2.default_sampling_params.update(top_p=1.0, top_k=-1)
    return baseline


def test_qwen3_omni_code2wav_cleanup_preserves_legacy_stage_contract():
    deploy = load_deploy_config(_DEPLOY_PATH)
    before = merge_pipeline_deploy(QWEN3_OMNI_PIPELINE, _before_cleanup(deploy))
    after = merge_pipeline_deploy(QWEN3_OMNI_PIPELINE, deepcopy(deploy))

    assert [stage.stage_id for stage in after] == [0, 1, 2]
    assert before[:2] == after[:2]  # Thinker and Talker: full resolved equality.

    code2wav = after[2]
    assert code2wav.model_stage == "code2wav"
    assert code2wav.worker_type == "generation"
    assert QWEN3_OMNI_PIPELINE.get_stage(2).execution_type == StageExecutionType.LLM_GENERATION

    params = code2wav.yaml_extras["default_sampling_params"]
    assert params["temperature"] == 0.0
    assert params["max_tokens"] == 65536
    assert params["repetition_penalty"] == 1.1
    assert params["detokenize"] is True
    assert "top_p" not in params
    assert "top_k" not in params

    # Code2Wav's generation worker does not sample tokens. Only these two
    # non-restrictive greedy filters may differ from the pre-cleanup contract;
    # preserve topology, connectors, runtime layout and all engine arguments.
    baseline_code2wav = deepcopy(before[2])
    baseline_params = baseline_code2wav.yaml_extras["default_sampling_params"]
    assert baseline_params.pop("top_p") == 1.0
    assert baseline_params.pop("top_k") == -1
    assert baseline_code2wav == code2wav


def test_qwen3_omni_code2wav_cleanup_preserves_structured_sampling_contract():
    deploy = load_deploy_config(_DEPLOY_PATH)
    before = VllmOmniConfig.from_pipeline_config(QWEN3_OMNI_PIPELINE, user_deploy_config=_before_cleanup(deploy))
    after = VllmOmniConfig.from_pipeline_config(QWEN3_OMNI_PIPELINE, user_deploy_config=deepcopy(deploy))

    for stage_id in (0, 1):
        assert (
            before.stage_by_id(stage_id).model_config.default_sampling_params
            == after.stage_by_id(stage_id).model_config.default_sampling_params
        )

    old_params = dict(before.stage_by_id(2).model_config.default_sampling_params)
    new_params = after.stage_by_id(2).model_config.default_sampling_params
    assert old_params.pop("top_p") == 1.0
    assert old_params.pop("top_k") == -1
    assert old_params == new_params

    for stage_id in (0, 1, 2):
        old_stage = before.stage_by_id(stage_id)
        new_stage = after.stage_by_id(stage_id)
        assert old_stage.runtime_config.devices == new_stage.runtime_config.devices
        assert old_stage.cache_config.gpu_memory_utilization == new_stage.cache_config.gpu_memory_utilization
        assert old_stage.connector_config.input_connectors == new_stage.connector_config.input_connectors
        assert old_stage.connector_config.output_connectors == new_stage.connector_config.output_connectors
