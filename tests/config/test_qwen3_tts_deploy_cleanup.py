# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen3-TTS connector-extra cleanup regression tests (#6176)."""

from copy import deepcopy
from pathlib import Path

import pytest

from vllm_omni.config.omni_config import VllmOmniConfig
from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy
from vllm_omni.model_executor.models.qwen3_tts.pipeline import QWEN3_TTS_PIPELINE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DEPLOY_DIR = Path(__file__).parents[2] / "vllm_omni" / "deploy"

_PROFILES = (
    "qwen3_tts.yaml",
    "qwen3_tts_high_concurrency.yaml",
)

_REMOVED_EXTRAS = {
    "codec_streaming": True,
    "connector_get_sleep_s": 0.01,
    "connector_get_max_wait_first_chunk": 3000,
    "connector_get_max_wait": 300,
}


def _before_and_after(profile):
    after = load_deploy_config(_DEPLOY_DIR / profile)
    before = deepcopy(after)

    old_extra = before.connectors["connector_of_shared_memory"]["extra"]
    new_extra = after.connectors["connector_of_shared_memory"]["extra"]

    assert not _REMOVED_EXTRAS.keys() & new_extra.keys()

    # Reconstruct the pre-cleanup YAML contract in memory.
    old_extra.update(_REMOVED_EXTRAS)

    assert {key: value for key, value in old_extra.items() if key not in _REMOVED_EXTRAS} == new_extra

    return before, after


@pytest.mark.parametrize("profile", _PROFILES)
def test_qwen3_tts_legacy_stage_contract(profile):
    before, after = _before_and_after(profile)

    old_stages = merge_pipeline_deploy(QWEN3_TTS_PIPELINE, deepcopy(before))
    new_stages = merge_pipeline_deploy(QWEN3_TTS_PIPELINE, deepcopy(after))

    assert [stage.stage_id for stage in new_stages] == [0, 1]
    assert old_stages == new_stages

    assert new_stages[0].model_stage == "qwen3_tts"
    assert new_stages[1].model_stage == "code2wav"


@pytest.mark.parametrize("profile", _PROFILES)
def test_qwen3_tts_structured_stage_contract(profile):
    before, after = _before_and_after(profile)

    old_config = VllmOmniConfig.from_pipeline_config(
        QWEN3_TTS_PIPELINE,
        user_deploy_config=deepcopy(before),
    )
    new_config = VllmOmniConfig.from_pipeline_config(
        QWEN3_TTS_PIPELINE,
        user_deploy_config=deepcopy(after),
    )

    for stage_id in (0, 1):
        old_stage = old_config.stage_by_id(stage_id)
        new_stage = new_config.stage_by_id(stage_id)

        assert old_stage.model_config.default_sampling_params == new_stage.model_config.default_sampling_params
        assert old_stage.runtime_config.devices == new_stage.runtime_config.devices
        assert old_stage.cache_config.gpu_memory_utilization == new_stage.cache_config.gpu_memory_utilization
        assert old_stage.connector_config.input_connectors == new_stage.connector_config.input_connectors
        assert old_stage.connector_config.output_connectors == new_stage.connector_config.output_connectors
