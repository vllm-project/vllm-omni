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

_REMOVED_EXTRAS = frozenset(
    {
        "codec_streaming",
        "connector_get_sleep_s",
        "connector_get_max_wait_first_chunk",
        "connector_get_max_wait",
    }
)


@pytest.mark.parametrize("profile", _PROFILES)
def test_qwen3_tts_removed_connector_extras(profile):
    deploy = load_deploy_config(_DEPLOY_DIR / profile)
    extras = deploy.connectors["connector_of_shared_memory"]["extra"]

    unexpected = _REMOVED_EXTRAS.intersection(extras)
    assert not unexpected, f"{profile} still contains unused connector keys: {sorted(unexpected)}"


@pytest.mark.parametrize("profile", _PROFILES)
def test_qwen3_tts_config_paths_resolve(profile):
    deploy = load_deploy_config(_DEPLOY_DIR / profile)

    # Both existing config paths must remain valid after the cleanup.
    legacy_stages = merge_pipeline_deploy(QWEN3_TTS_PIPELINE, deepcopy(deploy))
    assert [(stage.stage_id, stage.model_stage) for stage in legacy_stages] == [
        (0, "qwen3_tts"),
        (1, "code2wav"),
    ]

    structured = VllmOmniConfig.from_pipeline_config(
        QWEN3_TTS_PIPELINE,
        user_deploy_config=deepcopy(deploy),
    )
    assert structured.stage_by_id(0).model_config.default_sampling_params["temperature"] == 0.9
    assert structured.stage_by_id(1).model_config.default_sampling_params["temperature"] == 0.0
