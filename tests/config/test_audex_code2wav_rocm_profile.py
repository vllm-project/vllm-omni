# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pin the ROCm metadata shim for Audex's Code2Wav decoder."""

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.stage_config import (
    PipelineConfig,
    _apply_platform_overrides,
    load_deploy_config,
    merge_pipeline_deploy,
)
from vllm_omni.model_executor.models.audex.pipeline import (
    AUDEX_S2S_PIPELINE,
    AUDEX_TTS_PIPELINE,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_CODE2WAV_PROFILES = (
    ("audex_tts.yaml", AUDEX_TTS_PIPELINE),
    ("audex_tts_30b.yaml", AUDEX_TTS_PIPELINE),
    ("audex_s2s.yaml", AUDEX_S2S_PIPELINE),
    ("audex_s2s_30b.yaml", AUDEX_S2S_PIPELINE),
)


@pytest.mark.parametrize(("profile", "pipeline"), _CODE2WAV_PROFILES)
def test_rocm_supplies_supported_code2wav_head_metadata(
    profile: str,
    pipeline: PipelineConfig,
) -> None:
    deploy = _apply_platform_overrides(
        load_deploy_config(get_deploy_config_path(profile)),
        platform="rocm",
    )
    stages = merge_pipeline_deploy(pipeline, deploy)

    # AudexCode2Wav has 2048 hidden channels over 32 attention heads. The
    # override supplies that derived head size only to vLLM's metadata path.
    assert stages[1].yaml_engine_args["hf_overrides"]["head_dim"] == 64


@pytest.mark.parametrize(("profile", "pipeline"), _CODE2WAV_PROFILES)
def test_code2wav_head_metadata_override_is_rocm_only(
    profile: str,
    pipeline: PipelineConfig,
) -> None:
    del pipeline
    deploy = _apply_platform_overrides(
        load_deploy_config(get_deploy_config_path(profile)),
        platform="cuda",
    )

    assert "hf_overrides" not in deploy.stages[1].engine_extras
