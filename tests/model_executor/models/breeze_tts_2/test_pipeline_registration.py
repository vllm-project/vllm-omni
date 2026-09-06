# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path

import pytest
import yaml

from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.model_executor.models.breeze_tts_2.pipeline import BREEZE_TTS_2_PIPELINE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_breeze_pipeline_is_registered_with_talker_and_codec_stages():
    assert OMNI_PIPELINES["breeze"] is BREEZE_TTS_2_PIPELINE
    assert OMNI_PIPELINES["breeze_tts_2"] is BREEZE_TTS_2_PIPELINE
    assert [stage.model_stage for stage in BREEZE_TTS_2_PIPELINE.stages] == [
        "breeze_tts_2",
        "breeze_tts_2_codec",
    ]
    assert BREEZE_TTS_2_PIPELINE.stages[1].requires_full_payload_input is True
    assert BREEZE_TTS_2_PIPELINE.stages[0].async_chunk_process_next_stage_input_func.endswith(
        "talker2codec_async_chunk"
    )
    assert BREEZE_TTS_2_PIPELINE.stages[1].retains_state_across_chunks is True


def test_breeze_default_deploy_is_streaming():
    deploy_path = Path(__file__).parents[4] / "vllm_omni" / "deploy" / "breeze_tts_2.yaml"
    config = yaml.safe_load(deploy_path.read_text(encoding="utf-8"))
    assert config["async_chunk"] is True
    assert len(config["stages"]) == 2
    assert config["connectors"]["connector_of_shared_memory"]["extra"]["breeze_codec_chunk_frames"] == 8
