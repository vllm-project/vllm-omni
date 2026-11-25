# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json

import pytest

from vllm_omni.config.pipeline_registry import resolve_pipeline_config
from vllm_omni.model_executor.models.ming3_tts.config_ming3_tts import LingV3TTSConfig, Ming3TTSConfig
from vllm_omni.model_executor.models.ming3_tts.pipeline import MING3_TTS_PIPELINE
from vllm_omni.model_executor.models.ming3_tts.prompt_assembly import create_ming3_instruction

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]


def test_public_hf_names_resolve_to_canonical_pipeline():
    assert resolve_pipeline_config("lingv3_tts") is MING3_TTS_PIPELINE
    assert resolve_pipeline_config("ming3_tts") is MING3_TTS_PIPELINE
    assert "LingV3TTSForConditionalGeneration" in MING3_TTS_PIPELINE.hf_architectures
    config = LingV3TTSConfig.from_dict(
        {
            "model_type": "lingv3_tts",
            "architectures": ["LingV3TTSForConditionalGeneration"],
            "tts_config": {"task_contract": "lingv3_sft_multitask_v1"},
            "llm_config": {"num_attention_heads": 16, "hidden_size": 1536},
        }
    )
    assert isinstance(config, Ming3TTSConfig)
    assert config.model_type == "lingv3_tts"
    assert config.tts_config["task_contract"] == "lingv3_sft_multitask_v1"
    assert config.get_text_config().num_attention_heads == 16


@pytest.mark.parametrize("control", [{"风格": "年轻女声"}, {"情感": "高兴"}])
def test_short_controls_keep_vendor_task_field_order(control):
    caption = json.loads(create_ming3_instruction(control))["audio_sequence"][0]
    fields = list(caption)
    if "情感" in control:
        assert fields == ["序号", "说话人", "方言", "风格", "语速", "基频", "音量", "情感", "BGM", "IP"]
    else:
        assert fields == ["序号", "说话人", "方言", "情感", "风格", "语速", "基频", "音量", "BGM", "IP"]
    assert all(caption[key] == value for key, value in control.items())


def test_full_caption_bytes_are_preserved():
    caption = '{"audio_sequence": [{"风格": "calm", "情感": null}]}'
    assert create_ming3_instruction(caption) == caption
