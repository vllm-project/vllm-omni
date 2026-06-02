# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.stage_input_processors.aura_omni import (
    SILENT_TEXT,
    asr2aura,
    aura2tts,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _source_output(text: str, request_id: str = "req-1"):
    output = SimpleNamespace(text=text, cumulative_token_ids=[1, 2, 3], multimodal_output={})
    return SimpleNamespace(request_id=request_id, outputs=[output])


def test_asr2aura_carries_video_payload_and_transcript():
    prompt = {
        "multi_modal_data": {"video": ["frame-0", "frame-1"]},
        "additional_information": {"aura_system_prompt": ["system"]},
    }

    [next_input] = asr2aura([_source_output("现在发生了什么？")], prompt=[prompt])

    assert next_input["multi_modal_data"] == {"video": ["frame-0", "frame-1"]}
    assert "<|video_pad|>" in next_input["prompt"]
    assert "现在发生了什么？" in next_input["prompt"]
    assert next_input["prompt"].startswith("<|im_start|>system\nsystem")


def test_asr2aura_drops_audio_before_qwen3_vl_stage():
    prompt = {
        "multi_modal_data": {
            "audio": ("wave", 16000),
            "video": ["frame-0", "frame-1"],
        },
    }

    [next_input] = asr2aura([_source_output("看看视频")], prompt=[prompt])

    assert next_input["multi_modal_data"] == {"video": ["frame-0", "frame-1"]}
    assert "<|video_pad|>" in next_input["prompt"]


def test_asr2aura_supports_video_only_observation():
    prompt = {"multi_modal_data": {"video": ["frame-0", "frame-1"]}}

    [next_input] = asr2aura([_source_output("")], prompt=[prompt])

    assert "<|video_pad|>" in next_input["prompt"]
    assert "<|im_start|>assistant" in next_input["prompt"]


def test_aura2tts_builds_qwen3_tts_prompt_information():
    prompt = {
        "additional_information": {
            "tts_language": ["Chinese"],
            "tts_speaker": ["Vivian"],
            "tts_instruct": ["Calm voice."],
        }
    }

    [tts_input] = aura2tts([_source_output("你好。")], prompt=[prompt])

    assert len(tts_input["prompt_token_ids"]) >= 32
    assert tts_input["additional_information"]["text"] == ["你好。"]
    assert tts_input["additional_information"]["language"] == ["Chinese"]
    assert tts_input["additional_information"]["speaker"] == ["Vivian"]
    assert tts_input["additional_information"]["instruct"] == ["Calm voice."]


def test_aura2tts_drops_silent_response():
    assert aura2tts([_source_output(SILENT_TEXT)]) == []
