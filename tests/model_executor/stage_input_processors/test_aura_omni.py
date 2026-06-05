# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
    PRECOMPUTED_TEXT_IDS_KEY,
)
from vllm_omni.model_executor.stage_input_processors.aura_omni import (
    DEFAULT_QWEN3_TTS_REF_AUDIO,
    DEFAULT_QWEN3_TTS_REF_TEXT,
    SILENT_TEXT,
    asr2aura,
    aura2tts,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _source_output(text: str, request_id: str = "req-1", token_ids: list[int] | None = None):
    output = SimpleNamespace(text=text, cumulative_token_ids=token_ids or [1, 2, 3], multimodal_output={})
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


def test_asr2aura_reads_video_stashed_for_downstream_aura_stage():
    prompt = {
        "multi_modal_data": {"audio": ("wave", 16000)},
        "additional_information": {
            "aura_multi_modal_data": {"video": ["frame-0", "frame-1"]},
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
            "tts_instruct": ["Calm voice."],
            "tts_ref_text": ["测试参考文本。"],
        }
    }

    [tts_input] = aura2tts([_source_output("你好。")], prompt=[prompt])

    assert len(tts_input["prompt_token_ids"]) >= 32
    assert "text" not in tts_input["additional_information"]
    assert tts_input["additional_information"][PRECOMPUTED_TEXT_IDS_KEY] == [
        [151644, 77091, 198, 1, 2, 3, 151645, 198, 151644, 77091, 198]
    ]
    assert tts_input["additional_information"]["task_type"] == ["Base"]
    assert tts_input["additional_information"]["language"] == ["Chinese"]
    assert tts_input["additional_information"]["ref_audio"] == [DEFAULT_QWEN3_TTS_REF_AUDIO]
    assert tts_input["additional_information"]["ref_text"] == [DEFAULT_QWEN3_TTS_REF_TEXT]
    assert tts_input["additional_information"]["x_vector_only_mode"] == [False]
    assert tts_input["additional_information"]["instruct"] == ["Calm voice."]


def test_aura2tts_supports_base_ref_audio_override():
    prompt = {
        "additional_information": {
            "tts_ref_audio": ["custom.wav"],
            "tts_ref_text": ["custom transcript"],
        }
    }

    [tts_input] = aura2tts([_source_output("你好。")], prompt=[prompt])

    assert tts_input["additional_information"]["task_type"] == ["Base"]
    assert tts_input["additional_information"]["ref_audio"] == ["custom.wav"]
    assert tts_input["additional_information"]["ref_text"] == ["custom transcript"]
    assert tts_input["additional_information"]["x_vector_only_mode"] == [False]


def test_aura2tts_supports_x_vector_only_mode_for_base():
    prompt = {
        "additional_information": {
            "tts_task_type": ["Base"],
            "tts_x_vector_only_mode": [True],
        }
    }

    [tts_input] = aura2tts([_source_output("你好。")], prompt=[prompt])

    assert tts_input["additional_information"]["x_vector_only_mode"] == [True]


def test_aura2tts_supports_custom_voice_mode():
    prompt = {
        "additional_information": {
            "tts_task_type": ["CustomVoice"],
            "tts_speaker": ["vivian"],
        }
    }

    [tts_input] = aura2tts([_source_output("你好。")], prompt=[prompt])

    assert tts_input["additional_information"]["task_type"] == ["CustomVoice"]
    assert tts_input["additional_information"]["speaker"] == ["Vivian"]
    assert "ref_audio" not in tts_input["additional_information"]


def test_aura2tts_passes_aura_token_ids_to_qwen3_tts_by_default():
    prompt = {
        "additional_information": {
            "tts_ref_text": ["测试参考文本。"],
        }
    }

    [tts_input] = aura2tts(
        [
            _source_output(
                "你好。",
                token_ids=[151644, 77091, 198, 108386, 1773, 151645, 198],
            )
        ],
        prompt=[prompt],
    )

    assert tts_input["additional_information"][PRECOMPUTED_TEXT_IDS_KEY] == [
        [151644, 77091, 198, 108386, 1773, 151645, 198, 151644, 77091, 198]
    ]
    assert "text" not in tts_input["additional_information"]


def test_aura2tts_drops_silent_response():
    assert aura2tts([_source_output(SILENT_TEXT)]) == []
