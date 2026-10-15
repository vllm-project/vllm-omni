# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
    PRECOMPUTED_TEXT_IDS_KEY,
)
from vllm_omni.model_executor.stage_input_processors.aura_omni import (
    SILENT_TEXT,
    _estimate_tts_prompt_len_from_token_ids,
    _estimate_tts_prompt_len_official,
    _normalize_asr_transcript,
    asr2aura,
    aura2tts,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _source_output(text: str, request_id: str = "req-1", token_ids: list[int] | None = None):
    output = SimpleNamespace(text=text, cumulative_token_ids=token_ids or [1, 2, 3], multimodal_output={})
    return SimpleNamespace(request_id=request_id, outputs=[output])


def _source_delta_final_output(cumulative_text: str, request_id: str = "req-1"):
    output = SimpleNamespace(
        text="",
        cumulative_text=cumulative_text,
        cumulative_token_ids=[1, 2, 3],
        multimodal_output={},
    )
    return SimpleNamespace(request_id=request_id, outputs=[output])


def test_asr2aura_carries_video_payload_and_transcript():
    prompt = {
        "multi_modal_data": {"video": ["frame-0", "frame-1"]},
        "additional_information": {"aura_system_prompt": ["system"]},
    }

    [next_input] = asr2aura([_source_output("What is happening now?")], prompt=[prompt])

    assert next_input["multi_modal_data"] == {"video": ["frame-0", "frame-1"]}
    assert "<|video_pad|>" in next_input["prompt"]
    assert "What is happening now?" in next_input["prompt"]
    assert next_input["prompt"].startswith("<|im_start|>system\nsystem")


def test_asr2aura_drops_audio_before_qwen3_vl_stage():
    prompt = {
        "multi_modal_data": {
            "audio": ("wave", 16000),
            "video": ["frame-0", "frame-1"],
        },
    }

    [next_input] = asr2aura([_source_output("Check the video")], prompt=[prompt])

    assert next_input["multi_modal_data"] == {"video": ["frame-0", "frame-1"]}
    assert "<|video_pad|>" in next_input["prompt"]


def test_asr2aura_reads_video_stashed_for_downstream_stage():
    prompt = {
        "multi_modal_data": {"audio": ("wave", 16000)},
        "additional_information": {
            "deferred_multi_modal_data": {"video": ["frame-0", "frame-1"]},
        },
    }

    [next_input] = asr2aura([_source_output("Check the video")], prompt=[prompt])

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
            "tts_ref_audio": ["ref.wav"],
            "tts_ref_text": ["Reference transcript sample."],
        }
    }

    [tts_input] = aura2tts([_source_output("Hello.")], prompt=[prompt])

    assert len(tts_input["prompt_token_ids"]) > 0
    assert tts_input["additional_information"]["text"] == ["Hello."]
    assert PRECOMPUTED_TEXT_IDS_KEY not in tts_input["additional_information"]
    assert tts_input["additional_information"]["task_type"] == ["Base"]
    assert tts_input["additional_information"]["language"] == ["Chinese"]
    assert tts_input["additional_information"]["ref_audio"] == ["ref.wav"]
    assert tts_input["additional_information"]["ref_text"] == ["Reference transcript sample."]
    assert tts_input["additional_information"]["x_vector_only_mode"] == [False]
    assert tts_input["additional_information"]["instruct"] == ["Calm voice."]


def test_aura2tts_prefers_streaming_cumulative_text():
    prompt = {
        "additional_information": {
            "tts_ref_audio": ["ref.wav"],
            "tts_ref_text": ["Reference transcript sample."],
        }
    }

    [tts_input] = aura2tts(
        [_source_delta_final_output("The complete AURA reply.")],
        prompt=[prompt],
    )

    assert tts_input["additional_information"]["text"] == ["The complete AURA reply."]


def test_aura2tts_supports_base_ref_audio_override():
    prompt = {
        "additional_information": {
            "tts_ref_audio": ["custom.wav"],
            "tts_ref_text": ["custom transcript"],
        }
    }

    [tts_input] = aura2tts([_source_output("Hello.")], prompt=[prompt])

    assert tts_input["additional_information"]["task_type"] == ["Base"]
    assert tts_input["additional_information"]["ref_audio"] == ["custom.wav"]
    assert tts_input["additional_information"]["ref_text"] == ["custom transcript"]
    assert tts_input["additional_information"]["x_vector_only_mode"] == [False]


def test_aura2tts_supports_x_vector_only_mode_for_base():
    prompt = {
        "additional_information": {
            "tts_task_type": ["Base"],
            "tts_x_vector_only_mode": [True],
            "tts_ref_audio": ["ref.wav"],
            "tts_ref_text": ["Reference transcript sample."],
        }
    }

    [tts_input] = aura2tts([_source_output("Hello.")], prompt=[prompt])

    assert tts_input["additional_information"]["x_vector_only_mode"] == [True]


def test_aura2tts_supports_custom_voice_mode():
    prompt = {
        "additional_information": {
            "tts_task_type": ["CustomVoice"],
            "tts_speaker": ["vivian"],
        }
    }

    [tts_input] = aura2tts([_source_output("Hello.")], prompt=[prompt])

    assert tts_input["additional_information"]["task_type"] == ["CustomVoice"]
    assert tts_input["additional_information"]["speaker"] == ["Vivian"]
    assert "ref_audio" not in tts_input["additional_information"]
    assert len(tts_input["prompt_token_ids"]) > 0


def test_aura2tts_passes_token_ids_to_qwen3_tts_when_enabled():
    prompt = {
        "additional_information": {
            "tts_ref_audio": ["ref.wav"],
            "tts_ref_text": ["Reference transcript sample."],
            "tts_pass_token_ids": [True],
        }
    }

    [tts_input] = aura2tts(
        [
            _source_output(
                "Hello.",
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
    assert aura2tts([_source_output(f"{SILENT_TEXT}<|im_end|>")]) == []


def test_aura2tts_customvoice_uses_official_prompt_len_not_aura_token_count():
    text = "当然可以，我正看着你呢。"
    prompt = {
        "additional_information": {
            "tts_task_type": ["CustomVoice"],
            "tts_speaker": ["Vivian"],
            "tts_language": ["Chinese"],
        }
    }
    aura_token_ids = list(range(80))
    tts_info = {
        "task_type": ["CustomVoice"],
        "language": ["Chinese"],
        "instruct": [""],
        "text": [text],
        "speaker": ["Vivian"],
    }
    official = _estimate_tts_prompt_len_official(tts_info, task_type="CustomVoice")
    if official is None:
        pytest.skip("Qwen3-TTS tokenizer unavailable for official prompt_len")
    old_heuristic = _estimate_tts_prompt_len_from_token_ids(
        aura_token_ids,
        task_type="CustomVoice",
        language="Chinese",
        instruct="",
    )
    assert official != old_heuristic

    [tts_input] = aura2tts(
        [_source_output(text, token_ids=aura_token_ids)],
        prompt=[prompt],
    )
    assert len(tts_input["prompt_token_ids"]) == official
    assert tts_input["additional_information"]["instruct"] == [""]


def test_aura2tts_strips_im_end_from_spoken_text():
    prompt = {
        "additional_information": {
            "tts_ref_audio": ["ref.wav"],
            "tts_ref_text": ["Reference transcript sample."],
        }
    }
    [tts_input] = aura2tts([_source_output("你好。<|im_end|>")], prompt=[prompt])
    assert tts_input["additional_information"]["text"] == ["你好。"]


def test_aura2tts_does_not_treat_chinese_silence_as_special_token():
    # [沉默] is not <|silent|>; prompt/session must instruct the real token.
    prompt = {
        "additional_information": {
            "tts_ref_audio": ["ref.wav"],
            "tts_ref_text": ["Reference transcript sample."],
        }
    }
    assert len(aura2tts([_source_output("[沉默]")], prompt=[prompt])) == 1


def test_normalize_asr_transcript_strips_qwen3_asr_markup() -> None:
    raw = "language Chinese<asr_text>出现《古韵》这本书的时候，提醒我。"
    assert _normalize_asr_transcript(raw) == "出现《古韵》这本书的时候，提醒我。"
    assert _normalize_asr_transcript("出现古韵这本书的时候提醒我。") == "出现古韵这本书的时候提醒我。"


def test_next_duplex_sentence_chunk_batches_until_min_chars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_AURA_SENTENCE_TTS", raising=False)
    monkeypatch.delenv("VLLM_AURA_SENTENCE_TTS_MIN_CHARS", raising=False)
    from vllm_omni.model_executor.stage_input_processors.aura_omni import (
        next_duplex_sentence_chunk,
    )

    state: dict[str, object] = {}
    first = "甲" * 12 + "。"
    assert next_duplex_sentence_chunk(state, first, finished=False) is None
    second = first + "乙" * 20 + "。"
    chunk = next_duplex_sentence_chunk(state, second, finished=False)
    assert chunk is not None
    assert chunk.startswith("甲" * 12)
    assert "乙" * 20 in chunk


def test_next_duplex_sentence_chunk_holds_think_silent_and_flushes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_AURA_SENTENCE_TTS", "1")
    from vllm_omni.model_executor.stage_input_processors.aura_omni import (
        next_duplex_sentence_chunk,
    )

    assert next_duplex_sentence_chunk({}, "<think>still thinking", finished=False) is None
    assert next_duplex_sentence_chunk({}, "<|silent|>", finished=True) is None
    assert next_duplex_sentence_chunk({}, '<tool_call>{"a":1}</tool_call>', finished=False) is None

    state: dict[str, object] = {}
    assert next_duplex_sentence_chunk(state, "你好。", finished=False) is None
    assert next_duplex_sentence_chunk(state, "你好。", finished=True) == "你好。"


def test_next_duplex_sentence_chunk_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_AURA_SENTENCE_TTS", "0")
    from vllm_omni.model_executor.stage_input_processors.aura_omni import (
        next_duplex_sentence_chunk,
    )

    assert next_duplex_sentence_chunk({}, "甲" * 40 + "。", finished=False) is None
