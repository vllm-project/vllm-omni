# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.helpers.runtime import (
    _build_qwen3_tts_additional_information,
    _validate_qwen3_tts_offline_request,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]

SPK_EMB = [0.1, 0.2, 0.3]


# --- Rejection tests ---


def test_rejects_empty_input():
    with pytest.raises(ValueError, match="Input text cannot be empty"):
        _validate_qwen3_tts_offline_request("", {"task_type": "CustomVoice"})


def test_rejects_whitespace_input():
    with pytest.raises(ValueError, match="Input text cannot be empty"):
        _validate_qwen3_tts_offline_request(" \t\n ", {"task_type": "CustomVoice"})


def test_base_rejects_missing_ref_audio_and_embedding():
    with pytest.raises(ValueError, match="Base task requires 'ref_audio' or 'speaker_embedding'"):
        _validate_qwen3_tts_offline_request("Hello", {"task_type": "Base"})


def test_base_rejects_missing_ref_text():
    with pytest.raises(ValueError, match="Base task requires non-empty 'ref_text'"):
        _validate_qwen3_tts_offline_request(
            "Hello",
            {
                "task_type": "Base",
                "ref_audio": "data:audio/wav;base64,AAA=",
            },
        )


def test_base_rejects_whitespace_ref_text():
    with pytest.raises(ValueError, match="Base task requires non-empty 'ref_text'"):
        _validate_qwen3_tts_offline_request(
            "Hello",
            {
                "task_type": "Base",
                "ref_audio": "data:audio/wav;base64,AAA=",
                "ref_text": "   ",
            },
        )


def test_embedding_only_valid_for_base():
    with pytest.raises(ValueError, match="'speaker_embedding' is only valid for Base task"):
        _validate_qwen3_tts_offline_request(
            "Hello",
            {
                "task_type": "CustomVoice",
                "speaker_embedding": [0.0, 0.1],
            },
        )


def test_embedding_must_be_non_empty():
    with pytest.raises(ValueError, match="'speaker_embedding' must be a non-empty list"):
        _validate_qwen3_tts_offline_request(
            "Hello",
            {
                "task_type": "Base",
                "speaker_embedding": [],
            },
        )


# --- Acceptance tests ---


def test_accepts_customvoice_valid():
    text, x_vector = _validate_qwen3_tts_offline_request("Hello", {"task_type": "CustomVoice"})
    assert text == "Hello"
    assert x_vector is False


def test_accepts_base_with_ref_audio_and_ref_text():
    text, x_vector = _validate_qwen3_tts_offline_request(
        "Hello",
        {
            "task_type": "Base",
            "ref_audio": "data:audio/wav;base64,AAA=",
            "ref_text": "reference transcript",
        },
    )
    assert text == "Hello"
    assert x_vector is False


def test_accepts_base_x_vector_only_without_ref_text():
    text, x_vector = _validate_qwen3_tts_offline_request(
        "Hello",
        {
            "task_type": "Base",
            "ref_audio": "data:audio/wav;base64,AAA=",
            "x_vector_only_mode": True,
        },
    )
    assert text == "Hello"
    assert x_vector is True


def test_accepts_base_speaker_embedding_without_ref_text():
    text, x_vector = _validate_qwen3_tts_offline_request(
        "Hello",
        {
            "task_type": "Base",
            "speaker_embedding": [0.0, 0.1, 0.2],
        },
    )
    assert text == "Hello"
    assert x_vector is True


def test_accepts_base_speaker_embedding_with_ref_audio():
    text, x_vector = _validate_qwen3_tts_offline_request(
        "Hello",
        {
            "task_type": "Base",
            "speaker_embedding": [0.0, 0.1],
            "ref_audio": "data:audio/wav;base64,AAA=",
        },
    )
    assert text == "Hello"
    assert x_vector is True


def test_accepts_customvoice_default_task():
    text, x_vector = _validate_qwen3_tts_offline_request("Hello", {})
    assert text == "Hello"
    assert x_vector is False


# --- additional_information builder tests ---


def test_builder_embedding_schema():
    """Embedding lands in voice_clone_prompt[0].ref_spk_embedding, not speaker_embedding."""
    info = _build_qwen3_tts_additional_information(
        "Hello", {"task_type": "Base", "speaker_embedding": SPK_EMB}, x_vector_only_mode=True
    )
    vcp = info["voice_clone_prompt"]
    assert isinstance(vcp, list) and vcp[0]["ref_spk_embedding"] == SPK_EMB
    assert "speaker_embedding" not in info
    assert info["x_vector_only_mode"] == [True]
    assert "ref_audio" not in info


def test_builder_embedding_with_ref_audio():
    """embedding + ref_audio: embedding in voice_clone_prompt, not silently dropped."""
    info = _build_qwen3_tts_additional_information(
        "Hello",
        {"task_type": "Base", "speaker_embedding": SPK_EMB, "ref_audio": "data:audio/wav;base64,AAA="},
        x_vector_only_mode=True,
    )
    vcp = info["voice_clone_prompt"]
    assert vcp[0]["ref_spk_embedding"] == SPK_EMB
    assert "speaker_embedding" not in info


def test_builder_ref_audio_no_embedding():
    """ref_audio path: no voice_clone_prompt, no speaker_embedding leak."""
    info = _build_qwen3_tts_additional_information(
        "Hello",
        {"task_type": "Base", "ref_audio": "data:audio/wav;base64,AAA=", "ref_text": "transcript"},
    )
    assert info["ref_audio"] == ["data:audio/wav;base64,AAA="]
    assert "voice_clone_prompt" not in info
    assert "speaker_embedding" not in info


def test_validate_then_build_embedding():
    """End-to-end: validate + build for embedding path."""
    tts_kw = {"task_type": "Base", "speaker_embedding": SPK_EMB}
    text_str, x_vector = _validate_qwen3_tts_offline_request("Hi", tts_kw)
    info = _build_qwen3_tts_additional_information(text_str, tts_kw, x_vector)
    assert x_vector is True
    assert info["voice_clone_prompt"][0]["ref_spk_embedding"] == SPK_EMB
    assert info["x_vector_only_mode"] == [True]
    assert "speaker_embedding" not in info
