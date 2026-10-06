# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest

from vllm_omni.model_executor.models.minicpmo_4_5.duplex.policy import (
    MiniCPMO45DuplexPolicy,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("has_ref_audio", [False, True])
@pytest.mark.parametrize(
    "initial_user_text",
    [
        "What is the capital of France?",
        "Watch the video and only speak when a red ball appears.",
        "The quick brown fox.",
    ],
)
def test_initial_text_remains_a_completed_user_message_before_native_unit(has_ref_audio, initial_user_text):
    prefix, suffix = MiniCPMO45DuplexPolicy.session_context_texts(
        "Speak exactly.",
        has_ref_audio,
        initial_user_text,
    )

    assert prefix == "<|im_start|>system\nSpeak exactly." + ("\n<|audio_start|>" if has_ref_audio else "")
    expected_suffix = "<|audio_end|>" if has_ref_audio else ""
    expected_suffix += f"<|im_end|>\n<|im_start|>user\n{initial_user_text}<|im_end|>"
    assert suffix == expected_suffix
    assert (suffix + "<unit>").endswith(f"{initial_user_text}<|im_end|><unit>")
    assert "<|im_start|>assistant" not in suffix


@pytest.mark.parametrize("has_ref_audio", [False, True])
@pytest.mark.parametrize("initial_user_text", [None, ""])
@pytest.mark.parametrize("initial_user_text_is_tts", [False, True])
def test_unseeded_native_duplex_context_is_unchanged(has_ref_audio, initial_user_text, initial_user_text_is_tts):
    prefix, suffix = MiniCPMO45DuplexPolicy.session_context_texts(
        None, has_ref_audio, initial_user_text, initial_user_text_is_tts=initial_user_text_is_tts
    )

    assert prefix == "<|im_start|>system\nStreaming Omni Conversation." + ("\n<|audio_start|>" if has_ref_audio else "")
    assert suffix == ("<|audio_end|>" if has_ref_audio else "") + "<|im_end|>"


@pytest.mark.parametrize("has_ref_audio", [False, True])
def test_explicit_base_tts_opens_a_tts_response(has_ref_audio):
    _, suffix = MiniCPMO45DuplexPolicy.session_context_texts(
        "Speak exactly.", has_ref_audio, "The quick brown fox.", initial_user_text_is_tts=True
    )

    assert suffix == ("<|audio_end|>" if has_ref_audio else "") + (
        "<|im_end|>\n<|im_start|>user\nThe quick brown fox.<|im_end|>"
        "\n<|im_start|>assistant\n<think>\n\n</think>\n\n<|tts_bos|>"
    )
