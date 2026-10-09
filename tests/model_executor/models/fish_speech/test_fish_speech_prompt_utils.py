# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.model_executor.models.fish_speech.prompt_utils import normalize_fish_speech_text

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_normalize_fish_speech_text_allows_phoneme_control_tokens():
    text = "Deploy with <|phoneme_start|>K UW2 B ER0 N EH1 T IY0 Z<|phoneme_end|>."

    assert normalize_fish_speech_text(text) == text


def test_normalize_fish_speech_text_rejects_unknown_control_token():
    with pytest.raises(ValueError, match=r"unsupported control token\(s\): <\|foo\|>"):
        normalize_fish_speech_text("<|foo|>hello")


def test_normalize_fish_speech_text_normalizes_legacy_speaker_tag():
    assert normalize_fish_speech_text("<speaker:0>hello") == "<|speaker:0|>hello"
