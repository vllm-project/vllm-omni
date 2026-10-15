# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for Kimi-Audio input preparation."""

import numpy as np
import pytest
import torch
from transformers import WhisperFeatureExtractor

from vllm_omni.model_executor.models.kimi_audio.audio_processing import (
    prepare_whisper_inputs,
)
from vllm_omni.model_executor.models.kimi_audio.prompt import (
    KimiAudioEncodedAudio,
    KimiAudioPromptBuilder,
    KimiAudioSpecialTokens,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.omni]


@pytest.fixture
def prompt_builder():
    return KimiAudioPromptBuilder(
        encode_text=lambda text: list(text.encode("utf-8")),
        special_tokens=KimiAudioSpecialTokens(
            msg_end=0,
            media_begin=1,
            media_end=2,
            kimia_text_blank=3,
            kimia_text_eos=4,
            kimia_user_msg_start=5,
            kimia_assistant_msg_start=6,
            kimia_speech_ct_id=7,
            kimia_speech_ctd_id=8,
        ),
        audio_token_offset=256,
        audio_vocab_size=128,
        audio_delay=2,
        continuous_feature_size=4,
    )


@pytest.fixture(scope="module")
def feature_extractor():
    return WhisperFeatureExtractor(feature_size=128)


@pytest.mark.parametrize("samples,expected", [(1280, (1,)), (1281, (2,)), (480000, (375,)), (480001, (375, 1))])
def test_real_feature_extractor_preserves_audio_boundaries(feature_extractor, samples, expected):
    waveform = np.zeros(samples, dtype=np.float32)
    inputs = prepare_whisper_inputs(waveform, feature_extractor, sampling_rate=16000)
    assert inputs.token_lengths == expected
    assert inputs.input_features.shape == (len(expected), 128, 3000)
    assert inputs.input_features.dtype == torch.float32
    assert np.count_nonzero(waveform) == 0


def test_recording_after_thirty_seconds_is_not_truncated(feature_extractor):
    waveform = np.zeros(480000 + 16000, dtype=np.float32)
    waveform[480000:] = np.sin(np.arange(16000) * (2 * np.pi * 440 / 16000))
    inputs = prepare_whisper_inputs(waveform, feature_extractor, sampling_rate=16000)
    assert inputs.token_lengths == (375, 13)
    assert not torch.equal(inputs.input_features[0], inputs.input_features[1])


def test_audio_prompt_aligns_features_without_sharing_request_buffers(prompt_builder):
    messages = [
        {"role": "user", "message_type": "text", "content": "Hi"},
        {"role": "user", "message_type": "audio", "content": "audio"},
    ]
    codes = [10, 11, 12]
    features = torch.ones(len(codes), prompt_builder.continuous_feature_size)
    inputs = {1: KimiAudioEncodedAudio(codes, features)}
    first = prompt_builder.build(messages, audio_inputs=inputs)
    second = prompt_builder.build(messages, audio_inputs=inputs)

    start, stop = second.audio_spans[1]
    assert second.text_token_ids[1:3] == list(b"Hi")
    assert second.audio_token_ids[start:stop] == [prompt_builder.audio_token_offset + code for code in codes]
    assert second.text_token_ids[start:stop] == [prompt_builder.tokens.kimia_text_blank] * len(codes)
    assert [i for i, value in enumerate(second.is_continuous_mask) if value] == list(range(start, stop))
    assert len(second.text_token_ids) == len(second.audio_token_ids) == len(second.is_continuous_mask)
    assert second.continuous_features[0] is features

    first.text_token_ids.clear()
    first.audio_token_ids.clear()
    first.is_continuous_mask.clear()
    first.continuous_features.clear()
    first.audio_spans.clear()
    assert second.text_token_ids and second.audio_token_ids and second.is_continuous_mask
    assert second.continuous_features and second.audio_spans


def test_audio_text_history_preserves_delay_and_alignment(prompt_builder):
    codes = [10, 11, 12]
    prompt = prompt_builder.build(
        [{"role": "assistant", "message_type": "audio-text", "content": ("audio", "Hi")}],
        audio_inputs={0: KimiAudioEncodedAudio(codes)},
        add_assistant_start_msg=False,
    )
    start, stop = prompt.audio_spans[0]
    assert start == 1 + prompt_builder.audio_delay
    assert prompt.audio_token_ids[1:start] == [prompt_builder.tokens.kimia_text_blank] * prompt_builder.audio_delay
    assert prompt.audio_token_ids[start:stop] == [prompt_builder.audio_token_offset + code for code in codes]
    assert prompt.text_token_ids[1:3] == list(b"Hi")
    assert len(prompt.text_token_ids) == len(prompt.audio_token_ids) == len(prompt.is_continuous_mask)
    assert not any(prompt.is_continuous_mask)
    assert not prompt.continuous_features
