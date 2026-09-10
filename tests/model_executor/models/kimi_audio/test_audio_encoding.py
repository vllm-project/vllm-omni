# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Audio encoder call contracts and prompt handoff; no pretrained weights.

GLM and Whisper are recording doubles. The actual preprocessing, encoding
entry point, and prompt builder execute normally; this is not a GPU test.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import WhisperFeatureExtractor

from vllm_omni.model_executor.models.kimi_audio.audio_processing import (
    KimiAudioWhisperInputs,
    prepare_whisper_inputs,
)
from vllm_omni.model_executor.models.kimi_audio.kimi_audio_ar_stage import KimiAudioInputEncoder
from vllm_omni.model_executor.models.kimi_audio.prompt import KimiAudioPromptBuilder, KimiAudioSpecialTokens

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.omni]

REFERENCE = json.loads((Path(__file__).parent / "fixtures/prompt_reference.json").read_text(encoding="utf-8"))
FEATURE_SIZE = REFERENCE["input_config"]["continuous_feature_size"]


@pytest.fixture
def input_encoder():
    # Encoding tests inject already-loaded networks; no loader config is used.
    encoder = KimiAudioInputEncoder(vllm_config=SimpleNamespace())
    encoder.glm_feature_extractor = WhisperFeatureExtractor(feature_size=128)
    return encoder


class RecordingGlm(torch.nn.Module):
    def __init__(self, count):
        super().__init__()
        self.codes = torch.arange(count).unsqueeze(0)
        self.calls = []
        self.conv1 = torch.nn.Conv1d(128, 1, 1, bias=False)
        self.conv2 = torch.nn.Conv1d(1, 1, 1, stride=2, bias=False)
        self.config = SimpleNamespace(pooling_kernel_size=4)

    def forward(self, *, input_features, attention_mask):
        assert not torch.is_grad_enabled()
        self.calls.append((input_features.clone(), attention_mask.clone()))
        mask = attention_mask[:, ::2][:, ::4].bool()
        if mask.sum() != self.codes.numel():
            return SimpleNamespace(quantized_token_ids=self.codes)
        codes = torch.full(mask.shape, -999, dtype=torch.long)
        codes[mask] = self.codes[0]
        return SimpleNamespace(quantized_token_ids=codes)


class RecordingWhisper(torch.nn.Module):
    def __init__(self, valid_frames):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(128, 1, 1, bias=False, dtype=torch.float64)
        self.valid_frames = valid_frames
        self.calls = []

    def forward(self, features):
        assert not torch.is_grad_enabled()
        assert features.shape == (1, 128, 3000)
        assert features.dtype == self.conv1.weight.dtype
        assert features.device == self.conv1.weight.device
        index = len(self.calls)
        self.calls.append(features.shape)
        hidden = features.new_full((1, 1500, FEATURE_SIZE // 4), -999)
        count = self.valid_frames[index]
        hidden[0, :count] = (torch.arange(count) + index * 10000)[:, None]
        return hidden


def test_encode_two_chunks_and_hand_off_to_prompt_builder(input_encoder):
    waveform = np.zeros(480001, dtype=np.float32)
    inputs = prepare_whisper_inputs(waveform, WhisperFeatureExtractor(feature_size=128), sampling_rate=16000)
    glm = RecordingGlm(376)
    whisper = RecordingWhisper([1500, 4]).eval()
    input_encoder.audio_tokenizer = glm
    input_encoder.whisper_encoder = whisper

    encoded = input_encoder.encode_audio(waveform, sampling_rate=16000, whisper_inputs=inputs)

    assert len(glm.calls) == 1
    assert glm.calls[0][0].shape == (2, 128, 3000)
    assert glm.calls[0][1].sum(dim=1).tolist() == [3000, 1]
    assert len(whisper.calls) == 2
    assert encoded.codes == list(range(376))
    features = encoded.continuous_features
    assert features.shape == (376, FEATURE_SIZE)
    assert features.dtype == whisper.conv1.weight.dtype
    assert features[[0, 374, 375], :: FEATURE_SIZE // 4].tolist() == [
        [0, 1, 2, 3],
        [1496, 1497, 1498, 1499],
        [10000, 10001, 10002, 10003],
    ]
    assert not (features == -999).any()

    builder = KimiAudioPromptBuilder(
        encode_text=REFERENCE["text_tokens"].__getitem__,
        special_tokens=KimiAudioSpecialTokens.from_vocab(REFERENCE["special_tokens"]),
        **REFERENCE["input_config"],
    )
    prompt = builder.build(
        [{"role": "user", "message_type": "audio", "content": "recording"}], audio_inputs={0: encoded}
    )
    assert sum(prompt.is_continuous_mask) == len(features)
    assert prompt.continuous_features[0] is features
    offset = REFERENCE["input_config"]["audio_token_offset"]
    assert [token for token, continuous in zip(prompt.audio_token_ids, prompt.is_continuous_mask) if continuous] == [
        offset + code for code in encoded.codes
    ]


def test_audio_text_history_uses_only_glm(input_encoder):
    glm = RecordingGlm(2)
    input_encoder.audio_tokenizer = glm
    encoded = input_encoder.encode_audio(np.zeros(1281, dtype=np.float32), sampling_rate=16000)
    assert encoded.codes == [0, 1]
    assert encoded.continuous_features is None
    assert input_encoder.whisper_encoder is None


def test_rejects_glm_length_mismatch_without_truncating_codes(input_encoder):
    input_encoder.audio_tokenizer = RecordingGlm(1)
    with pytest.raises(ValueError, match="GLM codes do not match"):
        input_encoder.encode_audio(np.zeros(1281, dtype=np.float32), sampling_rate=16000)


def test_rejects_whisper_lengths_for_a_different_recording(input_encoder):
    glm = RecordingGlm(376)
    whisper = RecordingWhisper([]).eval()
    input_encoder.audio_tokenizer = glm
    input_encoder.whisper_encoder = whisper
    # The total length matches, but the lengths for each chunk are wrong.
    inputs = KimiAudioWhisperInputs(torch.zeros(2, 128, 3000), (374, 2))
    with pytest.raises(ValueError, match="chunk lengths"):
        input_encoder.encode_audio(
            np.zeros(480001, dtype=np.float32),
            sampling_rate=16000,
            whisper_inputs=inputs,
        )
    assert glm.calls == [] and whisper.calls == []


@pytest.mark.parametrize("component", ["GLM", "Whisper"])
def test_audio_request_requires_startup_loading(input_encoder, component):
    glm = RecordingGlm(1)
    inputs = KimiAudioWhisperInputs(torch.zeros(1, 128, 3000), (1,))
    if component == "Whisper":
        input_encoder.audio_tokenizer = glm
    with pytest.raises(RuntimeError, match=f"Load {component} weights during worker initialization"):
        input_encoder.encode_audio(
            np.zeros(1280, dtype=np.float32),
            sampling_rate=16000,
            whisper_inputs=inputs if component == "Whisper" else None,
        )
    assert input_encoder.whisper_encoder is None
    assert glm.calls == []
