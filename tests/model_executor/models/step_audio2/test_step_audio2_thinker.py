# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Step-Audio2 thinker processor, encoder, and token handling."""

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def create_dummy_audio(sample_rate: int = 16000, duration_sec: float = 1.0) -> tuple[np.ndarray, int]:
    """Create a dummy audio signal (sine wave) for testing."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return audio, sample_rate


def test_processor_loads_correctly():
    """Test that Step-Audio2 processor can be imported and initialized."""
    from vllm_omni.model_executor.models.step_audio2.step_audio2_thinker import (
        StepAudio2MultiModalProcessor,
        StepAudio2ProcessingInfo,
        StepAudio2Processor,
    )

    assert StepAudio2MultiModalProcessor is not None
    assert StepAudio2ProcessingInfo is not None
    assert StepAudio2Processor is not None


def test_audio_preprocessing():
    """Test that audio preprocessing produces correct tensor shapes."""
    from vllm_omni.model_executor.models.step_audio2.step_audio2_thinker import (
        log_mel_spectrogram,
        padding_mels,
    )

    audio, _sample_rate = create_dummy_audio(sample_rate=16000, duration_sec=1.0)

    mel = log_mel_spectrogram(audio)

    assert mel.ndim == 2, f"Expected 2D tensor, got {mel.ndim}D"
    assert mel.shape[0] == 128, f"Expected 128 mel bins, got {mel.shape[0]}"

    mels = [mel, mel]
    padded_mels, _mel_lens = padding_mels(mels)

    assert padded_mels.ndim == 3, f"Expected 3D tensor, got {padded_mels.ndim}D"
    assert padded_mels.shape[0] == 2, f"Expected batch size 2, got {padded_mels.shape[0]}"
    assert padded_mels.shape[1] == 128, f"Expected 128 mel bins, got {padded_mels.shape[1]}"


def test_feature_length_calculation():
    """Test that audio feature length calculation is correct."""
    from vllm_omni.model_executor.models.step_audio2.step_audio2_thinker import (
        calculate_audio_feature_length,
    )

    assert calculate_audio_feature_length(1000) == 125

    assert calculate_audio_feature_length(100) > 0

    assert calculate_audio_feature_length(1) >= 1


def test_token_config_constants():
    """Test that token configuration constants are set correctly."""
    from vllm_omni.model_executor.models.step_audio2.step_audio2_constants import (
        DEFAULT_TOKEN_CONFIG,
        STEP_AUDIO2_AUDIO_END,
        STEP_AUDIO2_AUDIO_PATCH_TOKEN_ID,
        STEP_AUDIO2_AUDIO_START,
        STEP_AUDIO2_AUDIO_VOCAB_SIZE,
        STEP_AUDIO2_TEXT_MAX,
    )

    assert STEP_AUDIO2_TEXT_MAX < STEP_AUDIO2_AUDIO_START, "Text tokens should come before audio tokens"
    assert STEP_AUDIO2_AUDIO_END == STEP_AUDIO2_AUDIO_START + STEP_AUDIO2_AUDIO_VOCAB_SIZE - 1
    assert STEP_AUDIO2_AUDIO_PATCH_TOKEN_ID > STEP_AUDIO2_TEXT_MAX
    assert STEP_AUDIO2_AUDIO_PATCH_TOKEN_ID < STEP_AUDIO2_AUDIO_START

    assert DEFAULT_TOKEN_CONFIG.text_max == STEP_AUDIO2_TEXT_MAX
    assert DEFAULT_TOKEN_CONFIG.audio_start == STEP_AUDIO2_AUDIO_START


def test_mm_field_config_structure():
    """Test that multimodal field config is properly structured for vLLM batching."""
    from vllm.multimodal.inputs import MultiModalFieldConfig

    audio_lens = torch.tensor([100, 200, 150], dtype=torch.int32)

    field_config = MultiModalFieldConfig.flat_from_sizes("audio", audio_lens)

    assert field_config is not None


def test_audio_encoder_output_shape():
    """Test that audio encoder produces correct output shapes."""
    from vllm_omni.model_executor.models.step_audio2.step_audio2_thinker import (
        Adaptor,
        AudioEncoder,
    )

    encoder = AudioEncoder(n_mels=128, n_ctx=1500, n_state=512, n_head=8, n_layer=6)

    batch_size = 2
    n_mels = 128
    time_steps = 400
    x = torch.randn(batch_size, n_mels, time_steps)
    x_len = torch.tensor([time_steps, time_steps // 2], dtype=torch.int32)

    encoded, _encoded_lens = encoder(x, x_len)

    assert encoded.ndim == 3, f"Expected 3D tensor, got {encoded.ndim}D"
    assert encoded.shape[0] == batch_size
    assert encoded.shape[2] == 512

    adapter = Adaptor(n_state=512, n_hidden=4096, kernel_size=3, stride=2)
    adapted = adapter(encoded)

    assert adapted.ndim == 3
    assert adapted.shape[0] == batch_size
    assert adapted.shape[2] == 4096


def test_token_separation():
    """Test that token separation works correctly."""
    from vllm_omni.model_executor.models.step_audio2.step_audio2_thinker import (
        StepAudio2ThinkerForConditionalGeneration,
    )

    token_ids = [100, 200, 151700, 300, 151800, 400]

    text_tokens, audio_tokens = StepAudio2ThinkerForConditionalGeneration.separate_tokens(token_ids)

    assert text_tokens == [100, 200, 300, 400]
    assert audio_tokens == [4, 104]


def test_has_audio_output():
    """Test detection of audio tokens in output."""
    from vllm_omni.model_executor.models.step_audio2.step_audio2_thinker import (
        StepAudio2ThinkerForConditionalGeneration,
    )

    text_only = [100, 200, 300]
    assert not StepAudio2ThinkerForConditionalGeneration.has_audio_output(text_only)

    with_audio = [100, 200, 151700, 300]
    assert StepAudio2ThinkerForConditionalGeneration.has_audio_output(with_audio)
