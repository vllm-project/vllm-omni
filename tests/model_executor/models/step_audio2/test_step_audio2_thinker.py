# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
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


@pytest.fixture
def step_audio2_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<unk>": 0, "<audio_patch>": 1}, unk_token="<unk>")),
        unk_token="<unk>",
        additional_special_tokens=["<audio_patch>"],
    )


@pytest.fixture
def step_audio2_prompt_updates(step_audio2_tokenizer, monkeypatch):
    """Exercise real prompt updates while isolating model-config/tokenizer loading."""
    from transformers.feature_extraction_utils import BatchFeature
    from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
    from vllm.multimodal.parse import AudioProcessorItems, MultiModalDataItems

    from vllm_omni.model_executor.models.step_audio2.step_audio2_thinker import (
        StepAudio2MultiModalProcessor,
        StepAudio2ProcessingInfo,
    )

    # Prompt updates only need processing info and a tokenizer, not a model config.
    info = object.__new__(StepAudio2ProcessingInfo)
    monkeypatch.setattr(info, "get_tokenizer", lambda **kwargs: step_audio2_tokenizer)
    processor = object.__new__(StepAudio2MultiModalProcessor)
    processor.info = info

    def get_updates(audio_lengths):
        mm_items = MultiModalDataItems(
            {"audio": AudioProcessorItems([np.zeros(321, dtype=np.float32) for _ in audio_lengths])}
            if audio_lengths
            else {}
        )
        mm_kwargs = MultiModalKwargsItems.from_hf_inputs(
            BatchFeature({"audio_lens": torch.tensor(audio_lengths, dtype=torch.int32)}),
            {"audio_lens": MultiModalFieldConfig.batched("audio")},
        )
        return processor._get_prompt_updates(mm_items, {}, mm_kwargs)

    return get_updates


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
    padded_mels, mel_lens = padding_mels(mels)

    assert padded_mels.ndim == 3, f"Expected 3D tensor, got {padded_mels.ndim}D"
    assert padded_mels.shape[0] == 2, f"Expected batch size 2, got {padded_mels.shape[0]}"
    assert padded_mels.shape[1] == 128, f"Expected 128 mel bins, got {padded_mels.shape[1]}"
    assert mel_lens.tolist() == [mel.shape[1] - 2, mel.shape[1] - 2]


@pytest.mark.parametrize("audio", [np.zeros(0, dtype=np.float32), torch.empty(0), []], ids=["numpy", "torch", "list"])
@pytest.mark.parametrize("padding", [0, 479])
def test_log_mel_spectrogram_rejects_empty_audio(audio, padding):
    from vllm_omni.model_executor.models.step_audio2.step_audio2_thinker import log_mel_spectrogram

    with pytest.raises(ValueError, match="Step-Audio2 audio input must contain at least one sample"):
        log_mel_spectrogram(audio, padding=padding)


@pytest.mark.parametrize(
    "empty_audio", [np.zeros(0, dtype=np.float32), torch.empty(0), []], ids=["numpy", "torch", "list"]
)
def test_processor_rejects_empty_audio_in_batch(step_audio2_tokenizer, empty_audio):
    from vllm_omni.model_executor.models.step_audio2.step_audio2_thinker import StepAudio2Processor

    processor = StepAudio2Processor(step_audio2_tokenizer)
    with pytest.raises(ValueError, match="Step-Audio2 audio input must contain at least one sample"):
        processor(text="", audio=[np.zeros(321, dtype=np.float32), empty_audio])


@pytest.mark.parametrize(
    "audio_lengths,error",
    [
        ([0], "Step-Audio2 audio lengths must be positive"),
        ([-1], "Step-Audio2 audio lengths must be positive"),
        ([9, 0, 10], "Step-Audio2 audio lengths must be positive"),
        ([1], "Step-Audio2 audio must produce at least one audio feature"),
        ([2], "Step-Audio2 audio must produce at least one audio feature"),
    ],
)
def test_prompt_updates_reject_invalid_audio_lengths(step_audio2_prompt_updates, audio_lengths, error):
    """Invalid items must fail rather than shift later lengths or fabricate tokens."""
    with pytest.raises(ValueError, match=error):
        step_audio2_prompt_updates(audio_lengths)


def test_prompt_updates_keep_valid_audio_lengths(step_audio2_prompt_updates):
    from vllm_omni.model_executor.models.step_audio2.step_audio2_constants import STEP_AUDIO2_AUDIO_PATCH_TOKEN_ID

    updates = step_audio2_prompt_updates([9, 10])

    assert len(updates) == 1
    assert updates[0].replacement(0) == [STEP_AUDIO2_AUDIO_PATCH_TOKEN_ID]
    assert updates[0].replacement(1) == [STEP_AUDIO2_AUDIO_PATCH_TOKEN_ID]


def test_prompt_updates_allow_no_audio_items(step_audio2_prompt_updates):
    assert len(step_audio2_prompt_updates([])) == 1


@pytest.mark.parametrize("sample_count", [1, 160, 161, 320, 321])
@pytest.mark.parametrize("padding", [0, 479])
def test_short_audio_placeholder_matches_encoder_embedding(sample_count, padding):
    """Very short non-empty audio produces one placeholder and one embedding."""
    from vllm_omni.model_executor.models.step_audio2.step_audio2_thinker import (
        Adaptor,
        AudioEncoder,
        calculate_audio_feature_length,
        log_mel_spectrogram,
        padding_mels,
    )

    mel = log_mel_spectrogram(np.zeros(sample_count, dtype=np.float32), padding=padding)
    padded_mels, mel_lens = padding_mels([mel])

    assert mel.shape == (128, 5)
    assert mel_lens.tolist() == [3]
    placeholder_count = calculate_audio_feature_length(mel_lens.item())
    assert placeholder_count == 1

    encoder = AudioEncoder(n_mels=128, n_ctx=16, n_state=16, n_head=2, n_layer=1)
    adapter = Adaptor(n_state=16, n_hidden=32, kernel_size=3, stride=2, adapter_state=32)
    with torch.inference_mode():
        encoded, encoded_lens = encoder(padded_mels, mel_lens)
        adapted = adapter(encoded)
        feature_lens = (encoded_lens - 1) // 2 + 1
        embedding = adapted[0, : feature_lens.item()]

    assert feature_lens.tolist() == [placeholder_count]
    assert adapted.shape[1] == placeholder_count
    assert embedding.shape == (placeholder_count, 32)
    assert torch.isfinite(embedding).all()


def test_feature_length_calculation():
    """Test that audio feature length calculation is correct."""
    from vllm_omni.model_executor.models.step_audio2.step_audio2_thinker import (
        calculate_audio_feature_length,
    )

    assert calculate_audio_feature_length(1000) == 125
    assert calculate_audio_feature_length(100) > 0
    assert calculate_audio_feature_length(3) == 1
    assert calculate_audio_feature_length(2) == 0


@pytest.mark.parametrize("mel_length,valid_length,feature_length", [(11, 9, 1), (12, 10, 1), (2502, 2500, 313)])
def test_feature_lengths_preserve_reference_convention(mel_length, valid_length, feature_length):
    """Ordinary clips keep excluding the two STFT boundary frames."""
    from vllm_omni.model_executor.models.step_audio2.step_audio2_thinker import (
        calculate_audio_feature_length,
        padding_mels,
    )

    _, mel_lens = padding_mels([torch.zeros(128, mel_length)])

    assert mel_lens.tolist() == [valid_length]
    assert calculate_audio_feature_length(mel_lens.item()) == feature_length


def test_feature_lengths_for_full_chunks_and_short_tail():
    """A non-empty tail must add one placeholder and one model embedding."""
    from vllm_omni.model_executor.models.step_audio2.step_audio2_thinker import (
        calculate_audio_feature_length,
        log_mel_spectrogram,
        padding_mels,
    )

    full_chunk = log_mel_spectrogram(np.zeros(25 * 16000, dtype=np.float32))
    short_tail = log_mel_spectrogram(np.zeros(1, dtype=np.float32))
    padded_mels, mel_lens = padding_mels([full_chunk] * 10 + [short_tail])

    assert padded_mels.shape == (11, 128, 2502)
    assert mel_lens.tolist() == [2500] * 10 + [3]
    feature_lens = [calculate_audio_feature_length(length) for length in mel_lens.tolist()]

    assert feature_lens == [313] * 10 + [1]
    assert sum(feature_lens) == 3131


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
