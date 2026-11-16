# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for Step-Audio2 thinker processor, encoder, and token handling."""

from unittest.mock import Mock

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

    assert calculate_audio_feature_length(1) == 0
    assert calculate_audio_feature_length(2) == 0
    assert calculate_audio_feature_length(3) == 1


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


@pytest.mark.usefixtures("single_threaded_cpu")
def test_mm_field_config_structure(audio_processor):
    """Chunk metadata survives batching independently processed audio items."""
    from vllm.multimodal.inputs import MultiModalKwargsItems

    hf_processor, processor = audio_processor
    items = []
    for duration in (1, 40):
        inputs = hf_processor(text="", audio=[create_dummy_audio(duration_sec=duration)[0]])
        fields = processor._get_mm_fields_config(inputs, {})
        items.extend(MultiModalKwargsItems.from_hf_inputs(inputs, fields)["audio"])
    data = MultiModalKwargsItems({"audio": items}).get_data()
    assert data["audio_num_chunks"].tolist() == [1, 2]
    assert data["audio_mels"].shape == (3, 128, 2502)
    assert data["audio_lens"].tolist() == [100, 2500, 1500]


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


@pytest.fixture
def audio_processor():
    from transformers import PreTrainedTokenizerBase
    from vllm.multimodal.parse import MultiModalDataParser

    from vllm_omni.model_executor.models.step_audio2.step_audio2_thinker import (
        StepAudio2DummyInputsBuilder,
        StepAudio2MultiModalProcessor,
        StepAudio2ProcessingInfo,
        StepAudio2Processor,
    )

    tokenizer = Mock(spec=PreTrainedTokenizerBase, return_value={})
    tokens = {"<audio_start>": [151689], "<audio_patch>": [151690], "<audio_end>": [151691]}
    tokenizer.encode.side_effect = lambda text, **kwargs: tokens[text]
    hf_processor = StepAudio2Processor(tokenizer)
    info = Mock(spec=StepAudio2ProcessingInfo, ctx=Mock(tokenizer=tokenizer))
    info.get_tokenizer.return_value = tokenizer
    info.get_hf_processor.return_value = hf_processor
    info.get_data_parser.return_value = MultiModalDataParser(target_sr=16000)
    return hf_processor, StepAudio2MultiModalProcessor(info, StepAudio2DummyInputsBuilder(info))


@pytest.mark.usefixtures("single_threaded_cpu")
@pytest.mark.parametrize(
    "durations, expected_chunks, expected_lengths",
    [
        ((1 / 16000,), [1], [0]),
        ((16 / 16000,), [1], [0]),
        ((320 / 16000,), [1], [0]),
        ((321 / 16000,), [1], [1]),
        ((25,), [1], [313]),
        ((30,), [2], [376]),
        ((40,), [2], [501]),
        ((50,), [2], [626]),
        ((60,), [3], [751]),
        ((400160 / 16000,), [2], [313]),
        ((800001 / 16000,), [3], [626]),
        ((16 / 16000, 1), [1, 1], [0, 13]),
        ((1, 40, 2), [1, 2, 1], [13, 501, 25]),
    ],
)
def test_audio_chunks_preserve_items_and_prompt_lengths(audio_processor, durations, expected_chunks, expected_lengths):
    """Short and chunked audio preserve items and match patches to embeddings."""
    from vllm.multimodal.inputs import MultiModalKwargsItems

    from vllm_omni.model_executor.models.step_audio2.step_audio2_thinker import (
        Adaptor,
        AudioEncoder,
        StepAudio2ThinkerForConditionalGeneration,
    )

    hf_processor, processor = audio_processor
    audios = [
        (0.5 * np.sin(2 * np.pi * 440 * np.arange(round(duration * 16000)) / 16000)).astype(np.float32)
        for duration in durations
    ]
    hf_inputs = hf_processor(text="", audio=audios)
    mm_items = processor.data_parser.parse_mm_data({"audio": audios})
    fields = processor._get_mm_fields_config(hf_inputs, {})
    mm_kwargs = MultiModalKwargsItems.from_hf_inputs(hf_inputs, fields)
    assert len(mm_kwargs["audio"]) == len(audios)

    model = StepAudio2ThinkerForConditionalGeneration.__new__(StepAudio2ThinkerForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.encoder = AudioEncoder(128, 1500, 16, 2, 1)
    model.adapter = Adaptor(n_state=16, n_hidden=32, adapter_state=32)
    model.eval()
    with torch.inference_mode():
        embeddings = model.embed_multimodal(**mm_kwargs.get_data())

    assert len(embeddings) == len(audios)
    assert [len(embedding) for embedding in embeddings] == expected_lengths
    replacement = processor._get_prompt_updates(mm_items, {}, mm_kwargs)[0].replacement
    for i, n_chunks in enumerate(expected_chunks):
        item_data = mm_kwargs["audio"][i].get_data()
        assert item_data["audio_mels"].shape[0] == n_chunks
        assert item_data["audio_mels"].shape[-1] <= 2502
        assert len(item_data["audio_lens"]) == n_chunks
        update = replacement(i)
        assert update.full.count(151689) == n_chunks
        assert update.full.count(151691) == n_chunks
        assert update.is_embed(update.full).sum().item() == embeddings[i].shape[0]


def test_text_only_audio_fields(audio_processor):
    from vllm.multimodal.inputs import MultiModalKwargsItems

    hf_processor, processor = audio_processor
    inputs = hf_processor(text="")
    fields = processor._get_mm_fields_config(inputs, {})
    assert not MultiModalKwargsItems.from_hf_inputs(inputs, fields)


@pytest.mark.usefixtures("single_threaded_cpu")
@pytest.mark.parametrize("num_samples", [959999, 960000, 960001])
def test_audio_duration_limit_and_budget(audio_processor, num_samples):
    from vllm_omni.model_executor.models.step_audio2.step_audio2_thinker import (
        StepAudio2DummyInputsBuilder,
        StepAudio2ProcessingInfo,
        calculate_audio_feature_length,
    )

    hf_processor, processor = audio_processor
    audio = np.zeros(num_samples, dtype=np.float32)
    if num_samples > 960000:
        with pytest.raises(ValueError, match="60"):
            hf_processor(text="", audio=[audio])
        return
    inputs = hf_processor(text="", audio=[audio, audio])
    assert inputs["audio_num_chunks"].tolist() == [3, 3]
    info = StepAudio2ProcessingInfo.__new__(StepAudio2ProcessingInfo)
    info.get_hf_processor = lambda **kwargs: hf_processor
    dummy = StepAudio2DummyInputsBuilder(info).get_dummy_mm_data(8192, {"audio": 1})
    assert len(dummy["audio"][0]) == 960000
    dummy_inputs = hf_processor(text="", audio=dummy["audio"])
    count = sum(calculate_audio_feature_length(int(n)) for n in dummy_inputs["audio_lens"])
    assert info.get_mm_max_tokens_per_item(8192, {"audio": 1}) == {"audio": count}
    assert count == 751


@pytest.mark.parametrize(
    "counts, expected",
    [
        ([1, 2], [1, 2]),
        (torch.tensor([1, 2]), [1, 2]),
        ([0, 3], None),
        ([-1, 4], None),
        ([1.5, 1.5], None),
        ([True, 2], None),
        ([1, 1], None),
        ([2, 2], None),
        ([], None),
    ],
)
def test_audio_chunk_metadata(counts, expected):
    from vllm_omni.model_executor.models.step_audio2.step_audio2_thinker import (
        StepAudio2ThinkerForConditionalGeneration,
    )

    model = StepAudio2ThinkerForConditionalGeneration.__new__(StepAudio2ThinkerForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.register_parameter("test_parameter", torch.nn.Parameter(torch.zeros(1)))
    kwargs = {"audio_mels": torch.zeros(3, 128, 10), "audio_lens": torch.tensor([8, 8, 8])}
    kwargs["audio_num_chunks"] = counts
    if expected is None:
        with pytest.raises(ValueError, match="audio_num_chunks"):
            model._parse_and_validate_audio_input(**kwargs)
    else:
        inputs = model._parse_and_validate_audio_input(**kwargs)
        assert inputs["audio_num_chunks"].tolist() == expected


@pytest.mark.parametrize("explicit_none", [False, True], ids=["missing", "explicit_none"])
def test_audio_chunk_metadata_required(explicit_none):
    from vllm_omni.model_executor.models.step_audio2.step_audio2_thinker import (
        StepAudio2ThinkerForConditionalGeneration,
    )

    model = StepAudio2ThinkerForConditionalGeneration.__new__(StepAudio2ThinkerForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.register_parameter("test_parameter", torch.nn.Parameter(torch.zeros(1)))
    kwargs = {"audio_mels": torch.zeros(3, 128, 10), "audio_lens": torch.tensor([8, 8, 8])}
    if explicit_none:
        kwargs["audio_num_chunks"] = None
    with pytest.raises(ValueError, match="audio_num_chunks is required when audio_mels is provided"):
        model._parse_and_validate_audio_input(**kwargs)
