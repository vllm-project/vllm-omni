# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PretrainedConfig, PreTrainedTokenizerFast
from vllm.config.multimodal import MultiModalConfig
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.parse import MultiModalDataParser
from vllm.multimodal.processing import InputProcessingContext

from vllm_omni.model_executor.models.registry import _OMNI_MODELS
from vllm_omni.model_executor.models.vibevoice.processing_vibevoice import (
    AUDIO_BOS_TOKEN,
    AUDIO_EOS_TOKEN,
    AUDIO_TOKEN,
    MAX_AUDIO_ITEMS,
    MAX_AUDIO_SAMPLES,
    MAX_AUDIO_TOKENS,
    SAMPLE_RATE,
    VibeVoiceDummyInputsBuilder,
    VibeVoiceMultiModalProcessor,
    VibeVoiceProcessingInfo,
)
from vllm_omni.model_executor.models.vibevoice.vibevoice import (
    VibeVoiceForConditionalGeneration,
    _flatten_audio_token_counts,
    _pad_ragged_audio_batch,
)
from vllm_omni.model_executor.models.vibevoice.vllm_compat import (
    get_audio_with_sr_from_parent,
    get_stage0_tokenizer,
    merge_multimodal_embeddings,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_ragged_audio_batch_is_right_padded_without_changing_valid_samples():
    inputs = [torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0, 5.0, 6.0]])]
    masks = [torch.tensor([[1, 1]]), torch.tensor([[1, 1, 1, 0]])]

    padded_inputs, padded_masks = _pad_ragged_audio_batch(inputs, masks)

    assert padded_inputs.tolist() == [[[1.0, 2.0, 0.0, 0.0]], [[3.0, 4.0, 5.0, 6.0]]]
    assert padded_masks.tolist() == [[1, 1, 0, 0], [1, 1, 1, 0]]


def test_ragged_audio_batch_rejects_misaligned_input_and_mask_lengths():
    with pytest.raises(ValueError, match="mismatched sample lengths"):
        _pad_ragged_audio_batch(
            [torch.tensor([[1.0, 2.0]])],
            [torch.tensor([[1]])],
        )


def test_nested_audio_token_counts_are_flattened_in_item_order():
    counts = _flatten_audio_token_counts([torch.tensor([2]), torch.tensor([[4], [3]])])
    assert counts.tolist() == [2, 4, 3]


def _make_tokenizer() -> PreTrainedTokenizerFast:
    backend = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "Speaker": 1,
                "0:": 2,
                "1:": 3,
                "then": 4,
                AUDIO_BOS_TOKEN: 5,
                AUDIO_EOS_TOKEN: 6,
                AUDIO_TOKEN: 7,
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        additional_special_tokens=[
            AUDIO_BOS_TOKEN,
            AUDIO_EOS_TOKEN,
            AUDIO_TOKEN,
        ],
    )
    return tokenizer


def _make_processor(
    *,
    user_audio_limit: int = 8,
) -> tuple[VibeVoiceMultiModalProcessor, VibeVoiceProcessingInfo]:
    tokenizer = _make_tokenizer()
    hf_config = PretrainedConfig()
    hf_config.audio_bos_token_id = tokenizer.convert_tokens_to_ids(AUDIO_BOS_TOKEN)
    hf_config.audio_eos_token_id = tokenizer.convert_tokens_to_ids(AUDIO_EOS_TOKEN)
    hf_config.audio_token_id = tokenizer.convert_tokens_to_ids(AUDIO_TOKEN)

    multimodal_config = MultiModalConfig(limit_per_prompt={"audio": user_audio_limit})
    model_config = SimpleNamespace(
        model="test-vibevoice",
        hf_config=hf_config,
        multimodal_config=multimodal_config,
        dtype=torch.float32,
        encoder_config=None,
        max_model_len=4096,
        get_multimodal_config=lambda: multimodal_config,
    )
    context = InputProcessingContext(model_config, tokenizer=tokenizer)
    info = VibeVoiceProcessingInfo(context)
    processor = VibeVoiceMultiModalProcessor(
        info,
        VibeVoiceDummyInputsBuilder(info),
        cache=None,
    )
    return processor, info


def test_vibevoice_model_registers_the_standard_multimodal_processor():
    assert _OMNI_MODELS["VibeVoiceForConditionalGeneration"] == (
        "vibevoice",
        "vibevoice",
        "VibeVoiceForConditionalGeneration",
    )
    factories = VibeVoiceForConditionalGeneration._processor_factory

    assert VibeVoiceForConditionalGeneration.supports_multimodal is True
    assert factories.info is VibeVoiceProcessingInfo
    assert factories.processor is VibeVoiceMultiModalProcessor
    assert factories.dummy_inputs is VibeVoiceDummyInputsBuilder
    assert VibeVoiceForConditionalGeneration.get_placeholder_str("audio", 0) == (
        f"{AUDIO_BOS_TOKEN}{AUDIO_TOKEN}{AUDIO_EOS_TOKEN}"
    )


def test_single_prompt_multiple_reference_audios_preserves_item_order():
    processor, info = _make_processor()
    tokenizer = info.get_tokenizer()

    prompt = (
        f"Speaker 0: {AUDIO_BOS_TOKEN}{AUDIO_TOKEN}{AUDIO_EOS_TOKEN} "
        f"then Speaker 1: {AUDIO_BOS_TOKEN}{AUDIO_TOKEN}{AUDIO_EOS_TOKEN}"
    )
    first_audio = np.linspace(-0.25, 0.25, 3_201, dtype=np.float32)
    second_audio = np.sin(np.linspace(0, 20 * np.pi, 8_001, dtype=np.float32)).astype(np.float32)
    mm_items = info.parse_mm_data(
        {
            "audio": [
                (first_audio, SAMPLE_RATE),
                (second_audio, SAMPLE_RATE),
            ]
        }
    )

    result = processor(prompt, mm_items=mm_items)

    expected_prompt = (
        f"Speaker 0: {AUDIO_BOS_TOKEN}{AUDIO_TOKEN * 2}{AUDIO_EOS_TOKEN} "
        f"then Speaker 1: {AUDIO_BOS_TOKEN}{AUDIO_TOKEN * 3}{AUDIO_EOS_TOKEN}"
    )
    assert result["prompt_token_ids"] == tokenizer.encode(
        expected_prompt,
        add_special_tokens=False,
    )

    placeholder_ranges = result["mm_placeholders"]["audio"]
    assert [item.length for item in placeholder_ranges] == [2, 3]
    audio_token_id = tokenizer.convert_tokens_to_ids(AUDIO_TOKEN)
    expected_offsets = [idx for idx, token_id in enumerate(result["prompt_token_ids"]) if token_id == audio_token_id]
    assert [item.offset for item in placeholder_ranges] == [
        expected_offsets[0],
        expected_offsets[2],
    ]

    mm_data = result["mm_kwargs"].get_data()
    assert mm_data["audio_num_tokens"].flatten().tolist() == [2, 3]
    assert mm_data["padding_mask"].sum(dim=-1).flatten().tolist() == [3_201, 8_001]

    input_values = mm_data["input_values"]
    assert input_values.shape[0] == 2
    assert torch.count_nonzero(input_values[0, ..., 3_201:]) == 0
    assert torch.count_nonzero(input_values[1, ..., :8_001]) > 0


@pytest.mark.parametrize(
    ("num_audios", "num_segments"),
    [(2, 1), (1, 2)],
)
def test_reference_audio_and_placeholder_counts_must_match(
    num_audios,
    num_segments,
):
    processor, info = _make_processor()
    segment = f"{AUDIO_BOS_TOKEN}{AUDIO_TOKEN}{AUDIO_EOS_TOKEN}"
    prompt = " ".join(segment for _ in range(num_segments))
    audio = np.ones(3_201, dtype=np.float32)

    with pytest.raises(
        RuntimeError,
        match=(
            rf"Expected there to be {num_audios} prompt placeholders .* "
            rf"found {num_segments} prompt placeholders"
        ),
    ):
        processor(
            prompt,
            mm_items=info.parse_mm_data({"audio": [(audio, SAMPLE_RATE)] * num_audios}),
        )


def test_ninth_audio_is_rejected_by_processor_supported_limit():
    _, info = _make_processor(user_audio_limit=16)
    audio = np.ones(3_201, dtype=np.float32)

    with pytest.raises(VLLMValidationError, match=r"At most 8 audio\(s\)"):
        info.parse_mm_data({"audio": [(audio, SAMPLE_RATE)] * 9})


def test_user_audio_limit_is_enforced_before_processing():
    _, info = _make_processor(user_audio_limit=2)
    audio = np.ones(3_201, dtype=np.float32)

    with pytest.raises(VLLMValidationError, match=r"At most 2 audio\(s\)"):
        info.parse_mm_data({"audio": [(audio, SAMPLE_RATE)] * 3})


def test_processing_limits_match_scheduler_budget():
    _, info = _make_processor()

    assert info.get_supported_mm_limits() == {"audio": MAX_AUDIO_ITEMS}
    assert info.get_mm_max_tokens_per_item(65_536) == {"audio": MAX_AUDIO_TOKENS}
    assert MAX_AUDIO_TOKENS == 450


def test_sixty_second_audio_expands_to_scheduler_maximum():
    processor, info = _make_processor()
    prompt = f"{AUDIO_BOS_TOKEN}{AUDIO_TOKEN}{AUDIO_EOS_TOKEN}"
    audio = np.zeros(MAX_AUDIO_SAMPLES, dtype=np.float32)

    result = processor(
        prompt,
        mm_items=info.parse_mm_data({"audio": [(audio, SAMPLE_RATE)]}),
    )

    assert result["mm_kwargs"].get_data()["audio_num_tokens"].item() == 450
    assert result["mm_placeholders"]["audio"][0].length == 450


def test_audio_longer_than_sixty_seconds_is_rejected():
    processor, info = _make_processor()
    prompt = f"{AUDIO_BOS_TOKEN}{AUDIO_TOKEN}{AUDIO_EOS_TOKEN}"
    audio = np.zeros(MAX_AUDIO_SAMPLES + 1, dtype=np.float32)

    with pytest.raises(ValueError, match=r"60\.00s; the maximum is 60s"):
        processor(
            prompt,
            mm_items=info.parse_mm_data({"audio": [(audio, SAMPLE_RATE)]}),
        )


def test_stereo_audio_is_downmixed_before_feature_extraction(caplog):
    processor, info = _make_processor()
    prompt = f"{AUDIO_BOS_TOKEN}{AUDIO_TOKEN}{AUDIO_EOS_TOKEN}"
    stereo = np.stack(
        [
            np.linspace(-0.5, 0.5, 3_201, dtype=np.float32),
            np.linspace(0.5, -0.5, 3_201, dtype=np.float32),
        ]
    )

    mm_items = info.parse_mm_data({"audio": [(stereo, SAMPLE_RATE)]})
    result = processor(prompt, mm_items=mm_items)

    assert "automatically downmixed to mono" in caplog.text
    input_values = result["mm_kwargs"].get_data()["input_values"]
    assert input_values.shape == (1, 1, 6_400)


@pytest.mark.parametrize(
    "stereo",
    [
        np.zeros((2, 3_201), dtype=np.float32),
        torch.zeros((2, 3_201), dtype=torch.float32),
    ],
)
def test_bare_2d_audio_is_rejected_as_ambiguous(stereo):
    _, info = _make_processor()

    with pytest.raises(ValueError, match="Ambiguous bare 2D VibeVoice audio"):
        info.parse_mm_data({"audio": stereo})


def test_vllm_private_multimodal_compatibility_smoke():
    upstream_signature = inspect.signature(MultiModalDataParser._get_audio_with_sr)
    assert list(upstream_signature.parameters) == ["self", "audio"]

    parser = MultiModalDataParser()
    waveform = np.ones(16, dtype=np.float32)
    parsed_waveform, sample_rate = get_audio_with_sr_from_parent(
        parser,
        (waveform, SAMPLE_RATE),
    )
    assert parsed_waveform is waveform
    assert sample_rate == SAMPLE_RATE

    inputs_embeds = torch.zeros((3, 2), dtype=torch.float32)
    mm_embeds = [torch.tensor([[7.0, 8.0]], dtype=torch.float32)]
    merged = merge_multimodal_embeddings(
        inputs_embeds,
        mm_embeds,
        torch.tensor([False, True, False]),
    )
    torch.testing.assert_close(merged[1], mm_embeds[0][0])

    tokenizer = object()
    engine_client = SimpleNamespace(
        engine=SimpleNamespace(
            input_processor=SimpleNamespace(renderer=SimpleNamespace(get_tokenizer=lambda: tokenizer))
        )
    )
    assert get_stage0_tokenizer(engine_client) is tokenizer
    with pytest.raises(RuntimeError, match="stage-0 tokenizer"):
        get_stage0_tokenizer(SimpleNamespace())


def test_request_scoped_audio_uuids_control_processor_hashes():
    processor, info = _make_processor()
    prompt = f"{AUDIO_BOS_TOKEN}{AUDIO_TOKEN}{AUDIO_EOS_TOKEN}"
    audio = np.linspace(-0.2, 0.2, 3_201, dtype=np.float32)
    mm_items = info.parse_mm_data({"audio": [(audio, SAMPLE_RATE)]})

    first = processor(
        prompt,
        mm_items=mm_items,
        mm_uuid_items={"audio": ["request-1:audio:0"]},
    )
    repeated = processor(
        prompt,
        mm_items=mm_items,
        mm_uuid_items={"audio": ["request-1:audio:0"]},
    )
    second_request = processor(
        prompt,
        mm_items=mm_items,
        mm_uuid_items={"audio": ["request-2:audio:0"]},
    )

    assert first["mm_hashes"] == repeated["mm_hashes"]
    assert first["mm_hashes"] != second_request["mm_hashes"]
