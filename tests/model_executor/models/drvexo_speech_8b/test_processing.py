# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, WhisperFeatureExtractor
from vllm.config import ModelConfig, MultiModalConfig
from vllm.multimodal.processing import InputProcessingContext

from vllm_omni.model_executor.models.drvexo_speech_8b.processing import (
    DRVEXODummyInputsBuilder,
    DRVEXOMultiModalProcessor,
    DRVEXOProcessingInfo,
    extract_audio_mels,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("seconds", [0.2, 1.0])
def test_current_vllm_media_only_frontend_matches_native_mels(seconds):
    config = object.__new__(ModelConfig)
    config.dtype = torch.bfloat16
    config.multimodal_config = MultiModalConfig()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")))
    info = DRVEXOProcessingInfo(InputProcessingContext(config, tokenizer))
    info.__dict__["feature_extractor"] = WhisperFeatureExtractor(feature_size=128)
    processor = DRVEXOMultiModalProcessor(info, DRVEXODummyInputsBuilder(info))
    audio = np.sin(np.arange(int(16000 * seconds), dtype=np.float32) * 0.05)
    items = info.parse_mm_data({"audio": [audio]})
    actual = processor._apply_hf_processor_main(items, {})
    mels, lengths = extract_audio_mels(info.feature_extractor, [audio])
    torch.testing.assert_close(actual["audio_mels"], mels.to(torch.bfloat16))
    torch.testing.assert_close(actual["audio_mel_lengths"], lengths)
    assert "input_ids" not in actual
