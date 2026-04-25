# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for F5 HF utility helpers."""

import pytest
from pytest_mock import MockerFixture

from vllm_omni.diffusion.models.f5_tts.hf_utils import (
    build_f5_transformer_config,
    is_f5_model,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    (
        "model_id",
        "expected_ckpt",
        "expected_vocoder",
        "expected_vocab_model",
        "expected_text_mask_padding",
        "expected_pe_attn_head",
    ),
    [
        (
            "SWivid/F5-TTS/F5TTS_v1_Base_no_zero_init",
            "model_1250000.safetensors",
            "vocos",
            "SWivid/F5-TTS/F5TTS_v1_Base",
            True,
            None,
        ),
        (
            "SWivid/F5-TTS/F5TTS_Base",
            "model_1200000.safetensors",
            "vocos",
            "SWivid/F5-TTS/F5TTS_Base",
            False,
            1,
        ),
        (
            "SWivid/F5-TTS/F5TTS_Base_bigvgan",
            "model_1250000.pt",
            "bigvgan",
            "SWivid/F5-TTS/F5TTS_Base",
            False,
            1,
        ),
    ],
)
def test_is_f5_model_builds_expected_variant_config(
    model_id: str,
    expected_ckpt: str,
    expected_vocoder: str,
    expected_vocab_model: str,
    expected_text_mask_padding: bool,
    expected_pe_attn_head: int | None,
    mocker: MockerFixture,
):
    assert is_f5_model(model_id) is True

    mock_download = mocker.patch(
        "vllm_omni.diffusion.models.f5_tts.hf_utils.download_f5_file",
        return_value="/tmp/f5_vocab.txt",
    )
    mocker.patch("vllm_omni.diffusion.models.f5_tts.hf_utils._count_vocab_entries", return_value=4096)

    config = build_f5_transformer_config(model_id)

    mock_download.assert_called_once_with(expected_vocab_model, "vocab.txt", revision=None, cache_dir=None)
    assert config["hf_weight_filename"] == expected_ckpt
    assert config["vocoder_name"] == expected_vocoder
    assert config["mel_spec"]["mel_spec_type"] == expected_vocoder
    assert config["text_mask_padding"] is expected_text_mask_padding
    assert config["pe_attn_head"] == expected_pe_attn_head
    assert config["tokenizer_type"] == "pinyin"
