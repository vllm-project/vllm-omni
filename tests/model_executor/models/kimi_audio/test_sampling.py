# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for Kimi-Audio dual-stream sampling."""

import pytest
import torch

from vllm_omni.model_executor.models.kimi_audio.prompt import KimiAudioSpecialTokens
from vllm_omni.model_executor.models.kimi_audio.sampling import KimiAudioSamplingParams, sample_kimi_audio_step

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.omni]


@pytest.fixture
def special_tokens():
    return KimiAudioSpecialTokens(
        msg_end=0,
        media_begin=1,
        media_end=2,
        kimia_text_blank=3,
        kimia_text_eos=4,
        kimia_user_msg_start=5,
        kimia_assistant_msg_start=6,
        kimia_speech_ct_id=7,
        kimia_speech_ctd_id=8,
    )


@pytest.mark.parametrize("top_k", [0, 5])
def test_low_temperature_sampling_remains_valid(special_tokens, top_k):
    logits = torch.full((16,), -100.0, dtype=torch.float16)
    result = sample_kimi_audio_step(
        logits,
        logits,
        text_history=[],
        audio_history=[],
        text_finished=False,
        output_type="both",
        special_tokens=special_tokens,
        audio_delay=0,
        params=KimiAudioSamplingParams(
            text_temperature=1e-4, audio_temperature=1e-4, text_top_k=top_k, audio_top_k=top_k
        ),
        generator=torch.Generator().manual_seed(42),
    )
    assert 0 <= result.text_token < len(logits)
    assert 0 <= result.audio_token < len(logits)


@pytest.mark.parametrize("step", [0, 1, 2])
def test_audio_is_blank_until_delay_elapses(special_tokens, step):
    logits = torch.zeros(16)
    logits[10] = 1.0
    result = sample_kimi_audio_step(
        logits,
        logits,
        text_history=[10] * step,
        audio_history=[special_tokens.kimia_text_blank] * step,
        text_finished=False,
        output_type="both",
        special_tokens=special_tokens,
        audio_delay=2,
    )
    assert result.text_token == 10
    assert result.audio_token == (special_tokens.kimia_text_blank if step < 2 else 10)
    assert not result.finished


@pytest.mark.parametrize(
    "output_type,already_finished,audio_ends,expected_finished",
    [
        ("text", False, False, True),
        ("both", False, False, False),
        ("both", True, False, False),
        ("both", True, True, True),
        ("both", False, True, True),
    ],
)
def test_stream_end_conditions(special_tokens, output_type, already_finished, audio_ends, expected_finished):
    text_logits, audio_logits = torch.zeros(16), torch.zeros(16)
    text_logits[10 if already_finished or audio_ends else special_tokens.kimia_text_eos] = 1.0
    audio_logits[special_tokens.media_end if audio_ends else 10] = 1.0
    result = sample_kimi_audio_step(
        text_logits,
        audio_logits,
        text_history=[special_tokens.kimia_text_eos] if already_finished else [],
        audio_history=[10] if already_finished else [],
        text_finished=already_finished,
        output_type=output_type,
        special_tokens=special_tokens,
        audio_delay=0,
    )
    assert result.text_finished == (already_finished or not audio_ends)
    assert result.finished == expected_finished
    if already_finished:
        assert result.text_token == special_tokens.kimia_text_blank
    if output_type == "text":
        assert result.audio_token == special_tokens.kimia_text_blank
