# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU coverage for the Confucius4-TTS offline helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm import SamplingParams

from vllm_omni.model_extras import confucius4_tts

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_prompt_builds_placeholder_and_conditioning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        confucius4_tts,
        "tokenize_confucius4_prompt",
        lambda model, text, lang: [10, 11, 12],
    )

    prompt = confucius4_tts.confucius4_tts_prompt(
        "model",
        " hello ",
        "reference.wav",
        lang="en",
    )

    assert prompt["prompt_token_ids"] == [0] * 5
    assert prompt["additional_information"] == {
        "text": ["hello"],
        "lang": ["en"],
        "text_token_ids": [[10, 11, 12]],
        "voice": ["reference.wav"],
    }


def test_prompt_rejects_invalid_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="text must not be empty"):
        confucius4_tts.confucius4_tts_prompt(
            "model",
            " ",
            "reference.wav",
        )
    with pytest.raises(ValueError, match="ref_audio is required"):
        confucius4_tts.confucius4_tts_prompt(
            "model",
            "hello",
            None,
        )

    monkeypatch.setattr(
        confucius4_tts,
        "tokenize_confucius4_prompt",
        lambda model, text, lang: [0] * 521,
    )
    with pytest.raises(ValueError, match="exceeds 520 tokens"):
        confucius4_tts.confucius4_tts_prompt(
            "model",
            "hello",
            "reference.wav",
        )


def test_sampling_params_use_requested_budget() -> None:
    talker, decoder = confucius4_tts.confucius4_tts_sampling_params(
        max_new_tokens=10,
        seed=7,
    )

    assert isinstance(talker, SamplingParams)
    assert isinstance(decoder, SamplingParams)
    assert talker.max_tokens == 10
    assert talker.stop_token_ids == [8193]
    assert talker.seed == 7
    assert decoder.max_tokens == 1
    assert decoder.seed == 7


@pytest.mark.parametrize("max_new_tokens", [0, 1519])
def test_sampling_params_reject_invalid_budget(max_new_tokens: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 1518"):
        confucius4_tts.confucius4_tts_sampling_params(max_new_tokens=max_new_tokens)


def test_audio_extracts_direct_and_nested_outputs() -> None:
    direct = SimpleNamespace(multimodal_output={"audio": [torch.tensor([1.0]), torch.tensor([2.0, 3.0])]})
    nested = SimpleNamespace(
        multimodal_output=None,
        outputs=[SimpleNamespace(multimodal_output={"model_outputs": torch.tensor([[4.0, 5.0]])})],
    )

    assert torch.equal(
        confucius4_tts.confucius4_tts_audio(direct),
        torch.tensor([1.0, 2.0, 3.0]),
    )
    assert torch.equal(
        confucius4_tts.confucius4_tts_audio(nested),
        torch.tensor([4.0, 5.0]),
    )


def test_audio_rejects_missing_waveform() -> None:
    with pytest.raises(ValueError, match="no multimodal output"):
        confucius4_tts.confucius4_tts_audio(SimpleNamespace())
    with pytest.raises(ValueError, match="no audio"):
        confucius4_tts.confucius4_tts_audio(SimpleNamespace(multimodal_output={"image": torch.zeros(1)}))
