# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for TTS feedback preprocessing and eager EOS write-back."""

import functools

import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@functools.lru_cache(maxsize=1)
def _voxtral_tts_model_cls():
    from tests.model_executor.helpers import bootstrap_vllm_layer_custom_op_modules

    bootstrap_vllm_layer_custom_op_modules()
    import vllm.model_executor.models.utils  # noqa: F401

    from vllm_omni.model_executor.models.voxtral_tts.voxtral_tts import (
        VoxtralTTSForConditionalGeneration,
    )

    return VoxtralTTSForConditionalGeneration


class FakeAudioGeneration(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_multimodal_calls = []

    def embed_multimodal(self, **kwargs):
        self.embed_multimodal_calls.append(kwargs)
        return [torch.full((1, 4), 7.0)]


def _make_voxtral_tts_model():
    model_cls = _voxtral_tts_model_cls()
    model = model_cls.__new__(model_cls)
    nn.Module.__init__(model)
    model.model_stage = "audio_generation"
    model.model = FakeAudioGeneration()
    model._audio_token_id = 42
    return model


def test_tts_preprocess_consumes_nested_codes_audio_feedback():
    model = _make_voxtral_tts_model()
    input_ids = torch.tensor([model._audio_token_id])
    input_embeds = torch.zeros((1, 4))
    audio_tokens = torch.tensor([[1, 2, 3, 4]])

    _, output_embeds, _ = model.tts_preprocess(
        input_ids=input_ids,
        input_embeds=input_embeds,
        codes={"audio": audio_tokens},
    )

    assert len(model.model.embed_multimodal_calls) == 1
    torch.testing.assert_close(model.model.embed_multimodal_calls[0]["audio_tokens"], audio_tokens)
    torch.testing.assert_close(output_embeds, torch.full((1, 4), 7.0))


def test_tts_preprocess_keeps_legacy_top_level_audio_feedback():
    model = _make_voxtral_tts_model()
    input_ids = torch.tensor([model._audio_token_id])
    input_embeds = torch.zeros((1, 4))
    audio_tokens = torch.tensor([[5, 6, 7, 8]])

    _, output_embeds, _ = model.tts_preprocess(
        input_ids=input_ids,
        input_embeds=input_embeds,
        audio=audio_tokens,
    )

    assert len(model.model.embed_multimodal_calls) == 1
    torch.testing.assert_close(model.model.embed_multimodal_calls[0]["audio_tokens"], audio_tokens)
    torch.testing.assert_close(output_embeds, torch.full((1, 4), 7.0))


class _FixedAudioCodes(nn.Module):
    def __init__(self, codes: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("codes", codes)

    def forward(self, llm_hidden: torch.Tensor, cfg_alpha: torch.Tensor) -> torch.Tensor:
        return self.codes


@pytest.fixture
def audio_generation_model():
    model = _make_voxtral_tts_model()
    from vllm_omni.model_executor.models.voxtral_tts.voxtral_tts_audio_generation import (
        AudioSpecialTokens,
        VoxtralTTSAudioGenerationForConditionalGeneration,
    )

    end_audio_id = AudioSpecialTokens.id(AudioSpecialTokens.end_audio)
    audio_codes = torch.tensor([[end_audio_id, 2, 3], [2, 4, 5]], dtype=torch.long)

    # Exercise the real EOS producer and its indexed-assignment consumer;
    # the acoustic transformer only supplies deterministic integer codes.
    generator = VoxtralTTSAudioGenerationForConditionalGeneration.__new__(
        VoxtralTTSAudioGenerationForConditionalGeneration
    )
    nn.Module.__init__(generator)
    generator.acoustic_transformer = _FixedAudioCodes(audio_codes)
    generator._end_audio_token_id = end_audio_id

    model.model = generator
    model._cudagraph_acoustic_transformer = None
    return model, audio_codes


@pytest.mark.parametrize(
    "model_dtype", [torch.float16, torch.bfloat16, torch.float32], ids=["float16", "bfloat16", "float32"]
)
def test_eager_eos_matches_hidden_states_dtype(audio_generation_model, model_dtype: torch.dtype):
    model, audio_codes = audio_generation_model
    logits_indices = torch.tensor([1, 3])

    # Reuse the model across dtypes to catch stale cached EOS constants too.
    for dtype in (torch.bfloat16, model_dtype):
        hidden_states = torch.full((4, 8), 0.25, dtype=dtype)
        expected = torch.full_like(hidden_states, 0.25)
        expected[1, 0] = 1.0
        expected[3, 0] = 0.0

        with torch.inference_mode():
            output = model.make_omni_output(hidden_states, logits_index=logits_indices)

        assert output.text_hidden_states.dtype == dtype
        torch.testing.assert_close(output.text_hidden_states, expected, rtol=0, atol=0)
        actual_codes = torch.cat(output.multimodal_outputs["codes"]["audio"], dim=0).squeeze(1)
        torch.testing.assert_close(actual_codes, audio_codes)
