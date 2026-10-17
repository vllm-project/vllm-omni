# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

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


class FakeAudioGenerationWithAcoustic(FakeAudioGeneration):
    def __init__(self):
        super().__init__()
        self.compute_mm_logits_calls = []
        self.fake_eos = torch.tensor([0.25, 0.75])
        self.multimodal_outputs = {"codes": {"audio": ["audio-code-1", "audio-code-2"]}}

    def compute_mm_logits(self, hidden_states, cfg_alpha):
        self.compute_mm_logits_calls.append((hidden_states.clone(), cfg_alpha.clone()))
        return self.fake_eos, self.multimodal_outputs


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


def test_make_omni_output_uses_logits_index_for_acoustic_path():
    from vllm_omni.model_executor.models.output_templates import OmniOutput

    model = _make_voxtral_tts_model()
    model.model = FakeAudioGenerationWithAcoustic()
    model._cudagraph_acoustic_transformer = None

    hidden_states = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    original_hidden_states = hidden_states.clone()
    logits_index = torch.tensor([1, 3])

    output = model.make_omni_output(hidden_states, logits_index=logits_index)

    assert isinstance(output, OmniOutput)
    assert output.multimodal_outputs is model.model.multimodal_outputs
    assert len(model.model.compute_mm_logits_calls) == 1

    selected_hidden_states, cfg_alpha = model.model.compute_mm_logits_calls[0]
    torch.testing.assert_close(selected_hidden_states, original_hidden_states[logits_index])
    torch.testing.assert_close(cfg_alpha, torch.full((2,), model._DEFAULT_CFG_ALPHA))

    expected_hidden_states = original_hidden_states.clone()
    expected_hidden_states[logits_index, 0] = model.model.fake_eos
    torch.testing.assert_close(output.text_hidden_states, expected_hidden_states)
    torch.testing.assert_close(output.text_hidden_states[[0, 2, 4]], original_hidden_states[[0, 2, 4]])
