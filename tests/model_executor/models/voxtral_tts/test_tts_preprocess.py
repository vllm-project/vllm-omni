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


def test_tts_preprocess_seeds_noise_generator_once():
    from vllm_omni.model_executor.models.voxtral_tts.voxtral_tts import _NOISE_GENERATOR_KEY

    model = _make_voxtral_tts_model()
    input_ids = torch.tensor([1, 2, 3])
    input_embeds = torch.zeros((3, 4))

    _, _, info = model.tts_preprocess(input_ids=input_ids, input_embeds=input_embeds, _omni_seed=42)
    generator = info[_NOISE_GENERATOR_KEY]
    expected = torch.Generator().manual_seed(42)
    assert torch.equal(torch.randn(8, generator=generator), torch.randn(8, generator=expected))

    # Later steps keep drawing from the same stream instead of reseeding.
    _, _, info = model.tts_preprocess(input_ids=input_ids[:1], input_embeds=input_embeds[:1], **info)
    assert info[_NOISE_GENERATOR_KEY] is generator

    _, _, info = model.tts_preprocess(input_ids=input_ids, input_embeds=input_embeds, _omni_seed=None)
    assert _NOISE_GENERATOR_KEY not in info


def test_fill_seeded_noise_ignores_global_rng_and_batch_position():
    _voxtral_tts_model_cls()
    from vllm_omni.model_executor.models.voxtral_tts.voxtral_tts_audio_generation import fill_seeded_noise

    def seeded_row(batch_size: int, row: int, global_seed: int) -> torch.Tensor:
        torch.manual_seed(global_seed)
        noise = torch.randn(batch_size, 36)
        generators = [None] * batch_size
        generators[row] = torch.Generator().manual_seed(7)
        fill_seeded_noise(noise, generators)
        return noise[row]

    reference = seeded_row(batch_size=1, row=0, global_seed=0)
    assert torch.equal(seeded_row(batch_size=4, row=2, global_seed=123), reference)

    torch.manual_seed(5)
    noise = torch.randn(2, 36)
    unseeded = noise.clone()
    fill_seeded_noise(noise, None)
    assert torch.equal(noise, unseeded)
