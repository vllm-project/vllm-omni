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


def test_tts_postprocess_uses_explicit_batch_row_without_advancing_v1_counter():
    model = _make_voxtral_tts_model()
    model.post_process_idx = 0
    frames = [torch.tensor([[1, 2]]), torch.tensor([[3, 4]])]

    update = model.tts_postprocess(
        torch.zeros(1, 4),
        multimodal_outputs={"codes": {"audio": frames}},
        _omni_batch_row=1,
    )

    torch.testing.assert_close(update["codes"]["audio"], frames[1])
    assert model.post_process_idx == 0


def test_tts_postprocess_retains_v1_request_order_fallback():
    model = _make_voxtral_tts_model()
    model.post_process_idx = 0
    frames = [torch.tensor([[1, 2]]), torch.tensor([[3, 4]])]

    first = model.tts_postprocess(torch.zeros(1, 4), multimodal_outputs={"codes": {"audio": frames}})
    second = model.tts_postprocess(torch.zeros(1, 4), multimodal_outputs={"codes": {"audio": frames}})

    torch.testing.assert_close(first["codes"]["audio"], frames[0])
    torch.testing.assert_close(second["codes"]["audio"], frames[1])
    assert model.post_process_idx == 2
