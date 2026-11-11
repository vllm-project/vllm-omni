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


def _make_voxtral_tts_model_with_real_embeddings(hidden_size: int = 8):
    """A Voxtral TTS wrapper whose generation model uses the real audio-token embedding path."""
    from vllm_omni.model_executor.models.voxtral_tts.voxtral_tts_audio_generation import (
        VoxtralTTSAudioGenerationForConditionalGeneration as GenerationModel,
    )
    from vllm_omni.model_executor.models.voxtral_tts.voxtral_tts_audio_tokenizer import (
        MultiVocabEmbeddings,
        VoxtralTTSAudioTokenizer,
    )

    audio_model_args = {
        "semantic_codebook_size": 16,
        "acoustic_codebook_size": 5,
        "n_acoustic_codebook": 3,
        "acoustic_transformer_args": {"input_dim": hidden_size},
    }

    class AudioTokenizer(nn.Module):
        encode_tokens = VoxtralTTSAudioTokenizer.encode_tokens

        def __init__(self):
            super().__init__()
            self.audio_token_embedding = MultiVocabEmbeddings(audio_model_args, embedding_dim=hidden_size)

    class AudioGeneration(nn.Module):
        embed_multimodal = GenerationModel.embed_multimodal
        _parse_and_validate_audio_arrays = GenerationModel._parse_and_validate_audio_arrays

        def __init__(self):
            super().__init__()
            self.audio_tokenizer = AudioTokenizer()
            self.text_embedding = nn.Embedding(64, hidden_size)

        def embed_input_ids(self, input_ids):
            return self.text_embedding(input_ids)

    model = _make_voxtral_tts_model()
    model.model = AudioGeneration()
    codebook_sizes = model.model.audio_tokenizer.audio_token_embedding.codebook_sizes
    return model, codebook_sizes


def _random_frame(codebook_sizes: list[int]) -> torch.Tensor:
    # Shape matches what tts_postprocess stores per request: (1, 1, num_codebooks).
    return torch.stack([torch.randint(0, size, ()) for size in codebook_sizes]).view(1, 1, -1)


def test_preprocess_decode_batch_matches_per_row_tts_preprocess():
    torch.manual_seed(0)
    model, codebook_sizes = _make_voxtral_tts_model_with_real_embeddings()
    input_ids = torch.tensor([model._audio_token_id] * 4)
    req_infos = [{"codes": {"audio": _random_frame(codebook_sizes)}} for _ in range(4)]

    with torch.no_grad():
        expected = torch.cat(
            [
                model.tts_preprocess(
                    input_ids[i : i + 1],
                    model.model.embed_input_ids(input_ids[i : i + 1]),
                    **dict(info),
                )[1].reshape(1, -1)
                for i, info in enumerate(req_infos)
            ]
        )
        out_ids, embeds, updates = model.preprocess_decode_batch(input_ids=input_ids, req_infos=req_infos)

    assert out_ids is input_ids
    assert updates == [{}, {}, {}, {}]
    torch.testing.assert_close(embeds, expected, rtol=0, atol=0)


def test_preprocess_decode_batch_row_without_frame_keeps_token_embedding():
    torch.manual_seed(0)
    model, codebook_sizes = _make_voxtral_tts_model_with_real_embeddings()
    input_ids = torch.tensor([model._audio_token_id] * 3)
    req_infos = [
        {"codes": {"audio": _random_frame(codebook_sizes)}},
        {},
        {"codes": {"audio": _random_frame(codebook_sizes)}},
    ]

    with torch.no_grad():
        _, embeds, updates = model.preprocess_decode_batch(input_ids=input_ids, req_infos=req_infos)
        for row in (0, 2):
            _, expected, _ = model.tts_preprocess(
                input_ids[row : row + 1], model.model.embed_input_ids(input_ids[row : row + 1]), **dict(req_infos[row])
            )
            torch.testing.assert_close(embeds[row : row + 1], expected.reshape(1, -1), rtol=0, atol=0)
        torch.testing.assert_close(embeds[1], model.model.embed_input_ids(input_ids[1:2])[0], rtol=0, atol=0)
    assert updates == [{}, {}, {}]
