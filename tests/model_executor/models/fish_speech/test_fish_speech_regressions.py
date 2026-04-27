import functools
import math

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@functools.lru_cache(maxsize=1)
def _fish_speech_regression_modules():
    from vllm_omni.model_executor.models.fish_speech import fish_speech_slow_ar as slow_ar_module
    from vllm_omni.model_executor.models.fish_speech.fish_speech_dac_decoder import FishSpeechDACDecoder
    from vllm_omni.model_executor.models.fish_speech.fish_speech_slow_ar import (
        FishSpeechSlowARForConditionalGeneration,
    )

    return slow_ar_module, FishSpeechDACDecoder, FishSpeechSlowARForConditionalGeneration


class _FakeCodec:
    def decode(self, codes_bqf: torch.Tensor, feature_lengths: torch.Tensor):
        del codes_bqf, feature_lengths
        wav = torch.arange(100, dtype=torch.float32).view(1, 1, 100)
        audio_lengths = torch.tensor([100], dtype=torch.long)
        return wav, audio_lengths


class _FakeTokenizer:
    def __init__(self, mapping, unk_token_id=-1):
        self._mapping = mapping
        self.unk_token_id = unk_token_id

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._mapping.get(token, self.unk_token_id)


def test_dac_decoder_mixed_batch_empty_request_does_not_misalign_indices():
    _, FishSpeechDACDecoder, _ = _fish_speech_regression_modules()
    decoder = object.__new__(FishSpeechDACDecoder)
    torch.nn.Module.__init__(decoder)
    decoder._codec = _FakeCodec()
    decoder._num_codebooks = 10
    decoder._output_sample_rate = 44100
    decoder._hop_length = 512
    decoder._logged_codec_stats = False
    decoder._ensure_codec_loaded = lambda: None
    decoder._split_request_ids = lambda ids, seq_token_counts=None: [
        torch.empty((0,), dtype=torch.long),
        torch.arange(20, dtype=torch.long),
    ]

    out = decoder.forward(
        input_ids=torch.arange(20, dtype=torch.long),
        runtime_additional_information=[{}, {"meta": {"left_context_size": 1}}],
    )

    audios = out.multimodal_outputs["model_outputs"]
    assert len(audios) == 2
    assert audios[0].numel() == 0
    # 2 total frames with 1 frame of left context => proportional trim removes half the samples.
    assert audios[1].shape[0] == 50


def test_structured_voice_clone_prefill_adds_full_codebooks_with_decode_scale(monkeypatch):
    slow_ar_module, _, FishSpeechSlowARForConditionalGeneration = _fish_speech_regression_modules()
    model = object.__new__(FishSpeechSlowARForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model._num_codebooks = 2
    model._codebook_size = 8
    model._semantic_begin_id = 100
    model.model_path = "unused"

    hidden_size = 3
    text_embed = torch.nn.Embedding(256, hidden_size)
    codebook_embed = torch.nn.Embedding(model._num_codebooks * model._codebook_size, hidden_size)
    with torch.no_grad():
        text_embed.weight.zero_()
        text_embed.weight[20] = torch.tensor([1.0, 2.0, 3.0])
        text_embed.weight[21] = torch.tensor([4.0, 5.0, 6.0])
        codebook_embed.weight.zero_()
        codebook_embed.weight[1] = torch.tensor([10.0, 0.0, 0.0])
        codebook_embed.weight[10] = torch.tensor([0.0, 20.0, 0.0])
        codebook_embed.weight[3] = torch.tensor([30.0, 0.0, 0.0])
        codebook_embed.weight[12] = torch.tensor([0.0, 40.0, 0.0])

    model.embed_input_ids = lambda ids: text_embed(ids)
    model.codebook_embeddings = codebook_embed
    model._get_tokenizer = lambda: _FakeTokenizer({"<|audio_start|>": 10, "<|audio_end|>": 11})

    monkeypatch.setattr(
        slow_ar_module,
        "encode_reference_audio_codes",
        lambda *args, **kwargs: torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
    )
    monkeypatch.setattr(
        slow_ar_module,
        "build_fish_voice_clone_prompt_ids",
        lambda tokenizer, text, ref_text, semantic_token_ids: ([1, 10, 20, 21, 11, 2], None, None),
    )

    prefill = model._build_structured_voice_clone_prefill_embeds(
        {
            "ref_text": "ref",
            "text": "target",
            "ref_audio_wav": torch.tensor([0.0]),
            "ref_audio_sr": 16000,
        }
    )

    expected_0 = (torch.tensor([1.0, 2.0, 3.0]) + torch.tensor([10.0, 20.0, 0.0])) / math.sqrt(3.0)
    expected_1 = (torch.tensor([4.0, 5.0, 6.0]) + torch.tensor([30.0, 40.0, 0.0])) / math.sqrt(3.0)
    assert torch.allclose(prefill[2].to(dtype=torch.float32), expected_0, atol=2e-2, rtol=0)
    assert torch.allclose(prefill[3].to(dtype=torch.float32), expected_1, atol=2e-2, rtol=0)
