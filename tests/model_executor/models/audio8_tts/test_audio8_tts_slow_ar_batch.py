# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.audio8_tts import audio8_tts_slow_ar as slow_ar_module
from vllm_omni.model_executor.models.audio8_tts.prompt_utils import build_voice_clone_prompt_ids

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_model(*, window: int = 0):
    cls = slow_ar_module.Audio8TTSSlowARForConditionalGeneration
    model = object.__new__(cls)
    torch.nn.Module.__init__(model)
    model._ras_window_size = window
    model._ras_recent_gpu = None
    model._ras_recent_staging = []
    model._ras_req_ids = []
    model._ras_intermediate_buffer = None
    model.vllm_config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=4))
    model.text_config = SimpleNamespace(hidden_size=4)
    model._num_semantic_ids = 8
    model._semantic_begin_id = 0
    model._semantic_end_id = 7
    model._eos_token_id = 8
    model._ras_temperature = 0.5
    model._ras_top_p = 0.9
    model._num_codebooks = 10
    model._pad_token_id = 0
    model._codec_kwargs = lambda: {}
    model.model_path = "unused"
    model.embed_input_ids = lambda ids: ids.float().unsqueeze(-1).expand(*ids.shape, 4)
    return model


def test_preprocess_decode_batch_embeds_once_and_preserves_row_controls():
    model = _make_model()
    calls = []

    def embed(ids):
        calls.append(tuple(ids.shape))
        return ids.float().unsqueeze(-1).expand(*ids.shape, 4)

    model.embed_input_ids = embed
    ids = torch.tensor([2, 3, 4])
    hidden = torch.full((4,), 7.0)
    result = model.preprocess_decode_batch(
        input_ids=ids,
        req_infos=[
            {"hidden_states": {"last": hidden}},
            {},
            {"additional_information": {"hidden_states": {"last": hidden + 1}}},
        ],
    )
    output_ids, embeds, last_hidden, text_step, updates = result
    assert calls == [(3, 1)]
    torch.testing.assert_close(output_ids, ids)
    assert embeds.shape == last_hidden.shape == text_step.shape == (3, 4)
    torch.testing.assert_close(last_hidden[0], hidden.to(torch.bfloat16))
    torch.testing.assert_close(last_hidden[1], torch.zeros(4, dtype=torch.bfloat16))
    torch.testing.assert_close(last_hidden[2], (hidden + 1).to(torch.bfloat16))
    assert text_step[:, 0].tolist() == [1, 0, 1]
    assert updates == [{}, {}, {}]
    _, scalar_embed, scalar_update = model._preprocess_decode(ids[:1], {"hidden_states": {"last": hidden}})
    torch.testing.assert_close(embeds[:1], scalar_embed)
    torch.testing.assert_close(last_hidden[:1], scalar_update["mtp_inputs"][0])
    torch.testing.assert_close(text_step[:1], scalar_update["mtp_inputs"][1])


def test_preprocess_batch_encodes_reference_audio_in_one_list_call(monkeypatch):
    model = _make_model()
    encoded_batches = []
    cached = {}

    class FakeCodec:
        frame_length = 2

        def encode(self, waveforms):
            encoded_batches.append([wav.numel() for wav in waveforms])
            return [torch.full((10, (wav.numel() + 1) // 2), index) for index, wav in enumerate(waveforms)]

    monkeypatch.setattr(slow_ar_module, "load_arktts_codec", lambda *args, **kwargs: FakeCodec())
    monkeypatch.setattr(slow_ar_module, "prepare_reference_waveform", lambda wav, sr, **kwargs: wav)
    model._speaker_cache = SimpleNamespace(
        make_cache_key=lambda name, **kwargs: ("audio8_tts", name, 0),
        get=lambda key: cached.get(key),
        put=lambda key, value: cached.__setitem__(key, value),
    )
    model._embed_voice_clone_prompt = lambda text, ref_text, codes: torch.full((2, 4), codes[0, 0].item())
    infos = {
        "a": {
            "audio8_structured_voice_clone": True,
            "text": "a",
            "ref_text": "ref",
            "voice_name": "a",
            "ref_audio_sr": 44100,
            "ref_audio_wav": torch.zeros(3),
        },
        "b": {
            "audio8_structured_voice_clone": True,
            "text": "b",
            "ref_text": "ref",
            "voice_name": "b",
            "ref_audio_sr": 44100,
            "ref_audio_wav": torch.zeros(5),
        },
    }
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", lambda self: (_ for _ in ()).throw(AssertionError("CPU readback")))
        model.preprocess_batch(req_ids=["a", "b"], model_intermediate_buffer=infos, device=torch.device("cpu"))
    assert encoded_batches == [[3, 5]]
    assert cached[("audio8_tts", "a", 0)]["ref_codes_fq"].shape == (2, 10)
    assert infos["a"]["embed"]["prefill"].shape == (2, 4)
    assert infos["b"]["embed"]["prefill"].shape == (2, 4)
    input_ids, embeds, update = model._preprocess_prefill(torch.tensor([1, 2]), infos["a"], 2)
    assert input_ids.tolist() == [0, 0]
    assert embeds.shape == (2, 4)
    assert update["meta"]["prefill_offset"] == 2
    model.preprocess_batch(req_ids=["a", "b"], model_intermediate_buffer=infos, device=torch.device("cpu"))
    assert encoded_batches == [[3, 5]]


def test_resident_prefill_buffer_is_not_replaced_between_chunks():
    model = _make_model()
    prompt = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
    info = {"embed": {"prefill": prompt}, "meta": {"prefill_offset": 0}}
    _, first, first_update = model._preprocess_prefill(torch.tensor([1, 2]), info, 2)
    torch.testing.assert_close(first, prompt[:2])
    assert "embed" not in first_update
    info["meta"].update(first_update["meta"])
    _, second, second_update = model._preprocess_prefill(torch.tensor([3, 4]), info, 2)
    torch.testing.assert_close(second, prompt[2:])
    assert second_update["embed"]["prefill"] is None


def test_single_token_prefill_uses_runner_phase_flag():
    model = _make_model()
    token_ids, embeds, update = model.preprocess(
        input_ids=torch.tensor([7]),
        input_embeds=None,
        _omni_is_prefill=True,
    )
    assert token_ids.tolist() == [model._pad_token_id]
    assert embeds.shape == (1, 4)
    assert update["meta"]["prefill_offset"] == 1
    assert update["codes"]["audio"].shape == (1, model._num_codebooks)


def test_gpu_resident_ras_history_follows_request_ids_across_reordering():
    model = _make_model(window=3)
    infos = {"a": {}, "b": {}}
    model.preprocess_batch(req_ids=["a", "b"], model_intermediate_buffer=infos, device=torch.device("cpu"))
    assert model._resident_recent_local_ids(2, torch.device("cpu")) is None
    model._update_resident_recent_local_ids(torch.tensor([2, 4]), 2)
    model.preprocess_batch(req_ids=["b", "a"], model_intermediate_buffer=infos, device=torch.device("cpu"))
    recent = model._resident_recent_local_ids(2, torch.device("cpu"))
    assert recent.tolist() == [[-1, -1, 4], [-1, -1, 2]]
    assert model.skips_model_sampler_output_token_history is True


def test_sample_uses_resident_history_without_host_output_token_ids(monkeypatch):
    model = _make_model(window=3)
    infos = {"a": {}, "b": {}}
    model.preprocess_batch(req_ids=["a", "b"], model_intermediate_buffer=infos, device=torch.device("cpu"))
    seen = []

    def fake_ras(logits, recent, **kwargs):
        seen.append(None if recent is None else recent.tolist())
        return torch.tensor([2, 4] if len(seen) == 1 else [3, 5])

    monkeypatch.setattr(slow_ar_module, "ras_sample_batch", fake_ras)
    metadata = SimpleNamespace(
        max_num_logprobs=None,
        no_penalties=True,
        bad_words_token_ids={},
        allowed_token_ids_mask=None,
        logitsprocs=SimpleNamespace(non_argmax_invariant=[]),
        temperature=torch.ones(2),
        all_greedy=False,
        top_p=None,
        top_k=None,
        output_token_ids=[],
        generators={},
    )
    logits = torch.zeros(2, 9)
    model.sample(logits, metadata)
    model.sample(logits, metadata)
    assert seen == [None, [[-1, -1, 2], [-1, -1, 4]]]

    metadata.no_penalties = False
    assert model.sample(logits, metadata) is None
    assert model.skips_model_sampler_output_token_history is False
    metadata.no_penalties = True
    metadata.output_token_ids = [[2], [4]]
    model.sample(logits, metadata)
    assert seen[-1] == [[-1, -1, 2], [-1, -1, 4]]


def test_talker_mtp_keeps_missing_hidden_row_text_only():
    model = _make_model()
    model._codebook_size = 16
    model.codebook_embeddings = torch.nn.Embedding(160, 4)
    model.codebook_embeddings.weight.data.fill_(1)
    model.fast_ar = lambda **kwargs: torch.ones((2, 10), dtype=torch.long)
    control = torch.zeros((2, 4), dtype=torch.bfloat16)
    control[0, 0] = 1
    embeds, codes = model.talker_mtp(
        input_ids=torch.tensor([2, 3]),
        input_embeds=torch.zeros((2, 4), dtype=torch.bfloat16),
        last_talker_hidden=torch.zeros((2, 4), dtype=torch.bfloat16),
        text_step=control,
    )
    torch.testing.assert_close(embeds[0], torch.full((4,), 10, dtype=torch.bfloat16))
    torch.testing.assert_close(embeds[1], torch.zeros(4, dtype=torch.bfloat16))
    assert codes[0].tolist() == [1] * 10
    assert codes[1].tolist() == [0] * 10


def test_voice_clone_prompt_embeds_splice_semantic_codes_without_tensor_tolist(monkeypatch):
    model = _make_model()
    model._codebook_size = 16
    model._semantic_begin_id = 100
    model.codebook_embeddings = torch.nn.Embedding(160, 4)
    model.codebook_embeddings.weight.data.fill_(1)

    class Tokenizer:
        def __init__(self):
            self.vocab = {}

        def encode(self, text, add_special_tokens=False):
            return [self.vocab.setdefault(text, 20 + len(self.vocab))]

    tokenizer = Tokenizer()
    model._get_tokenizer = lambda: tokenizer
    codes = torch.tensor([[1] * 10, [2] * 10], dtype=torch.long)
    expected_ids, ref_start, _, _ = build_voice_clone_prompt_ids(tokenizer, "target", "reference", [101, 102])

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "tolist", lambda self: (_ for _ in ()).throw(AssertionError("GPU readback")))
        embeds = model._embed_voice_clone_prompt("target", "reference", codes)

    expected = torch.tensor(expected_ids, dtype=torch.bfloat16).unsqueeze(-1).expand(-1, 4).clone()
    expected[ref_start : ref_start + 2] += 10
    torch.testing.assert_close(embeds, expected)
