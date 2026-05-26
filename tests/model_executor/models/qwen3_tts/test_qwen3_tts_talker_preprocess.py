from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import torch

from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
    _NORMALIZED_REF_AUDIO_KEY,
    _PRECOMPUTED_REF_CODE_KEY,
    _PRECOMPUTED_REF_IDS_KEY,
    _PRECOMPUTED_TEXT_IDS_KEY,
    Qwen3TTSTalkerForConditionalGeneration,
)


def _make_minimal_talker():
    model = Qwen3TTSTalkerForConditionalGeneration.__new__(Qwen3TTSTalkerForConditionalGeneration)
    model.talker_config = SimpleNamespace(codec_pad_id=7, num_code_groups=16)
    model._ref_audio_artifact_cache_max_entries = 256
    model._ref_audio_artifact_cache = OrderedDict()
    return model


def test_single_token_prefill_uses_prefill_path():
    model = _make_minimal_talker()
    full_prompt_embeds = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    trailing_text = torch.ones((2, 4), dtype=torch.float32)
    tts_pad = torch.full((1, 4), 0.5, dtype=torch.float32)
    ref_code = torch.arange(32, dtype=torch.long).reshape(2, 16)

    def fake_build_prompt_embeds(*, task_type, info_dict):
        return full_prompt_embeds, trailing_text, tts_pad, 2, ref_code

    model._build_prompt_embeds = fake_build_prompt_embeds

    input_ids = torch.tensor([123], dtype=torch.long)
    out_ids, out_embeds, update = model.preprocess(
        input_ids=input_ids,
        input_embeds=None,
        text=["hello"],
        task_type=["CustomVoice"],
        _omni_is_prefill=True,
        _omni_num_computed_tokens=0,
        _omni_prompt_len=3,
    )

    assert out_ids.tolist() == [7]
    assert torch.equal(out_embeds.cpu(), full_prompt_embeds[:1].to(torch.bfloat16))
    assert update["meta"]["talker_prefill_offset"] == 1
    assert update["meta"]["talker_text_offset"] == 0
    assert update["meta"]["ref_code_len"] == 2
    assert torch.equal(update["embed"]["prefill"], full_prompt_embeds)
    assert torch.equal(update["embed"]["tts_pad"], tts_pad)
    assert torch.equal(update["hidden_states"]["trailing_text"], trailing_text)
    assert torch.equal(update["codes"]["ref"], ref_code)
    assert update["codes"]["audio"].shape == (1, 16)


def test_single_token_prefill_can_be_inferred_from_token_progress():
    model = _make_minimal_talker()
    full_prompt_embeds = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    trailing_text = torch.ones((1, 4), dtype=torch.float32)
    tts_pad = torch.zeros((1, 4), dtype=torch.float32)

    def fake_build_prompt_embeds(*, task_type, info_dict):
        return full_prompt_embeds, trailing_text, tts_pad, None, None

    model._build_prompt_embeds = fake_build_prompt_embeds

    out_ids, out_embeds, update = model.preprocess(
        input_ids=torch.tensor([123], dtype=torch.long),
        input_embeds=None,
        text=["hello"],
        task_type=["CustomVoice"],
        _omni_num_computed_tokens=0,
        _omni_prompt_len=2,
    )

    assert out_ids.tolist() == [7]
    assert torch.equal(out_embeds.cpu(), full_prompt_embeds[:1].to(torch.bfloat16))
    assert update["meta"]["talker_prefill_offset"] == 1


def test_decode_advances_trailing_text_by_offset_without_rewriting_tail():
    model = _make_minimal_talker()

    def fake_embed_input_ids(input_ids):
        return input_ids.to(torch.float32).reshape(1, 1, 1).expand(1, 1, 4)

    model.embed_input_ids = fake_embed_input_ids
    trailing_text = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    last_hidden = torch.full((4,), 2.0, dtype=torch.float32)
    tts_pad = torch.full((1, 4), -1.0, dtype=torch.float32)

    out_ids, out_embeds, update = model.preprocess(
        input_ids=torch.tensor([123], dtype=torch.long),
        input_embeds=None,
        text=["hello"],
        task_type=["CustomVoice"],
        hidden_states={"trailing_text": trailing_text, "last": last_hidden},
        embed={"tts_pad": tts_pad},
        meta={"talker_text_offset": 1},
        _omni_is_prefill=False,
        _omni_num_computed_tokens=2,
        _omni_prompt_len=2,
    )

    assert out_ids.tolist() == [123]
    assert torch.equal(out_embeds.cpu(), torch.full((1, 4), 123.0, dtype=torch.bfloat16))
    assert "hidden_states" not in update
    assert update["meta"]["talker_text_offset"] == 2
    past_hidden, text_step = update["mtp_inputs"]
    assert torch.equal(past_hidden.cpu(), last_hidden.reshape(1, -1).to(torch.bfloat16))
    assert torch.equal(text_step.cpu(), trailing_text[1:2].to(torch.bfloat16))


def test_decode_advances_trailing_text_offset_across_multiple_steps():
    model = _make_minimal_talker()

    def fake_embed_input_ids(input_ids):
        return input_ids.to(torch.float32).reshape(1, 1, 1).expand(1, 1, 4)

    model.embed_input_ids = fake_embed_input_ids
    trailing_text = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    state_tail = trailing_text
    last_hidden = torch.full((4,), 2.0, dtype=torch.float32)
    tts_pad = torch.full((1, 4), -1.0, dtype=torch.float32)
    meta = {"talker_text_offset": 0}
    seen_steps = []

    for _ in range(3):
        _, _, update = model.preprocess(
            input_ids=torch.tensor([123], dtype=torch.long),
            input_embeds=None,
            text=["hello"],
            task_type=["CustomVoice"],
            hidden_states={"trailing_text": state_tail, "last": last_hidden},
            embed={"tts_pad": tts_pad},
            meta=meta,
            _omni_is_prefill=False,
            _omni_num_computed_tokens=2,
            _omni_prompt_len=2,
        )
        seen_steps.append(update["mtp_inputs"][1].cpu())
        if "hidden_states" in update and "trailing_text" in update["hidden_states"]:
            state_tail = update["hidden_states"]["trailing_text"]
        meta = update["meta"]

    assert torch.equal(seen_steps[0], trailing_text[0:1].to(torch.bfloat16))
    assert torch.equal(seen_steps[1], trailing_text[1:2].to(torch.bfloat16))
    assert torch.equal(seen_steps[2], tts_pad.to(torch.bfloat16))
    assert meta["talker_text_offset"] == 0
    assert state_tail.numel() == 0


def test_decode_compacts_long_trailing_text_after_large_offset():
    model = _make_minimal_talker()

    def fake_embed_input_ids(input_ids):
        return input_ids.to(torch.float32).reshape(1, 1, 1).expand(1, 1, 4)

    model.embed_input_ids = fake_embed_input_ids
    trailing_text = torch.arange(130 * 4, dtype=torch.float32).reshape(130, 4)
    last_hidden = torch.full((4,), 2.0, dtype=torch.float32)
    tts_pad = torch.full((1, 4), -1.0, dtype=torch.float32)

    _, _, update = model.preprocess(
        input_ids=torch.tensor([123], dtype=torch.long),
        input_embeds=None,
        text=["hello"],
        task_type=["CustomVoice"],
        hidden_states={"trailing_text": trailing_text, "last": last_hidden},
        embed={"tts_pad": tts_pad},
        meta={"talker_text_offset": 64},
        _omni_is_prefill=False,
        _omni_num_computed_tokens=2,
        _omni_prompt_len=2,
    )

    assert torch.equal(update["mtp_inputs"][1].cpu(), trailing_text[64:65].to(torch.bfloat16))
    assert update["meta"]["talker_text_offset"] == 0
    assert torch.equal(update["hidden_states"]["trailing_text"], trailing_text[65:])


def test_decode_batch_preprocess_matches_decode_state_updates():
    model = _make_minimal_talker()

    def fake_embed_input_ids(input_ids):
        return input_ids.to(torch.float32).reshape(-1, 1, 1).expand(-1, 1, 4)

    model.embed_input_ids = fake_embed_input_ids
    trailing_a = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    trailing_b = torch.arange(8, dtype=torch.float32).reshape(2, 4) + 100
    last_a = torch.full((4,), 2.0, dtype=torch.float32)
    last_b = torch.full((4,), 3.0, dtype=torch.float32)
    tts_pad = torch.full((1, 4), -1.0, dtype=torch.float32)

    out_ids, out_embeds, past_hidden, text_step, updates = model.preprocess_decode_batch(
        input_ids=torch.tensor([101, 202], dtype=torch.long),
        req_infos=[
            {
                "text": ["hello"],
                "task_type": ["Base"],
                "hidden_states": {"trailing_text": trailing_a, "last": last_a},
                "embed": {"tts_pad": tts_pad},
                "meta": {"talker_text_offset": 1},
            },
            {
                "text": ["world"],
                "task_type": ["CustomVoice"],
                "hidden_states": {"trailing_text": trailing_b, "last": last_b},
                "embed": {"tts_pad": tts_pad},
                "meta": {"talker_text_offset": 2},
            },
        ],
    )

    assert out_ids.tolist() == [101, 202]
    assert torch.equal(out_embeds.cpu(), torch.tensor([[101.0] * 4, [202.0] * 4], dtype=torch.bfloat16))
    assert torch.equal(past_hidden.cpu(), torch.stack([last_a, last_b]).to(torch.bfloat16))
    assert torch.equal(text_step[0].cpu(), trailing_a[1].to(torch.bfloat16))
    assert torch.equal(text_step[1].cpu(), tts_pad.reshape(-1).to(torch.bfloat16))
    assert updates[0]["meta"]["talker_text_offset"] == 2
    assert updates[0]["meta"]["codec_streaming"] is True
    assert "hidden_states" not in updates[0]
    assert updates[1]["meta"]["talker_text_offset"] == 0
    assert updates[1]["meta"]["codec_streaming"] is False
    assert updates[1]["hidden_states"]["trailing_text"].numel() == 0


def test_base_voice_clone_normalizes_ref_audio_once_for_ref_code_and_speaker():
    model = _make_minimal_talker()
    device_param = torch.nn.Parameter(torch.empty(0))
    model.parameters = lambda: iter([device_param])
    model.config = SimpleNamespace(
        tts_bos_token_id=100,
        tts_eos_token_id=101,
        tts_pad_token_id=102,
    )
    model.talker_config = SimpleNamespace(
        codec_nothink_id=10,
        codec_think_bos_id=11,
        codec_think_eos_id=12,
        codec_think_id=13,
        codec_language_id={},
        codec_pad_id=7,
        codec_bos_id=8,
        num_code_groups=2,
        spk_is_dialect={},
    )

    class FakeTokenizer:
        def __call__(self, *_args, **_kwargs):
            return {"input_ids": torch.arange(8, dtype=torch.long).reshape(1, -1)}

    model._get_tokenizer = lambda: FakeTokenizer()
    model.text_embedding = lambda ids: torch.ones((*ids.shape, 4), device=ids.device)
    model.text_projection = lambda embeds: embeds
    model.embed_input_ids = lambda ids: torch.zeros((*ids.shape, 4), device=ids.device)
    model._generate_icl_prompt = lambda **kwargs: (
        torch.ones((1, 2, 4), device=kwargs["ref_code"].device),
        torch.ones((1, 4), device=kwargs["ref_code"].device),
    )

    normalize_calls = []
    ref_audio = np.arange(1024, dtype=np.float32)
    model._normalize_ref_audio = lambda raw: normalize_calls.append(raw) or (ref_audio, 16000)

    ref_audio_ids = []
    model._encode_ref_audio_to_code = lambda wav, _sr: (
        ref_audio_ids.append(id(wav)) or torch.ones((2, 2), dtype=torch.long)
    )
    model._extract_speaker_embedding = lambda wav, _sr: (
        ref_audio_ids.append(id(wav)) or torch.ones(4, dtype=torch.bfloat16)
    )

    _prompt, _trailing, _pad, ref_code_len, ref_code = model._build_prompt_embeds(
        task_type="Base",
        info_dict={
            "text": ["hello"],
            "ref_audio": ["ref.wav"],
            "ref_ids": torch.arange(8, dtype=torch.long).reshape(1, -1),
            "non_streaming_mode": [False],
        },
    )

    assert normalize_calls == ["ref.wav"]
    assert ref_audio_ids == [id(ref_audio), id(ref_audio)]
    assert ref_code_len == 2
    assert torch.equal(ref_code, torch.ones((2, 2), dtype=torch.long))


def test_base_voice_clone_batch_preprocess_encodes_ref_code_by_sample_rate():
    model = _make_minimal_talker()
    wav1 = np.arange(2048, dtype=np.float32)
    wav2 = np.arange(3072, dtype=np.float32)
    normalize_calls = []
    model._normalize_ref_audio = lambda raw: (
        normalize_calls.append(raw)
        or (
            wav1 if raw == "a.wav" else wav2,
            16000,
        )
    )

    class FakeSpeechTokenizer:
        def __init__(self):
            self.calls = []

        def encode(self, audios, *, sr, return_dict):
            self.calls.append((audios, sr, return_dict))
            return SimpleNamespace(
                audio_codes=[
                    torch.full((2, 2), 11, dtype=torch.long),
                    torch.full((3, 2), 22, dtype=torch.long),
                ]
            )

    tok = FakeSpeechTokenizer()
    model._ensure_speech_tokenizer_loaded = lambda: tok

    class FakeTextTokenizer:
        def __init__(self):
            self.calls = []

        def __call__(self, texts, *, padding=False):
            self.calls.append((texts, padding))
            return {"input_ids": [[idx + 1, idx + 2, idx + 3] for idx, _ in enumerate(texts)]}

    text_tok = FakeTextTokenizer()
    model._get_tokenizer = lambda: text_tok
    buf = {
        "r1": {
            "task_type": ["Base"],
            "text": ["one"],
            "ref_audio": ["a.wav"],
            "ref_text": ["hello"],
            "x_vector_only_mode": [False],
        },
        "r2": {
            "task_type": ["Base"],
            "text": ["two"],
            "ref_audio": ["b.wav"],
            "ref_text": ["world"],
            "x_vector_only_mode": [False],
        },
    }

    model.preprocess_batch(
        req_ids=["r1", "r2"],
        model_intermediate_buffer=buf,
        device=torch.device("cpu"),
    )

    assert normalize_calls == ["a.wav", "b.wav"]
    assert len(tok.calls) == 1
    audios, sr, return_dict = tok.calls[0]
    assert audios[0] is wav1
    assert audios[1] is wav2
    assert sr == 16000
    assert return_dict is True
    assert torch.equal(buf["r1"]["codes"][_PRECOMPUTED_REF_CODE_KEY], torch.full((2, 2), 11))
    assert torch.equal(buf["r2"]["codes"][_PRECOMPUTED_REF_CODE_KEY], torch.full((3, 2), 22))
    assert buf["r1"][_NORMALIZED_REF_AUDIO_KEY][0] is wav1
    assert buf["r2"][_NORMALIZED_REF_AUDIO_KEY][0] is wav2
    assert len(text_tok.calls) == 2
    assert torch.equal(buf["r1"][_PRECOMPUTED_TEXT_IDS_KEY], torch.tensor([1, 2, 3]))
    assert torch.equal(buf["r2"][_PRECOMPUTED_TEXT_IDS_KEY], torch.tensor([2, 3, 4]))
    assert torch.equal(buf["r1"][_PRECOMPUTED_REF_IDS_KEY], torch.tensor([1, 2, 3]))
    assert torch.equal(buf["r2"][_PRECOMPUTED_REF_IDS_KEY], torch.tensor([2, 3, 4]))


def test_base_voice_clone_batch_preprocess_reuses_singleton_normalized_audio_without_speech_tokenizer():
    model = _make_minimal_talker()
    wav = np.arange(2048, dtype=np.float32)
    model._normalize_ref_audio = lambda raw: (wav, 16000)
    model._ensure_speech_tokenizer_loaded = lambda: (_ for _ in ()).throw(
        AssertionError("singleton should not load speech tokenizer")
    )

    class FakeTextTokenizer:
        def __call__(self, texts, *, padding=False):
            return {"input_ids": [[7, 8, 9] for _ in texts]}

    model._get_tokenizer = lambda: FakeTextTokenizer()
    buf = {
        "r1": {
            "task_type": ["Base"],
            "text": ["one"],
            "ref_audio": ["a.wav"],
            "ref_text": ["hello"],
            "x_vector_only_mode": [False],
        }
    }

    model.preprocess_batch(
        req_ids=["r1"],
        model_intermediate_buffer=buf,
        device=torch.device("cpu"),
    )

    assert buf["r1"][_NORMALIZED_REF_AUDIO_KEY][0] is wav
    assert torch.equal(buf["r1"][_PRECOMPUTED_TEXT_IDS_KEY], torch.tensor([7, 8, 9]))
    assert torch.equal(buf["r1"][_PRECOMPUTED_REF_IDS_KEY], torch.tensor([7, 8, 9]))


def test_base_voice_clone_batch_preprocess_skips_after_initial_prefill_state_exists():
    model = _make_minimal_talker()
    model._normalize_ref_audio = lambda _raw: (_ for _ in ()).throw(AssertionError("normalize not expected"))
    model._get_tokenizer = lambda: (_ for _ in ()).throw(AssertionError("tokenizer not expected"))
    model._ensure_speech_tokenizer_loaded = lambda: (_ for _ in ()).throw(
        AssertionError("speech tokenizer not expected")
    )
    buf = {
        "r1": {
            "task_type": ["Base"],
            "text": ["one"],
            "ref_audio": ["a.wav"],
            "ref_text": ["hello"],
            "x_vector_only_mode": [False],
            "embed": {"prefill": torch.ones((1, 4))},
        }
    }

    model.preprocess_batch(
        req_ids=["r1"],
        model_intermediate_buffer=buf,
        device=torch.device("cpu"),
    )

    assert _PRECOMPUTED_TEXT_IDS_KEY not in buf["r1"]
    assert _NORMALIZED_REF_AUDIO_KEY not in buf["r1"]


def test_base_voice_clone_uses_batched_ref_code_without_serial_encode():
    model = _make_minimal_talker()
    device_param = torch.nn.Parameter(torch.empty(0))
    model.parameters = lambda: iter([device_param])
    model.config = SimpleNamespace(
        tts_bos_token_id=100,
        tts_eos_token_id=101,
        tts_pad_token_id=102,
    )
    model.talker_config = SimpleNamespace(
        codec_nothink_id=10,
        codec_think_bos_id=11,
        codec_think_eos_id=12,
        codec_think_id=13,
        codec_language_id={},
        codec_pad_id=7,
        codec_bos_id=8,
        num_code_groups=2,
        spk_is_dialect={},
    )

    class FakeTokenizer:
        def __call__(self, *_args, **_kwargs):
            return {"input_ids": torch.arange(8, dtype=torch.long).reshape(1, -1)}

    model._get_tokenizer = lambda: FakeTokenizer()
    model.text_embedding = lambda ids: torch.ones((*ids.shape, 4), device=ids.device)
    model.text_projection = lambda embeds: embeds
    model.embed_input_ids = lambda ids: torch.zeros((*ids.shape, 4), device=ids.device)
    model._generate_icl_prompt = lambda **kwargs: (
        torch.ones((1, 2, 4), device=kwargs["ref_code"].device),
        torch.ones((1, 4), device=kwargs["ref_code"].device),
    )

    ref_audio = np.arange(2048, dtype=np.float32)
    ref_code = torch.arange(4, dtype=torch.long).reshape(2, 2)
    model._normalize_ref_audio = lambda _raw: (_ for _ in ()).throw(AssertionError("serial normalize not expected"))
    model._encode_ref_audio_to_code = lambda _wav, _sr: (_ for _ in ()).throw(
        AssertionError("serial encode not expected")
    )
    speaker_wav_ids = []
    model._extract_speaker_embedding = lambda wav, _sr: (
        speaker_wav_ids.append(id(wav)) or torch.ones(4, dtype=torch.bfloat16)
    )

    _prompt, _trailing, _pad, ref_code_len, out_ref_code = model._build_prompt_embeds(
        task_type="Base",
        info_dict={
            "text": ["hello"],
            "ref_audio": ["ref.wav"],
            "ref_ids": torch.arange(8, dtype=torch.long).reshape(1, -1),
            "non_streaming_mode": [False],
            "codes": {_PRECOMPUTED_REF_CODE_KEY: ref_code},
            _NORMALIZED_REF_AUDIO_KEY: (ref_audio, 16000),
        },
    )

    assert speaker_wav_ids == [id(ref_audio)]
    assert ref_code_len == 2
    assert torch.equal(out_ref_code, ref_code)
