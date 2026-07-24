import base64
import os
import struct
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    MiniCPMO45OmniTTSForConditionalGeneration,
    _drain_native_duplex_emitted_text,
    _queue_native_duplex_segment_text,
    _TalkerTurnState,
)
from vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni import (
    _extract_first_audio_ref,
    _native_duplex_segment_output_ids,
    llm2tts,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_native_duplex_transcript_waits_for_buffered_vocoder_audio():
    state = _TalkerTurnState(None, None, turn_id=1)

    _queue_native_duplex_segment_text(state, "好")
    assert _drain_native_duplex_emitted_text(state, has_audio=False) == ""

    _queue_native_duplex_segment_text(state, "的，刚")
    assert _drain_native_duplex_emitted_text(state, has_audio=True) == "好的，刚"
    assert _drain_native_duplex_emitted_text(state, has_audio=True) == ""


def test_native_duplex_token2wav_cache_isolated_across_interleaved_sessions():
    model = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    model.audio_tokenizer = SimpleNamespace(stream_cache=None, hift_cache_dict={})
    observed = []

    def stream_window(token_list, _prompt, *, last_chunk):
        del token_list, last_chunk
        marker = model.audio_tokenizer.stream_cache["marker"]
        observed.append(marker)
        prefix = marker[0]
        index = int(marker[1:]) + 1
        model.audio_tokenizer.stream_cache = {"marker": f"{prefix}{index}"}
        model.audio_tokenizer.hift_cache_dict = {"marker": f"{prefix}{index}"}
        return torch.ones(1)

    model._t2w_stream_window = stream_window
    a = _TalkerTurnState(None, None)
    b = _TalkerTurnState(None, None)
    a.stream_cache = {"marker": "A0"}
    a.hift_cache_dict = {"marker": "A0"}
    a.vocoder_initialized = True
    b.stream_cache = {"marker": "B0"}
    b.hift_cache_dict = {"marker": "B0"}
    b.vocoder_initialized = True

    model._run_vocoder_window(a, [1], last_chunk=False)
    model._run_vocoder_window(b, [1], last_chunk=False)
    model._run_vocoder_window(a, [1], last_chunk=False)
    model._run_vocoder_window(b, [1], last_chunk=True)

    assert observed == ["A0", "B0", "A1", "B1"]
    assert a.stream_cache == {"marker": "A2"}
    assert b.stream_cache == {"marker": "B2"}
    assert model.audio_tokenizer.stream_cache is None
    assert model.audio_tokenizer.hift_cache_dict == {}


def test_extract_first_audio_ref_accepts_dict_stereo_audio():
    ref = _extract_first_audio_ref(
        {
            "audio": {
                "array": [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]],
                "sampling_rate": 16000,
            }
        }
    )

    assert ref is not None
    waveform, sample_rate = ref
    assert sample_rate == 16000
    assert torch.allclose(waveform, torch.tensor([1.5, 3.5, 5.5]))


def test_llm2tts_carries_request_ref_audio_to_talker_payload():
    latent = torch.arange(20, dtype=torch.float16).reshape(5, 4)
    output = SimpleNamespace(
        token_ids=[11, 12, 9002],
        text="hello",
        multimodal_output={
            "latent": latent,
            "meta": {
                "tts_bos_token_id": 9001,
                "tts_eos_token_id": 9002,
            },
        },
    )
    llm_output = SimpleNamespace(
        request_id="req-1",
        prompt_token_ids=[101, 9001],
        outputs=[output],
    )
    ref_waveform = torch.tensor([0.1, 0.2, 0.3])

    converted = llm2tts(
        [llm_output],
        prompt=[
            {
                "multi_modal_data": {
                    "audio": (ref_waveform, 22050),
                }
            }
        ],
    )

    assert "additional_information" not in converted[0]
    info = converted[0]["model_intermediate_buffer"]
    assert info["codes"]["ref"] == ref_waveform.tolist()
    assert info["meta"]["ref_audio_sr"] == 22050
    assert info["ids"]["tts"] == [11, 12]
    assert torch.equal(torch.as_tensor(info["hidden_states"]["tts"]), latent[2:4].float())
    assert "tts_token_ids" not in info
    assert "tts_hidden_states" not in info


def test_llm2tts_uses_duplex_prompt_token_ids_for_tts_boundary():
    latent = torch.arange(24, dtype=torch.float16).reshape(6, 4)
    output = SimpleNamespace(
        token_ids=[11, 12, 9002],
        text="hello",
        multimodal_output={
            "latent": latent,
            "duplex_prompt_token_ids": [101, 102, 9001],
            "meta": {
                "tts_bos_token_id": 9001,
                "tts_eos_token_id": 9002,
            },
        },
    )
    llm_output = SimpleNamespace(
        request_id="duplex-req",
        prompt_token_ids=[0, 0, 0],
        outputs=[output],
    )

    converted = llm2tts([llm_output], prompt=[{}])

    assert "additional_information" not in converted[0]
    info = converted[0]["model_intermediate_buffer"]
    assert info["prompt_token_ids"] == [101, 102, 9001]
    assert converted[0]["prompt_token_ids"] == [11, 12]
    assert info["ids"]["tts"] == [11, 12]
    assert torch.equal(torch.as_tensor(info["hidden_states"]["tts"]), latent[3:5].float())
    assert "tts_token_ids" not in info
    assert "tts_hidden_states" not in info


def test_llm2tts_accepts_flat_tokenizer_derived_special_token_metadata():
    latent = torch.arange(24, dtype=torch.float16).reshape(6, 4)
    output = SimpleNamespace(
        token_ids=[11, 12, 9302],
        text="hello",
        multimodal_output={
            "latent": latent,
            "duplex_prompt_token_ids": [101, 102, 9301],
            "meta.tts_bos_token_id": torch.tensor([9301]),
            "meta.tts_eos_token_id": torch.tensor([9302]),
            "meta.listen_token_id": torch.tensor([9303]),
            "meta.speak_token_id": torch.tensor([9304]),
            "meta.chunk_eos_token_id": torch.tensor([9305]),
            "meta.chunk_tts_eos_token_id": torch.tensor([9306]),
            "meta.turn_eos_token_id": torch.tensor([9307]),
        },
    )
    llm_output = SimpleNamespace(
        request_id="duplex-flat-meta",
        prompt_token_ids=[0, 0, 0],
        outputs=[output],
    )

    converted = llm2tts([llm_output], prompt=[{}])

    assert "additional_information" not in converted[0]
    info = converted[0]["model_intermediate_buffer"]
    assert converted[0]["prompt_token_ids"] == [11, 12]
    assert info["ids"]["tts"] == [11, 12]
    assert info["stream_output"] is True
    assert info["native_duplex"] is True
    assert torch.equal(torch.as_tensor(info["hidden_states"]["tts"]), latent[3:5].float())
    assert "tts_token_ids" not in info
    assert "tts_hidden_states" not in info


def test_llm2tts_stops_at_tokenizer_derived_duplex_state_tokens():
    latent = torch.arange(32, dtype=torch.float16).reshape(8, 4)
    output = SimpleNamespace(
        token_ids=[11, 12, 9303, 13, 9302],
        text="hello",
        multimodal_output={
            "latent": latent,
            "duplex_prompt_token_ids": [101, 102, 9301],
            "meta": {
                "tts_bos_token_id": 9301,
                "tts_eos_token_id": 9302,
                "listen_token_id": 9303,
                "speak_token_id": 9304,
                "tts_pad_token_id": 9305,
                "unit_token_id": 9306,
                "unit_end_token_id": 9307,
                "chunk_eos_token_id": 9308,
                "chunk_tts_eos_token_id": 9309,
                "turn_eos_token_id": 9310,
            },
        },
    )
    llm_output = SimpleNamespace(
        request_id="duplex-state-boundary",
        prompt_token_ids=[0, 0, 0],
        outputs=[output],
    )

    converted = llm2tts([llm_output], prompt=[{}])

    info = converted[0]["model_intermediate_buffer"]
    assert converted[0]["prompt_token_ids"] == [11, 12]
    assert info["ids"]["tts"] == [11, 12]
    assert torch.equal(torch.as_tensor(info["hidden_states"]["tts"]), latent[3:5].float())
    assert "tts_token_ids" not in info
    assert "tts_hidden_states" not in info


def test_llm2tts_native_duplex_uses_speak_region_without_tts_bos_prompt():
    latent = torch.arange(36, dtype=torch.float16).reshape(9, 4)
    output = SimpleNamespace(
        token_ids=[9304, 11, 12, 9308, 13],
        text="hello",
        multimodal_output={
            "latent": latent,
            "duplex_prompt_token_ids": [101, 102, 9306, 9306],
            "meta": {
                "tts_bos_token_id": 9301,
                "tts_eos_token_id": 9302,
                "listen_token_id": 9303,
                "speak_token_id": 9304,
                "tts_pad_token_id": 9305,
                "unit_token_id": 9306,
                "unit_end_token_id": 9307,
                "chunk_eos_token_id": 9308,
                "chunk_tts_eos_token_id": 9309,
                "turn_eos_token_id": 9310,
            },
        },
    )
    llm_output = SimpleNamespace(
        request_id="duplex-speak-region",
        prompt_token_ids=[0, 0, 0, 0],
        outputs=[output],
    )

    converted = llm2tts([llm_output], prompt=[{}])

    info = converted[0]["model_intermediate_buffer"]
    assert converted[0]["prompt_token_ids"] == [11, 12]
    assert info["ids"]["tts"] == [11, 12]
    assert torch.equal(torch.as_tensor(info["hidden_states"]["tts"]), latent[5:7].float())
    assert "tts_token_ids" not in info
    assert "tts_hidden_states" not in info


def test_llm2tts_native_duplex_uses_plain_text_region_without_speak_marker():
    latent = torch.arange(32, dtype=torch.float16).reshape(8, 4)
    output = SimpleNamespace(
        token_ids=[11, 12, 13, 9308],
        text="hello",
        multimodal_output={
            "latent": latent,
            "duplex_prompt_token_ids": [101, 102, 9306, 9306],
            "meta": {
                "tts_bos_token_id": 9301,
                "tts_eos_token_id": 9302,
                "listen_token_id": 9303,
                "speak_token_id": 9304,
                "tts_pad_token_id": 9305,
                "unit_token_id": 9306,
                "unit_end_token_id": 9307,
                "chunk_eos_token_id": 9308,
                "chunk_tts_eos_token_id": 9309,
                "turn_eos_token_id": 9310,
            },
        },
    )
    llm_output = SimpleNamespace(
        request_id="duplex-plain-text-region",
        prompt_token_ids=[0, 0, 0, 0],
        outputs=[output],
    )

    converted = llm2tts([llm_output], prompt=[{}])

    info = converted[0]["model_intermediate_buffer"]
    assert converted[0]["prompt_token_ids"] == [12, 13]
    assert info["ids"]["tts"] == [12, 13]
    assert torch.equal(torch.as_tensor(info["hidden_states"]["tts"]), latent[5:7].float())


def test_llm2tts_native_duplex_conditions_on_turn_eos_and_midunit_speak():
    latent = torch.arange(36, dtype=torch.float16).reshape(9, 4)
    # Official duplex includes mid-unit <|speak|> tokens AND the <|turn_eos|>
    # token+hidden in the talker condition (its embedding is the trained
    # stop signal); only chunk terminators bound the slice.
    output = SimpleNamespace(
        token_ids=[9304, 11, 9304, 12, 9310, 9308],
        text="hello",
        multimodal_output={
            "latent": latent,
            "duplex_prompt_token_ids": [101, 102, 9306],
            "meta": {
                "tts_bos_token_id": 9301,
                "tts_eos_token_id": 9302,
                "listen_token_id": 9303,
                "speak_token_id": 9304,
                "tts_pad_token_id": 9305,
                "unit_token_id": 9306,
                "unit_end_token_id": 9307,
                "chunk_eos_token_id": 9308,
                "chunk_tts_eos_token_id": 9309,
                "turn_eos_token_id": 9310,
            },
        },
    )
    llm_output = SimpleNamespace(
        request_id="duplex-turn-eos",
        prompt_token_ids=[0, 0, 0],
        outputs=[output],
    )

    converted = llm2tts([llm_output], prompt=[{}])

    info = converted[0]["model_intermediate_buffer"]
    assert info["ids"]["tts"] == [11, 9304, 12, 9310]
    assert torch.equal(torch.as_tensor(info["hidden_states"]["tts"]), latent[4:8].float())
    assert info["meta"]["turn_eos_token_id"] == 9310


def test_llm2tts_native_duplex_ignores_stale_tts_bos_inside_folded_prompt():
    latent = torch.arange(36, dtype=torch.float16).reshape(9, 4)
    # A continuation unit: the resumable prompt has an EARLIER reply's
    # <|tts_bos|> folded mid-prompt, and the model decided to LISTEN (no new
    # speak text). The stale boundary must not re-slice already-spoken text.
    output = SimpleNamespace(
        token_ids=[9303, 9309],
        text="",
        multimodal_output={
            "latent": latent,
            "duplex_prompt_token_ids": [101, 9301, 41, 42, 9306, 9306],
            "meta": {
                "tts_bos_token_id": 9301,
                "tts_eos_token_id": 9302,
                "listen_token_id": 9303,
                "speak_token_id": 9304,
                "tts_pad_token_id": 9305,
                "unit_token_id": 9306,
                "unit_end_token_id": 9307,
                "chunk_eos_token_id": 9308,
                "chunk_tts_eos_token_id": 9309,
                "turn_eos_token_id": 9310,
            },
        },
    )
    llm_output = SimpleNamespace(
        request_id="duplex-stale-bos",
        prompt_token_ids=[0, 0, 0, 0, 0, 0],
        outputs=[output],
    )

    converted = llm2tts([llm_output], prompt=[{}])

    assert converted == []


def test_llm2tts_rejects_native_duplex_output_without_special_token_metadata():
    latent = torch.arange(16, dtype=torch.float16).reshape(4, 4)
    output = SimpleNamespace(
        token_ids=[11, 12],
        text="hello",
        multimodal_output={
            "latent": latent,
            "duplex_prompt_token_ids": [101, 102, 9301],
        },
    )
    llm_output = SimpleNamespace(
        request_id="duplex-missing-meta",
        prompt_token_ids=[0, 0, 0],
        outputs=[output],
    )

    with pytest.raises(ValueError, match=r"<\|tts_bos\|>"):
        llm2tts([llm_output], prompt=[{}])


def test_minicpmo_talker_normalizes_list_handoff_payload(monkeypatch):
    model = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    captured = {}

    monkeypatch.setattr(model, "_should_stream_output", lambda *args, **kwargs: False)

    def fake_generate_speech(tts_token_ids, tts_hidden_states, **kwargs):
        captured["tts_token_ids"] = tts_token_ids
        captured["tts_hidden_states"] = tts_hidden_states
        return torch.tensor([0.0, 0.1], dtype=torch.float32)

    monkeypatch.setattr(model, "generate_speech", fake_generate_speech)

    waveform, mel = model.forward(
        additional_information={
            "tts_token_ids": [11, 12],
            "tts_hidden_states": [[0.1, 0.2], [0.3, 0.4]],
        }
    )

    assert mel is None
    assert torch.equal(waveform, torch.tensor([0.0, 0.1], dtype=torch.float32))
    assert torch.equal(captured["tts_token_ids"], torch.tensor([11, 12]))
    assert captured["tts_hidden_states"].dtype == torch.float32
    assert captured["tts_hidden_states"].shape == (2, 2)


def test_minicpmo_talker_accepts_structured_omni_payload_handoff(monkeypatch):
    model = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    captured = {}

    monkeypatch.setattr(model, "_should_stream_output", lambda *args, **kwargs: False)

    def fake_generate_speech(tts_token_ids, tts_hidden_states, **kwargs):
        captured["tts_token_ids"] = tts_token_ids
        captured["tts_hidden_states"] = tts_hidden_states
        return torch.tensor([0.0, 0.1], dtype=torch.float32)

    monkeypatch.setattr(model, "generate_speech", fake_generate_speech)

    waveform, mel = model.forward(
        additional_information={
            "ids": {"tts": [11, 12]},
            "hidden_states": {"tts": torch.tensor([[0.1, 0.2], [0.3, 0.4]])},
        }
    )

    assert mel is None
    assert torch.equal(waveform, torch.tensor([0.0, 0.1], dtype=torch.float32))
    assert torch.equal(captured["tts_token_ids"], torch.tensor([11, 12]))
    assert captured["tts_hidden_states"].dtype == torch.float32
    assert captured["tts_hidden_states"].shape == (2, 2)


def test_token2wav_prompt_cache_resets_when_reference_changes():
    model = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    model.audio_tokenizer = SimpleNamespace(cache="old")

    model._reset_token2wav_cache_if_needed("/tmp/ref_a.wav")
    assert model.audio_tokenizer.cache is None

    model.audio_tokenizer.cache = "prepared"
    model._reset_token2wav_cache_if_needed("/tmp/ref_a.wav")
    assert model.audio_tokenizer.cache == "prepared"

    model._reset_token2wav_cache_if_needed("/tmp/ref_b.wav")
    assert model.audio_tokenizer.cache is None


def test_request_ref_audio_prompt_wav_uses_content_cache():
    model = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    ref_audio = torch.tensor([0.0, 0.1, 0.2], dtype=torch.float32)

    path1 = model._write_ref_audio_prompt_wav(ref_audio, 16000)
    path2 = model._write_ref_audio_prompt_wav(ref_audio.clone(), 16000)

    try:
        assert path1 == path2
        assert os.path.exists(path1)
        assert model._is_cached_ref_audio_prompt_wav(path1)
    finally:
        for path in set(getattr(model, "_ref_audio_prompt_files", {}).values()):
            try:
                os.unlink(path)
            except OSError:
                pass


def test_native_duplex_talker_uses_generate_chunk_continuation(monkeypatch):
    class _FakeTTS:
        audio_bos_token_id = 88

        def __init__(self):
            self.config = SimpleNamespace(num_audio_tokens=128)
            self.model = SimpleNamespace(config=SimpleNamespace())
            self.emb_text = torch.nn.Embedding(256, 4)
            self.calls = []
            self._responses = [
                (torch.tensor([[[10], [11], [12]]]), "kv1"),
                (torch.tensor([[[20]]]), "kv2"),
            ]

        def generate_chunk(self, **kwargs):
            self.calls.append(kwargs)
            return self._responses[len(self.calls) - 1]

    model = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    tts = _FakeTTS()
    windows = []
    model.tts_obj = tts
    model.audio_tokenizer = SimpleNamespace()
    model._talker_turn_states = {}
    model._talker_consumed_tokens = {}

    monkeypatch.setattr(model, "_lazy_init_tts", lambda: None)
    monkeypatch.setattr(
        model, "_build_tts_sampling_params", lambda: SimpleNamespace(temperature=0.8, repetition_penalty=1.05)
    )
    monkeypatch.setattr(model, "_tts_runtime_config", lambda: SimpleNamespace(streaming_generator_chunk=2))
    monkeypatch.setattr(model, "_resolve_prompt_wav_path", lambda ref, sr: ("/tmp/ref.wav", None))
    monkeypatch.setattr(model, "_begin_turn_vocoder_cache", lambda prompt, **kwargs: None)
    monkeypatch.setattr(model, "_t2w_pre_lookahead", lambda: 1)
    monkeypatch.setattr(
        model,
        "_build_tts_condition_embeds",
        lambda ids, hidden: torch.ones((ids.numel(), 4), dtype=torch.float32),
    )

    def fake_stream_window(token_list, prompt_wav_path, *, last_chunk):
        windows.append((list(token_list), last_chunk))
        return torch.tensor([float(len(token_list))])

    monkeypatch.setattr(model, "_t2w_stream_window", fake_stream_window)

    base_info = {
        "request_id": "duplex-generate-chunk",
        "meta": {"turn_eos_token_id": 9310},
        "ids": {"tts": [21]},
        "hidden_states": {"tts": [[0.1, 0.2, 0.3, 0.4]]},
        "codes": {"ref": None},
        "native_duplex": True,
    }
    list(model._create_native_duplex_stream_gen(base_info))

    turn_end_info = {
        **base_info,
        "ids": {"tts": [21, 9310]},
        "hidden_states": {"tts": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]},
    }
    list(model._create_native_duplex_stream_gen(turn_end_info))

    assert len(tts.calls) == 2
    first, second = tts.calls
    assert first["past_key_values"] is None
    assert first["text_start_pos"] == 0
    assert first["min_new_tokens"] == 0
    assert first["max_new_token"] == 3
    assert torch.equal(first["eos_token"], torch.tensor([127]))
    assert float(first["temperature"].item()) == pytest.approx(0.8)
    assert first["repetition_penalty"] == pytest.approx(1.05)

    assert second["past_key_values"] == "kv1"
    assert second["text_start_pos"] == 5
    assert second["min_new_tokens"] == 0
    assert second["max_new_token"] == 3
    assert "duplex-generate-chunk" not in model._talker_turn_states
    assert windows[-1] == ([20], True)


def test_minicpmo_native_duplex_talker_turn_end_metadata_flushes_tail(monkeypatch):
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        MiniCPMO45OmniTTSForConditionalGeneration,
    )

    class _FakeEmbText:
        weight = torch.ones((200, 4), dtype=torch.float32)

        def __call__(self, ids):
            return torch.ones((ids.numel(), 4), dtype=torch.float32)

    class _FakeTTS:
        audio_bos_token_id = 1
        config = SimpleNamespace(num_audio_tokens=128)
        emb_text = _FakeEmbText()
        model = SimpleNamespace(config=SimpleNamespace(rope_theta=10000.0))

        def generate_chunk(self, **kwargs):
            del kwargs
            return torch.tensor([[[33]]]), None

    model = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    model.tts_obj = _FakeTTS()
    model.audio_tokenizer = SimpleNamespace()
    model._talker_turn_states = {}
    model._talker_consumed_tokens = {}

    monkeypatch.setattr(model, "_lazy_init_tts", lambda: None)
    monkeypatch.setattr(
        model, "_build_tts_sampling_params", lambda: SimpleNamespace(temperature=0.8, repetition_penalty=1.05)
    )
    monkeypatch.setattr(model, "_tts_runtime_config", lambda: SimpleNamespace(streaming_generator_chunk=25))
    monkeypatch.setattr(model, "_resolve_prompt_wav_path", lambda ref, sr: ("/tmp/ref.wav", None))
    monkeypatch.setattr(model, "_begin_turn_vocoder_cache", lambda prompt, **kwargs: None)
    monkeypatch.setattr(model, "_t2w_pre_lookahead", lambda: 5)
    monkeypatch.setattr(
        model,
        "_build_tts_condition_embeds",
        lambda ids, hidden: torch.ones((ids.numel(), 4), dtype=torch.float32),
    )

    windows = []

    def fake_stream_window(token_list, prompt_wav_path, *, last_chunk):
        del prompt_wav_path
        windows.append((list(token_list), last_chunk))
        return torch.tensor([float(len(token_list))])

    monkeypatch.setattr(model, "_t2w_stream_window", fake_stream_window)

    info = {
        "request_id": "duplex-metadata-tail-flush",
        "meta": {"turn_eos_token_id": 9310},
        "ids": {"tts": [21]},
        "hidden_states": {"tts": [[0.1, 0.2, 0.3, 0.4]]},
        "codes": {"ref": None},
        "end_of_turn": True,
        "native_duplex": True,
    }

    list(model._create_native_duplex_stream_gen(info))

    assert windows[-1][1] is True
    assert "duplex-metadata-tail-flush" not in model._talker_turn_states
    assert "duplex-metadata-tail-flush" not in model._talker_consumed_tokens


def test_minicpmo_native_duplex_terminal_only_new_turn_does_not_start_tts(monkeypatch):
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        MiniCPMO45OmniTTSForConditionalGeneration,
    )

    class _FakeEmbText:
        weight = torch.ones((200, 4), dtype=torch.float32)

        def __call__(self, ids):
            return torch.ones((ids.numel(), 4), dtype=torch.float32)

    class _FakeTTS:
        audio_bos_token_id = 1
        config = SimpleNamespace(num_audio_tokens=128)
        emb_text = _FakeEmbText()
        model = SimpleNamespace(config=SimpleNamespace(rope_theta=10000.0))

        def __init__(self):
            self.calls = 0

        def generate_chunk(self, **kwargs):
            del kwargs
            self.calls += 1
            return torch.tensor([[[33]]]), None

    model = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    tts = _FakeTTS()
    model.tts_obj = tts
    model.audio_tokenizer = SimpleNamespace()
    model._talker_turn_states = {}
    model._talker_consumed_tokens = {}

    monkeypatch.setattr(model, "_lazy_init_tts", lambda: None)
    monkeypatch.setattr(
        model, "_build_tts_sampling_params", lambda: SimpleNamespace(temperature=0.8, repetition_penalty=1.05)
    )
    monkeypatch.setattr(model, "_tts_runtime_config", lambda: SimpleNamespace(streaming_generator_chunk=25))
    monkeypatch.setattr(model, "_resolve_prompt_wav_path", lambda ref, sr: ("/tmp/ref.wav", None))
    monkeypatch.setattr(model, "_begin_turn_vocoder_cache", lambda prompt, **kwargs: None)
    monkeypatch.setattr(model, "_t2w_pre_lookahead", lambda: 5)
    monkeypatch.setattr(
        model,
        "_build_tts_condition_embeds",
        lambda ids, hidden: torch.ones((ids.numel(), 4), dtype=torch.float32),
    )
    monkeypatch.setattr(
        model,
        "_t2w_stream_window",
        lambda token_list, prompt_wav_path, *, last_chunk: torch.tensor([float(len(token_list))]),
    )

    outputs = list(
        model._create_native_duplex_stream_gen(
            {
                "request_id": "duplex-terminal-only",
                "duplex": {"epoch": 0, "turn_id": 1},
                "meta": {"turn_eos_token_id": 9310},
                "ids": {"tts": [9310]},
                "hidden_states": {"tts": [[0.1, 0.2, 0.3, 0.4]]},
                "codes": {"ref": None},
                "native_duplex": True,
            }
        )
    )

    assert tts.calls == 0
    assert len(outputs) == 1
    assert outputs[0][0].numel() == 0
    assert outputs[0][1] is True
    assert model._talker_turn_states == {}
    assert model._talker_consumed_tokens == {}


def test_minicpmo_native_duplex_talker_finished_request_closes_session_keyed_turn_state():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        MiniCPMO45OmniTTSForConditionalGeneration,
        _TalkerTurnState,
    )

    model = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    model.audio_tokenizer = SimpleNamespace(stream_cache={"stale": True}, hift_cache_dict={"stale": True})
    model._stream_gens = {}
    model._talker_turn_states = {
        "sid-stage1": _TalkerTurnState(prompt_wav_path=None, temp_prompt_wav_path=None),
    }
    model._talker_consumed_tokens = {"sid-stage1": 12}
    model._talker_request_keys = {"req-stage1": "sid-stage1"}

    model.on_requests_finished({"req-stage1"})

    assert "sid-stage1" not in model._talker_turn_states
    assert "sid-stage1" not in model._talker_consumed_tokens
    assert "req-stage1" not in model._talker_request_keys
    assert model.audio_tokenizer.stream_cache is None
    assert model.audio_tokenizer.hift_cache_dict == {}


def test_minicpmo_native_duplex_talker_new_turn_reopens_session_keyed_state(monkeypatch):
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        MiniCPMO45OmniTTSForConditionalGeneration,
        _TalkerTurnState,
    )

    class _FakeTTS:
        audio_bos_token_id = 88

        def __init__(self):
            self.config = SimpleNamespace(num_audio_tokens=128)
            self.model = SimpleNamespace(config=SimpleNamespace())
            self.emb_text = torch.nn.Embedding(256, 4)
            self.calls = []

        def generate_chunk(self, **kwargs):
            self.calls.append(kwargs)
            return torch.tensor([[[10]]]), "new-kv"

    model = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    tts = _FakeTTS()
    model.tts_obj = tts
    model.audio_tokenizer = SimpleNamespace(stream_cache={"old": True}, hift_cache_dict={"old": True})
    model._talker_turn_states = {
        "sid-stage1": _TalkerTurnState(
            prompt_wav_path=None,
            temp_prompt_wav_path=None,
            epoch=0,
            turn_id=1,
        ),
    }
    model._talker_turn_states["sid-stage1"].past_key_values = "old-kv"
    model._talker_consumed_tokens = {"sid-stage1": 9}
    model._talker_request_keys = {}

    monkeypatch.setattr(model, "_lazy_init_tts", lambda: None)
    monkeypatch.setattr(
        model, "_build_tts_sampling_params", lambda: SimpleNamespace(temperature=0.8, repetition_penalty=1.05)
    )
    monkeypatch.setattr(model, "_tts_runtime_config", lambda: SimpleNamespace(streaming_generator_chunk=25))
    monkeypatch.setattr(model, "_resolve_prompt_wav_path", lambda ref, sr: ("/tmp/ref.wav", None))
    monkeypatch.setattr(model, "_begin_turn_vocoder_cache", lambda prompt, **kwargs: None)
    monkeypatch.setattr(model, "_t2w_pre_lookahead", lambda: 5)
    monkeypatch.setattr(
        model,
        "_build_tts_condition_embeds",
        lambda ids, hidden: torch.ones((ids.numel(), 4), dtype=torch.float32),
    )
    monkeypatch.setattr(
        model,
        "_t2w_stream_window",
        lambda token_list, prompt_wav_path, *, last_chunk: torch.tensor([float(len(token_list))]),
    )

    info = {
        "request_id": "sid-stage1",
        "duplex": {"epoch": 0, "turn_id": 2},
        "meta": {"turn_eos_token_id": 9310},
        "ids": {"tts": [21]},
        "hidden_states": {"tts": [[0.1, 0.2, 0.3, 0.4]]},
        "codes": {"ref": None},
        "native_duplex": True,
    }

    list(model._create_native_duplex_stream_gen(info))

    assert tts.calls[0]["past_key_values"] is None
    assert tts.calls[0]["text_start_pos"] == 0
    assert model._talker_consumed_tokens["sid-stage1"] == 1
    assert model._talker_turn_states["sid-stage1"].turn_id == 2


def test_minicpmo_native_duplex_talker_segment_end_preserves_token2wav_stream_until_turn_end(monkeypatch):
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
        MiniCPMO45OmniTTSForConditionalGeneration,
    )

    class _FakeEmbText:
        weight = torch.ones((200, 4), dtype=torch.float32)

        def __call__(self, ids):
            return torch.ones((ids.numel(), 4), dtype=torch.float32)

    class _FakeTTS:
        audio_bos_token_id = 1
        config = SimpleNamespace(num_audio_tokens=128)
        emb_text = _FakeEmbText()
        model = SimpleNamespace(config=SimpleNamespace(rope_theta=10000.0))

        def generate_chunk(self, **kwargs):
            del kwargs
            return torch.tensor([[[33]]]), "kv"

    model = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    model.tts_obj = _FakeTTS()
    model.audio_tokenizer = SimpleNamespace()
    model._talker_turn_states = {}
    model._talker_consumed_tokens = {}

    monkeypatch.setattr(model, "_lazy_init_tts", lambda: None)
    monkeypatch.setattr(
        model, "_build_tts_sampling_params", lambda: SimpleNamespace(temperature=0.8, repetition_penalty=1.05)
    )
    monkeypatch.setattr(model, "_tts_runtime_config", lambda: SimpleNamespace(streaming_generator_chunk=25))
    monkeypatch.setattr(model, "_resolve_prompt_wav_path", lambda ref, sr: ("/tmp/ref.wav", None))
    monkeypatch.setattr(model, "_begin_turn_vocoder_cache", lambda prompt, **kwargs: None)
    monkeypatch.setattr(model, "_t2w_pre_lookahead", lambda: 5)
    monkeypatch.setattr(
        model,
        "_build_tts_condition_embeds",
        lambda ids, hidden: torch.ones((ids.numel(), 4), dtype=torch.float32),
    )

    windows = []

    def fake_stream_window(token_list, prompt_wav_path, *, last_chunk):
        del prompt_wav_path
        windows.append((list(token_list), last_chunk))
        return torch.tensor([float(len(token_list))])

    monkeypatch.setattr(model, "_t2w_stream_window", fake_stream_window)

    info = {
        "request_id": "duplex-segment-tail-flush",
        "meta": {"turn_eos_token_id": 9310, "segment_end": True},
        "ids": {"tts": [21]},
        "hidden_states": {"tts": [[0.1, 0.2, 0.3, 0.4]]},
        "codes": {"ref": None},
        "native_duplex": True,
    }

    list(model._create_native_duplex_stream_gen(info))

    state = model._talker_turn_states["duplex-segment-tail-flush"]
    assert windows == []
    assert state.token2wav_buffer == [4218, 4218, 4218, 33]

    turn_end_info = {
        **info,
        "ids": {"tts": [21, 9310]},
        "hidden_states": {
            "tts": [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
            ]
        },
        "meta": {"turn_eos_token_id": 9310, "turn_end": True},
    }
    list(model._create_native_duplex_stream_gen(turn_end_info))

    assert windows == [([4218, 4218, 4218, 33, 33], True)]
    assert "duplex-segment-tail-flush" not in model._talker_turn_states


def test_tts_scheduler_eos_uses_tokenizer_im_end_when_config_has_no_eos():
    class _Tokenizer:
        unk_token_id = 0

        def convert_tokens_to_ids(self, token):
            return {"<|im_end|>": 77}.get(token, self.unk_token_id)

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [77] if text == "<|im_end|>" else [self.unk_token_id]

    model = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    model.config = SimpleNamespace(eos_token_id=None, vocab_size=100)
    model.vllm_config = SimpleNamespace(model_config=SimpleNamespace())
    model._text_tokenizer = _Tokenizer()
    model._ar_last_chunk_flags = [True]

    logits = model.compute_logits(torch.zeros(1, 4))

    assert int(torch.argmax(logits[0]).item()) == 77


def test_tts_streaming_non_final_chunk_does_not_mark_extra_rows_eos():
    class _Tokenizer:
        unk_token_id = 0

        def convert_tokens_to_ids(self, token):
            return {"<|im_end|>": 77}.get(token, self.unk_token_id)

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [77] if text == "<|im_end|>" else [self.unk_token_id]

    model = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    model.config = SimpleNamespace(eos_token_id=None, vocab_size=100)
    model.vllm_config = SimpleNamespace(model_config=SimpleNamespace())
    model._text_tokenizer = _Tokenizer()
    model._ar_last_chunk_flags = [False]

    logits = model.compute_logits(torch.zeros(3, 4))
    sampled = torch.argmax(logits, dim=-1).tolist()

    assert sampled == [1, 1, 1]


def test_native_duplex_turn_end_detection_is_not_segment_end_detection():
    model = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    turn_eos_id = 9310

    assert model._native_duplex_input_ends_turn(
        {
            "native_duplex": True,
            "ids": {"tts": [9304, 42, turn_eos_id]},
            "meta": {"turn_eos_token_id": turn_eos_id},
        }
    )
    assert not model._native_duplex_input_ends_turn(
        {
            "native_duplex": True,
            "ids": {"tts": [9304, 42, 9308]},
            "meta": {
                "turn_eos_token_id": turn_eos_id,
                "segment_end": True,
            },
        }
    )


def _native_duplex_meta():
    return {
        "tts_bos_token_id": 9301,
        "tts_eos_token_id": 9302,
        "listen_token_id": 9303,
        "speak_token_id": 9304,
        "tts_pad_token_id": 9305,
        "unit_token_id": 9306,
        "unit_end_token_id": 9307,
        "chunk_eos_token_id": 9308,
        "chunk_tts_eos_token_id": 9309,
        "turn_eos_token_id": 9310,
    }


def _native_duplex_handoff(request_id, prompt_ids, cumulative_output_ids, *, text="hello"):
    latent = torch.arange(
        (len(prompt_ids) + len(cumulative_output_ids)) * 4,
        dtype=torch.float16,
    ).reshape(-1, 4)
    output = SimpleNamespace(
        token_ids=cumulative_output_ids,
        text=text,
        multimodal_output={
            "latent": latent,
            "duplex_prompt_token_ids": list(prompt_ids),
            "meta": _native_duplex_meta(),
        },
    )
    return SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=[0, 0, 0],
        outputs=[output],
    )


def _install_native_duplex_decoder(streaming_context, token_text):
    def _decode(token_ids):
        return "".join(token_text.get(int(token_id), "") for token_id in token_ids)

    streaming_context.source_token_decoder = _decode
    return streaming_context


def test_native_duplex_segment_text_comes_from_token_delta_decode():
    streaming_context = _install_native_duplex_decoder(
        SimpleNamespace(bridge_states={}),
        {
            21: "今早",
            22: "吃东。",
            31: "西了吗？",
        },
    )

    ids1, text1 = _native_duplex_segment_output_ids(
        [9304, 21, 22, 9308],
        "今早吃东。",
        streaming_context,
        request_id="duplex-token-text",
    )
    assert ids1 == [9304, 21, 22, 9308]
    assert text1 == "今早吃东。"

    ids2, text2 = _native_duplex_segment_output_ids(
        [9304, 21, 22, 9308, 9304, 31, 9308],
        "今早吃东。西了吗？如需陪聊，",
        streaming_context,
        request_id="duplex-token-text",
    )
    assert ids2 == [9304, 31, 9308]
    assert text2 == "西了吗？"


def test_native_duplex_segment_restart_does_not_reuse_character_cursor():
    streaming_context = _install_native_duplex_decoder(
        SimpleNamespace(bridge_states={}),
        {
            21: "你好",
            31: "你好",
            32: "，有什么想聊的吗？",
        },
    )

    _native_duplex_segment_output_ids(
        [9304, 21, 9310],
        "你好，第一轮尾巴",
        streaming_context,
        request_id="duplex-restart-text",
    )

    ids2, text2 = _native_duplex_segment_output_ids(
        [9304, 31, 32, 9310],
        "你好，第一轮尾巴你好，有什么想聊的吗？",
        streaming_context,
        request_id="duplex-restart-text",
    )
    assert ids2 == [9304, 31, 32, 9310]
    assert text2 == "你好，有什么想聊的吗？"


def test_native_duplex_segment_turn_id_change_keeps_cumulative_token_cursor():
    streaming_context = _install_native_duplex_decoder(
        SimpleNamespace(bridge_states={"duplex": {"turn_id": 1}}),
        {
            21: "你好",
            32: "，有什么想聊的吗？",
        },
    )

    ids1, text1 = _native_duplex_segment_output_ids(
        [9304, 21],
        "你好",
        streaming_context,
        request_id="duplex-turn-id-reset",
    )
    assert ids1 == [9304, 21]
    assert text1 == "你好"

    streaming_context.bridge_states["duplex"]["turn_id"] = 2
    ids2, text2 = _native_duplex_segment_output_ids(
        [9304, 21, 32],
        "你好，有什么想聊的吗？",
        streaming_context,
        request_id="duplex-turn-id-reset",
    )
    assert ids2 == [32]
    assert text2 == "，有什么想聊的吗？"


def test_llm2tts_native_duplex_hands_off_segment_deltas():
    """Each duplex handoff must carry only the current segment's tokens.

    The thinker's resumable request reports cumulative output ids while
    earlier segments are already folded into the prompt, so re-forwarding
    the full list misaligns the hidden states and grows the talker prompt
    without bound.
    """
    streaming_context = SimpleNamespace(bridge_states={})

    # Segment 1: speak + two text tokens + chunk_eos.
    seg1_output = [9304, 21, 22, 9308]
    handoff1 = _native_duplex_handoff("duplex-delta", [101, 102], list(seg1_output))
    converted1 = llm2tts([handoff1], prompt=[{}], _streaming_context=streaming_context)
    info1 = converted1[0]["model_intermediate_buffer"]
    assert info1["ids"]["output"] == seg1_output
    assert info1["llm_output_text"] == ["hello"]
    assert info1["meta"]["native_duplex_segment_text"] == "hello"
    assert "llm_output_text" in info1["meta"]["override_keys"]
    assert ["meta", "native_duplex_segment_text"] in info1["meta"]["override_keys"]
    assert converted1[0]["prompt_token_ids"] == [21, 22]
    assert info1["meta"]["segment_end"] is True

    # Segment 2: prompt now contains the folded segment-1 tokens; the
    # cumulative output list still carries segment 1 ahead of segment 2.
    seg2_output = [9304, 31, 9308]
    prompt2 = [101, 102, *seg1_output, 555]
    handoff2 = _native_duplex_handoff("duplex-delta", prompt2, [*seg1_output, *seg2_output])
    converted2 = llm2tts([handoff2], prompt=[{}], _streaming_context=streaming_context)
    info2 = converted2[0]["model_intermediate_buffer"]
    assert info2["ids"]["output"] == seg2_output
    # Cumulative text "hello" was fully delivered with segment 1; only the
    # delta (empty here) rides along with segment 2.
    assert info2["llm_output_text"] == [""]
    assert info2["meta"]["native_duplex_segment_text"] == ""
    assert converted2[0]["prompt_token_ids"] == [31]
    assert info2["meta"]["segment_end"] is True


def test_llm2tts_native_duplex_marks_talker_handoff_as_data_plane():
    streaming_context = SimpleNamespace(
        bridge_states={
            "duplex": {
                "session_id": "sid-stage1-stop",
                "incarnation": 2,
                "epoch": 0,
                "turn_id": 7,
                "session_config": {
                    "extra_body": {
                        "duplex_stage_sampling_params": {
                            "1": {"stop_token_ids": [151645]},
                        },
                    },
                },
            },
        },
    )
    handoff = _native_duplex_handoff("duplex-stage1-stop", [101, 102], [9304, 21, 9308])

    converted = llm2tts([handoff], prompt=[{}], _streaming_context=streaming_context)

    duplex = converted[0]["model_intermediate_buffer"]["duplex"]
    assert duplex["data_plane"] is True
    assert duplex["session_id"] == "sid-stage1-stop"
    assert duplex["incarnation"] == 2
    assert duplex["epoch"] == 0
    assert duplex["turn_id"] == 7
    assert duplex["session_config"]["extra_body"]["duplex_stage_sampling_params"]["1"]["stop_token_ids"] == [151645]


def test_llm2tts_native_duplex_carries_runtime_ref_audio_to_talker_payload():
    ref_audio = base64.b64encode(struct.pack("<3f", 0.1, 0.2, 0.3)).decode("ascii")
    streaming_context = SimpleNamespace(
        bridge_states={
            "duplex": {
                "session_id": "sid-runtime-ref",
                "runtime_config": {
                    "ref_audio_data": ref_audio,
                    "ref_audio_format": "pcm_f32le",
                    "ref_audio_sample_rate_hz": 16000,
                },
            },
        },
    )
    handoff = _native_duplex_handoff("duplex-runtime-ref", [101, 102], [9304, 21, 9308])

    converted = llm2tts([handoff], prompt=[{}], _streaming_context=streaming_context)

    info = converted[0]["model_intermediate_buffer"]
    assert torch.allclose(torch.tensor(info["codes"]["ref"]), torch.tensor([0.1, 0.2, 0.3]))
    assert info["meta"]["ref_audio_sr"] == 16000
    assert info["duplex"]["runtime_config"]["ref_audio_data"] == ref_audio


def test_llm2tts_native_duplex_advances_model_turn_only_after_turn_eos():
    streaming_context = SimpleNamespace(
        bridge_states={
            "duplex": {
                "session_id": "sid-model-turn",
                "epoch": 0,
                "turn_id": 9,
                "model_turn_id": 0,
            },
        }
    )

    first = _native_duplex_handoff(
        "duplex-model-turn",
        [101],
        [9304, 21, 9310],
        text="first",
    )
    converted_first = llm2tts([first], prompt=[{}], _streaming_context=streaming_context)

    first_duplex = converted_first[0]["model_intermediate_buffer"]["duplex"]
    assert first_duplex["turn_id"] == 0
    assert streaming_context.bridge_states["duplex"]["model_turn_id"] == 1

    second = _native_duplex_handoff(
        "duplex-model-turn",
        [101, 9304, 21, 9310, 555],
        [9304, 21, 9310, 9304, 22, 9308],
        text="second",
    )
    converted_second = llm2tts([second], prompt=[{}], _streaming_context=streaming_context)

    second_duplex = converted_second[0]["model_intermediate_buffer"]["duplex"]
    assert second_duplex["turn_id"] == 1
    assert streaming_context.bridge_states["duplex"]["model_turn_id"] == 1


def test_llm2tts_native_duplex_accumulates_tts_condition_across_handoffs():
    """Every handoff must carry the FULL accumulated tts condition.

    The runner's resume-prefill path REPLACES the streaming buffer (only
    in-place updates merge), so per-segment tts payloads were silently lost
    for alternating segments; the talker then vocalized text it never saw.
    Handing the complete history per handoff makes replacement lossless.
    """
    streaming_context = SimpleNamespace(bridge_states={})

    seg1_output = [9304, 21, 22, 9308]
    handoff1 = _native_duplex_handoff("duplex-acc", [101, 102], list(seg1_output))
    converted1 = llm2tts([handoff1], prompt=[{}], _streaming_context=streaming_context)
    info1 = converted1[0]["model_intermediate_buffer"]
    assert info1["ids"]["tts"] == [21, 22]
    assert len(info1["hidden_states"]["tts"]) == 2

    seg2_output = [9304, 31, 9308]
    prompt2 = [101, 102, *seg1_output, 555]
    handoff2 = _native_duplex_handoff("duplex-acc", prompt2, [*seg1_output, *seg2_output])
    converted2 = llm2tts([handoff2], prompt=[{}], _streaming_context=streaming_context)
    info2 = converted2[0]["model_intermediate_buffer"]
    assert info2["ids"]["tts"] == [21, 22, 31]
    assert len(info2["hidden_states"]["tts"]) == 3


def test_llm2tts_native_duplex_preserves_text_cursor_across_turn_reset():
    streaming_context = _install_native_duplex_decoder(
        SimpleNamespace(bridge_states={}),
        {
            21: "hello",
            31: "hello",
        },
    )

    seg1_output = [9304, 21, 9310]
    handoff1 = _native_duplex_handoff("duplex-turn-text", [101, 102], seg1_output, text="hello")
    converted1 = llm2tts([handoff1], prompt=[{}], _streaming_context=streaming_context)
    info1 = converted1[0]["model_intermediate_buffer"]
    assert info1["meta"]["native_duplex_segment_text"] == "hello"

    seg2_output = [9304, 31, 9310]
    handoff2 = _native_duplex_handoff(
        "duplex-turn-text",
        [101, 102, *seg1_output],
        seg2_output,
        text="hellohello",
    )
    converted2 = llm2tts([handoff2], prompt=[{}], _streaming_context=streaming_context)
    info2 = converted2[0]["model_intermediate_buffer"]
    assert info2["ids"]["tts"] == [31, 9310]
    assert info2["meta"]["native_duplex_segment_text"] == "hello"


def test_llm2tts_native_duplex_preserves_output_cursor_across_turn_reset():
    streaming_context = _install_native_duplex_decoder(
        SimpleNamespace(bridge_states={}),
        {
            21: "hello",
            31: "hello",
        },
    )

    seg1_output = [9304, 21, 9310]
    handoff1 = _native_duplex_handoff("duplex-turn-output", [101, 102], seg1_output, text="hello")
    converted1 = llm2tts([handoff1], prompt=[{}], _streaming_context=streaming_context)
    info1 = converted1[0]["model_intermediate_buffer"]
    assert info1["ids"]["output"] == seg1_output
    assert info1["ids"]["tts"] == [21, 9310]

    seg2_output = [9304, 31, 9310]
    handoff2 = _native_duplex_handoff(
        "duplex-turn-output",
        [101, 102, *seg1_output],
        [*seg1_output, *seg2_output],
        text="hellohello",
    )
    converted2 = llm2tts([handoff2], prompt=[{}], _streaming_context=streaming_context)
    info2 = converted2[0]["model_intermediate_buffer"]
    assert info2["ids"]["output"] == seg2_output
    assert info2["ids"]["tts"] == [31, 9310]
    assert info2["meta"]["native_duplex_segment_text"] == "hello"


def test_llm2tts_native_duplex_resets_tts_condition_on_new_turn_without_resetting_output_cursor():
    streaming_context = _install_native_duplex_decoder(
        SimpleNamespace(bridge_states={"duplex": {"turn_id": 0}}),
        {
            21: "first",
            31: "second",
        },
    )

    seg1_output = [9304, 21, 9308]
    handoff1 = _native_duplex_handoff("duplex-turn-condition", [101, 102], seg1_output, text="first")
    converted1 = llm2tts([handoff1], prompt=[{}], _streaming_context=streaming_context)
    info1 = converted1[0]["model_intermediate_buffer"]
    assert info1["ids"]["output"] == seg1_output
    assert info1["ids"]["tts"] == [21]

    streaming_context.bridge_states["duplex"]["turn_id"] = 1
    seg2_output = [9304, 31, 9308]
    handoff2 = _native_duplex_handoff(
        "duplex-turn-condition",
        [101, 102, *seg1_output],
        [*seg1_output, *seg2_output],
        text="firstsecond",
    )
    converted2 = llm2tts([handoff2], prompt=[{}], _streaming_context=streaming_context)
    info2 = converted2[0]["model_intermediate_buffer"]
    assert info2["ids"]["output"] == seg2_output
    assert info2["ids"]["tts"] == [31]
    assert info2["meta"]["native_duplex_segment_text"] == "second"


def test_llm2tts_never_aliases_thinker_token_list():
    """The talker prompt must never be the thinker's live token list object.

    CompletionOutput.token_ids can alias the upstream detokenizer's internal
    list; forwarding that object lets the stage-1 streaming update extend the
    list with itself, doubling the recorded output every segment.
    """
    live_token_list = [9304, 21, 22, 9308]
    handoff = _native_duplex_handoff("duplex-alias", [101, 102], live_token_list)
    converted = llm2tts([handoff], prompt=[{}])
    scheduler_prompt = converted[0]["prompt_token_ids"]
    assert scheduler_prompt is not live_token_list
    scheduler_prompt.extend(scheduler_prompt)
    assert live_token_list == [9304, 21, 22, 9308]
