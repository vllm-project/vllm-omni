# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU request transport and prefill boundaries, with substituted encoders.

Official prompt fixtures supply expected IDs/positions. Real HF mel extraction,
Omni payload serialization, and the AR preprocess method execute here. Neural
audio encoding and the distributed AR network are not exercised by this file.
"""

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import msgspec
import numpy as np
import pytest
import torch
from transformers import WhisperFeatureExtractor

from vllm_omni.model_executor.models.kimi_audio.audio_processing import prepare_kimi_audio_inputs
from vllm_omni.model_executor.models.kimi_audio.kimi_audio_ar_stage import KimiAudioARStage, KimiAudioInputEncoder
from vllm_omni.model_executor.models.kimi_audio.prompt import (
    KimiAudioEncodedAudio,
    KimiAudioPromptBuilder,
    KimiAudioSpecialTokens,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.omni]
REFERENCE = json.loads((Path(__file__).parent / "fixtures/prompt_reference.json").read_text(encoding="utf-8"))


@pytest.fixture
def handoff_runtime(monkeypatch):
    config = REFERENCE["input_config"]
    builder = KimiAudioPromptBuilder(
        REFERENCE["text_tokens"].__getitem__, KimiAudioSpecialTokens.from_vocab(REFERENCE["special_tokens"]), **config
    )
    runtime = SimpleNamespace(builder=builder, extractor=WhisperFeatureExtractor(feature_size=128), calls=[])
    # AR construction/loading is tested separately. Here CPU embeddings and a
    # small linear projection make the stream fusion independently inspectable.
    stage = KimiAudioARStage.__new__(KimiAudioARStage)
    torch.nn.Module.__init__(stage)
    stage.config = SimpleNamespace(
        hidden_size=4,
        kimia_adaptor_input_dim=config["continuous_feature_size"],
        kimia_token_offset=config["audio_token_offset"],
        vocab_size=config["audio_token_offset"] + config["audio_vocab_size"],
    )
    stage.embed_tokens = torch.nn.Embedding(stage.config.vocab_size, 4)
    stage.vq_adaptor = torch.nn.Linear(stage.config.kimia_adaptor_input_dim, 4)
    with torch.no_grad():
        stage.embed_tokens.weight.copy_(
            torch.rand(stage.config.vocab_size, 4, generator=torch.Generator().manual_seed(712))
        )
        stage.vq_adaptor.weight.zero_()
        stage.vq_adaptor.weight[:, :4].copy_(torch.eye(4))
        stage.vq_adaptor.bias.fill_(0.25)
    stage.input_encoder = KimiAudioInputEncoder(vllm_config=SimpleNamespace())

    def encode_audio(waveform, *, sampling_rate, whisper_inputs):
        runtime.calls.append((waveform.copy(), whisper_inputs))
        assert sampling_rate == 16000
        assert not torch.is_grad_enabled()
        spec = next(spec for spec in REFERENCE["audio_inputs"].values() if spec["feature_value"] == waveform[0])
        features = None
        if whisper_inputs is not None:
            assert sum(whisper_inputs.token_lengths) == len(spec["codes"])
            assert whisper_inputs.input_features.shape == (1, 128, 3000)
            features = torch.full((len(spec["codes"]), stage.config.kimia_adaptor_input_dim), spec["feature_value"])
        return KimiAudioEncodedAudio(spec["codes"], features)

    monkeypatch.setattr(stage.input_encoder, "encode_audio", encode_audio)
    runtime.stage = stage.eval()
    return runtime


@pytest.mark.parametrize(
    "case_name", ["text", "empty_history", "group_text_two_audios", "delayed_history", "audio_both_output"]
)
def test_request_roundtrip_and_chunked_prefill_match_official_layout(handoff_runtime, case_name):
    runtime = handoff_runtime
    case = next(case for case in REFERENCE["cases"] if case["name"] == case_name)
    messages = copy.deepcopy(case["messages"])
    audios = {}
    for index, message in enumerate(messages):
        kind = message["message_type"]
        if kind not in ("audio", "audio-text"):
            continue
        source = message["content"] if kind == "audio" else message["content"][0]
        spec = REFERENCE["audio_inputs"][source]
        audios[index] = np.full(len(spec["codes"]) * 1280 - 1, spec["feature_value"], dtype=np.float32)
    prepared = prepare_kimi_audio_inputs(
        messages,
        runtime.builder,
        audio_inputs=audios,
        feature_extractor=runtime.extractor,
        output_type=case["output_type"],
        add_assistant_start_msg=case["add_assistant_start_msg"],
    )
    assert runtime.calls == []
    assert messages == case["messages"]
    # Generic dict IPC has no tensor type annotations. The complete request
    # must survive plain MessagePack without any custom object/pickle support.
    restored = msgspec.msgpack.decode(msgspec.msgpack.encode(prepared))
    assert restored == prepared
    for waveform in audios.values():
        waveform[:] = -99  # queued bytes and cache salt must be independent
    info = restored["model_intermediate_buffer"]
    size = len(prepared["prompt_token_ids"])
    placeholders = torch.tensor(prepared["prompt_token_ids"])
    boundaries = sorted({0, 1, size - 1, size})
    chunks = list(zip(boundaries, boundaries[1:]))
    actual_ids, actual_embeds = [], []
    for start, end in chunks:
        ids, embeds, update = runtime.stage.preprocess(
            placeholders[start:end],
            None,
            **info,
            _omni_is_prefill=True,
            _omni_num_computed_tokens=start,
            _omni_prompt_len=size,
        )
        info.update(update)
        actual_ids.extend(ids.tolist())
        actual_embeds.append(embeds)
    assert actual_ids == case["audio_token_ids"]
    assert len(runtime.calls) == len(audios)
    assert [whisper is not None for _, whisper in runtime.calls] == [
        message["message_type"] == "audio"
        for message in case["messages"]
        if message["message_type"] in ("audio", "audio-text")
    ]
    assert prepared["prompt_token_ids"] == placeholders.tolist()
    # Expected positions come from the OFFICIAL PromptManager fixture.
    with torch.no_grad():
        expected = runtime.stage.embed_tokens(torch.tensor(case["audio_token_ids"]))
        feature_values = []
        for message in case["messages"]:
            if message["message_type"] == "audio":
                spec = REFERENCE["audio_inputs"][message["content"]]
                feature_values.extend([spec["feature_value"]] * len(spec["codes"]))
        for position, value in zip(case["continuous_positions"], feature_values, strict=True):
            expected[position] = (expected[position] + value + 0.25) * math.sqrt(2)
        expected += runtime.stage.embed_tokens(torch.tensor(case["text_token_ids"]))
    torch.testing.assert_close(torch.cat(actual_embeds), expected)
    # Replay an earlier chunk using its absolute scheduler offset, not a
    # mutable advancing cursor; the encoders must not run a second time.
    _, replay, update = runtime.stage.preprocess(
        placeholders[:1],
        None,
        **info,
        _omni_is_prefill=True,
        _omni_num_computed_tokens=0,
        _omni_prompt_len=size,
    )
    torch.testing.assert_close(replay, expected[:1])
    assert update == {} and len(runtime.calls) == len(audios)


def test_cache_identity_covers_text_audio_and_request_ownership(handoff_runtime):
    runtime = handoff_runtime
    messages = [{"role": "user", "message_type": "audio", "content": "same-path.wav"}]
    audio = np.ones(100, dtype=np.float32)
    first = prepare_kimi_audio_inputs(
        messages, runtime.builder, audio_inputs={0: audio}, feature_extractor=runtime.extractor
    )
    repeated = prepare_kimi_audio_inputs(
        messages, runtime.builder, audio_inputs={0: audio}, feature_extractor=runtime.extractor
    )
    audio[-1] = 0.5
    changed = prepare_kimi_audio_inputs(
        messages, runtime.builder, audio_inputs={0: audio}, feature_extractor=runtime.extractor
    )
    assert first["prompt_token_ids"] == changed["prompt_token_ids"]
    assert first["cache_salt"] == repeated["cache_salt"] != changed["cache_salt"]
    # Same placeholder audio stream, different text conditioning.
    texts = [key for key, ids in REFERENCE["text_tokens"].items() if ids]
    pair = next(
        (a, b)
        for a in texts
        for b in texts
        if a != b and len(REFERENCE["text_tokens"][a]) == len(REFERENCE["text_tokens"][b])
    )
    requests = [
        prepare_kimi_audio_inputs([{"role": "user", "message_type": "text", "content": text}], runtime.builder)
        for text in pair
    ]
    assert requests[0]["prompt_token_ids"] == requests[1]["prompt_token_ids"]
    assert requests[0]["cache_salt"] != requests[1]["cache_salt"]
    first["prompt_token_ids"].clear()
    assert repeated["prompt_token_ids"]


def test_truncated_prompt_and_decode_do_not_silently_use_prefill(handoff_runtime):
    runtime = handoff_runtime
    case = next(case for case in REFERENCE["cases"] if case["name"] == "text")
    prepared = prepare_kimi_audio_inputs(case["messages"], runtime.builder)
    ids = torch.tensor(prepared["prompt_token_ids"])
    with pytest.raises(ValueError, match="length differs"):
        runtime.stage.preprocess(
            ids[:1],
            None,
            **prepared["model_intermediate_buffer"],
            _omni_is_prefill=True,
            _omni_num_computed_tokens=0,
            _omni_prompt_len=len(ids) - 1,
        )
    with pytest.raises(ValueError, match="generation state for decode"):
        runtime.stage.preprocess(ids[:1], None, _omni_is_prefill=False)
    assert runtime.calls == []
