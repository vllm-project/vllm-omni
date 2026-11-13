# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from vllm.transformers_utils.config import get_config

from tests.model_executor.models.test_zonos3_modeling import release_params
from vllm_omni.config.resolver import resolve_omni_config
from vllm_omni.engine.serialization import deserialize_additional_information, serialize_additional_information
from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
from vllm_omni.entrypoints.openai.tts_adapters.zonos3 import Zonos3TTSAdapter
from vllm_omni.model_executor.models.zonos3.ar_config import Zonos3Architecture
from vllm_omni.model_executor.models.zonos3.prompt import SILENCE_FRAME, Zonos3PromptBuilder
from vllm_omni.transformers_utils.configs.zonos3 import Zonos3Config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_raw_speaker_adapter_reserves_one_row_and_serializes_waveform(tmp_path):
    pytest.importorskip("zyphra_textnorm")
    config = Zonos3Config({**release_params(), "text_vocab": 448, "speaker_embedding_codec": "ownmect_s2"})
    waveform = torch.linspace(-0.1, 0.1, 16000)
    server = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config, model=str(tmp_path), revision=None),
        _apply_uploaded_speaker=Mock(return_value=None),
        _validate_ref_audio_format=Mock(return_value=None),
        _resolve_ref_audio_array=AsyncMock(return_value=(waveform.numpy(), 16000, "reference")),
    )
    adapter = Zonos3TTSAdapter(SpeechServingContext(server=server))
    request = OpenAICreateSpeechRequest(
        input="hello",
        ref_audio="https://example.com/reference.wav",
        extra_params={"silence_prefill_frames": 0, "text_normalization": False},
    )
    assert adapter.validate(request) is None
    prepared = asyncio.run(adapter.build(request, [], True))
    prompt = prepared.prompt
    assert len(prompt["prompt_token_ids"]) == 1 + 3 + 7
    serialized = serialize_additional_information(prompt["additional_information"])
    restored = deserialize_additional_information(serialized)
    # Serialization flattens nested metadata keys at the EngineCore boundary.
    from vllm_omni.data_entry_keys import unflatten_payload

    metadata = unflatten_payload(restored)["zonos3_prompt"]
    assert metadata["condition_start"] == 1 and metadata["speaker_row"] == 0
    assert "speaker_embedding" not in metadata
    assert metadata["reference_sample_rate"] == 16000
    torch.testing.assert_close(metadata["reference_waveform"], waveform, rtol=0, atol=0)
    server._resolve_ref_audio_array.assert_awaited_once_with(request.ref_audio)


def test_speaker_embedding_validation_and_request_chunk_settings(tmp_path):
    pytest.importorskip("zyphra_textnorm")
    config = Zonos3Config({**release_params(), "text_vocab": 448, "speaker_embedding_codec": "ownmect_s2"})
    server = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config, model=str(tmp_path), revision=None),
        _apply_uploaded_speaker=Mock(return_value=None),
        _validate_ref_audio_format=Mock(return_value=None),
        _resolve_ref_audio_array=AsyncMock(),
    )
    adapter = Zonos3TTSAdapter(SpeechServingContext(server=server))
    malformed = OpenAICreateSpeechRequest(
        input="hello",
        speaker_embedding=[1.0, 2.0],
        extra_params={"text_normalization": False},
    )
    assert "speaker_embedding" in adapter.validate(malformed)

    request = OpenAICreateSpeechRequest(
        input="hello",
        extra_params={
            "text_normalization": False,
            "silence_prefill_frames": 0,
            "chunk_size": 7,
            "chunk_ramp": [1, 3],
            "first_chunk_codebooks": 6,
        },
    )
    assert adapter.validate(request) is None
    prompt = adapter._build_prompt(request)
    metadata = prompt["additional_information"]["zonos3_prompt"]
    assert metadata["codec_chunking"] == {"chunk_size": 7, "chunk_ramp": [1, 3], "first_chunk_codebooks": 6}

    invalid = OpenAICreateSpeechRequest(input="hello", extra_params={"chunk_ramp": [0], "text_normalization": False})
    assert "chunk_ramp" in adapter.validate(invalid)


def test_reference_codes_and_raw_waveform_are_distinct_clone_modes(tmp_path):
    pytest.importorskip("zyphra_textnorm")
    params = {
        **release_params(),
        "text_vocab": 448,
        "speaker_embedding_codec": "ownmect_s2",
        "speaker_reference_prefix_enabled": True,
        "audio_codec": "opus_dual_3n_600k_sched600k_r4096_step600000",
    }
    config = Zonos3Config(params)
    server = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config, model=str(tmp_path), revision=None),
        _apply_uploaded_speaker=Mock(return_value=None),
        _validate_ref_audio_format=Mock(return_value=None),
        _resolve_ref_audio_array=AsyncMock(),
    )
    adapter = Zonos3TTSAdapter(SpeechServingContext(server=server))
    request = OpenAICreateSpeechRequest(
        input="hello",
        ref_text="reference transcript",
        ref_audio_codes=[[i, i + 1] for i in range(10)],
        extra_params={"text_normalization": False, "silence_prefill_frames": 0},
    )
    assert adapter.validate(request) is None
    prompt = adapter._build_prompt(request)
    metadata = prompt["additional_information"]["zonos3_prompt"]
    assert prompt["prompt_token_ids"] == [0] * len(metadata["frame_ids"])
    assert metadata["frame_ids"][:11, 10].tolist() == [448] * 11


def test_zonos3_rejects_uploaded_audio_voice_without_reference_codec_encoder(tmp_path):
    pytest.importorskip("zyphra_textnorm")
    config = Zonos3Config({**release_params(), "text_vocab": 448, "speaker_embedding_codec": "ownmect_s2"})
    server = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config, model=str(tmp_path), revision=None),
        _apply_uploaded_speaker=Mock(side_effect=lambda request: setattr(request, "ref_audio", "file:///voice.wav")),
        _validate_ref_audio_format=Mock(return_value=None),
        _resolve_ref_audio_array=AsyncMock(),
    )
    adapter = Zonos3TTSAdapter(SpeechServingContext(server=server))
    request = OpenAICreateSpeechRequest(input="hello", voice="uploaded")
    assert "unsupported reference-prefix encoder" in adapter.validate(request)


def test_deployment_resolves_both_native_stages_with_release_parser(tmp_path):
    (tmp_path / "params.json").write_text(json.dumps(release_params()))
    deploy = Path(__file__).resolve().parents[3] / "vllm_omni/deploy/zonos3_mini.yaml"
    resolved = resolve_omni_config(
        str(tmp_path),
        trust_remote_code=False,
        deploy_config_path=str(deploy),
        cli_overrides={},
        stage_overrides=None,
        strategy_config_path=None,
    )
    assert len(resolved.stage_configs) == 2
    ar, codec = resolved.stage_configs
    for stage in (ar, codec):
        assert stage.load_config.config_format == "zonos3"
        assert stage.model_config.use_v2_model_runner
        assert stage.model_config.supports_native_mrv2_data_plane
        assert stage.connector_config.async_chunk
        assert stage.cache_config.enable_prefix_caching is False
    assert ar.model_config.recompute_preemption == "fail"
    assert ar.model_config.enforce_eager is False
    assert "FULL_DECODE_ONLY" in str(ar.compilation_config)
    assert codec.stage_pipeline_config.hf_config_name == "codec_config"
    assert codec.stage_pipeline_config.retains_state_across_chunks
    assert codec.runtime_config.additional_config["zonos3_codec_cuda_graph"] is True


def test_release_parser_roundtrip_and_conditioning_prompt_lengths(tmp_path):
    pytest.importorskip("zyphra_textnorm")
    params = {
        **release_params(),
        "text_vocab": 448,
        "speaker_reference_prefix_enabled": True,
        "audio_codec": "opus_dual_3n_600k_sched600k_r4096_step600000",
    }
    (tmp_path / "params.json").write_text(json.dumps(params))
    config = get_config(str(tmp_path), trust_remote_code=False, config_format="zonos3")
    restored = Zonos3Config.from_dict(config.to_dict())
    assert restored.zonos3_params == params
    assert restored.vocab_size == 3 and restored.codec_config.architectures == ["Zonos3CodecModel"]
    builder = Zonos3PromptBuilder(Zonos3Architecture.from_params(params), str(tmp_path), "lm/zonos3_mini")
    assert builder.canonical_language("Chinese") == "cmn"
    tokens = [2, *(byte + 192 for byte in "你好".encode()), 3]
    assert builder.text_ids("你好", "zh", False) == tokens
    for mode in ("text", "speaker", "prefix"):
        options = {}
        prefix_length = 0
        if mode == "speaker":
            options["speaker_embedding"] = torch.ones(192)
            prefix_length = 1
        elif mode == "prefix":
            options["reference_codes"] = torch.arange(20).reshape(2, 10)
            prefix_length = 11
        prompt = builder.build("你好", normalize=False, silence_frames=3, **options)
        metadata = prompt["additional_information"]["zonos3_prompt"]
        frames = metadata["frame_ids"]
        assert len(prompt["prompt_token_ids"]) == len(frames) == prefix_length + 3 + len(tokens) + 3
        assert metadata["condition_start"] == prefix_length
        assert frames[prefix_length + 3 : -3, 10].tolist() == tokens
        assert frames[-3:, 0].tolist() == [SILENCE_FRAME[0]] * 3
        assert frames[-3:, 9].tolist() == [4097] * 3
        assert metadata["has_audio_prefix"]
    with pytest.raises(ValueError, match="reference codes"):
        builder.build("test", reference_codes=torch.zeros(2, 10), silence_frames=0)
