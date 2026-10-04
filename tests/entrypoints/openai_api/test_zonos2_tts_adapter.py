# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""ZONOS2 P1 registration and P4 serving contracts, without downloads."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from vllm import SamplingParams

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.tts_adapters import all_tts_stage_keys, detect_tts_model_type, resolve_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
from vllm_omni.entrypoints.openai.tts_adapters.zonos2 import Zonos2Adapter

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    "language,expected",
    [
        ("English", "en_us"),
        ("en-GB", "en_gb"),
        ("Chinese", "cmn"),
        ("zh-CN", "cmn"),
        ("Japanese", "ja"),
        ("Korean", "ko"),
        ("French", "fr_fr"),
        ("de", "de"),
        ("Spanish", "es"),
        ("pt-BR", "pt_br"),
        ("Italian", "it"),
    ],
)
def test_language_aliases(language, expected):
    adapter = _adapter()
    request = OpenAICreateSpeechRequest(input="Test", language=language)
    assert adapter.validate(request) is None
    assert adapter._conditioning(request)["language"] == expected


@pytest.mark.parametrize(
    "extra",
    [
        {"quality_values": {"lufs": float("nan")}},
        {"quality_buckets": {"unknown": 0}},
        {"quality_buckets": {"lufs": 1.5}},
        {"speaking_rate_bucket": 1.5},
        {"speaking_rate": float("inf")},
        {"emotion_cfg_scale": 1.5},
        {"cfg_scale": 0},
        {"emotion": "happy"},
        {"quality_buckets": {"lufs": 99}},
        {"speaking_rate_bucket": 99},
        {"speaking_rate": 0},
        {"text_normalization": "no"},
        {"quality_buckets": {}, "quality_values": {}},
        {"speaking_rate": 10, "speaking_rate_bucket": 1},
    ],
)
def test_conditioning_rejections(extra):
    assert _adapter().validate(OpenAICreateSpeechRequest(input="Test", extra_params=extra))


def test_rate_quality_conditioning_and_default_unchanged():
    adapter = _adapter()
    assert adapter._conditioning(OpenAICreateSpeechRequest(input="Test"))["speaking_rate_bucket"] is None
    request = OpenAICreateSpeechRequest(
        input="Test", speed=2, extra_params={"quality_buckets": {"lufs": 3}, "emotion_cfg_scale": 1}
    )
    assert adapter.validate(request) is None
    values = adapter._conditioning(request)
    assert values["speaking_rate_bucket"] == 2
    assert values["quality_buckets"] == [3, None, None, None, None, None]
    assert adapter.validate(OpenAICreateSpeechRequest(input="Test", language="Auto"))


def test_reference_audio_adapter_reuses_resolver_and_encoder():
    import torch

    adapter = _adapter()
    called = []

    async def resolve(value):
        called.append(value)
        return [0.0] * 24000, 24000, "reference-key"

    adapter.ctx.server._resolve_ref_audio = resolve
    adapter._speaker = SimpleNamespace(encode=lambda wav, sr: torch.ones(2048))
    adapter._get_processor().normalizer.normalize = lambda text, language: text
    prepared = asyncio.run(
        adapter.build(
            OpenAICreateSpeechRequest(input="Test", ref_audio="data:audio/wav;base64,AAAA", language="Chinese"),
            [],
            False,
        )
    )
    info = prepared.prompt["additional_information"]
    assert info["zonos2_speaker_embedding"].shape == (2048,)
    assert info["zonos2_frames"][0, 9] == 519
    assert prepared.tts_params == {"ref_audio_cache_key": "reference-key", "zonos2_token_budget": 1024}
    assert len(called) == 1


def test_audio_decode_backend_error_is_actionable():
    adapter = _adapter()

    async def fail(value):
        raise FileNotFoundError("ffmpeg")

    adapter.ctx.server._resolve_ref_audio = fail
    with pytest.raises(ValueError, match="ffmpeg"):
        asyncio.run(adapter.build(OpenAICreateSpeechRequest(input="Test", ref_audio="bad"), [], False))


def test_p4_overrides_only_stage0_and_no_default_mutation():
    params = [SamplingParams(max_tokens=1024), SamplingParams(max_tokens=10)]
    request = OpenAICreateSpeechRequest(input="Test", seed=123, extra_params={"min_p": 0.3})
    result = _adapter().apply_sampling_overrides(params, request)
    assert result[0].seed == 123 and result[0].extra_args["min_p"] == 0.3
    assert result[1].max_tokens == 10 and params[0].extra_args is None


def _adapter(formats=("dummy", "dummy"), *, legacy=False):
    stages = []
    for stage_id, (stage_name, fmt) in enumerate(zip(("zonos2", "dac_decoder"), formats)):
        args = SimpleNamespace(model_stage=stage_name, model_arch=None, worker_type="ar", load_format=fmt)
        if legacy:
            stage = SimpleNamespace(engine_args=args)
        else:
            from vllm_omni.config.stage_config import StagePipelineConfig

            stage = SimpleNamespace(
                stage_pipeline_config=StagePipelineConfig(stage_id=stage_id, model_stage=stage_name),
                model_config=SimpleNamespace(model_arch=None),
                load_config=SimpleNamespace(load_format=fmt),
                worker_type="ar",
            )
        stages.append(stage)
    from vllm_omni.model_executor.models.zonos2.configuration_zonos2 import Zonos2Config

    counts = {
        "lufs": 12,
        "estimated_snr": 12,
        "max_pause": 12,
        "estimated_bandlimit_hz": 8,
        "leading_silence_s": 8,
        "trailing_silence_s": 8,
    }
    config = Zonos2Config(
        quality_features=list(counts), quality_buckets={k: list(map(str, range(n))) for k, n in counts.items()}
    )
    engine = SimpleNamespace(stage_configs=stages, model_config=SimpleNamespace(hf_config=config))
    return Zonos2Adapter(
        SpeechServingContext(
            server=SimpleNamespace(_validate_ref_audio_format=lambda value: None), engine_client=engine
        )
    )


def test_zonos2_registration_and_talker_detection():
    assert resolve_adapter("zonos2") is Zonos2Adapter
    assert "zonos2" in all_tts_stage_keys()
    for arch in (None, "Zonos2ForConditionalGeneration", "Zonos2TalkerForConditionalGeneration"):
        assert detect_tts_model_type("zonos2", arch) == "zonos2"
    assert detect_tts_model_type(None, "Zonos2ForConditionalGeneration") == "zonos2"
    assert detect_tts_model_type("dac_decoder", "Zonos2Code2WavForConditionalGeneration") is None


@pytest.mark.parametrize("legacy", [False, True])
def test_serving_discovers_the_talker_and_accepts_dummy_request(legacy):
    adapter = _adapter(legacy=legacy)
    server = OmniOpenAIServingSpeech.__new__(OmniOpenAIServingSpeech)
    server.engine_client = adapter.ctx.engine_client
    server._tts_stage = server._find_tts_stage()
    assert server._tts_stage is server.engine_client.stage_configs[0]
    assert server._detect_tts_model_type() == "zonos2"
    request = OpenAICreateSpeechRequest(input="Hello.", voice="default", max_new_tokens=16)
    assert adapter.validate(request) is None
    assert adapter.load_capabilities().supported_speakers == {"default"}


@pytest.mark.parametrize("formats", [("auto", "auto"), ("dummy", "auto"), ("auto", "dummy")])
def test_p4_accepts_real_weight_deployments(formats):
    assert _adapter(formats).validate(OpenAICreateSpeechRequest(input="Hello.")) is None


@pytest.mark.parametrize("text", ["", " ", "\n\t"])
def test_rejects_empty_input(text):
    assert "empty" in _adapter().validate(OpenAICreateSpeechRequest(input=text)).lower()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"voice": "vivian"},
        {"speaker_embedding": [0.0]},
        {"instructions": "happy"},
        {"sample_rate": 24000},
        {"max_new_tokens": 1025},
        {"word_timestamps": True},
    ],
)
def test_p4_rejects_unsupported_features(kwargs):
    assert _adapter().validate(OpenAICreateSpeechRequest(input="Hello.", **kwargs)) is not None


def test_build_uses_the_p2_processor_without_a_hf_tokenizer():
    adapter = _adapter()
    calls = []

    def build_prompt(text, **kwargs):
        calls.append(text)
        return {"prompt_token_ids": [519, 2, 3], "additional_information": {"zonos2_frames": "sentinel"}}

    adapter._processor = SimpleNamespace(
        build_prompt=build_prompt, config=adapter.ctx.engine_client.model_config.hf_config
    )
    request = OpenAICreateSpeechRequest(input="中文 dummy.", max_new_tokens=16)
    prepared = asyncio.run(adapter.build(request, [], False))
    assert calls == ["中文 dummy."]
    assert prepared.model_type == "zonos2"
    assert prepared.prompt["additional_information"]["zonos2_frames"] == "sentinel"
    assert prepared.tts_params == {"zonos2_token_budget": 16}
    assert request.input == "中文 dummy."


def test_max_tokens_override_does_not_mutate_defaults_or_codec_params():
    adapter = _adapter()
    params = [SamplingParams(max_tokens=64), SamplingParams(max_tokens=128)]
    request = OpenAICreateSpeechRequest(input="Hello.", max_new_tokens=16)
    result = adapter.apply_sampling_overrides(params, request)
    assert result[0].max_tokens == 16
    assert result[1].max_tokens == 128
    assert params[0].max_tokens == 64


def test_p3_adapter_accepts_request_seed_and_audio_sampling_controls():
    request = OpenAICreateSpeechRequest(
        input="Hello.",
        seed=7,
        extra_params={
            "temperature": 0.7,
            "top_k": 8,
            "top_p": 0.9,
            "min_p": 0.2,
            "repetition_window": 3,
            "repetition_penalty": 1.5,
        },
    )
    assert _adapter().validate(request) is None


@pytest.mark.parametrize(
    "extra",
    [
        {"min_p": 1.1},
        {"top_k": -2},
        {"repetition_window": -1},
        {"repetition_penalty": 0.5},
        {"repetition_codebooks": 9},
        {"emotion": 1},
    ],
)
def test_p3_adapter_rejects_invalid_or_out_of_scope_sampling_fields(extra):
    assert _adapter().validate(OpenAICreateSpeechRequest(input="Hello.", extra_params=extra)) is not None


@pytest.mark.parametrize("speed,bucket", [(0.25, 1), (0.5, 2), (1.0, 4), (2.0, 7), (4.0, 7)])
def test_official_speed_ranges(speed, bucket):
    adapter = _adapter()
    config = adapter.ctx.engine_client.model_config.hf_config
    config.speaking_rate_buckets = ["0-3", "3-6", "6-9", "9-12", "12-15", "15-18", "18-21", "21+"]
    request = OpenAICreateSpeechRequest(input="Test", speed=speed)
    assert adapter._conditioning(request)["speaking_rate_bucket"] == bucket


@pytest.mark.parametrize("value,bucket", [(-100, 0), (-60, 0), (-55, 1), (-0.5, 11), (0, 11), (99, 11)])
def test_quality_negative_ranges_and_outside_clamping(value, bucket):
    adapter = _adapter()
    config = adapter.ctx.engine_client.model_config.hf_config
    config.quality_buckets["lufs"] = [f"{low}-{low + 5}" for low in range(-60, 0, 5)]
    request = OpenAICreateSpeechRequest(input="Test", extra_params={"quality_values": {"lufs": value}})
    assert adapter._conditioning(request)["quality_buckets"][0] == bucket


@pytest.mark.parametrize("finish,tokens", [("length", 10), ("stop", 1024)])
def test_codec_budget_does_not_silently_return_truncated_speech(finish, tokens):
    from vllm_omni.entrypoints.openai.tts_adapters.base import TTSGenerationError

    with pytest.raises(TTSGenerationError, match="incomplete"):
        _adapter().validate_generation({"zonos2_token_budget": 1024}, stage0_finish_reason=finish, output_tokens=tokens)


def test_natural_eos_below_budget_is_valid():
    _adapter().validate_generation({"zonos2_token_budget": 1024}, stage0_finish_reason="stop", output_tokens=300)
