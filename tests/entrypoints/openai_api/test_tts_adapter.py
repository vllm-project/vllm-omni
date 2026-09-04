# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the TTS serving adapter registry (RFC #4327).

Pure-Python registry/resolution logic; no model or GPU resources are loaded.
"""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from vllm.sampling_params import SamplingParams

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.tts_adapters import (
    TTS_ADAPTER_REGISTRY,
    ARTTSAdapter,
    DiffusionTTSAdapter,
    SpeechServingContext,
    all_tts_model_types,
    detect_tts_model_type,
    resolve_adapter,
)
from vllm_omni.entrypoints.openai.tts_adapters.higgs_audio_v2 import HiggsAudioV2Adapter
from vllm_omni.entrypoints.openai.tts_adapters.indextts2 import (
    IndexTTS2Adapter,
    IndexTTS25Adapter,
    indextts2_conditioning_cache_salt,
)
from vllm_omni.entrypoints.openai.tts_adapters.moss_tts import (
    MossTTSAdapter,
    MossTTSNanoAdapter,
)
from vllm_omni.entrypoints.openai.tts_adapters.qwen3_tts import (
    QWEN3_TTS_EFFECTIVE_MAX_TOKENS_KEY,
    Qwen3TTSAdapter,
    Qwen3TTSCodecLimitError,
)
from vllm_omni.model_executor.models.indextts2 import prompt_utils
from vllm_omni.model_executor.models.indextts2.tokenizer_v2_5 import (
    INDEXTTS25_TOKENIZER_FILE,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# Every dedicated TTS model-type must have an adapter so the orchestrator's
# uniform ``self._adapter.build(...)`` dispatch covers it.
EXPECTED_MODEL_TYPES = {
    "qwen3_tts",
    "voxcpm2",
    "voxtral_tts",
    "fish_tts",
    "cosyvoice3",
    "omnivoice",
    "covo_audio",
    "ming_tts",
    "moss_tts_nano",
    "moss_tts",
    "higgs_audio_v2",
    "higgs_audio_v3",
    "glm_tts",
    "step_audio2",
    "indextts2",
    "indextts2_5",
}


def test_all_model_types_registered():
    assert EXPECTED_MODEL_TYPES <= all_tts_model_types()


def test_registry_keyed_by_name():
    for name, cls in TTS_ADAPTER_REGISTRY.items():
        assert cls.name == name


def test_resolve_each_model_type():
    for model_type in EXPECTED_MODEL_TYPES:
        cls = resolve_adapter(model_type)
        assert cls is not None, model_type
        assert cls.name == model_type


def test_resolve_qwen3_tts_class():
    assert resolve_adapter("qwen3_tts") is Qwen3TTSAdapter


def test_resolve_unknown_returns_none():
    assert resolve_adapter("not_a_real_model") is None
    assert resolve_adapter(None) is None


def test_voxcpm2_resolves():
    """VoxCPM2 (the served ``latent_generator`` model) resolves cleanly.

    Detection never returns the legacy ``voxcpm`` type, so there is no shared
    stage-key ambiguity to resolve.
    """
    assert resolve_adapter("voxcpm2") is not None
    assert resolve_adapter("voxcpm") is None


def test_all_adapters_are_ar_or_diffusion():
    for cls in TTS_ADAPTER_REGISTRY.values():
        assert issubclass(cls, (ARTTSAdapter, DiffusionTTSAdapter))
        assert cls.backend in ("ar", "diffusion")


@pytest.mark.parametrize("adapter_cls", [MossTTSAdapter, MossTTSNanoAdapter])
def test_moss_tts_applies_request_max_new_tokens(adapter_cls):
    adapter = adapter_cls(SimpleNamespace(server=object()))
    stage_defaults = [SimpleNamespace(max_tokens=4096)]

    overridden = adapter.apply_sampling_overrides(
        stage_defaults,
        SimpleNamespace(max_new_tokens=512),
    )

    assert overridden[0].max_tokens == 512
    assert stage_defaults[0].max_tokens == 4096


def _build_moss_tts_request(adapter_cls, mocker, *, request_seed):
    server = mocker.Mock()
    server._build_moss_tts_params = mocker.AsyncMock(return_value={})
    adapter = adapter_cls(SpeechServingContext(server=server))
    request = OpenAICreateSpeechRequest(input="hello", seed=request_seed)

    return asyncio.run(
        adapter.build(
            request,
            [SamplingParams(seed=42)],
            has_inline_ref_audio=False,
        )
    )


# Full-family coverage pins the adapter contract; only Nano consumes this seed end to end today.
@pytest.mark.parametrize("adapter_cls", [MossTTSAdapter, MossTTSNanoAdapter])
@pytest.mark.parametrize("request_seed", [0, 1234])
def test_moss_tts_request_seed_overrides_stage_default(adapter_cls, request_seed, mocker):
    prepared = _build_moss_tts_request(
        adapter_cls,
        mocker,
        request_seed=request_seed,
    )

    assert prepared.tts_params["seed"] == [request_seed]


@pytest.mark.parametrize("adapter_cls", [MossTTSAdapter, MossTTSNanoAdapter])
def test_moss_tts_seed_falls_back_to_stage_default(adapter_cls, mocker):
    prepared = _build_moss_tts_request(
        adapter_cls,
        mocker,
        request_seed=None,
    )

    assert prepared.tts_params["seed"] == [42]


def test_qwen3_tts_metadata():
    assert Qwen3TTSAdapter.backend == "ar"
    assert issubclass(Qwen3TTSAdapter, ARTTSAdapter)


@pytest.mark.parametrize(
    ("task_type", "text_tokens", "request_cap", "expected_cap"),
    [
        ("Base", 0, None, 4096),
        ("Base", 10, None, 192),
        ("Base", 23, None, 276),
        ("Base", 23, 128, 128),
        ("Base", 23, 512, 512),
        ("Base", 400, None, 4096),
        ("CustomVoice", 10, None, 4096),
        ("CustomVoice", 10, 256, 256),
    ],
)
def test_qwen3_tts_applies_text_scaled_codec_safety_limit(task_type, text_tokens, request_cap, expected_cap, mocker):
    server = mocker.Mock()
    server._count_usage_text_tokens.return_value = text_tokens
    adapter = Qwen3TTSAdapter(SpeechServingContext(server=server))
    stage_defaults = [SamplingParams(max_tokens=4096, min_tokens=2)]
    request = OpenAICreateSpeechRequest(
        input="test text",
        task_type=task_type,
        max_new_tokens=request_cap,
    )
    prompt: dict[str, Any] = {"additional_information": {}}

    overridden = adapter.apply_sampling_overrides(stage_defaults, request, prompt)

    assert overridden[0].max_tokens == expected_cap
    assert prompt["additional_information"][QWEN3_TTS_EFFECTIVE_MAX_TOKENS_KEY] == [expected_cap]
    assert stage_defaults[0].max_tokens == 4096


def test_qwen3_tts_rejects_only_length_finished_base_audio(mocker):
    adapter = Qwen3TTSAdapter(SpeechServingContext(server=mocker.Mock()))
    params = {
        "task_type": ["Base"],
        QWEN3_TTS_EFFECTIVE_MAX_TOKENS_KEY: [192],
    }

    # EOS at the exact budget is valid; the token count alone is not a
    # sufficient failure signal.
    adapter.validate_generation(params, stage0_finish_reason="stop", output_tokens=192)
    adapter.validate_generation(params, stage0_finish_reason=None, output_tokens=192)

    # Some frontends expose 191 decoded frames for max_new_tokens=192. The
    # engine terminal reason still makes this an unambiguous limit failure.
    with pytest.raises(Qwen3TTSCodecLimitError, match="191/192"):
        adapter.validate_generation(params, stage0_finish_reason="length", output_tokens=191)


def test_indextts_adapters_are_versioned():
    assert resolve_adapter("indextts2") is IndexTTS2Adapter
    assert resolve_adapter("indextts2_5") is IndexTTS25Adapter
    assert IndexTTS25Adapter.stage_keys == frozenset({"indextts2_5_talker"})
    assert detect_tts_model_type("indextts2_5_talker", None) == "indextts2_5"


def test_indextts25_validates_explicit_language():
    adapter = IndexTTS25Adapter(type("Context", (), {"server": object()})())

    assert adapter._validate_extra_params({"lang": "ja"}) is None
    assert "Unsupported IndexTTS 2.5 language" in adapter._validate_extra_params({"lang": "xx-invalid"})


def _indextts25_adapter_and_request(*, speed: float):
    server = SimpleNamespace(
        uploaded_speakers={},
        _validate_ref_audio_format=lambda ref_audio: None,
    )
    adapter = IndexTTS25Adapter(SimpleNamespace(server=server))
    request = SimpleNamespace(
        input="hello",
        voice="alloy",
        ref_audio=object(),
        max_new_tokens=None,
        extra_params=None,
        speed=speed,
    )
    return adapter, request


def test_indextts25_uses_native_speed_control_duration_factor():
    adapter, fast_request = _indextts25_adapter_and_request(speed=2.0)
    _, slow_request = _indextts25_adapter_and_request(speed=0.5)

    assert adapter.native_speed_control is True
    assert adapter.validate(fast_request) is None
    assert adapter.validate(slow_request) is None
    assert asyncio.run(adapter._build_params(fast_request))["duration_factor"] == [0.5]
    assert asyncio.run(adapter._build_params(slow_request))["duration_factor"] == [2.0]


@pytest.mark.parametrize("speed", [0.49, 2.01])
def test_indextts25_rejects_out_of_range_native_speed(speed):
    adapter, request = _indextts25_adapter_and_request(speed=speed)

    assert adapter.validate(request) == "IndexTTS 2.5 speed must be between 0.5 and 2.0"


def test_indextts25_speed_does_not_change_conditioning_cache_salt():
    adapter, fast_request = _indextts25_adapter_and_request(speed=2.0)
    _, slow_request = _indextts25_adapter_and_request(speed=0.5)
    slow_request.ref_audio = fast_request.ref_audio

    fast_params = asyncio.run(adapter._build_params(fast_request))
    slow_params = asyncio.run(adapter._build_params(slow_request))

    assert indextts2_conditioning_cache_salt(
        fast_request,
        fast_params,
    ) == indextts2_conditioning_cache_salt(slow_request, slow_params)


def test_indextts2_conditioning_cache_salt_changes_with_ref_audio_cache_key():
    request = SimpleNamespace(input="hello", ref_audio="file:///data/spk.wav")
    salt_a = indextts2_conditioning_cache_salt(request, {"ref_audio_cache_key": ["key_aaa"]})
    salt_b = indextts2_conditioning_cache_salt(request, {"ref_audio_cache_key": ["key_bbb"]})
    assert salt_a != salt_b


@pytest.mark.parametrize(
    ("hf_config", "expected_tokenizer_file"),
    [
        (SimpleNamespace(tokenizer_file="custom-tokenizer.tiktoken"), "custom-tokenizer.tiktoken"),
        (SimpleNamespace(), INDEXTTS25_TOKENIZER_FILE),
    ],
)
def test_indextts25_build_uses_configured_tokenizer_file(
    monkeypatch,
    hf_config,
    expected_tokenizer_file,
):
    captured = {}

    def fake_estimate(*args, **kwargs):
        captured.update(kwargs)
        return 4

    async def fake_build_params(request):
        return {"lang": ["en"], "text_normalization": [True]}

    monkeypatch.setattr(
        prompt_utils,
        "estimate_indextts2_prefill_prompt_len",
        fake_estimate,
    )
    server = SimpleNamespace(
        engine_client=SimpleNamespace(
            model_config=SimpleNamespace(
                model="/model",
                hf_config=hf_config,
            )
        )
    )
    adapter = IndexTTS25Adapter(SimpleNamespace(server=server))
    monkeypatch.setattr(adapter, "_build_params", fake_build_params)
    request = SimpleNamespace(input="hello", ref_audio=None)

    prepared = asyncio.run(adapter.build(request, [], False))

    assert prepared.prompt["prompt_token_ids"] == [1] * 4
    assert captured["tokenizer_file"] == expected_tokenizer_file


def test_diffusion_adapter_extra_body_params_fallback():
    class _DiffAdapter(DiffusionTTSAdapter):
        name = "diff_probe"

        async def build(self, request, sampling_params_list):  # pragma: no cover
            raise NotImplementedError

    assert _DiffAdapter.extra_body_params() == frozenset()


def _higgs_v2_adapter() -> HiggsAudioV2Adapter:
    server = SimpleNamespace(
        _apply_uploaded_speaker=lambda request: None,
        uploaded_speakers={},
    )
    return HiggsAudioV2Adapter(SimpleNamespace(server=server))


def _higgs_v2_request(**overrides: Any) -> SimpleNamespace:
    fields: dict[str, Any] = {
        "input": "Hello world.",
        "ref_audio": None,
        "ref_text": None,
        "voice": None,
        "x_vector_only_mode": None,
        "speaker_embedding": None,
        "instructions": None,
        "task_type": None,
        "language": None,
        "speed": None,
        "max_new_tokens": None,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize(
    "overrides, err_substr",
    [
        pytest.param({"ref_audio": "data:audio/wav;base64,AA=="}, "ref_text", id="ref_audio_without_ref_text"),
        pytest.param({"ref_text": "some transcript"}, "ref_audio", id="ref_text_without_ref_audio"),
        pytest.param({"task_type": "Base"}, "task_type", id="task_type"),
        pytest.param({"language": "Chinese"}, "language", id="language_override"),
        pytest.param({"input": "[SPEAKER0] hi"}, "multi-speaker", id="multi_speaker_tag"),
        pytest.param({"input": "   "}, "empty", id="input_whitespace_only"),
    ],
)
def test_higgs_audio_v2_validate_rejects_out_of_scope_fields(overrides: dict[str, object], err_substr: str) -> None:
    """Adapter-only policy checks formerly covered by invalid_param e2e on a live V2 server."""
    adapter = _higgs_v2_adapter()
    err = adapter.validate(_higgs_v2_request(**overrides))
    assert err is not None
    assert err_substr.lower() in err.lower()


def test_higgs_audio_v2_validate_accepts_plain_text_and_paired_clone() -> None:
    adapter = _higgs_v2_adapter()
    assert adapter.validate(_higgs_v2_request()) is None
    assert (
        adapter.validate(_higgs_v2_request(ref_audio="data:audio/wav;base64,AA==", ref_text="some transcript")) is None
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
