# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the TTS serving adapter registry (RFC #4327).

Pure-Python registry/resolution logic; no model or GPU resources are loaded.
"""

import asyncio
from types import SimpleNamespace

import pytest

from vllm_omni.entrypoints.openai.tts_adapters import (
    TTS_ADAPTER_REGISTRY,
    ARTTSAdapter,
    DiffusionTTSAdapter,
    all_tts_model_types,
    detect_tts_model_type,
    resolve_adapter,
)
from vllm_omni.entrypoints.openai.tts_adapters.indextts2 import (
    IndexTTS2Adapter,
    IndexTTS25Adapter,
    indextts2_conditioning_cache_salt,
)
from vllm_omni.entrypoints.openai.tts_adapters.moss_tts import (
    MossTTSAdapter,
    MossTTSNanoAdapter,
)
from vllm_omni.entrypoints.openai.tts_adapters.qwen3_tts import Qwen3TTSAdapter
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


def test_qwen3_tts_metadata():
    assert Qwen3TTSAdapter.backend == "ar"
    assert issubclass(Qwen3TTSAdapter, ARTTSAdapter)


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
