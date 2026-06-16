# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.tts_adapters import SpeechServingContext
from vllm_omni.entrypoints.openai.tts_adapters.indextts2 import (
    IndexTTS2Adapter,
    indextts2_conditioning_cache_salt,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]


class _AsyncResolver:
    def __init__(self):
        self.sources: list[str] = []

    async def __call__(self, source: str):
        self.sources.append(source)
        if source == "data:audio/wav;base64,REF_A":
            return [0.1, 0.2, 0.3], 16000
        if source == "data:audio/wav;base64,REF_B":
            return [0.4, 0.5, 0.6], 16000
        if source == "data:audio/wav;base64,EMO":
            return [0.7, 0.8], 16000
        if source == "data:audio/wav;base64,UPLOADED":
            return [0.9, 1.0], 16000
        return [0.0], 16000


def _serving() -> OmniOpenAIServingSpeech:
    serving = object.__new__(OmniOpenAIServingSpeech)
    serving.uploaded_speakers = {}
    return serving


def _adapter(serving: OmniOpenAIServingSpeech) -> IndexTTS2Adapter:
    return IndexTTS2Adapter(SpeechServingContext(server=serving))


@pytest.mark.asyncio
async def test_indextts2_inline_ref_audio_ignores_voice_name_cache_key() -> None:
    serving = _serving()
    serving.uploaded_speakers = {"default": {"created_at": 123}}
    resolver = _AsyncResolver()
    serving._resolve_ref_audio = resolver  # type: ignore[method-assign]
    serving._get_uploaded_audio_data = lambda voice: "data:audio/wav;base64,UPLOADED"  # type: ignore[method-assign]

    request = OpenAICreateSpeechRequest(
        input="hello",
        voice="default",
        ref_audio="data:audio/wav;base64,REF_A",
    )

    params = await _adapter(serving)._build_params(request)

    assert params["voice"] == [[[0.1, 0.2, 0.3], 16000]]
    assert "voice_name" not in params
    assert "voice_created_at" not in params
    assert resolver.sources == ["data:audio/wav;base64,REF_A"]


@pytest.mark.asyncio
async def test_indextts2_uploaded_voice_sets_generation_cache_key() -> None:
    serving = _serving()
    serving.uploaded_speakers = {"alice": {"created_at": 456}}
    resolver = _AsyncResolver()
    serving._resolve_ref_audio = resolver  # type: ignore[method-assign]
    serving._get_uploaded_audio_data = lambda voice: "data:audio/wav;base64,UPLOADED"  # type: ignore[method-assign]

    request = OpenAICreateSpeechRequest(input="hello", voice="alice")

    assert _adapter(serving).validate(request) is None
    params = await _adapter(serving)._build_params(request)

    assert params["voice"] == [[[0.9, 1.0], 16000]]
    assert params["voice_name"] == ["alice"]
    assert params["voice_created_at"] == [456]
    assert resolver.sources == ["data:audio/wav;base64,UPLOADED"]


@pytest.mark.asyncio
async def test_indextts2_emotion_audio_is_resolved_separately() -> None:
    serving = _serving()
    resolver = _AsyncResolver()
    serving._resolve_ref_audio = resolver  # type: ignore[method-assign]

    request = OpenAICreateSpeechRequest(
        input="hello",
        ref_audio="data:audio/wav;base64,REF_A",
        extra_params={"emo_audio": "data:audio/wav;base64,EMO"},
    )

    assert _adapter(serving).validate(request) is None
    params = await _adapter(serving)._build_params(request)

    assert params["voice"] == [[[0.1, 0.2, 0.3], 16000]]
    assert params["emo_audio"] == [[[0.7, 0.8], 16000]]
    assert resolver.sources == ["data:audio/wav;base64,REF_A", "data:audio/wav;base64,EMO"]


def test_indextts2_validates_emotion_extra_params() -> None:
    serving = _serving()
    serving._get_uploaded_audio_data = lambda voice: None  # type: ignore[method-assign]

    request = OpenAICreateSpeechRequest(
        input="hello",
        ref_audio="data:audio/wav;base64,REF_A",
        extra_params={"use_emo_text": "false"},
    )
    assert _adapter(serving).validate(request) == "extra_params.use_emo_text must be a boolean"

    request = OpenAICreateSpeechRequest(
        input="hello",
        ref_audio="data:audio/wav;base64,REF_A",
        extra_params={"emo_vector": [0.0] * 7},
    )
    assert _adapter(serving).validate(request) == "extra_params.emo_vector must be a list of 8 numbers"

    request = OpenAICreateSpeechRequest(
        input="hello",
        ref_audio="data:audio/wav;base64,REF_A",
        extra_params={"emo_audio": 1},
    )
    assert "extra_params.emo_audio" in _adapter(serving).validate(request)


def test_indextts2_conditioning_salt_tracks_text_speaker_and_emotion() -> None:
    request_a = OpenAICreateSpeechRequest(input="hello", ref_audio="data:audio/wav;base64,REF_A")
    request_b = OpenAICreateSpeechRequest(input="hallo", ref_audio="data:audio/wav;base64,REF_A")
    params_a = {"text": ["hello"], "voice": [[[0.1, 0.2], 16000]], "emo_vector": [[1.0] + [0.0] * 7]}
    params_b = {"text": ["hallo"], "voice": [[[0.1, 0.2], 16000]], "emo_vector": [[1.0] + [0.0] * 7]}
    params_c = {"text": ["hello"], "voice": [[[0.3, 0.4], 16000]], "emo_vector": [[1.0] + [0.0] * 7]}
    params_d = {"text": ["hello"], "voice": [[[0.1, 0.2], 16000]], "emo_vector": [[0.0, 1.0] + [0.0] * 6]}

    salt_a = indextts2_conditioning_cache_salt(request_a, params_a, request_id="req-a")

    assert indextts2_conditioning_cache_salt(request_b, params_b, request_id="req-b") != salt_a
    assert indextts2_conditioning_cache_salt(request_a, params_c, request_id="req-c") != salt_a
    assert indextts2_conditioning_cache_salt(request_a, params_d, request_id="req-d") != salt_a


def test_indextts2_inline_ref_audio_voice_field_does_not_change_salt() -> None:
    params = {"text": ["hello"], "voice": [[[0.1, 0.2], 16000]]}
    request_a = OpenAICreateSpeechRequest(
        input="hello",
        voice="default",
        ref_audio="data:audio/wav;base64,REF_A",
    )
    request_b = OpenAICreateSpeechRequest(
        input="hello",
        voice="ignored",
        ref_audio="data:audio/wav;base64,REF_A",
    )

    assert indextts2_conditioning_cache_salt(
        request_a, params, request_id="req-a"
    ) == indextts2_conditioning_cache_salt(request_b, params, request_id="req-b")


def test_indextts2_uploaded_voice_created_at_changes_salt() -> None:
    request = OpenAICreateSpeechRequest(input="hello", voice="alice")
    params_a = {
        "text": ["hello"],
        "voice": [[[0.1, 0.2], 16000]],
        "voice_name": ["alice"],
        "voice_created_at": [1],
    }
    params_b = {
        "text": ["hello"],
        "voice": [[[0.1, 0.2], 16000]],
        "voice_name": ["alice"],
        "voice_created_at": [2],
    }

    assert indextts2_conditioning_cache_salt(
        request, params_a, request_id="req-a"
    ) != indextts2_conditioning_cache_salt(request, params_b, request_id="req-b")


def test_indextts2_unknown_extra_params_do_not_change_salt() -> None:
    params = {"text": ["hello"], "voice": [[[0.1, 0.2], 16000]]}
    request_a = OpenAICreateSpeechRequest(
        input="hello",
        ref_audio="data:audio/wav;base64,REF_A",
        extra_params={"ignored": "a"},
    )
    request_b = OpenAICreateSpeechRequest(
        input="hello",
        ref_audio="data:audio/wav;base64,REF_A",
        extra_params={"ignored": "b"},
    )

    assert indextts2_conditioning_cache_salt(
        request_a, params, request_id="req-a"
    ) == indextts2_conditioning_cache_salt(request_b, params, request_id="req-b")


def test_indextts2_random_emotion_salt_is_per_request() -> None:
    request = OpenAICreateSpeechRequest(
        input="hello",
        ref_audio="data:audio/wav;base64,REF_A",
        extra_params={"use_random": True},
    )
    params = {"text": ["hello"], "voice": [[[0.1, 0.2], 16000]], "use_random": [True]}

    assert indextts2_conditioning_cache_salt(request, params, request_id="req-a") != indextts2_conditioning_cache_salt(
        request, params, request_id="req-b"
    )
