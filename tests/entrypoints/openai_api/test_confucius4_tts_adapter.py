# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU coverage for the Confucius4-TTS OpenAI speech adapter."""

from dataclasses import dataclass
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from vllm import SamplingParams

from vllm_omni.entrypoints.openai.protocol.audio import (
    OpenAICreateSpeechRequest,
)
from vllm_omni.entrypoints.openai.tts_adapters import (
    detect_tts_model_type,
    resolve_adapter,
)
from vllm_omni.entrypoints.openai.tts_adapters.base import (
    SpeechServingContext,
)
from vllm_omni.entrypoints.openai.tts_adapters.confucius4_tts import (
    Confucius4TTSAdapter,
    confucius4_conditioning_cache_salt,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

REFERENCE = "data:audio/wav;base64,cHVibGljLXRlc3QtZGF0YQ=="


@dataclass
class _ModelConfigStub:
    model: str


@dataclass
class _EngineClientStub:
    model_config: _ModelConfigStub


@dataclass
class _ServerStub:
    engine_client: _EngineClientStub
    uploaded_speakers: dict
    _get_uploaded_audio_data: Mock
    _validate_ref_audio_format: Mock
    _resolve_ref_audio: AsyncMock
    _voice_created_at: Mock


@pytest.fixture
def adapter() -> Confucius4TTSAdapter:
    server = _ServerStub(
        engine_client=_EngineClientStub(
            model_config=_ModelConfigStub(
                model="netease-youdao/Confucius4-TTS",
            )
        ),
        uploaded_speakers={},
        _get_uploaded_audio_data=Mock(return_value=None),
        _validate_ref_audio_format=Mock(return_value=None),
        _resolve_ref_audio=AsyncMock(return_value=([0.1, -0.1], 22050, "audio-key")),
        _voice_created_at=Mock(return_value=0),
    )
    return Confucius4TTSAdapter(SpeechServingContext(server=server))


def test_registration_and_detection() -> None:
    assert resolve_adapter("confucius4_tts") is Confucius4TTSAdapter
    assert detect_tts_model_type("confucius4_tts_talker", None) == "confucius4_tts"
    assert (
        detect_tts_model_type(
            None,
            "Confucius4TTSTalkerForConditionalGeneration",
        )
        == "confucius4_tts"
    )
    assert (
        detect_tts_model_type(
            "audex_omni",
            "Confucius4TTSTalkerForConditionalGeneration",
        )
        == "audex"
    )


@pytest.mark.parametrize("text", ["", " ", "\n\t"])
def test_empty_input_is_rejected(
    adapter: Confucius4TTSAdapter,
    text: str,
) -> None:
    error = adapter.validate(OpenAICreateSpeechRequest(input=text, ref_audio=REFERENCE))
    assert error is not None and "empty" in error.lower()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stream": True},
        {"stream_format": "audio"},
        {"stream_format": "sse"},
    ],
)
def test_streaming_is_rejected(
    adapter: Confucius4TTSAdapter,
    kwargs: dict,
) -> None:
    error = adapter.validate(
        OpenAICreateSpeechRequest(
            input="Hello.",
            ref_audio=REFERENCE,
            **kwargs,
        )
    )
    assert error is not None and "does not support streaming" in error


def test_reference_audio_is_required(
    adapter: Confucius4TTSAdapter,
) -> None:
    error = adapter.validate(OpenAICreateSpeechRequest(input="Hello."))
    assert error is not None and "requires 'ref_audio'" in error


def test_reference_validation_error_is_preserved(
    adapter: Confucius4TTSAdapter,
) -> None:
    adapter.ctx.server._validate_ref_audio_format.return_value = "Unsupported reference format"
    request = OpenAICreateSpeechRequest(
        input="Hello.",
        ref_audio=REFERENCE,
    )
    assert adapter.validate(request) == "Unsupported reference format"


def test_multiple_reference_audios_are_rejected(
    adapter: Confucius4TTSAdapter,
) -> None:
    error = adapter.validate(
        OpenAICreateSpeechRequest(
            input="Hello.",
            ref_audio=[REFERENCE, REFERENCE],
        )
    )
    assert error is not None and "exactly one" in error


def test_invalid_language_is_rejected(
    adapter: Confucius4TTSAdapter,
) -> None:
    error = adapter.validate(
        OpenAICreateSpeechRequest(
            input="Hello.",
            ref_audio=REFERENCE,
            language="invalid",
        )
    )
    assert error is not None and "language/extra_params.lang" in error


def test_language_outside_official_fourteen_is_rejected(
    adapter: Confucius4TTSAdapter,
) -> None:
    error = adapter.validate(
        OpenAICreateSpeechRequest(
            input="Hello.",
            ref_audio=REFERENCE,
            language="tl",
        )
    )
    assert error is not None and "language/extra_params.lang" in error


@pytest.mark.asyncio
async def test_build_uses_token_count_plus_two_placeholder(
    adapter: Confucius4TTSAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token_ids = [11, 12, 13]
    monkeypatch.setattr(
        "vllm_omni.entrypoints.openai.tts_adapters.confucius4_tts.tokenize_confucius4_prompt",
        lambda *_args: token_ids,
    )
    request = OpenAICreateSpeechRequest(
        input="Hello.",
        ref_audio=REFERENCE,
        extra_params={"lang": "en"},
    )

    prepared = await adapter.build(request, [], True)

    assert prepared.prompt["prompt_token_ids"] == [0] * 5
    assert prepared.prompt["additional_information"] == prepared.tts_params
    assert prepared.tts_params["text_token_ids"] == [token_ids]
    assert prepared.tts_params["lang"] == ["en"]
    assert prepared.tts_params["voice"] == [[[0.1, -0.1], 22050]]
    assert prepared.tts_params["ref_audio_cache_key"] == ["audio-key"]
    adapter.ctx.server._resolve_ref_audio.assert_awaited_once_with(REFERENCE)


def test_conditioning_cache_salt_tracks_reference_and_text() -> None:
    baseline = confucius4_conditioning_cache_salt(
        text="Hello.",
        lang="en",
        ref_audio_cache_key="key-a",
    )
    assert baseline != confucius4_conditioning_cache_salt(
        text="Different.",
        lang="en",
        ref_audio_cache_key="key-a",
    )
    assert baseline != confucius4_conditioning_cache_salt(
        text="Hello.",
        lang="en",
        ref_audio_cache_key="key-b",
    )


@pytest.mark.asyncio
async def test_uploaded_voice_metadata_reaches_conditioning(
    adapter: Confucius4TTSAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter.ctx.server.uploaded_speakers = {"alice": {"created_at": 123}}
    adapter.ctx.server._get_uploaded_audio_data.return_value = REFERENCE
    adapter.ctx.server._voice_created_at.return_value = 123
    monkeypatch.setattr(
        "vllm_omni.entrypoints.openai.tts_adapters.confucius4_tts.tokenize_confucius4_prompt",
        lambda *_args: [11],
    )
    request = OpenAICreateSpeechRequest(input="Hello.", voice="Alice")

    assert adapter.validate(request) is None
    prepared = await adapter.build(request, [], False)

    assert prepared.tts_params["voice_name"] == ["alice"]
    assert prepared.tts_params["voice_created_at"] == [123]
    adapter.ctx.server._resolve_ref_audio.assert_awaited_once_with(REFERENCE)


def test_request_max_new_tokens_is_validated_and_applied(
    adapter: Confucius4TTSAdapter,
) -> None:
    too_large = OpenAICreateSpeechRequest(
        input="Hello.",
        ref_audio=REFERENCE,
        max_new_tokens=1519,
    )
    assert adapter.validate(too_large) == ("max_new_tokens cannot exceed 1518")

    defaults = [SamplingParams(max_tokens=1518)]
    request = OpenAICreateSpeechRequest(
        input="Hello.",
        ref_audio=REFERENCE,
        max_new_tokens=64,
    )
    overridden = adapter.apply_sampling_overrides(defaults, request)
    assert overridden[0].max_tokens == 64
    assert defaults[0].max_tokens == 1518


@pytest.mark.parametrize(
    ("audio_output", "rejected"),
    [
        ({"model_outputs": [torch.zeros(0)]}, True),
        ({"audio": torch.zeros(0)}, True),
        ({"model_outputs": [torch.zeros(0), torch.zeros(0)]}, True),
        ({"model_outputs": [torch.zeros(8)]}, False),
        ({"audio": torch.zeros(8)}, False),
        ({"sr": 22050}, False),
    ],
)
def test_empty_audio_is_rejected_instead_of_served(
    adapter: Confucius4TTSAdapter,
    audio_output: dict,
    rejected: bool,
) -> None:
    # An immediate EOS yields an empty waveform for that request. It must become
    # a failed request, not a successful zero-length audio file.
    collect: dict = {}

    adapter.collect_response_metadata(audio_output, collect)

    assert bool(collect.get("audio_synthesis_error")) is rejected
