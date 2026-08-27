# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from vllm import SamplingParams

from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.tts_adapters import (
    detect_tts_model_type,
    resolve_adapter,
)
from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
from vllm_omni.entrypoints.openai.tts_adapters.neutts_air import (
    NEUTTS_SPEECH_GENERATION_END_TOKEN_ID,
    NeuTTSAirAdapter,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeSpeechServer:
    def __init__(self) -> None:
        self.resolved_ref_audio: str | None = None

    def _validate_ref_audio_format(self, ref_audio: str) -> str | None:
        if ref_audio == "bad-audio":
            return "Invalid reference audio"
        return None

    async def _resolve_ref_audio(self, ref_audio: str) -> tuple[list[float], int]:
        self.resolved_ref_audio = ref_audio
        return [0.0, 0.25, -0.25], 16_000


def _request(**changes):
    values = {
        "input": "Hello, this is a test.",
        "ref_audio": "file:///tmp/reference.wav",
        "ref_text": "My name is Jo.",
        "max_new_tokens": None,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _adapter() -> tuple[NeuTTSAirAdapter, FakeSpeechServer]:
    server = FakeSpeechServer()
    adapter = NeuTTSAirAdapter(SpeechServingContext(server=server))
    return adapter, server


def test_neutts_air_adapter_is_registered_and_detectable():
    assert resolve_adapter("neutts_air") is NeuTTSAirAdapter
    assert detect_tts_model_type("neucodec", "NeuTTSAirCode2Wav") == "neutts_air"

    serving = object.__new__(OmniOpenAIServingSpeech)
    serving._tts_stage = SimpleNamespace(
        engine_args=SimpleNamespace(
            model_stage="neucodec",
            model_arch="NeuTTSAirCode2Wav",
        )
    )
    assert serving._detect_tts_model_type() == "neutts_air"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"input": "  "}, "Input text cannot be empty"),
        ({"ref_audio": None}, "requires 'ref_audio'"),
        ({"ref_audio": ["one.wav", "two.wav"]}, "exactly one 'ref_audio'"),
        ({"ref_audio": "bad-audio"}, "Invalid reference audio"),
        ({"ref_text": "  "}, "requires 'ref_text'"),
        ({"max_new_tokens": 49}, "must be at least 50"),
        ({"max_new_tokens": 4097}, "cannot exceed 4096"),
    ],
)
def test_neutts_air_adapter_validation(changes, message):
    adapter, _ = _adapter()
    assert message in (adapter.validate(_request(**changes)) or "")


def test_neutts_air_adapter_builds_the_offline_prompt_contract():
    adapter, server = _adapter()
    request = _request()

    prepared = asyncio.run(adapter.build(request, [], True))

    assert prepared.model_type == "neutts_air"
    assert prepared.prompt["prompt"] == request.input
    assert prepared.prompt["mm_processor_kwargs"] == {
        "ref_text": request.ref_text,
    }

    audio, sample_rate = prepared.prompt["multi_modal_data"]["audio"]
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    np.testing.assert_array_equal(
        audio,
        np.asarray([0.0, 0.25, -0.25], dtype=np.float32),
    )
    assert sample_rate == 16_000
    assert server.resolved_ref_audio == request.ref_audio


def test_neutts_air_adapter_applies_required_sampling_contract():
    adapter, _ = _adapter()
    original = [
        SamplingParams(
            temperature=0.7,
            top_p=0.6,
            top_k=20,
            max_tokens=2048,
        ),
        SamplingParams(max_tokens=2048),
    ]

    params = adapter.apply_sampling_overrides(
        original,
        _request(max_new_tokens=256),
    )

    assert params is not original
    assert original[0].max_tokens == 2048
    assert params[0].temperature == 0.7
    assert params[0].top_p == 0.6
    assert params[0].top_k == 20
    assert params[0].max_tokens == 256
    assert params[0].min_tokens == 50
    assert params[0].stop_token_ids == [NEUTTS_SPEECH_GENERATION_END_TOKEN_ID]
    assert params[0].ignore_eos
    assert not params[0].detokenize
    assert params[1].max_tokens == 1
    assert not params[1].detokenize
