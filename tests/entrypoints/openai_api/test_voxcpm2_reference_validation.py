# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""VoxCPM2 rejects unsupported references before audio resolution."""

import asyncio

import pytest

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext, validate_tts_request
from vllm_omni.entrypoints.openai.tts_adapters.voxcpm2 import VoxCPM2Adapter

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def adapter(mocker):
    server = object.__new__(OmniOpenAIServingSpeech)
    server._diffusion_mode = False
    server._tts_model_type = "voxcpm2"
    server.model_config = mocker.Mock(allowed_local_media_path="", async_chunk=True)
    server.engine_client = mocker.Mock(errored=False, default_sampling_params_list=[])
    server._get_available_speakers = mocker.Mock(return_value={"default"})
    server._validate_speech_sample_rate = mocker.Mock(return_value=None)
    server._check_model = mocker.AsyncMock(return_value=None)
    server._refresh_uploaded_speakers = mocker.Mock()
    result = VoxCPM2Adapter(SpeechServingContext(server=server, engine_client=server.engine_client))
    mocker.patch.object(result, "_encode", return_value=[1, 2, 3])
    server._adapter = result
    return result


@pytest.mark.parametrize(
    "reference",
    [["data:audio/wav;base64,AAAA"], "ftp://example.invalid/reference.wav"],
)
def test_voxcpm2_rejects_unsupported_reference(adapter, reference):
    request = OpenAICreateSpeechRequest(input="Hello", ref_audio=reference)

    error = validate_tts_request(adapter, request)

    assert error is not None
    assert "ref_audio must be" in error


@pytest.mark.parametrize(
    "reference",
    [
        None,
        "http://example.invalid/reference.wav",
        "https://example.invalid/reference.wav",
        "data:audio/wav;base64,AAAA",
        "file:///reference.wav",
    ],
)
def test_voxcpm2_accepts_supported_reference_format(adapter, reference):
    request = OpenAICreateSpeechRequest(input="Hello", ref_audio=reference)
    assert validate_tts_request(adapter, request) is None


def test_voxcpm2_reference_list_returns_client_error_before_resolution(adapter, mocker):
    server = adapter.ctx.server
    resolve = mocker.spy(server, "_resolve_ref_audio_array")
    request = OpenAICreateSpeechRequest(input="Hello", ref_audio=["data:audio/wav;base64,AAAA"])

    response = asyncio.run(server.create_speech(request))

    error = response.model_dump()["error"]
    assert error["code"] == 400
    assert "ref_audio must be" in error["message"]
    resolve.assert_not_awaited()
    server.engine_client.generate.assert_not_called()
