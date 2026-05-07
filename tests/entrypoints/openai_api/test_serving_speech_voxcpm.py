# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""UTs for VoxCPM OpenAI speech serving behavior."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pytest_mock import MockerFixture

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def voxcpm_server(mocker: MockerFixture):
    mocker.patch.object(OmniOpenAIServingSpeech, "_load_supported_speakers", return_value=set())
    mocker.patch.object(OmniOpenAIServingSpeech, "_load_codec_frame_rate", return_value=None)

    mock_engine_client = mocker.MagicMock()
    mock_engine_client.errored = False
    mock_engine_client.model_config = mocker.MagicMock(model="OpenBMB/VoxCPM1.5")
    mock_engine_client.default_sampling_params_list = [SimpleNamespace(max_tokens=2048)]
    mock_engine_client.tts_batch_max_items = 32
    mock_engine_client.generate = mocker.MagicMock(return_value="generator")
    mock_engine_client.stage_configs = [
        SimpleNamespace(
            engine_args=SimpleNamespace(
                model_stage="latent_generator",
                model_arch="VoxCPMForConditionalGeneration",
            ),
            tts_args={},
        ),
        SimpleNamespace(
            engine_args=SimpleNamespace(model_stage="vae"),
            tts_args={},
        ),
    ]

    mock_models = mocker.MagicMock()
    mock_models.is_base_model.return_value = True

    return OmniOpenAIServingSpeech(
        engine_client=mock_engine_client,
        models=mock_models,
        request_logger=mocker.MagicMock(),
    )


class TestVoxCPMServing:
    def test_voxcpm_model_type_detection(self, voxcpm_server):
        assert voxcpm_server._tts_model_type == "voxcpm"
        assert voxcpm_server._is_tts is True
        assert voxcpm_server.supported_speakers == set()

    @pytest.mark.parametrize(
        ("request_kwargs", "expected_substring"),
        [
            ({"voice": "alice"}, "voice"),
            ({"instructions": "whisper"}, "instructions"),
            ({"language": "en"}, "language"),
            ({"task_type": "CustomVoice"}, "plain tts"),
            ({"x_vector_only_mode": True}, "x_vector_only_mode"),
            ({"speaker_embedding": [0.1, 0.2]}, "speaker_embedding"),
            ({"initial_codec_chunk_frames": 4}, "initial_codec_chunk_frames"),
            ({"ref_text": "reference"}, "ref_audio"),
        ],
    )
    def test_validate_voxcpm_rejects_unsupported_fields(self, voxcpm_server, request_kwargs, expected_substring):
        request = OpenAICreateSpeechRequest(input="hello voxcpm", **request_kwargs)
        error = voxcpm_server._validate_voxcpm_request(request)
        assert error is not None
        assert expected_substring in error.lower()

    def test_validate_voxcpm_accepts_plain_tts_request(self, voxcpm_server):
        request = OpenAICreateSpeechRequest(input="hello voxcpm", max_new_tokens=256)
        assert voxcpm_server._validate_voxcpm_request(request) is None

    def test_validate_voxcpm_accepts_voice_clone_request(self, voxcpm_server):
        request = OpenAICreateSpeechRequest(
            input="clone this voice",
            ref_audio="data:audio/wav;base64,QUJD",
            ref_text="reference transcript",
            max_new_tokens=256,
        )
        assert voxcpm_server._validate_voxcpm_request(request) is None

    def test_prepare_speech_generation_voxcpm_text_only(self, voxcpm_server):
        request = OpenAICreateSpeechRequest(input="hello voxcpm", max_new_tokens=321)

        request_id, generator, tts_params = asyncio.run(voxcpm_server._prepare_speech_generation(request))

        assert request_id.startswith("speech-")
        assert generator == "generator"
        assert tts_params == {
            "text": ["hello voxcpm"],
            "cfg_value": [2.0],
            "inference_timesteps": [10],
            "min_len": [2],
            "max_new_tokens": [321],
        }

        voxcpm_server.engine_client.generate.assert_called_once()
        call = voxcpm_server.engine_client.generate.call_args
        assert call.kwargs["prompt"] == {
            "prompt_token_ids": [1],
            "additional_information": tts_params,
            "type": "token",
        }
        assert call.kwargs["output_modalities"] == ["audio"]

    def test_prepare_speech_generation_voxcpm_voice_clone_resolves_ref_audio(self, voxcpm_server):
        voxcpm_server._resolve_ref_audio = AsyncMock(return_value=([0.1, -0.1, 0.2], 16000))
        request = OpenAICreateSpeechRequest(
            input="clone this voice",
            ref_audio="data:audio/wav;base64,QUJD",
            ref_text="reference transcript",
            max_new_tokens=512,
        )

        request_id, generator, tts_params = asyncio.run(voxcpm_server._prepare_speech_generation(request))

        assert request_id.startswith("speech-")
        assert generator == "generator"
        assert tts_params == {
            "text": ["clone this voice"],
            "cfg_value": [2.0],
            "inference_timesteps": [10],
            "min_len": [2],
            "max_new_tokens": [512],
            "ref_text": ["reference transcript"],
            "ref_audio": [[[0.1, -0.1, 0.2], 16000]],
        }

        voxcpm_server._resolve_ref_audio.assert_awaited_once_with("data:audio/wav;base64,QUJD")
        call = voxcpm_server.engine_client.generate.call_args
        assert call.kwargs["prompt"] == {
            "prompt_token_ids": [1],
            "additional_information": tts_params,
            "type": "token",
        }
