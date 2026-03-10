# tests/entrypoints/openai/test_serving_speech.py
import logging
import os
from inspect import Signature, signature
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.params import File, Form
from fastapi.testclient import TestClient
from pydantic import ValidationError
from pytest_mock import MockerFixture

from vllm_omni.entrypoints.openai.audio_utils_mixin import AudioMixin
from vllm_omni.entrypoints.openai.protocol.audio import CreateAudio, OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import (
    OmniOpenAIServingSpeech,
)
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

logger = logging.getLogger(__name__)


class TestAudioMixin:
    @pytest.fixture
    def audio_mixin(self):
        return AudioMixin()

    def test_stereo_to_mono_conversion(self, audio_mixin, mocker: MockerFixture):
        stereo_tensor = np.random.rand(24000, 2).astype(np.float32)
        audio_obj = CreateAudio(audio_tensor=stereo_tensor)

        mock_speed = mocker.patch.object(
            audio_mixin, "_apply_speed_adjustment", side_effect=lambda tensor, speed, sr: (tensor, sr)
        )
        mocker.patch("soundfile.write")

        audio_mixin.create_audio(audio_obj)

        # Check that the tensor passed to speed adjustment is mono
        mock_speed.assert_called_once()
        adjusted_tensor = mock_speed.call_args[0][0]
        assert len(adjusted_tensor) == 24000

    def test_speed_adjustment(self, audio_mixin, mocker: MockerFixture):
        mock_time_stretch = mocker.patch("librosa.effects.time_stretch")
        mock_time_stretch.return_value = np.zeros(12000)
        audio_tensor = np.random.rand(24000).astype(np.float32)

        adjusted_audio, _ = audio_mixin._apply_speed_adjustment(audio_tensor, speed=2.0, sample_rate=24000)

        mock_time_stretch.assert_called_with(y=audio_tensor, rate=2.0)
        assert adjusted_audio.shape == (12000,)

    def test_unsupported_format_fallback(self, audio_mixin, caplog, mocker: MockerFixture):
        mock_write = mocker.patch("soundfile.write")
        audio_tensor = np.random.rand(24000).astype(np.float32)
        # Use a format that is not in the list of supported formats
        audio_obj = CreateAudio(audio_tensor=audio_tensor, response_format="vorbis")

        audio_mixin.create_audio(audio_obj)

        # Should fall back to 'wav'
        mock_write.assert_called_once()
        write_kwargs = mock_write.call_args.kwargs
        assert write_kwargs["format"] == "WAV"

    def test_mono_audio_preservation(self, audio_mixin, mocker: MockerFixture):
        """Test that mono (1D) audio tensors are processed correctly and passed to writer."""
        mono_tensor = np.random.rand(24000).astype(np.float32)
        audio_obj = CreateAudio(audio_tensor=mono_tensor)

        mock_write = mocker.patch("soundfile.write")
        audio_mixin.create_audio(audio_obj)

        mock_write.assert_called_once()
        # Verify the tensor passed to soundfile.write is the exact 1D tensor
        output_tensor = mock_write.call_args[0][1]
        assert output_tensor.ndim == 1
        assert output_tensor.shape == (24000,)
        assert np.array_equal(output_tensor, mono_tensor)

    def test_stereo_audio_preservation(self, audio_mixin, mocker: MockerFixture):
        """Test that stereo (2D) audio tensors are processed correctly and preserved."""
        stereo_tensor = np.random.rand(24000, 2).astype(np.float32)
        audio_obj = CreateAudio(audio_tensor=stereo_tensor)

        mock_write = mocker.patch("soundfile.write")
        audio_mixin.create_audio(audio_obj)

        mock_write.assert_called_once()
        # Verify the tensor passed to soundfile.write is the exact 2D tensor
        output_tensor = mock_write.call_args[0][1]
        assert output_tensor.ndim == 2
        assert output_tensor.shape == (24000, 2)
        assert np.array_equal(output_tensor, stereo_tensor)

    def test_speed_adjustment_bypass(self, audio_mixin, mocker: MockerFixture):
        """Test that speed=1.0 bypasses the expensive librosa time stretching."""
        audio_tensor = np.random.rand(24000).astype(np.float32)

        mock_time_stretch = mocker.patch("librosa.effects.time_stretch")
        # speed=1.0 should return immediately without calling librosa
        result, _ = audio_mixin._apply_speed_adjustment(audio_tensor, speed=1.0, sample_rate=24000)

        mock_time_stretch.assert_not_called()
        assert np.array_equal(result, audio_tensor)

    def test_speed_adjustment_stereo_handling(self, audio_mixin, mocker: MockerFixture):
        """Test that speed adjustment is attempted on stereo inputs."""
        mock_time_stretch = mocker.patch("librosa.effects.time_stretch")
        stereo_tensor = np.random.rand(24000, 2).astype(np.float32)
        # Mock return value representing a sped-up version (half length)
        mock_time_stretch.return_value = np.zeros((12000, 2), dtype=np.float32)

        result, _ = audio_mixin._apply_speed_adjustment(stereo_tensor, speed=2.0, sample_rate=24000)

        mock_time_stretch.assert_called_once()
        # Ensure the stereo tensor was passed to librosa
        call_args = mock_time_stretch.call_args
        assert np.array_equal(call_args.kwargs["y"], stereo_tensor)
        assert call_args.kwargs["rate"] == 2.0
        assert result.shape == (12000, 2)


# Helper to create mock model output for endpoint tests
def create_mock_audio_output_for_test(
    request_id: str = "speech-mock-123",
) -> OmniRequestOutput:
    class MockCompletionOutput:
        def __init__(self, index: int = 0):
            self.index = index
            self.text = ""
            self.token_ids = []
            self.finish_reason = "stop"
            self.stop_reason = None
            self.logprobs = None

    class MockRequestOutput:
        def __init__(self, request_id: str, audio_tensor: torch.Tensor):
            self.request_id = request_id
            self.outputs = [MockCompletionOutput(index=0)]
            self.multimodal_output = {"audio": audio_tensor}
            self.finished = True
            self.prompt_token_ids = None
            self.encoder_prompt_token_ids = None
            self.num_cached_tokens = None
            self.prompt_logprobs = None
            self.kv_transfer_params = None

    num_samples = 24000
    audio_tensor = torch.sin(torch.linspace(0, 440 * 2 * torch.pi, num_samples))
    mock_request_output = MockRequestOutput(request_id=request_id, audio_tensor=audio_tensor)

    return OmniRequestOutput(
        stage_id=0,
        final_output_type="audio",
        request_output=mock_request_output,
    )


@pytest.fixture
def test_app(mocker: MockerFixture):
    # Mock the engine client
    mock_engine_client = mocker.MagicMock()
    mock_engine_client.errored = False

    async def mock_generate_fn(*args, **kwargs):
        yield create_mock_audio_output_for_test(request_id=kwargs.get("request_id"))

    mock_engine_client.generate = mocker.MagicMock(side_effect=mock_generate_fn)
    mock_engine_client.default_sampling_params_list = [{}]

    # Mock models to have an is_base_model method
    mock_models = mocker.MagicMock()
    mock_models.is_base_model.return_value = True

    mock_request_logger = mocker.MagicMock()

    speech_server = OmniOpenAIServingSpeech(
        engine_client=mock_engine_client,
        models=mock_models,
        request_logger=mock_request_logger,
    )

    # Patch the signature of create_speech to remove 'raw_request' for FastAPI route introspection
    original_create_speech = speech_server.create_speech
    _ = mocker.MagicMock(side_effect=original_create_speech)

    sig = signature(original_create_speech)

    new_parameters = [param for name, param in sig.parameters.items() if name != "raw_request"]

    new_sig = Signature(parameters=new_parameters, return_annotation=sig.return_annotation)

    async def awaitable_patched_create_speech(*args, **kwargs):
        return await original_create_speech(*args, **kwargs)

    awaitable_patched_create_speech.__signature__ = new_sig
    speech_server.create_speech = awaitable_patched_create_speech

    app = FastAPI()
    app.add_api_route("/v1/audio/speech", speech_server.create_speech, methods=["POST"], response_model=None)

    # Add list_voices endpoint
    async def list_voices():
        speakers = sorted(speech_server.supported_speakers) if speech_server.supported_speakers else []
        uploaded_voices = []
        if hasattr(speech_server, "uploaded_speakers"):
            for voice_name, info in speech_server.uploaded_speakers.items():
                uploaded_voices.append(
                    {
                        "name": info.get("name", voice_name),
                        "consent": info.get("consent", ""),
                        "created_at": info.get("created_at", 0),
                        "file_size": info.get("file_size", 0),
                        "mime_type": info.get("mime_type", ""),
                    }
                )
        return {"voices": speakers, "uploaded_voices": uploaded_voices}

    app.add_api_route("/v1/audio/voices", list_voices, methods=["GET"])

    # Add upload_voice endpoint
    async def upload_voice(audio_sample: UploadFile = File(...), consent: str = Form(...), name: str = Form(...)):
        try:
            result = await speech_server.upload_voice(audio_sample, consent, name)
            return {"success": True, "voice": result}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception(f"Failed to upload voice: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to upload voice: {str(e)}")

    app.add_api_route("/v1/audio/voices", upload_voice, methods=["POST"])

    # Add delete_voice endpoint
    async def delete_voice(name: str):
        try:
            success = await speech_server.delete_voice(name)
            if not success:
                raise HTTPException(status_code=404, detail=f"Voice '{name}' not found")
            return {"success": True, "message": f"Voice '{name}' deleted successfully"}
        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.exception(f"Failed to delete voice '{name}': {e}")
            raise HTTPException(status_code=500, detail=f"Failed to delete voice: {str(e)}")

    app.add_api_route("/v1/audio/voices/{name}", delete_voice, methods=["DELETE"])

    return app


@pytest.fixture
def client(test_app):
    return TestClient(test_app)


class TestSpeechAPI:
    def test_create_speech_success(self, client):
        payload = {
            "input": "Hello world",
            "model": "tts-model",
            "voice": "alloy",
            "response_format": "wav",
        }
        response = client.post("/v1/audio/speech", json=payload)
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert len(response.content) > 0

    def test_create_speech_mp3_format(self, client):
        payload = {
            "input": "Hello world",
            "model": "tts-model",
            "voice": "alloy",
            "response_format": "mp3",
        }
        response = client.post("/v1/audio/speech", json=payload)
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/mpeg"
        assert len(response.content) > 0

    def test_create_speech_invalid_format(self, client):
        payload = {
            "input": "Hello world",
            "model": "tts-model",
            "voice": "alloy",
            "response_format": "invalid_format",
        }
        response = client.post("/v1/audio/speech", json=payload)
        assert response.status_code == 422  # Unprocessable Entity

    def test_speed_parameter_is_used(self, test_app, mocker: MockerFixture):
        mock_create_audio = mocker.patch(
            "vllm_omni.entrypoints.openai.serving_speech.OmniOpenAIServingSpeech.create_audio"
        )
        client = TestClient(test_app)

        mock_audio_response = mocker.MagicMock()
        mock_audio_response.audio_data = b"dummy_audio"
        mock_audio_response.media_type = "audio/wav"
        mock_create_audio.return_value = mock_audio_response

        payload = {
            "input": "This should be fast.",
            "model": "tts-model",
            "voice": "alloy",
            "response_format": "wav",
            "speed": 2.5,
        }
        client.post("/v1/audio/speech", json=payload)

        mock_create_audio.assert_called_once()
        call_args = mock_create_audio.call_args[0]
        audio_obj = call_args[0]
        assert isinstance(audio_obj, CreateAudio)
        assert audio_obj.speed == 2.5

    def test_list_voices_endpoint(self, client):
        response = client.get("/v1/audio/voices")
        assert response.status_code == 200
        assert "voices" in response.json()

    def test_upload_voice_success(self, client, tmp_path):
        """Test successful voice upload."""
        # Create a mock audio file
        audio_content = b"fake audio content" * 1000  # ~17KB
        files = {
            "audio_sample": ("test.wav", audio_content, "audio/wav"),
        }
        data = {
            "consent": "user_consent_123",
            "name": "test_voice",
        }

        response = client.post("/v1/audio/voices", files=files, data=data)
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert "voice" in result
        voice_info = result["voice"]
        assert voice_info["name"] == "test_voice"
        assert voice_info["consent"] == "user_consent_123"
        assert "created_at" in voice_info
        assert voice_info["mime_type"] == "audio/wav"
        assert voice_info["file_size"] == len(audio_content)
        response = client.delete("/v1/audio/voices/test_voice")

    def test_upload_voice_file_too_large(self, client):
        """Test voice upload with file exceeding size limit."""
        # Create a file larger than 10MB
        audio_content = b"x" * (11 * 1024 * 1024)  # 11MB
        files = {
            "audio_sample": ("test.wav", audio_content, "audio/wav"),
        }
        data = {
            "consent": "user_consent_123",
            "name": "test_voice",
        }

        response = client.post("/v1/audio/voices", files=files, data=data)
        assert response.status_code == 400
        result = response.json()
        assert "detail" in result
        assert "10MB" in result["detail"]

    def test_upload_voice_invalid_mime_type(self, client):
        """Test voice upload with invalid MIME type."""
        audio_content = b"fake audio content"
        files = {
            "audio_sample": ("test.txt", audio_content, "text/plain"),
        }
        data = {
            "consent": "user_consent_123",
            "name": "test_voice",
        }

        response = client.post("/v1/audio/voices", files=files, data=data)
        assert response.status_code == 400
        result = response.json()
        assert "detail" in result
        assert "MIME type" in result["detail"]

    def test_upload_voice_name_collision(self, client):
        """Test voice upload with duplicate name."""
        # First upload
        audio_content = b"fake audio content"
        files = {
            "audio_sample": ("test.wav", audio_content, "audio/wav"),
        }
        data = {
            "consent": "user_consent_123",
            "name": "test_voice",
        }

        response = client.post("/v1/audio/voices", files=files, data=data)
        assert response.status_code == 200

        # Second upload with same name
        response = client.post("/v1/audio/voices", files=files, data=data)
        assert response.status_code == 400
        result = response.json()
        assert "detail" in result
        assert "already exists" in result["detail"]
        response = client.delete("/v1/audio/voices/test_voice")

    def test_upload_voice_missing_parameters(self, client):
        """Test voice upload with missing required parameters."""
        audio_content = b"fake audio content"
        files = {
            "audio_sample": ("test.wav", audio_content, "audio/wav"),
        }

        # Missing consent
        data = {"name": "test_voice5"}
        response = client.post("/v1/audio/voices", files=files, data=data)
        assert response.status_code == 422  # Validation error

        # Missing name
        data = {"consent": "user_consent_123"}
        response = client.post("/v1/audio/voices", files=files, data=data)
        assert response.status_code == 422  # Validation error

        # Missing file
        data = {
            "consent": "user_consent_123",
            "name": "test_voice6",
        }
        response = client.post("/v1/audio/voices", data=data)
        assert response.status_code == 422  # Validation error

    def test_delete_voice_success(self, client):
        """Test successful voice deletion."""
        # First upload a voice
        audio_content = b"fake audio content"
        files = {
            "audio_sample": ("test.wav", audio_content, "audio/wav"),
        }
        data = {
            "consent": "user_consent_123",
            "name": "test_voice7",
        }

        response = client.post("/v1/audio/voices", files=files, data=data)
        assert response.status_code == 200

        # Then delete it
        response = client.delete("/v1/audio/voices/test_voice7")
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert "deleted successfully" in result["message"]

        # Verify it's gone by trying to delete again
        response = client.delete("/v1/audio/voices/test_voice7")
        assert response.status_code == 404
        result = response.json()
        assert "not found" in result["detail"]

    def test_delete_voice_not_found(self, client):
        """Test deleting a non-existent voice."""
        response = client.delete("/v1/audio/voices/nonexistent")
        assert response.status_code == 404
        result = response.json()
        assert "not found" in result["detail"]


class TestTTSMethods:
    """Unit tests for TTS validation and parameter building."""

    @pytest.fixture
    def speech_server(self, mocker: MockerFixture):
        mock_engine_client = mocker.MagicMock()
        mock_engine_client.errored = False
        mock_engine_client.stage_list = None
        mock_engine_client.tts_max_instructions_length = None
        mock_models = mocker.MagicMock()
        mock_models.is_base_model.return_value = True
        return OmniOpenAIServingSpeech(
            engine_client=mock_engine_client,
            models=mock_models,
            request_logger=mocker.MagicMock(),
        )

    def test_is_tts_detection_no_stage(self, speech_server):
        """Test TTS model detection when no TTS stage exists."""
        # Fixture creates server with stage_list = None -> _is_tts should be False
        assert speech_server._is_tts is False
        assert speech_server._tts_stage is None

    def test_is_tts_detection_with_tts_stage(self, mocker: MockerFixture):
        """Test TTS model detection when TTS stage exists."""
        mock_engine_client = mocker.MagicMock()
        mock_engine_client.errored = False
        mock_engine_client.tts_max_instructions_length = None

        # Create a TTS stage
        mock_stage = mocker.MagicMock()
        mock_stage.model_stage = "qwen3_tts"
        mock_stage.tts_args = {}
        mock_engine_client.stage_list = [mock_stage]

        mock_models = mocker.MagicMock()
        mock_models.is_base_model.return_value = True

        server = OmniOpenAIServingSpeech(
            engine_client=mock_engine_client,
            models=mock_models,
            request_logger=mocker.MagicMock(),
        )

        assert server._is_tts is True
        assert server._tts_stage is mock_stage

    def test_estimate_prompt_len_fallback(self, speech_server):
        """Test prompt length estimation falls back to 2048 when model is unavailable."""
        tts_params = {"text": ["Hello"], "task_type": ["CustomVoice"]}
        result = speech_server._estimate_prompt_len(tts_params)
        # Without a real model, it should fall back to 2048.
        assert result == 2048

    def test_validate_tts_request_basic(self, speech_server):
        """Test basic validation cases."""
        # Empty input
        req = OpenAICreateSpeechRequest(input="")
        assert speech_server._validate_tts_request(req) == "Input text cannot be empty"

        # Invalid language
        req = OpenAICreateSpeechRequest(input="Hello", language="InvalidLang")
        assert "Invalid language" in speech_server._validate_tts_request(req)

        # CustomVoice on model with no speakers -> rejected
        req = OpenAICreateSpeechRequest(input="Hello", voice="Invalid")
        assert "does not support CustomVoice" in speech_server._validate_tts_request(req)

        # CustomVoice without voice on model with no speakers -> also rejected
        req = OpenAICreateSpeechRequest(input="Hello")
        assert "does not support CustomVoice" in speech_server._validate_tts_request(req)

    def test_validate_tts_request_task_types(self, speech_server):
        """Test task-specific validation."""
        # Base task requires ref_audio
        req = OpenAICreateSpeechRequest(input="Hello", task_type="Base")
        assert "ref_audio" in speech_server._validate_tts_request(req)

        # VoiceDesign requires instructions
        req = OpenAICreateSpeechRequest(input="Hello", task_type="VoiceDesign")
        assert "instructions" in speech_server._validate_tts_request(req)

        # ref_text without task_type auto-infers Base, then fails on missing ref_audio
        req = OpenAICreateSpeechRequest(input="Hello", ref_text="text")
        assert "ref_audio" in speech_server._validate_tts_request(req)

    def test_validate_tts_request_auto_infer_base(self, speech_server):
        """Test auto-inference of Base task when ref_audio/ref_text is provided."""
        # ref_audio without task_type -> infers Base, requires non-empty ref_text
        req = OpenAICreateSpeechRequest(input="Hello", ref_audio="data:audio/wav;base64,abc")
        result = speech_server._validate_tts_request(req)
        assert "ref_text" in result
        assert req.task_type == "Base"

        # ref_text without task_type -> infers Base, requires ref_audio
        req = OpenAICreateSpeechRequest(input="Hello", ref_text="some text")
        result = speech_server._validate_tts_request(req)
        assert "ref_audio" in result
        assert req.task_type == "Base"

    def test_validate_tts_request_base_empty_ref_text(self, speech_server):
        """Empty ref_text on Base task returns 400 instead of crashing engine."""
        req = OpenAICreateSpeechRequest(
            input="Hello", task_type="Base", ref_audio="data:audio/wav;base64,abc", ref_text=""
        )
        result = speech_server._validate_tts_request(req)
        assert "non-empty 'ref_text'" in result

        # x_vector_only_mode bypasses ref_text requirement
        req = OpenAICreateSpeechRequest(
            input="Hello", task_type="Base", ref_audio="data:audio/wav;base64,abc", ref_text="", x_vector_only_mode=True
        )
        assert speech_server._validate_tts_request(req) is None

    def test_validate_tts_request_customvoice_no_speakers(self, speech_server):
        """CustomVoice on a model with no speakers returns 400 instead of crashing engine."""
        req = OpenAICreateSpeechRequest(input="Hello", task_type="CustomVoice")
        result = speech_server._validate_tts_request(req)
        assert "does not support CustomVoice" in result

    def test_build_tts_params(self, speech_server):
        """Test TTS parameter building."""
        req = OpenAICreateSpeechRequest(input="Hello", voice="Ryan", language="English")
        params = speech_server._build_tts_params(req)

        assert params["text"] == ["Hello"]
        assert params["speaker"] == ["Ryan"]
        assert params["language"] == ["English"]
        assert params["task_type"] == ["CustomVoice"]

    def test_load_supported_speakers(self, mocker: MockerFixture):
        """Test _load_supported_speakers."""
        mock_engine_client = mocker.MagicMock()
        mock_engine_client.errored = False
        mock_engine_client.stage_list = None

        # Mock talker_config with mixed-case speaker names
        mock_talker_config = mocker.MagicMock()
        mock_talker_config.spk_id = {"Ryan": 0, "Vivian": 1, "Aiden": 2}
        mock_engine_client.model_config.hf_config.talker_config = mock_talker_config

        mock_models = mocker.MagicMock()
        mock_models.is_base_model.return_value = True

        server = OmniOpenAIServingSpeech(
            engine_client=mock_engine_client,
            models=mock_models,
            request_logger=mocker.MagicMock(),
        )

        # Verify speakers are normalized to lowercase
        assert server.supported_speakers == {"ryan", "vivian", "aiden"}

    def test_build_tts_params_with_uploaded_voice(self, speech_server):
        """Test _build_tts_params auto-sets ref_audio for uploaded voices."""
        # Mock an uploaded speaker
        speech_server.uploaded_speakers = {
            "custom_voice": {
                "name": "custom_voice",
                "file_path": "/tmp/voice_samples/custom_voice_consent_123.wav",
                "mime_type": "audio/wav",
            }
        }
        speech_server.supported_speakers = {"ryan", "vivian", "custom_voice"}

        # Mock _get_uploaded_audio_data to return base64 data
        with patch.object(speech_server, "_get_uploaded_audio_data") as mock_get_audio:
            mock_get_audio.return_value = "data:audio/wav;base64,ZmFrZWF1ZGlv"

            req = OpenAICreateSpeechRequest(input="Hello", voice="custom_voice", task_type="Base")

            params = speech_server._build_tts_params(req)

            # Verify ref_audio was auto-set
            assert "ref_audio" in params
            assert params["ref_audio"] == ["data:audio/wav;base64,ZmFrZWF1ZGlv"]
            assert "x_vector_only_mode" in params
            assert params["x_vector_only_mode"] == [True]
            mock_get_audio.assert_called_once_with("custom_voice")

    def test_build_tts_params_without_uploaded_voice(self, speech_server):
        """Test _build_tts_params does not auto-set ref_audio for non-uploaded voices."""
        # No uploaded speakers
        speech_server.uploaded_speakers = {}
        speech_server.supported_speakers = {"ryan", "vivian"}

        req = OpenAICreateSpeechRequest(input="Hello", voice="ryan", task_type="Base")

        params = speech_server._build_tts_params(req)

        # Verify ref_audio was NOT auto-set
        assert "ref_audio" not in params
        assert "x_vector_only_mode" not in params

    def test_build_tts_params_with_explicit_ref_audio(self, speech_server):
        """Test _build_tts_params uses explicit ref_audio even for uploaded voices."""
        # Mock an uploaded speaker
        speech_server.uploaded_speakers = {
            "custom_voice": {
                "name": "custom_voice",
                "file_path": "/tmp/voice_samples/custom_voice_consent_123.wav",
                "mime_type": "audio/wav",
            }
        }
        speech_server.supported_speakers = {"ryan", "vivian", "custom_voice"}

        req = OpenAICreateSpeechRequest(
            input="Hello", voice="custom_voice", task_type="Base", ref_audio="data:audio/wav;base64,ZXhwbGljaXQ="
        )

        params = speech_server._build_tts_params(req)

        # _build_tts_params should NOT auto-set ref_audio when explicit ref_audio
        # is provided (request.ref_audio is not None skips the auto-set branch).
        # The explicit ref_audio is resolved later in create_speech() via
        # _resolve_ref_audio(), not in _build_tts_params().
        assert "ref_audio" not in params
        # x_vector_only_mode should not be set when explicit ref_audio is provided
        assert "x_vector_only_mode" not in params

    def test_get_uploaded_audio_data(self, speech_server):
        """Test _get_uploaded_audio_data function."""
        # Mock file operations
        with (
            patch("builtins.open", create=True) as mock_open,
            patch("base64.b64encode") as mock_b64encode,
            patch("pathlib.Path.exists") as mock_exists,
        ):
            mock_exists.return_value = True
            mock_b64encode.return_value = b"ZmFrZWF1ZGlv"

            # Setup mock file
            mock_file = MagicMock()
            mock_file.read.return_value = b"fakeaudio"
            mock_open.return_value.__enter__.return_value = mock_file

            # Setup uploaded speaker
            speech_server.uploaded_speakers = {
                "test_voice": {"name": "test_voice", "file_path": "/tmp/test.wav", "mime_type": "audio/wav"}
            }
            result = speech_server._get_uploaded_audio_data("test_voice")

            assert result == "data:audio/wav;base64,ZmFrZWF1ZGlv"
            mock_open.assert_called_once_with(Path("/tmp/test.wav"), "rb")
            mock_b64encode.assert_called_once_with(b"fakeaudio")

    def test_get_uploaded_audio_data_missing_file(self, speech_server):
        """Test _get_uploaded_audio_data when file is missing."""
        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = False

            # Setup uploaded speaker
            speech_server.uploaded_speakers = {
                "test_voice": {"name": "test_voice", "file_path": "/tmp/test.wav", "mime_type": "audio/wav"}
            }

            result = speech_server._get_uploaded_audio_data("test_voice")

            assert result is None

    def test_get_uploaded_audio_data_voice_not_found(self, speech_server):
        """Test _get_uploaded_audio_data when voice is not in uploaded_speakers."""
        speech_server.uploaded_speakers = {}

        result = speech_server._get_uploaded_audio_data("nonexistent")

        assert result is None

    def test_max_instructions_length_default(self, speech_server):
        """Test default max instructions length (500) when no config provided."""
        # Fixture creates server with no CLI override and no TTS stage
        assert speech_server._max_instructions_length == 500

    def test_max_instructions_length_cli_override(self, mocker: MockerFixture):
        """Test CLI override (stored in engine_client) takes highest priority."""
        mock_engine_client = mocker.MagicMock()
        mock_engine_client.errored = False
        mock_engine_client.stage_list = None
        # CLI override is stored in engine_client
        mock_engine_client.tts_max_instructions_length = 1000
        mock_models = mocker.MagicMock()
        mock_models.is_base_model.return_value = True

        server = OmniOpenAIServingSpeech(
            engine_client=mock_engine_client,
            models=mock_models,
            request_logger=mocker.MagicMock(),
        )
        # Value is cached during __init__
        assert server._max_instructions_length == 1000

    def test_max_instructions_length_stage_config(self, mocker: MockerFixture):
        """Test stage config value is used when no CLI override."""
        mock_engine_client = mocker.MagicMock()
        mock_engine_client.errored = False
        mock_engine_client.tts_max_instructions_length = None  # No CLI override
        mock_models = mocker.MagicMock()
        mock_models.is_base_model.return_value = True

        # Mock stage with tts_args
        mock_stage = mocker.MagicMock()
        mock_stage.model_stage = "qwen3_tts"
        mock_stage.tts_args = {"max_instructions_length": 750}
        mock_engine_client.stage_list = [mock_stage]

        server = OmniOpenAIServingSpeech(
            engine_client=mock_engine_client,
            models=mock_models,
            request_logger=mocker.MagicMock(),
        )
        # Value is cached during __init__
        assert server._max_instructions_length == 750

    def test_max_instructions_length_cli_overrides_stage_config(self, mocker: MockerFixture):
        """Test CLI override (in engine_client) takes precedence over stage config."""
        mock_engine_client = mocker.MagicMock()
        mock_engine_client.errored = False
        # CLI override stored in engine_client
        mock_engine_client.tts_max_instructions_length = 2000
        mock_models = mocker.MagicMock()
        mock_models.is_base_model.return_value = True

        # Mock stage with tts_args
        mock_stage = mocker.MagicMock()
        mock_stage.model_stage = "qwen3_tts"
        mock_stage.tts_args = {"max_instructions_length": 750}
        mock_engine_client.stage_list = [mock_stage]

        server = OmniOpenAIServingSpeech(
            engine_client=mock_engine_client,
            models=mock_models,
            request_logger=mocker.MagicMock(),
        )
        # CLI value (2000) should override stage config (750)
        assert server._max_instructions_length == 2000

    def test_validate_instructions_length_uses_cached_value(self, mocker: MockerFixture):
        """Test instructions length validation uses cached _max_instructions_length."""
        mock_engine_client = mocker.MagicMock()
        mock_engine_client.errored = False
        mock_engine_client.stage_list = None
        # CLI override with max length of 10 characters
        mock_engine_client.tts_max_instructions_length = 10
        mock_models = mocker.MagicMock()
        mock_models.is_base_model.return_value = True

        server = OmniOpenAIServingSpeech(
            engine_client=mock_engine_client,
            models=mock_models,
            request_logger=mocker.MagicMock(),
        )

        # Verify cached value
        assert server._max_instructions_length == 10

        # Instructions within limit should pass
        req = OpenAICreateSpeechRequest(
            input="Hello",
            task_type="VoiceDesign",
            instructions="short",
        )
        assert server._validate_tts_request(req) is None

        # Instructions exceeding limit should fail
        req = OpenAICreateSpeechRequest(
            input="Hello",
            task_type="VoiceDesign",
            instructions="this is too long",
        )
        error = server._validate_tts_request(req)
        assert error is not None
        assert "max 10 characters" in error


class TestFileValidationFunctions:
    """Unit tests for file validation helper functions."""

    def test_sanitize_filename(self):
        """Test _sanitize_filename function."""
        from vllm_omni.entrypoints.openai.serving_speech import _sanitize_filename

        # Test normal filenames
        assert _sanitize_filename("test.wav") == "test.wav"
        assert _sanitize_filename("test-file.mp3") == "test-file.mp3"
        assert _sanitize_filename("test_file.flac") == "test_file.flac"

        # Test path traversal attempts
        assert _sanitize_filename("../../../etc/passwd") == "passwd"
        assert _sanitize_filename("/absolute/path/file.wav") == "file.wav"

        # Test special characters
        assert _sanitize_filename("file with spaces.wav") == "file_with_spaces.wav"
        assert _sanitize_filename("file&with&special&chars.wav") == "file_with_special_chars.wav"
        assert _sanitize_filename("file@with#special$chars%.wav") == "file_with_special_chars_.wav"

        # Test empty filename
        assert _sanitize_filename("") == "file"

        # Test very long filename
        long_name = "a" * 300
        sanitized = _sanitize_filename(long_name)
        assert len(sanitized) == 255
        assert sanitized.startswith("a")

    def test_validate_path_within_directory(self, tmp_path):
        """Test _validate_path_within_directory function."""
        from vllm_omni.entrypoints.openai.serving_speech import _validate_path_within_directory

        # Create test directory structure
        base_dir = tmp_path / "uploads"
        base_dir.mkdir()

        # Valid paths within directory
        valid_file = base_dir / "test.wav"
        valid_subdir_file = base_dir / "subdir" / "test.wav"
        valid_subdir_file.parent.mkdir()

        assert _validate_path_within_directory(valid_file, base_dir) is True
        assert _validate_path_within_directory(valid_subdir_file, base_dir) is True

        # Invalid paths outside directory
        outside_file = tmp_path / "outside.wav"
        assert _validate_path_within_directory(outside_file, base_dir) is False

        # Test with symlink (should fail)
        if hasattr(os, "symlink"):
            link_target = tmp_path / "target.wav"
            link_target.touch()
            symlink = base_dir / "link.wav"
            os.symlink(link_target, symlink)
            # Symlinks to outside should be rejected
            assert _validate_path_within_directory(symlink, base_dir) is False

        # Test with non-existent file (should still validate path)
        non_existent = base_dir / "nonexistent.wav"
        assert _validate_path_within_directory(non_existent, base_dir) is True


class TestStreamingProtocolValidation:
    """Unit tests for the stream field validators in OpenAICreateSpeechRequest."""

    def test_stream_validation_errors(self):
        """stream=True requires response_format='pcm' and speed=1.0."""
        with pytest.raises(ValidationError, match="response_format='pcm'"):
            OpenAICreateSpeechRequest(input="Hello", stream=True, response_format="wav")
        with pytest.raises(ValidationError, match="Speed adjustment is not supported"):
            OpenAICreateSpeechRequest(input="Hello", stream=True, response_format="pcm", speed=2.0)

    def test_stream_valid(self):
        """stream=True + response_format='pcm' + speed=1.0 is accepted."""
        req = OpenAICreateSpeechRequest(input="Hello", stream=True, response_format="pcm")
        assert req.stream is True

    def test_sse_stream_format_is_blocked(self):
        """stream_format='sse' is blocked."""
        with pytest.raises(ValidationError, match="sse"):
            OpenAICreateSpeechRequest(input="Hello", stream_format="sse")


class TestStreamingResponse:
    """Integration tests for the streaming audio response path."""

    @pytest.fixture
    def streaming_app(self, mocker: MockerFixture):
        """Test app whose mock engine yields one intermediate chunk then a final chunk."""
        mock_engine_client = mocker.MagicMock()
        mock_engine_client.errored = False

        def _make_output(finished: bool) -> OmniRequestOutput:
            chunk = torch.sin(torch.linspace(0, 440 * 2 * torch.pi, 24000))

            class MockCompletionOutput:
                def __init__(self, index: int = 0):
                    self.index = index
                    self.text = ""
                    self.token_ids = []
                    self.finish_reason = "stop"
                    self.stop_reason = None
                    self.logprobs = None

            class MockRequestOutput:
                def __init__(self, audio_tensor: torch.Tensor):
                    self.request_id = "speech-stream-test"
                    self.outputs = [MockCompletionOutput(index=0)]
                    self.multimodal_output = {"audio": audio_tensor}
                    self.finished = finished
                    self.prompt_token_ids = None
                    self.encoder_prompt_token_ids = None
                    self.num_cached_tokens = None
                    self.prompt_logprobs = None
                    self.kv_transfer_params = None

            return OmniRequestOutput(
                stage_id=0,
                final_output_type="audio",
                request_output=MockRequestOutput(audio_tensor=chunk),
                finished=finished,
            )

        async def mock_generate_streaming(*args, **kwargs):
            yield _make_output(finished=False)
            yield _make_output(finished=True)

        mock_engine_client.generate = mocker.MagicMock(side_effect=mock_generate_streaming)
        mock_engine_client.default_sampling_params_list = [{}]
        mock_models = mocker.MagicMock()
        mock_models.is_base_model.return_value = True

        speech_server = OmniOpenAIServingSpeech(
            engine_client=mock_engine_client,
            models=mock_models,
            request_logger=mocker.MagicMock(),
        )

        original_create_speech = speech_server.create_speech
        sig = signature(original_create_speech)
        new_parameters = [p for name, p in sig.parameters.items() if name != "raw_request"]
        new_sig = Signature(parameters=new_parameters, return_annotation=sig.return_annotation)

        async def awaitable_create_speech(*args, **kwargs):
            return await original_create_speech(*args, **kwargs)

        awaitable_create_speech.__signature__ = new_sig
        speech_server.create_speech = awaitable_create_speech

        app = FastAPI()
        app.add_api_route("/v1/audio/speech", speech_server.create_speech, methods=["POST"], response_model=None)
        return app

    def test_streaming(self, streaming_app):
        """stream=True must return audio/pcm with non-empty body."""
        client = TestClient(streaming_app)
        response = client.post("/v1/audio/speech", json={"input": "Hello", "stream": True, "response_format": "pcm"})
        assert response.status_code == 200
        assert "audio/pcm" in response.headers["content-type"]
        assert len(response.content) > 0

    def test_non_streaming_unchanged(self, streaming_app):
        """Non-streaming path must still return audio/wav."""
        client = TestClient(streaming_app)
        response = client.post("/v1/audio/speech", json={"input": "Hello", "response_format": "wav"})
        assert response.status_code == 200
        assert "audio/wav" in response.headers["content-type"]
