# tests/entrypoints/openai/test_serving_speech.py
import logging
from inspect import Signature, signature
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_omni.entrypoints.openai.audio_utils_mixin import AudioMixin
from vllm_omni.entrypoints.openai.protocol.audio import CreateAudio
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.outputs import OmniRequestOutput

logger = logging.getLogger(__name__)


class TestAudioMixin:
    @pytest.fixture
    def audio_mixin(self):
        return AudioMixin()

    def test_stereo_to_mono_conversion(self, audio_mixin):
        stereo_tensor = np.random.rand(24000, 2).astype(np.float32)
        audio_obj = CreateAudio(audio_tensor=stereo_tensor)

        with (
            patch.object(
                audio_mixin, "_apply_speed_adjustment", side_effect=lambda tensor, speed, sr: (tensor, sr)
            ) as mock_speed,
            patch("soundfile.write") as _,
        ):
            audio_mixin.create_audio(audio_obj)

            # Check that the tensor passed to speed adjustment is mono
            mock_speed.assert_called_once()
            adjusted_tensor = mock_speed.call_args[0][0]
            assert len(adjusted_tensor) == 24000

    @patch("librosa.effects.time_stretch")
    def test_speed_adjustment(self, mock_time_stretch, audio_mixin):
        mock_time_stretch.return_value = np.zeros(12000)
        audio_tensor = np.random.rand(24000).astype(np.float32)

        adjusted_audio, _ = audio_mixin._apply_speed_adjustment(audio_tensor, speed=2.0, sample_rate=24000)

        mock_time_stretch.assert_called_with(y=audio_tensor, rate=2.0)
        assert adjusted_audio.shape == (12000,)

    @patch("soundfile.write")
    def test_unsupported_format_fallback(self, mock_write, audio_mixin, caplog):
        audio_tensor = np.random.rand(24000).astype(np.float32)
        # Use a format that is not in the list of supported formats
        audio_obj = CreateAudio(audio_tensor=audio_tensor, response_format="vorbis")

        audio_mixin.create_audio(audio_obj)

        # Should fall back to 'wav'
        mock_write.assert_called_once()
        write_kwargs = mock_write.call_args.kwargs
        assert write_kwargs["format"] == "WAV"

    def test_mono_audio_preservation(self, audio_mixin):
        """Test that mono (1D) audio tensors are processed correctly and passed to writer."""
        mono_tensor = np.random.rand(24000).astype(np.float32)
        audio_obj = CreateAudio(audio_tensor=mono_tensor)

        with patch("soundfile.write") as mock_write:
            audio_mixin.create_audio(audio_obj)

            mock_write.assert_called_once()
            # Verify the tensor passed to soundfile.write is the exact 1D tensor
            output_tensor = mock_write.call_args[0][1]
            assert output_tensor.ndim == 1
            assert output_tensor.shape == (24000,)
            assert np.array_equal(output_tensor, mono_tensor)

    def test_stereo_audio_preservation(self, audio_mixin):
        """Test that stereo (2D) audio tensors are processed correctly and preserved."""
        stereo_tensor = np.random.rand(24000, 2).astype(np.float32)
        audio_obj = CreateAudio(audio_tensor=stereo_tensor)

        with patch("soundfile.write") as mock_write:
            audio_mixin.create_audio(audio_obj)

            mock_write.assert_called_once()
            # Verify the tensor passed to soundfile.write is the exact 2D tensor
            output_tensor = mock_write.call_args[0][1]
            assert output_tensor.ndim == 2
            assert output_tensor.shape == (24000, 2)
            assert np.array_equal(output_tensor, stereo_tensor)

    def test_speed_adjustment_bypass(self, audio_mixin):
        """Test that speed=1.0 bypasses the expensive librosa time stretching."""
        audio_tensor = np.random.rand(24000).astype(np.float32)

        with patch("librosa.effects.time_stretch") as mock_time_stretch:
            # speed=1.0 should return immediately without calling librosa
            result, _ = audio_mixin._apply_speed_adjustment(audio_tensor, speed=1.0, sample_rate=24000)

            mock_time_stretch.assert_not_called()
            assert np.array_equal(result, audio_tensor)

    @patch("librosa.effects.time_stretch")
    def test_speed_adjustment_stereo_handling(self, mock_time_stretch, audio_mixin):
        """Test that speed adjustment is attempted on stereo inputs."""
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
def test_app():
    # Mock the engine client
    mock_engine_client = MagicMock()
    mock_engine_client.errored = False

    async def mock_generate_fn(*args, **kwargs):
        yield create_mock_audio_output_for_test(request_id=kwargs.get("request_id"))

    mock_engine_client.generate = MagicMock(side_effect=mock_generate_fn)
    mock_engine_client.default_sampling_params_list = [{}]

    # Mock models to have an is_base_model method
    mock_models = MagicMock()
    mock_models.is_base_model.return_value = True

    mock_request_logger = MagicMock()

    speech_server = OmniOpenAIServingSpeech(
        engine_client=mock_engine_client,
        models=mock_models,
        request_logger=mock_request_logger,
    )

    # Patch the signature of create_speech to remove 'raw_request' for FastAPI route introspection
    original_create_speech = speech_server.create_speech
    _ = MagicMock(side_effect=original_create_speech)

    sig = signature(original_create_speech)

    new_parameters = [param for name, param in sig.parameters.items() if name != "raw_request"]

    new_sig = Signature(parameters=new_parameters, return_annotation=sig.return_annotation)

    async def awaitable_patched_create_speech(*args, **kwargs):
        return await original_create_speech(*args, **kwargs)

    awaitable_patched_create_speech.__signature__ = new_sig
    speech_server.create_speech = awaitable_patched_create_speech

    app = FastAPI()
    app.add_api_route("/v1/audio/speech", speech_server.create_speech, methods=["POST"], response_model=None)

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

    @patch("vllm_omni.entrypoints.openai.serving_speech.OmniOpenAIServingSpeech.create_audio")
    def test_speed_parameter_is_used(self, mock_create_audio, test_app):
        client = TestClient(test_app)

        mock_audio_response = MagicMock()
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
