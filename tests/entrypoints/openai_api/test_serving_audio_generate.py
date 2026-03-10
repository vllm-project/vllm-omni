# tests/entrypoints/openai_api/test_serving_audio_generate.py
import logging
from inspect import Signature, signature
from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_omni.entrypoints.openai.protocol.audio import (
    CreateAudio,
    OpenAICreateAudioGenerateRequest,
)
from vllm_omni.entrypoints.openai.serving_audio_generate import (
    OmniOpenAIServingAudioGenerate,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

logger = logging.getLogger(__name__)


# Helper: create a mock audio output for endpoint tests
def create_mock_audio_output(
    request_id: str = "audiogen-mock-123",
    sample_rate: int = 44100,
    num_samples: int = 44100,
    audio_key: str = "audio",
) -> OmniRequestOutput:
    """Return an OmniRequestOutput mimicking diffusion audio model output."""

    audio_tensor = torch.sin(torch.linspace(0, 440 * 2 * torch.pi, num_samples))

    return OmniRequestOutput.from_diffusion(
        request_id=request_id,
        images=[],
        prompt=None,
        metrics={},
        multimodal_output={audio_key: audio_tensor, "sr": sample_rate},
    )


def _make_engine_client(*, audio_key: str = "audio", sample_rate: int = 44100):
    """Build a mock engine client producing audio output."""
    mock_engine_client = MagicMock()
    mock_engine_client.errored = False
    mock_engine_client.model_type = "StableAudioPipeline"
    mock_engine_client.default_sampling_params_list = [{}]

    async def mock_generate_fn(*args, **kwargs):
        yield create_mock_audio_output(
            request_id=kwargs.get("request_id", "audiogen-mock"),
            sample_rate=sample_rate,
            audio_key=audio_key,
        )

    mock_engine_client.generate = MagicMock(side_effect=mock_generate_fn)
    return mock_engine_client


def _make_server(engine_client=None):
    """Build an OmniOpenAIServingAudioGenerate with mocks."""
    if engine_client is None:
        engine_client = _make_engine_client()

    mock_models = MagicMock()
    mock_models.is_base_model.return_value = True

    return OmniOpenAIServingAudioGenerate(
        engine_client=engine_client,
        models=mock_models,
        request_logger=MagicMock(),
    )


@pytest.fixture
def test_app():
    server = _make_server()

    original_fn = server.create_audio_generate
    sig = signature(original_fn)
    new_params = [p for name, p in sig.parameters.items() if name != "raw_request"]
    new_sig = Signature(parameters=new_params, return_annotation=sig.return_annotation)

    async def patched_create_audio_generate(*args, **kwargs):
        return await original_fn(*args, **kwargs)

    patched_create_audio_generate.__signature__ = new_sig
    server.create_audio_generate = patched_create_audio_generate

    app = FastAPI()
    app.add_api_route(
        "/v1/audio/generate",
        server.create_audio_generate,
        methods=["POST"],
        response_model=None,
    )
    return app


@pytest.fixture
def client(test_app):
    return TestClient(test_app)


# Request Validation (Pydantic model)
class TestRequestValidation:
    """Validate OpenAICreateAudioGenerateRequest pydantic constraints."""

    def test_valid_minimal_request(self):
        req = OpenAICreateAudioGenerateRequest(input="A calm piano melody")
        assert req.input == "A calm piano melody"
        assert req.response_format == "wav"
        assert req.speed == 1.0

    def test_fields_are_wired_correctly(self):
        req = OpenAICreateAudioGenerateRequest(
            input="rain sounds",
            model="stable-audio",
            response_format="flac",
            speed=1.5,
            audio_length=10.0,
            audio_start=2.0,
            negative_prompt="noise",
            guidance_scale=7.5,
            num_inference_steps=100,
            seed=42,
        )
        assert req.input == "rain sounds"
        assert req.model == "stable-audio"
        assert req.response_format == "flac"
        assert req.speed == 1.5
        assert req.audio_length == 10.0
        assert req.audio_start == 2.0
        assert req.negative_prompt == "noise"
        assert req.guidance_scale == 7.5
        assert req.num_inference_steps == 100
        assert req.seed == 42

    def test_invalid_response_format(self):
        with pytest.raises(Exception):
            OpenAICreateAudioGenerateRequest(input="test", response_format="invalid_format")

    def test_speed_lower_bound(self):
        with pytest.raises(Exception):
            OpenAICreateAudioGenerateRequest(input="test", speed=0.1)

    def test_speed_upper_bound(self):
        with pytest.raises(Exception):
            OpenAICreateAudioGenerateRequest(input="test", speed=5.0)

    def test_speed_at_boundaries(self):
        req_low = OpenAICreateAudioGenerateRequest(input="test", speed=0.25)
        assert req_low.speed == 0.25
        req_high = OpenAICreateAudioGenerateRequest(input="test", speed=4.0)
        assert req_high.speed == 4.0

    def test_stream_format_sse_rejected(self):
        with pytest.raises(Exception):
            OpenAICreateAudioGenerateRequest(input="test", stream_format="sse")

    def test_stream_format_audio_accepted(self):
        req = OpenAICreateAudioGenerateRequest(input="test", stream_format="audio")
        assert req.stream_format == "audio"


# Constructor & Class Methods
class TestConstructor:
    def test_default_init(self):
        server = _make_server()
        assert server.diffusion_mode is False

    def test_for_diffusion_factory(self):
        engine_client = _make_engine_client()
        mock_models = MagicMock()
        mock_models.is_base_model.return_value = True

        server = OmniOpenAIServingAudioGenerate.for_diffusion(
            engine_client=engine_client,
            models=mock_models,
            request_logger=MagicMock(),
        )
        assert server.diffusion_mode is True

    def test_is_stable_audio_model_true(self):
        server = _make_server()
        assert server._is_stable_audio_model() is True

    def test_is_stable_audio_model_false(self):
        engine = _make_engine_client()
        engine.model_type = "SomeOtherModel"
        server = _make_server(engine_client=engine)
        assert server._is_stable_audio_model() is False


# Parameter Wiring — verify request params reach the engine
class TestParameterWiring:
    """Ensure request parameters are correctly forwarded to the engine."""

    @pytest.fixture
    def server_and_engine(self):
        engine = _make_engine_client()
        server = _make_server(engine_client=engine)
        return server, engine

    @pytest.mark.asyncio
    async def test_prompt_wiring(self, server_and_engine):
        server, engine = server_and_engine
        req = OpenAICreateAudioGenerateRequest(input="birds chirping")
        await server.create_audio_generate(req)

        engine.generate.assert_called_once()
        call_kwargs = engine.generate.call_args[1]
        assert call_kwargs["prompt"]["prompt"] == "birds chirping"
        assert call_kwargs["output_modalities"] == ["audio"]

    @pytest.mark.asyncio
    async def test_negative_prompt_wiring(self, server_and_engine):
        server, engine = server_and_engine
        req = OpenAICreateAudioGenerateRequest(input="a calm ocean", negative_prompt="noise distortion")
        await server.create_audio_generate(req)

        call_kwargs = engine.generate.call_args[1]
        assert call_kwargs["prompt"]["negative_prompt"] == "noise distortion"

    @pytest.mark.asyncio
    async def test_negative_prompt_absent(self, server_and_engine):
        server, engine = server_and_engine
        req = OpenAICreateAudioGenerateRequest(input="a calm ocean")
        await server.create_audio_generate(req)

        call_kwargs = engine.generate.call_args[1]
        assert "negative_prompt" not in call_kwargs["prompt"]

    @pytest.mark.asyncio
    async def test_guidance_scale_wiring(self, server_and_engine):
        server, engine = server_and_engine
        req = OpenAICreateAudioGenerateRequest(input="test", guidance_scale=12.0)
        await server.create_audio_generate(req)

        call_kwargs = engine.generate.call_args[1]
        sp = call_kwargs["sampling_params_list"][0]
        assert isinstance(sp, OmniDiffusionSamplingParams)
        assert sp.guidance_scale == 12.0

    @pytest.mark.asyncio
    async def test_num_inference_steps_wiring(self, server_and_engine):
        server, engine = server_and_engine
        req = OpenAICreateAudioGenerateRequest(input="test", num_inference_steps=200)
        await server.create_audio_generate(req)

        sp = engine.generate.call_args[1]["sampling_params_list"][0]
        assert sp.num_inference_steps == 200

    @pytest.mark.asyncio
    async def test_seed_creates_generator(self, server_and_engine):
        server, engine = server_and_engine
        req = OpenAICreateAudioGenerateRequest(input="test", seed=42)

        with patch("vllm_omni.entrypoints.openai.serving_audio_generate.torch") as mock_torch:
            mock_gen = MagicMock()
            mock_gen.manual_seed.return_value = mock_gen
            mock_torch.Generator.return_value = mock_gen

            await server.create_audio_generate(req)

            mock_torch.Generator.assert_called_once()
            mock_gen.manual_seed.assert_called_once_with(42)

    @pytest.mark.asyncio
    async def test_seed_none_skips_generator(self, server_and_engine):
        server, engine = server_and_engine
        req = OpenAICreateAudioGenerateRequest(input="test")

        await server.create_audio_generate(req)

        sp = engine.generate.call_args[1]["sampling_params_list"][0]
        assert sp.generator is None

    @pytest.mark.asyncio
    async def test_audio_length_wiring(self, server_and_engine):
        server, engine = server_and_engine
        req = OpenAICreateAudioGenerateRequest(input="test", audio_length=10.0, audio_start=2.0)
        await server.create_audio_generate(req)

        sp = engine.generate.call_args[1]["sampling_params_list"][0]
        assert sp.extra_args["audio_start_in_s"] == 2.0
        assert sp.extra_args["audio_end_in_s"] == 12.0  # start + length

    @pytest.mark.asyncio
    async def test_audio_length_default_start(self, server_and_engine):
        server, engine = server_and_engine
        req = OpenAICreateAudioGenerateRequest(input="test", audio_length=5.0)
        await server.create_audio_generate(req)

        sp = engine.generate.call_args[1]["sampling_params_list"][0]
        assert sp.extra_args["audio_start_in_s"] == 0.0
        assert sp.extra_args["audio_end_in_s"] == 5.0

    @pytest.mark.asyncio
    async def test_no_audio_length_skips_extra_args(self, server_and_engine):
        server, engine = server_and_engine
        req = OpenAICreateAudioGenerateRequest(input="test")
        await server.create_audio_generate(req)

        sp = engine.generate.call_args[1]["sampling_params_list"][0]
        assert sp.extra_args == {}

    @pytest.mark.asyncio
    async def test_defaults_not_set_when_omitted(self, server_and_engine):
        """Guidance scale and num_inference_steps keep dataclass defaults when not in request."""
        server, engine = server_and_engine
        req = OpenAICreateAudioGenerateRequest(input="test")
        await server.create_audio_generate(req)

        sp = engine.generate.call_args[1]["sampling_params_list"][0]
        defaults = OmniDiffusionSamplingParams()
        assert sp.guidance_scale == defaults.guidance_scale
        assert sp.num_inference_steps == defaults.num_inference_steps


# Audio Response Format
class TestAudioResponseFormat:
    def test_wav_response(self, client):
        payload = {"input": "a gentle rain", "response_format": "wav"}
        response = client.post("/v1/audio/generate", json=payload)
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert len(response.content) > 0

    def test_mp3_response(self, client):
        payload = {"input": "a gentle rain", "response_format": "mp3"}
        response = client.post("/v1/audio/generate", json=payload)
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/mpeg"
        assert len(response.content) > 0

    def test_flac_response(self, client):
        payload = {"input": "a gentle rain", "response_format": "flac"}
        response = client.post("/v1/audio/generate", json=payload)
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/flac"
        assert len(response.content) > 0

    def test_invalid_format_rejected(self, client):
        payload = {"input": "test", "response_format": "banana"}
        response = client.post("/v1/audio/generate", json=payload)
        assert response.status_code == 422

    @patch("vllm_omni.entrypoints.openai.serving_audio_generate.OmniOpenAIServingAudioGenerate.create_audio")
    def test_speed_parameter_forwarded(self, mock_create_audio, test_app):
        mock_audio_response = MagicMock()
        mock_audio_response.audio_data = b"dummy_audio"
        mock_audio_response.media_type = "audio/wav"
        mock_create_audio.return_value = mock_audio_response

        c = TestClient(test_app)
        payload = {"input": "test", "response_format": "wav", "speed": 2.5}
        c.post("/v1/audio/generate", json=payload)

        mock_create_audio.assert_called_once()
        audio_obj = mock_create_audio.call_args[0][0]
        assert isinstance(audio_obj, CreateAudio)
        assert audio_obj.speed == 2.5

    @patch("vllm_omni.entrypoints.openai.serving_audio_generate.OmniOpenAIServingAudioGenerate.create_audio")
    def test_sample_rate_from_output(self, mock_create_audio, test_app):
        mock_audio_response = MagicMock()
        mock_audio_response.audio_data = b"dummy"
        mock_audio_response.media_type = "audio/wav"
        mock_create_audio.return_value = mock_audio_response

        c = TestClient(test_app)
        payload = {"input": "test"}
        c.post("/v1/audio/generate", json=payload)

        audio_obj = mock_create_audio.call_args[0][0]
        assert audio_obj.sample_rate == 44100  # Stable Audio default


# Error Handling
class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_no_output_returns_error(self):
        engine = _make_engine_client()

        async def empty_gen(*args, **kwargs):
            return
            yield  # noqa: unreachable – makes this an async generator

        engine.generate = MagicMock(side_effect=empty_gen)
        server = _make_server(engine_client=engine)
        req = OpenAICreateAudioGenerateRequest(input="test")
        resp = await server.create_audio_generate(req)

        # create_error_response returns an ErrorResponse with .error.message
        assert "No output generated" in resp.error.message

    @pytest.mark.asyncio
    async def test_no_audio_in_output_returns_error(self):
        engine = _make_engine_client()

        async def gen_without_audio(*args, **kwargs):
            yield OmniRequestOutput.from_diffusion(
                request_id="test",
                images=[],
                prompt=None,
                metrics={},
                multimodal_output={},  # no audio key
            )

        engine.generate = MagicMock(side_effect=gen_without_audio)
        server = _make_server(engine_client=engine)
        req = OpenAICreateAudioGenerateRequest(input="test")
        resp = await server.create_audio_generate(req)

        assert "did not produce audio" in resp.error.message

    @pytest.mark.asyncio
    async def test_engine_errored_raises(self):
        engine = _make_engine_client()
        engine.errored = True
        engine.dead_error = RuntimeError("engine is dead")
        server = _make_server(engine_client=engine)

        req = OpenAICreateAudioGenerateRequest(input="test")
        with pytest.raises(RuntimeError, match="engine is dead"):
            await server.create_audio_generate(req)

    @pytest.mark.asyncio
    async def test_model_outputs_key_fallback(self):
        """Audio data under 'model_outputs' key should be accepted."""
        engine = _make_engine_client(audio_key="model_outputs")
        server = _make_server(engine_client=engine)
        req = OpenAICreateAudioGenerateRequest(input="test")
        resp = await server.create_audio_generate(req)

        # Should succeed and return a Response with audio bytes
        assert hasattr(resp, "body")
        assert len(resp.body) > 0

    @pytest.mark.asyncio
    async def test_value_error_returns_error_response(self):
        engine = _make_engine_client()

        async def gen_value_error(*args, **kwargs):
            raise ValueError("bad value")
            yield  # noqa: unreachable

        engine.generate = MagicMock(side_effect=gen_value_error)
        server = _make_server(engine_client=engine)
        req = OpenAICreateAudioGenerateRequest(input="test")
        resp = await server.create_audio_generate(req)

        assert "bad value" in resp.error.message

    @pytest.mark.asyncio
    async def test_generic_exception_returns_error_response(self):
        engine = _make_engine_client()

        async def gen_runtime_error(*args, **kwargs):
            raise RuntimeError("something went wrong")
            yield  # noqa: unreachable

        engine.generate = MagicMock(side_effect=gen_runtime_error)
        server = _make_server(engine_client=engine)
        req = OpenAICreateAudioGenerateRequest(input="test")
        resp = await server.create_audio_generate(req)

        assert "Audio generation failed" in resp.error.message


# End-to-End via TestClient
class TestAudioGenerateAPI:
    def test_basic_success(self, client):
        payload = {"input": "ambient forest sounds"}
        response = client.post("/v1/audio/generate", json=payload)
        assert response.status_code == 200
        assert len(response.content) > 0

    def test_with_all_params(self, client):
        payload = {
            "input": "gentle piano",
            "response_format": "wav",
            "speed": 1.0,
            "audio_length": 5.0,
            "audio_start": 0.0,
            "negative_prompt": "noise",
            "guidance_scale": 7.0,
            "num_inference_steps": 50,
            "seed": 123,
        }
        response = client.post("/v1/audio/generate", json=payload)
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"

    def test_missing_input_rejected(self, client):
        payload = {}
        response = client.post("/v1/audio/generate", json=payload)
        assert response.status_code == 422

    def test_extra_unknown_fields_ignored(self, client):
        payload = {"input": "test", "unknown_field": "value"}
        response = client.post("/v1/audio/generate", json=payload)
        # Pydantic v2 ignores extra fields by default
        assert response.status_code == 200
