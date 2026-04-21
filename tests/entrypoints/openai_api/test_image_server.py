# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for async image generation API endpoints.

This module contains unit tests and integration tests (with mocking) for the
OpenAI-compatible async text-to-image generation API endpoints in api_server.py.
"""

import base64
import io
from argparse import Namespace
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pytest_mock import MockerFixture
from vllm import SamplingParams

from vllm_omni.entrypoints.openai.image_api_utils import (
    encode_image_base64,
    parse_size,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Unit Tests


def test_parse_size_valid():
    """Test size parsing with valid inputs"""
    assert parse_size("1024x1024") == (1024, 1024)
    assert parse_size("512x768") == (512, 768)
    assert parse_size("256x256") == (256, 256)
    assert parse_size("1792x1024") == (1792, 1024)
    assert parse_size("1024x1792") == (1024, 1792)


def test_parse_size_invalid():
    """Test size parsing with invalid inputs"""
    with pytest.raises(ValueError, match="Invalid size format"):
        parse_size("invalid")

    with pytest.raises(ValueError, match="Invalid size format"):
        parse_size("1024")

    with pytest.raises(ValueError, match="Invalid size format"):
        parse_size("1024x")

    with pytest.raises(ValueError, match="Invalid size format"):
        parse_size("x1024")


def test_parse_size_negative():
    """Test size parsing with negative or zero dimensions"""
    with pytest.raises(ValueError, match="positive integers"):
        parse_size("0x1024")

    with pytest.raises(ValueError, match="positive integers"):
        parse_size("1024x0")

    with pytest.raises(ValueError):
        parse_size("-1024x1024")


def test_parse_size_edge_cases():
    """Test size parsing with edge cases like empty strings and non-integers"""
    # Empty string
    with pytest.raises(ValueError, match="non-empty string"):
        parse_size("")

    # Non-integer dimensions
    with pytest.raises(ValueError, match="must be integers"):
        parse_size("abc x def")

    with pytest.raises(ValueError, match="must be integers"):
        parse_size("1024.5x768.5")

    # Missing separator (user might forget 'x')
    with pytest.raises(ValueError, match="separator"):
        parse_size("1024 1024")


def test_encode_image_base64():
    """Test image encoding to base64"""
    # Create a simple test image
    img = Image.new("RGB", (64, 64), color="red")
    b64_str = encode_image_base64(img)

    # Should be valid base64
    assert isinstance(b64_str, str)
    assert len(b64_str) > 0

    # Should decode back to PNG
    decoded = base64.b64decode(b64_str)
    decoded_img = Image.open(io.BytesIO(decoded))

    # Verify properties
    assert decoded_img.size == (64, 64)
    assert decoded_img.format == "PNG"


# Integration Tests (with mocking)


class MockGenerationResult:
    """Mock result object compatible with current diffusion output shape."""

    def __init__(self, images):
        self.images = images
        self.request_output = SimpleNamespace(images=images)
        self.stage_durations = {}
        self.peak_memory_mb = 0.0


class FakeAsyncOmni:
    """Fake AsyncOmni that yields a single diffusion output."""

    def __init__(self, images=None):
        self.stage_configs = [
            SimpleNamespace(stage_type="llm", is_comprehension=True),
            SimpleNamespace(stage_type="diffusion", is_comprehension=False),
        ]
        self.default_sampling_params_list = [SamplingParams(temperature=0.1), OmniDiffusionSamplingParams()]
        self.captured_sampling_params_list = None
        self.captured_prompt = None
        self._images = images or [Image.new("RGB", (64, 64), color="green")]

    async def generate(self, prompt, request_id, sampling_params=None, sampling_params_list=None):
        if sampling_params_list is not None:
            self.captured_sampling_params_list = sampling_params_list
        else:
            self.captured_sampling_params_list = [sampling_params]
        self.captured_prompt = prompt
        images = [img.copy() for img in self._images]
        yield MockGenerationResult(images)

    def __class_getitem__(cls, item):
        return cls


@pytest.fixture
def mock_async_diffusion(mocker: MockerFixture):
    """Mock diffusion engine that matches the current async-generator API."""

    class MockAsyncDiffusion:
        def __init__(self) -> None:
            self.is_running = True
            self.check_health = mocker.AsyncMock()
            self.captured_sampling_params_list = None
            self.captured_prompt = None
            self.generate_calls = 0

        async def generate(self, **kwargs):
            self.generate_calls += 1
            n = kwargs["sampling_params_list"][0].num_outputs_per_prompt
            self.captured_sampling_params_list = kwargs["sampling_params_list"]
            self.captured_prompt = kwargs["prompt"]
            images = [Image.new("RGB", (64, 64), color="blue") for _ in range(n)]
            yield MockGenerationResult(images)

    return MockAsyncDiffusion()


@pytest.fixture
def test_client(mock_async_diffusion):
    """Create test client with mocked async diffusion engine"""
    from fastapi import FastAPI

    from vllm_omni.entrypoints.openai.api_server import router

    app = FastAPI()
    app.include_router(router)

    # Set up app state with diffusion engine
    app.state.engine_client = mock_async_diffusion
    app.state.diffusion_engine = mock_async_diffusion  # Also set for health endpoint
    app.state.stage_configs = [SimpleNamespace(stage_type="diffusion")]
    from vllm.entrypoints.openai.models.protocol import BaseModelPath

    from vllm_omni.entrypoints.openai.api_server import _DiffusionServingModels

    app.state.openai_serving_models = _DiffusionServingModels(
        [BaseModelPath(name="Qwen/Qwen-Image", model_path="Qwen/Qwen-Image")]
    )
    app.state.args = Namespace(
        default_sampling_params='{"0": {"num_inference_steps":4, "guidance_scale":7.5, "generator_device":"cpu"}}',
        max_generated_image_size=1024 * 1792,
    )

    return TestClient(app)


@pytest.fixture
def async_omni_test_client():
    """Create test client with mocked AsyncOmni engine."""
    from fastapi import FastAPI

    from vllm_omni.entrypoints.async_omni import AsyncOmni
    from vllm_omni.entrypoints.openai.api_server import router
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    class FakeAsyncOmniClass(AsyncOmni):
        def __init__(self):
            stage_configs = [
                SimpleNamespace(stage_type="llm", is_comprehension=True),
                SimpleNamespace(stage_type="diffusion", is_comprehension=False),
            ]
            default_sampling_params_list = [
                SamplingParams(temperature=0.1),
                OmniDiffusionSamplingParams(),
            ]
            self.engine = SimpleNamespace(
                stage_configs=stage_configs,
                default_sampling_params_list=default_sampling_params_list,
            )
            self.default_sampling_params_list = default_sampling_params_list
            self.captured_sampling_params_list = None
            self.captured_prompt = None
            self._images = [Image.new("RGB", (64, 64), color="green")]
            self.od_config = SimpleNamespace(supports_multimodal_inputs=True)

        async def generate(self, prompt, request_id, sampling_params=None, sampling_params_list=None):
            if sampling_params_list is not None:
                self.captured_sampling_params_list = sampling_params_list
            else:
                self.captured_sampling_params_list = [sampling_params]
            self.captured_prompt = prompt
            images = [img.copy() for img in self._images]
            yield MockGenerationResult(images)

        def __class_getitem__(cls, item):
            return cls

        def get_diffusion_od_config(self):
            return self.od_config

    app = FastAPI()
    app.include_router(router)

    engine = FakeAsyncOmniClass()
    chat_handler = object.__new__(OmniOpenAIServingChat)
    chat_handler.engine_client = engine
    chat_handler._diffusion_engine = None
    app.state.openai_serving_chat = chat_handler
    app.state.engine_client = engine
    app.state.stage_configs = [
        SimpleNamespace(stage_type="llm"),
        SimpleNamespace(stage_type="diffusion"),
    ]
    app.state.args = Namespace(
        default_sampling_params='{"1": {"num_inference_steps":4, "guidance_scale":7.5, "generator_device":"cpu"}}',
        max_generated_image_size=1048576,  # 1024*1024 to support resolution tests
    )
    return TestClient(app)


@pytest.fixture
def async_omni_rgba_test_client():
    """Create test client with mocked AsyncOmni engine returning RGBA output."""
    from fastapi import FastAPI

    from vllm_omni.entrypoints.async_omni import AsyncOmni
    from vllm_omni.entrypoints.openai.api_server import router
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    class FakeAsyncOmniClass(AsyncOmni):
        def __init__(self):
            stage_configs = [
                SimpleNamespace(stage_type="llm", is_comprehension=True),
                SimpleNamespace(stage_type="diffusion", is_comprehension=False),
            ]
            default_sampling_params_list = [
                SamplingParams(temperature=0.1),
                OmniDiffusionSamplingParams(),
            ]
            self.engine = SimpleNamespace(
                stage_configs=stage_configs,
                default_sampling_params_list=default_sampling_params_list,
            )
            self.default_sampling_params_list = default_sampling_params_list
            self.captured_sampling_params_list = None
            self.captured_prompt = None
            self._images = [Image.new("RGBA", (64, 64), color=(0, 255, 0, 128))]
            self.od_config = SimpleNamespace(supports_multimodal_inputs=True)

        async def generate(self, prompt, request_id, sampling_params=None, sampling_params_list=None):
            if sampling_params_list is not None:
                self.captured_sampling_params_list = sampling_params_list
            else:
                self.captured_sampling_params_list = [sampling_params]
            self.captured_prompt = prompt
            images = [img.copy() for img in self._images]
            yield MockGenerationResult(images)

        def __class_getitem__(cls, item):
            return cls

        def get_diffusion_od_config(self):
            return self.od_config

    app = FastAPI()
    app.include_router(router)

    engine = FakeAsyncOmniClass()
    chat_handler = object.__new__(OmniOpenAIServingChat)
    chat_handler.engine_client = engine
    chat_handler._diffusion_engine = None
    app.state.openai_serving_chat = chat_handler
    app.state.engine_client = engine
    app.state.stage_configs = [
        SimpleNamespace(stage_type="llm"),
        SimpleNamespace(stage_type="diffusion"),
    ]
    app.state.args = Namespace(
        default_sampling_params='{"1": {"num_inference_steps":4, "guidance_scale":7.5, "generator_device":"cpu"}}',
        max_generated_image_size=1048576,
    )
    return TestClient(app)


@pytest.fixture
def async_omni_stage_configs_only_client():
    """Create test client with refactored AsyncOmni compatibility surface only."""
    from fastapi import FastAPI

    from vllm_omni.entrypoints.async_omni import AsyncOmni
    from vllm_omni.entrypoints.openai.api_server import router
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    class FakeAsyncOmniClass(AsyncOmni):
        def __init__(self):
            stage_configs = [
                SimpleNamespace(stage_type="llm", is_comprehension=True),
                SimpleNamespace(stage_type="diffusion", is_comprehension=False),
            ]
            default_sampling_params_list = [
                SamplingParams(temperature=0.1),
                OmniDiffusionSamplingParams(),
            ]
            self.engine = SimpleNamespace(
                stage_configs=stage_configs,
                default_sampling_params_list=default_sampling_params_list,
            )
            self.default_sampling_params_list = default_sampling_params_list
            self.captured_sampling_params_list = None
            self.captured_prompt = None
            self._images = [Image.new("RGB", (64, 64), color="green")]
            self.od_config = SimpleNamespace(supports_multimodal_inputs=True)

        async def generate(self, prompt, request_id, sampling_params=None, sampling_params_list=None):
            if sampling_params_list is not None:
                self.captured_sampling_params_list = sampling_params_list
            else:
                self.captured_sampling_params_list = [sampling_params]
            self.captured_prompt = prompt
            images = [img.copy() for img in self._images]
            yield MockGenerationResult(images)

        def __class_getitem__(cls, item):
            return cls

        def get_diffusion_od_config(self):
            return self.od_config

    app = FastAPI()
    app.include_router(router)

    engine = FakeAsyncOmniClass()
    assert not hasattr(engine, "stage_list")
    app.state.engine_client = engine
    chat_handler = object.__new__(OmniOpenAIServingChat)
    chat_handler.engine_client = engine
    chat_handler._diffusion_engine = None
    app.state.openai_serving_chat = chat_handler
    app.state.args = Namespace(
        default_sampling_params='{"1": {"num_inference_steps":4, "guidance_scale":7.5, "generator_device":"cpu"}}',
        max_generated_image_size=1024 * 1792,
    )
    return TestClient(app)


def test_health_endpoint(test_client):
    """Test health check endpoint for diffusion mode"""
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_health_endpoint_no_engine():
    """Test health check endpoint when no engine is initialized"""
    from fastapi import FastAPI

    from vllm_omni.entrypoints.openai.api_server import router

    app = FastAPI()
    app.include_router(router)
    # Don't set any engine

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "unhealthy"


def test_health_endpoint_dead_engine():
    """Health returns 503 when the engine raises EngineDeadError."""
    from unittest.mock import AsyncMock

    from fastapi import FastAPI
    from vllm.v1.engine.exceptions import EngineDeadError

    from vllm_omni.entrypoints.openai.api_server import router

    app = FastAPI()
    app.include_router(router)

    dead_engine = AsyncMock()
    dead_engine.check_health = AsyncMock(side_effect=EngineDeadError())
    app.state.engine_client = dead_engine

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "unhealthy"


def test_models_endpoint(test_client):
    """Test /v1/models endpoint for diffusion mode"""
    response = test_client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "Qwen/Qwen-Image"
    assert data["data"][0]["object"] == "model"


def test_models_endpoint_no_engine():
    """Test /v1/models endpoint when no engine is initialized"""
    from fastapi import FastAPI

    from vllm_omni.entrypoints.openai.api_server import router

    app = FastAPI()
    app.include_router(router)
    # Don't set any engine

    client = TestClient(app)
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 0


def test_generate_single_image(test_client):
    """Test generating a single image"""
    # Single-stage path should not require openai_serving_chat.
    assert not hasattr(test_client.app.state, "openai_serving_chat")

    response = test_client.post(
        "/v1/images/generations",
        json={
            "prompt": "a cat",
            "n": 1,
            "size": "1024x1024",
        },
    )
    assert response.status_code == 200
    data = response.json()

    # Check response structure
    assert "created" in data
    assert isinstance(data["created"], int)
    assert "data" in data
    assert len(data["data"]) == 1
    assert "b64_json" in data["data"][0]

    # Verify image can be decoded
    img_bytes = base64.b64decode(data["data"][0]["b64_json"])
    img = Image.open(io.BytesIO(img_bytes))
    assert img.size == (64, 64)  # Our mock returns 64x64 images


def test_generate_images_async_omni_sampling_params(async_omni_test_client):
    """Test AsyncOmni path uses per-stage sampling params."""
    response = async_omni_test_client.post(
        "/v1/images/generations",
        json={
            "prompt": "a cat",
            "n": 2,
            "size": "256x256",
            "seed": 7,
        },
    )
    assert response.status_code == 200
    engine = async_omni_test_client.app.state.engine_client
    captured = engine.captured_sampling_params_list
    assert captured is not None
    assert len(captured) == 2
    assert captured[0].temperature == 0.1
    assert captured[1].num_outputs_per_prompt == 2
    assert captured[1].height == 256
    assert captured[1].width == 256
    assert captured[1].seed == 7


def test_generate_images_async_omni_stage_configs_only(async_omni_stage_configs_only_client):
    """Regression: image generation accepts refactored AsyncOmni without stage_list."""
    response = async_omni_stage_configs_only_client.post(
        "/v1/images/generations",
        json={
            "prompt": "a castle",
            "n": 1,
            "size": "256x256",
            "seed": 11,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) == 1
    engine = async_omni_stage_configs_only_client.app.state.engine_client
    captured = engine.captured_sampling_params_list
    assert captured is not None
    assert len(captured) == 2
    assert captured[1].seed == 11


def test_multistage_images_async_omni_construction(async_omni_test_client):
    """Regression: multistage image generation builds the expected chat-style payload."""
    response = async_omni_test_client.post(
        "/v1/images/generations",
        json={
            "prompt": "a cat",
            "n": 2,
            "size": "128x256",
            "seed": 7,
            "num_inference_steps": 12,
            "guidance_scale": 6.5,
        },
    )
    assert response.status_code == 200

    engine = async_omni_test_client.app.state.engine_client
    captured_prompt = engine.captured_prompt
    assert captured_prompt["prompt"] == "a cat"
    assert captured_prompt["modalities"] == ["image"]
    assert captured_prompt["mm_processor_kwargs"] == {
        "target_h": 256,
        "target_w": 128,
    }

    captured = engine.captured_sampling_params_list
    assert captured is not None
    assert len(captured) == 2
    assert captured[0].temperature == 0.1
    assert captured[0].seed == 7
    assert captured[1].num_outputs_per_prompt == 2
    assert captured[1].width == 128
    assert captured[1].height == 256
    assert captured[1].seed == 7
    assert captured[1].num_inference_steps == 12
    assert captured[1].guidance_scale == 6.5


def test_image_edits_async_omni_stage_configs_only(async_omni_stage_configs_only_client):
    """Regression: image edits accepts refactored AsyncOmni without stage_list."""
    img_bytes = make_test_image_bytes((16, 16))
    response = async_omni_stage_configs_only_client.post(
        "/v1/images/edits",
        files=[("image", img_bytes)],
        data={
            "prompt": "edit me",
            "size": "auto",
        },
    )
    assert response.status_code == 200
    engine = async_omni_stage_configs_only_client.app.state.engine_client
    captured = engine.captured_sampling_params_list
    assert captured is not None
    assert len(captured) == 2


def test_generate_images_max_size_rejected(async_omni_test_client):
    """Test that a size exceeding max_generated_image_size returns 400."""
    response = async_omni_test_client.post(
        "/v1/images/generations",
        json={
            "prompt": "a cat",
            "size": "2048x2048",  # 4,194,304 pixels > max_generated_image_size (1,048,576)
        },
    )
    assert response.status_code == 400


def test_generate_multiple_images(test_client):
    """Test generating multiple images"""
    response = test_client.post(
        "/v1/images/generations",
        json={
            "prompt": "a dog",
            "n": 3,
            "size": "512x512",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) == 3

    # All images should be valid
    for img_data in data["data"]:
        assert "b64_json" in img_data
        img_bytes = base64.b64decode(img_data["b64_json"])
        img = Image.open(io.BytesIO(img_bytes))
        assert img.format == "PNG"


def test_with_negative_prompt(test_client):
    """Test with negative prompt"""
    response = test_client.post(
        "/v1/images/generations",
        json={
            "prompt": "beautiful landscape",
            "negative_prompt": "blurry, low quality",
            "size": "1024x1024",
        },
    )
    assert response.status_code == 200


def test_with_seed(test_client):
    """Test with seed for reproducibility"""
    response = test_client.post(
        "/v1/images/generations",
        json={
            "prompt": "a tree",
            "seed": 42,
            "size": "1024x1024",
        },
    )
    assert response.status_code == 200


def test_with_seed_zero(test_client):
    """Test with seed=0 for reproducibility"""
    response = test_client.post(
        "/v1/images/generations",
        json={
            "prompt": "a tree",
            "seed": 0,
            "size": "1024x1024",
        },
    )
    assert response.status_code == 200
    engine = test_client.app.state.engine_client
    captured = engine.captured_sampling_params_list[0]
    # Verify that seed=0 is correctly passed
    assert captured.seed == 0, (
        f"Expected seed=0, but got seed={captured.seed}. This indicates the bug where seed=0 is treated as falsy."
    )


def test_with_custom_parameters(test_client):
    """Test with custom diffusion parameters"""
    response = test_client.post(
        "/v1/images/generations",
        json={
            "prompt": "a mountain",
            "size": "1024x1024",
            "num_inference_steps": 100,
            "true_cfg_scale": 5.5,
            "seed": 123,
        },
    )
    assert response.status_code == 200


def test_invalid_size(test_client):
    """Test with invalid size parameter - rejected by Pydantic"""
    response = test_client.post(
        "/v1/images/generations",
        json={
            "prompt": "a cat",
            "size": "invalid",
        },
    )
    # Pydantic validation errors return 422 (Unprocessable Entity)
    # "invalid" has no "x" so Pydantic rejects it
    assert response.status_code == 422
    # Check error detail contains size validation message
    detail = str(response.json()["detail"])
    assert "size" in detail.lower() or "invalid" in detail.lower()


def test_invalid_size_parse_error(test_client):
    """Test with malformed size - passes Pydantic but fails parse_size()"""
    response = test_client.post(
        "/v1/images/generations",
        json={
            "prompt": "a cat",
            "size": "1024x",  # Has "x" so Pydantic accepts, but parse_size() rejects
        },
    )
    # parse_size() raises ValueError → endpoint converts to 400 (Bad Request)
    assert response.status_code == 400
    detail = str(response.json()["detail"])
    assert "size" in detail.lower() or "invalid" in detail.lower()


def test_missing_prompt(test_client):
    """Test with missing required prompt field"""
    response = test_client.post(
        "/v1/images/generations",
        json={
            "size": "1024x1024",
        },
    )
    # Pydantic validation error
    assert response.status_code == 422


def test_invalid_n_parameter(test_client):
    """Test with invalid n parameter (out of range)"""
    # n < 1
    response = test_client.post(
        "/v1/images/generations",
        json={
            "prompt": "a cat",
            "n": 0,
        },
    )
    assert response.status_code == 422

    # n > 10
    response = test_client.post(
        "/v1/images/generations",
        json={
            "prompt": "a cat",
            "n": 11,
        },
    )
    assert response.status_code == 422


def test_url_response_format_not_supported(test_client):
    """Test that URL format returns error"""
    response = test_client.post(
        "/v1/images/generations",
        json={
            "prompt": "a cat",
            "response_format": "url",
        },
    )
    # Pydantic validation errors return 422 (Unprocessable Entity)
    assert response.status_code == 422
    # Check error mentions response_format or b64_json
    detail = str(response.json()["detail"])
    assert "b64_json" in detail.lower() or "response" in detail.lower()


def test_model_not_loaded():
    """Test error when diffusion engine is not initialized"""
    from fastapi import FastAPI

    from vllm_omni.entrypoints.openai.api_server import router

    app = FastAPI()
    app.include_router(router)
    # Don't set diffusion_engine to simulate uninitialized state
    app.state.diffusion_engine = None

    client = TestClient(app)
    response = client.post(
        "/v1/images/generations",
        json={
            "prompt": "a cat",
        },
    )
    assert response.status_code == 503
    assert "not initialized" in response.json()["detail"].lower()


def test_different_image_sizes(test_client):
    """Test various valid image sizes"""
    sizes = ["256x256", "512x512", "1024x1024", "1792x1024", "1024x1792"]

    for size in sizes:
        response = test_client.post(
            "/v1/images/generations",
            json={
                "prompt": "a test image",
                "size": size,
            },
        )
        assert response.status_code == 200, f"Failed for size {size}"


def test_parameter_validation():
    """Test Pydantic model validation"""
    from vllm_omni.entrypoints.openai.protocol.images import ImageGenerationRequest

    # Valid request - optional parameters default to None
    req = ImageGenerationRequest(prompt="test")
    assert req.prompt == "test"
    assert req.n == 1
    assert req.model is None
    assert req.size is None  # Engine will use model defaults
    assert req.num_inference_steps is None  # Engine will use model defaults
    assert req.true_cfg_scale is None  # Engine will use model defaults

    # Invalid num_inference_steps (out of range)
    with pytest.raises(ValueError):
        ImageGenerationRequest(prompt="test", num_inference_steps=0)

    with pytest.raises(ValueError):
        ImageGenerationRequest(prompt="test", num_inference_steps=201)

    # Invalid guidance_scale (out of range)
    with pytest.raises(ValueError):
        ImageGenerationRequest(prompt="test", guidance_scale=-1.0)

    with pytest.raises(ValueError):
        ImageGenerationRequest(prompt="test", guidance_scale=21.0)

    # Invalid layers for layered models (must stay within the backend-supported range)
    with pytest.raises(ValueError):
        ImageGenerationRequest(prompt="test", layers=2)

    with pytest.raises(ValueError):
        ImageGenerationRequest(prompt="test", layers=11)


# Pass-Through Tests


def test_parameters_passed_through(test_client, mock_async_diffusion):
    """Verify all parameters passed through without modification"""
    response = test_client.post(
        "/v1/images/generations",
        json={
            "prompt": "test",
            "num_inference_steps": 100,
            "guidance_scale": 7.5,
            "true_cfg_scale": 3.0,
            "seed": 42,
        },
    )
    assert response.status_code == 200

    assert mock_async_diffusion.generate_calls == 1
    captured = mock_async_diffusion.captured_sampling_params_list[0]
    assert captured.num_inference_steps == 100
    assert captured.guidance_scale == 7.5
    assert captured.true_cfg_scale == 3.0
    assert captured.seed == 42


def test_model_field_omitted_works(test_client):
    """Test that omitting model field works"""
    response = test_client.post(
        "/v1/images/generations",
        json={
            "prompt": "test",
            "size": "1024x1024",
            # model field omitted
        },
    )
    assert response.status_code == 200


def test_generate_images_rejects_model_mismatch(test_client):
    response = test_client.post(
        "/v1/images/generations",
        json={
            "prompt": "test",
            "model": "Qwen/Qwen-Image-2512",
            "size": "1024x1024",
        },
    )
    assert response.status_code == 400
    assert "model mismatch" in response.json()["detail"].lower()


def make_test_image_bytes(size=(64, 64)) -> bytes:
    img = Image.new(
        "RGB",
        size,
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_image_edit_images_processing(async_omni_test_client):
    img_bytes_1 = make_test_image_bytes((16, 16))
    img_bytes_2 = make_test_image_bytes((32, 32))

    # uploadfile with image key
    response = async_omni_test_client.post(
        "/v1/images/edits",
        files=[
            ("image", img_bytes_1),
            ("image", img_bytes_2),
        ],
        data={"prompt": "hello world."},
    )
    assert response.status_code == 200
    engine = async_omni_test_client.app.state.engine_client
    captured_prompt = engine.captured_prompt
    processed_images = captured_prompt["multi_modal_data"]["image"]
    assert len(processed_images) == 2
    assert isinstance(processed_images[0], Image.Image)
    assert isinstance(processed_images[1], Image.Image)
    assert processed_images[0].size == (16, 16)
    assert processed_images[1].size == (32, 32)

    # uploadfile with image[] key
    response = async_omni_test_client.post(
        "/v1/images/edits",
        files=[
            ("image[]", img_bytes_2),
            ("image[]", img_bytes_1),
        ],
        data={"prompt": "hello world."},
    )

    assert response.status_code == 200
    engine = async_omni_test_client.app.state.engine_client
    captured_prompt = engine.captured_prompt
    processed_images = captured_prompt["multi_modal_data"]["image"]
    assert len(processed_images) == 2
    assert isinstance(processed_images[0], Image.Image)
    assert isinstance(processed_images[1], Image.Image)
    assert processed_images[0].size == (32, 32)
    assert processed_images[1].size == (16, 16)

    # base64 url
    buf1 = io.BytesIO()
    img1 = Image.new("RGB", (16, 16))
    img1.save(buf1, format="PNG")
    b64_1 = "data:image/png;base64," + base64.b64encode(buf1.getvalue()).decode()

    buf2 = io.BytesIO()
    img2 = Image.new("RGB", (24, 24))
    img2.save(buf2, format="PNG")
    b64_2 = "data:image/png;base64," + base64.b64encode(buf2.getvalue()).decode()

    response = async_omni_test_client.post(
        "/v1/images/edits",
        data={
            "prompt": "hello from base64",
            "url": [b64_1, b64_2],
        },
    )
    assert response.status_code == 200
    processed_images = engine.captured_prompt["multi_modal_data"]["image"]
    assert len(processed_images) == 2
    assert isinstance(processed_images[0], Image.Image)
    assert isinstance(processed_images[1], Image.Image)
    assert processed_images[0].size == (16, 16)
    assert processed_images[1].size == (24, 24)


def test_image_edit_rejects_multiple_images_when_model_does_not_support_them(async_omni_test_client):
    img_bytes_1 = make_test_image_bytes((16, 16))
    img_bytes_2 = make_test_image_bytes((32, 32))

    engine = async_omni_test_client.app.state.engine_client
    engine.get_diffusion_od_config = lambda: SimpleNamespace(supports_multimodal_inputs=False)

    response = async_omni_test_client.post(
        "/v1/images/edits",
        files=[
            ("image", img_bytes_1),
            ("image", img_bytes_2),
        ],
        data={"prompt": "hello world."},
    )

    assert response.status_code == 400
    assert (
        response.json()["detail"] == "Received multiple input images. Only a single image is supported by this model."
    )
    assert engine.captured_prompt is None


def test_image_edit_rejects_model_mismatch(test_client):
    img_bytes = make_test_image_bytes((16, 16))
    response = test_client.post(
        "/v1/images/edits",
        files=[("image", img_bytes)],
        data={
            "prompt": "edit me",
            "model": "Qwen/Qwen-Image-Edit",
        },
    )
    assert response.status_code == 400
    assert "model mismatch" in response.json()["detail"].lower()


def test_image_edit_rejects_too_many_images_for_qwen_image_edit_2511(async_omni_test_client):
    engine = async_omni_test_client.app.state.engine_client
    engine.get_diffusion_od_config = lambda: SimpleNamespace(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=4,
    )

    response = async_omni_test_client.post(
        "/v1/images/edits",
        files=[
            ("image", make_test_image_bytes((16, 16))),
            ("image", make_test_image_bytes((16, 16))),
            ("image", make_test_image_bytes((16, 16))),
            ("image", make_test_image_bytes((16, 16))),
            ("image", make_test_image_bytes((16, 16))),
        ],
        data={"prompt": "hello world."},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Received 5 input images. At most 4 images are supported by this model."
    assert engine.captured_prompt is None


def test_image_edit_rejects_too_many_images_for_qwen_image_edit_2511_before_loading(
    async_omni_test_client, monkeypatch: pytest.MonkeyPatch
):
    import vllm_omni.entrypoints.openai.api_server as api_server_module

    engine = async_omni_test_client.app.state.engine_client
    engine.get_diffusion_od_config = lambda: SimpleNamespace(
        supports_multimodal_inputs=True,
        max_multimodal_image_inputs=4,
    )

    def _fail_load(*args, **kwargs):
        raise AssertionError("_load_input_images should not run for over-limit requests")

    monkeypatch.setattr(api_server_module, "_load_input_images", _fail_load)

    response = async_omni_test_client.post(
        "/v1/images/edits",
        files=[
            ("image", make_test_image_bytes((16, 16))),
            ("image", make_test_image_bytes((16, 16))),
            ("image", make_test_image_bytes((16, 16))),
            ("image", make_test_image_bytes((16, 16))),
            ("image", make_test_image_bytes((16, 16))),
        ],
        data={"prompt": "hello world."},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Received 5 input images. At most 4 images are supported by this model."
    assert engine.captured_prompt is None


def test_image_edit_parameter_pass(async_omni_test_client):
    img_bytes_1 = make_test_image_bytes((16, 16))

    # uploadfile with image key
    response = async_omni_test_client.post(
        "/v1/images/edits",
        files=[("image", img_bytes_1)],
        data={
            "prompt": "hello world.",
            "size": "16x24",
            "output_format": "jpeg",
            "num_inference_steps": 20,
            "guidance_scale": 8.0,
            "seed": 1234,
            "negative_prompt": "negative",
            "n": 2,
        },
    )
    assert response.status_code == 200
    engine = async_omni_test_client.app.state.engine_client
    captured_prompt = engine.captured_prompt
    captured_sampling_params = engine.captured_sampling_params_list[-1]

    assert captured_prompt["prompt"] == "hello world."
    assert captured_prompt["negative_prompt"] == "negative"
    assert captured_sampling_params.num_inference_steps == 20
    assert captured_sampling_params.guidance_scale == 8.0
    assert captured_sampling_params.seed == 1234
    assert captured_sampling_params.num_outputs_per_prompt == 2
    assert captured_sampling_params.width == 16
    assert captured_sampling_params.height == 24

    data = response.json()
    # All images should be valid
    for img_data in data["data"]:
        assert "b64_json" in img_data
        img_bytes = base64.b64decode(img_data["b64_json"])
        img = Image.open(io.BytesIO(img_bytes))
        assert img.format.lower() == "jpeg"
        assert data["output_format"] == "jpeg"
        assert data["size"] == "16x24"


def test_image_edit_layers_and_resolution(async_omni_test_client):
    """Test layers and resolution parameters for layered models."""
    img_bytes = make_test_image_bytes((16, 16))

    response = async_omni_test_client.post(
        "/v1/images/edits",
        files=[("image", img_bytes)],
        data={
            "prompt": "decompose into layers",
            "layers": 4,
            "resolution": 1024,
        },
    )
    assert response.status_code == 200
    engine = async_omni_test_client.app.state.engine_client
    captured_sampling_params = engine.captured_sampling_params_list[-1]
    assert captured_sampling_params.layers == 4
    assert captured_sampling_params.resolution == 1024


def test_image_edit_resolution_auto_size(async_omni_test_client):
    """Test that size='auto' with resolution lets pipeline calculate dimensions."""
    img_bytes = make_test_image_bytes((16, 16))

    response = async_omni_test_client.post(
        "/v1/images/edits",
        files=[("image", img_bytes)],
        data={
            "prompt": "test",
            "size": "auto",
            "resolution": 640,
        },
    )
    assert response.status_code == 200
    engine = async_omni_test_client.app.state.engine_client
    captured_sampling_params = engine.captured_sampling_params_list[-1]
    # When resolution is set with size=auto, width/height should be None
    # to let pipeline calculate based on resolution
    assert captured_sampling_params.width is None
    assert captured_sampling_params.height is None
    assert captured_sampling_params.resolution == 640


def test_image_edit_invalid_resolution(async_omni_test_client):
    """Test that invalid resolution values are rejected with 400."""
    img_bytes = make_test_image_bytes((16, 16))

    response = async_omni_test_client.post(
        "/v1/images/edits",
        files=[("image", img_bytes)],
        data={
            "prompt": "test",
            "resolution": 512,  # Invalid, only 640 or 1024 are supported
        },
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Invalid resolution" in detail
    assert "512" in detail


def test_image_edit_invalid_layers(async_omni_test_client):
    """Test that layered image edits reject out-of-range layers with 400."""
    img_bytes = make_test_image_bytes((16, 16))

    # Test layers below the supported range
    response = async_omni_test_client.post(
        "/v1/images/edits",
        files=[("image", img_bytes)],
        data={
            "prompt": "test",
            "layers": 2,
        },
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Invalid layers" in detail
    assert "layers must be between 3 and 10 inclusive" in detail

    # Test layers above the supported range
    response = async_omni_test_client.post(
        "/v1/images/edits",
        files=[("image", img_bytes)],
        data={
            "prompt": "test",
            "layers": 11,
        },
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Invalid layers" in detail
    assert "layers must be between 3 and 10 inclusive" in detail


def test_image_edit_resolution_and_size_conflict(async_omni_test_client):
    """Test that providing both resolution and explicit size raises 400."""
    img_bytes = make_test_image_bytes((16, 16))

    response = async_omni_test_client.post(
        "/v1/images/edits",
        files=[("image", img_bytes)],
        data={
            "prompt": "test",
            "resolution": 1024,
            "size": "512x512",  # Conflict: both resolution and explicit size
        },
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Cannot specify both" in detail
    assert "resolution" in detail
    assert "size" in detail


def test_image_edit_parameter_default(async_omni_test_client):
    img_bytes_1 = make_test_image_bytes((24, 16))

    # uploadfile with image key
    response = async_omni_test_client.post(
        "/v1/images/edits",
        files=[("image", img_bytes_1)],
        data={
            "prompt": "hello world.",
            "size": "auto",
        },
    )
    assert response.status_code == 200
    engine = async_omni_test_client.app.state.engine_client
    captured_sampling_params = engine.captured_sampling_params_list[-1]

    assert captured_sampling_params.width == 24
    assert captured_sampling_params.height == 16
    assert captured_sampling_params.num_outputs_per_prompt == 1
    assert captured_sampling_params.num_inference_steps == 4
    assert captured_sampling_params.guidance_scale == 7.5
    assert captured_sampling_params.generator_device == "cpu"

    # Test that a size exceeding max_generated_image_size returns 400
    response = async_omni_test_client.post(
        "/v1/images/edits",
        files=[("image", img_bytes_1)],
        data={
            "prompt": "hello world.",
            "size": "2048x2048",  # 4,194,304 pixels > max_generated_image_size (1,048,576)
        },
    )
    assert response.status_code == 400


def test_image_edit_parameter_default_single_stage(test_client):
    img_bytes_1 = make_test_image_bytes((24, 16))

    # uploadfile with image key
    response = test_client.post(
        "/v1/images/edits",
        files=[("image", img_bytes_1)],
        data={
            "prompt": "hello world.",
        },
    )
    assert response.status_code == 200
    engine = test_client.app.state.engine_client
    captured_sampling_params = engine.captured_sampling_params_list[0]

    assert captured_sampling_params.width == 24
    assert captured_sampling_params.height == 16
    assert captured_sampling_params.num_outputs_per_prompt == 1
    assert captured_sampling_params.num_inference_steps == 4
    assert captured_sampling_params.guidance_scale == 7.5
    assert captured_sampling_params.generator_device == "cpu"

    # Size exceeding max_generated_image_size (1024*1792) returns 400
    response = test_client.post(
        "/v1/images/edits",
        files=[("image", img_bytes_1)],
        data={
            "prompt": "hello world.",
            "size": "2048x2048",
        },
    )
    assert response.status_code == 400


def test_image_edit_compression_jpeg(test_client):
    img_bytes_1 = make_test_image_bytes((16, 16))
    # uploadfile with image key
    response = test_client.post(
        "/v1/images/edits",
        files=[("image", img_bytes_1)],
        data={"prompt": "hello world.", "output_format": "jpeg", "output_compression": 100},
    )
    assert response.status_code == 200
    data = response.json()
    img_bytes_100 = base64.b64decode(data["data"][0]["b64_json"])
    img = Image.open(io.BytesIO(img_bytes_100))
    assert img.format.lower() == "jpeg"

    response = test_client.post(
        "/v1/images/edits",
        files=[("image", img_bytes_1)],
        data={
            "prompt": "hello world.",
            "output_format": "jpeg",
            "output_compression": 50,
        },
    )
    assert response.status_code == 200
    data = response.json()
    img_bytes_50 = base64.b64decode(data["data"][0]["b64_json"])

    response = test_client.post(
        "/v1/images/edits",
        files=[("image", img_bytes_1)],
        data={
            "prompt": "hello world.",
            "output_format": "jpeg",
            "output_compression": 10,
        },
    )
    assert response.status_code == 200
    data = response.json()
    img_bytes_10 = base64.b64decode(data["data"][0]["b64_json"])

    assert len(img_bytes_10) < len(img_bytes_50)
    assert len(img_bytes_50) < len(img_bytes_100)


def test_image_edit_rgba_output_converts_to_jpeg(async_omni_rgba_test_client):
    img_bytes_1 = make_test_image_bytes((16, 16))

    response = async_omni_rgba_test_client.post(
        "/v1/images/edits",
        files=[("image", img_bytes_1)],
        data={
            "prompt": "hello world.",
            "output_format": "jpeg",
        },
    )
    assert response.status_code == 200

    data = response.json()
    img_bytes = base64.b64decode(data["data"][0]["b64_json"])
    img = Image.open(io.BytesIO(img_bytes))
    assert img.format.lower() == "jpeg"
    assert img.mode == "RGB"
    assert data["output_format"] == "jpeg"


def test_image_edit_compression_png(async_omni_test_client):
    img_bytes_1 = make_test_image_bytes((16, 16))
    # uploadfile with image key
    response = async_omni_test_client.post(
        "/v1/images/edits",
        files=[("image", img_bytes_1)],
        data={"prompt": "hello world.", "output_format": "PNG", "output_compression": 100},
    )
    assert response.status_code == 200
    data = response.json()
    img_bytes_100 = base64.b64decode(data["data"][0]["b64_json"])
    img = Image.open(io.BytesIO(img_bytes_100))
    assert img.format.lower() == "png"

    response = async_omni_test_client.post(
        "/v1/images/edits",
        files=[("image", img_bytes_1)],
        data={
            "prompt": "hello world.",
            "output_format": "PNG",
            "output_compression": 50,
        },
    )
    assert response.status_code == 200
    data = response.json()
    img_bytes_50 = base64.b64decode(data["data"][0]["b64_json"])

    response = async_omni_test_client.post(
        "/v1/images/edits",
        files=[("image", img_bytes_1)],
        data={
            "prompt": "hello world.",
            "output_format": "PNG",
            "output_compression": 10,
        },
    )
    assert response.status_code == 200
    data = response.json()
    img_bytes_10 = base64.b64decode(data["data"][0]["b64_json"])

    assert len(img_bytes_10) < len(img_bytes_50)
    assert len(img_bytes_50) < len(img_bytes_100)


def test_image_edit_with_seed_zero(async_omni_test_client):
    """Test that seed=0 is correctly handled in image editing.

    Previously, seed=0 was incorrectly replaced by a random seed due to the
    falsy value check using `or` operator. This test ensures seed=0 is
    properly passed through to the sampling parameters in image editing.
    """
    img_bytes_1 = make_test_image_bytes((16, 16))

    response = async_omni_test_client.post(
        "/v1/images/edits",
        files=[("image", img_bytes_1)],
        data={
            "prompt": "edit this image",
            "seed": 0,
        },
    )
    assert response.status_code == 200
    engine = async_omni_test_client.app.state.engine_client
    captured_sampling_params = engine.captured_sampling_params_list[-1]
    # Verify that seed=0 is correctly passed
    assert captured_sampling_params.seed == 0, (
        f"Expected seed=0, but got seed={captured_sampling_params.seed}. "
        "This indicates the bug where seed=0 is treated as falsy."
    )


def test_image_edit_with_seed_zero_single_stage(test_client):
    """Test that seed=0 is correctly handled in image editing (single stage).

    Test seed=0 handling in image editing with single stage path.
    """
    img_bytes_1 = make_test_image_bytes((16, 16))

    response = test_client.post(
        "/v1/images/edits",
        files=[("image", img_bytes_1)],
        data={
            "prompt": "edit this image",
            "seed": 0,
        },
    )
    assert response.status_code == 200
    engine = test_client.app.state.engine_client
    captured_sampling_params = engine.captured_sampling_params_list[0]
    # Verify that seed=0 is correctly passed
    assert captured_sampling_params.seed == 0, (
        f"Expected seed=0, but got seed={captured_sampling_params.seed}. "
        "This indicates the bug where seed=0 is treated as falsy."
    )


def test_normalize_image():
    """Test _normalize_image with various input types"""
    import numpy as np

    from vllm_omni.entrypoints.openai.api_server import _normalize_image

    # Test PIL Image input
    img = Image.new("RGB", (64, 64), color="red")
    result = _normalize_image(img)
    assert isinstance(result, Image.Image)
    assert result.size == (64, 64)

    # Test uint8 numpy array
    arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    result = _normalize_image(arr)
    assert isinstance(result, Image.Image)
    assert result.size == (64, 64)

    # Test float [0, 1] numpy array
    arr = np.random.rand(64, 64, 3).astype(np.float32)
    result = _normalize_image(arr)
    assert isinstance(result, Image.Image)
    assert result.size == (64, 64)

    # Test float [-1, 1] numpy array
    arr = np.random.rand(64, 64, 3).astype(np.float32) * 2 - 1
    result = _normalize_image(arr)
    assert isinstance(result, Image.Image)
    assert result.size == (64, 64)

    # Test batch dimensions (1, 1, H, W, C)
    arr = np.random.randint(0, 255, (1, 1, 64, 64, 3), dtype=np.uint8)
    result = _normalize_image(arr)
    assert isinstance(result, Image.Image)
    assert result.size == (64, 64)


def test_extract_images_from_result():
    """Test _extract_images_from_result with various result formats"""
    import numpy as np

    from vllm_omni.entrypoints.openai.api_server import _extract_images_from_result

    # Test empty result
    class EmptyResult:
        pass

    result = EmptyResult()
    images = _extract_images_from_result(result)
    assert images == []

    # Test nested batch: [np.array(shape=(3, 64, 64, 3))]
    batch = np.random.randint(0, 255, (3, 1, 64, 64, 3), dtype=np.uint8)

    class BatchResult:
        def __init__(self):
            self.images = [batch]

    result = BatchResult()
    images = _extract_images_from_result(result)
    assert len(images) == 3
    assert all(isinstance(img, Image.Image) for img in images)
    assert all(img.size == (64, 64) for img in images)

    # Test dict path: result.request_output["images"]
    class DictRequestOutput:
        def __init__(self):
            self.request_output = {"images": [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)]}

    result = DictRequestOutput()
    images = _extract_images_from_result(result)
    assert len(images) == 1
    assert isinstance(images[0], Image.Image)

    # Test attribute path: result.request_output.images
    class AttrRequestOutput:
        def __init__(self):
            self.request_output = type(
                "obj", (), {"images": [np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)]}
            )()

    result = AttrRequestOutput()
    images = _extract_images_from_result(result)
    assert len(images) == 1
    assert isinstance(images[0], Image.Image)
    assert images[0].size == (32, 32)
