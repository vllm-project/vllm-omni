# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for async image generation API endpoints.

This module contains unit tests and integration tests (with mocking) for the
OpenAI-compatible async text-to-image generation API endpoints in api_server.py.
"""

import base64
import io
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from vllm_omni.entrypoints.openai.image_api_utils import (
    encode_image_base64,
    parse_size,
)

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
    """Mock result object from AsyncOmniDiffusion.generate()"""

    def __init__(self, images):
        self.images = images


@pytest.fixture
def mock_async_diffusion():
    """Mock AsyncOmniDiffusion instance that returns fake images"""
    mock = Mock()

    async def generate(**kwargs):
        # Return n PIL images wrapped in result object
        n = kwargs.get("num_outputs_per_prompt", 1)
        images = [Image.new("RGB", (64, 64), color="blue") for _ in range(n)]
        return MockGenerationResult(images)

    mock.generate = AsyncMock(side_effect=generate)
    return mock


@pytest.fixture
def test_client(mock_async_diffusion):
    """Create test client with mocked async diffusion engine"""
    from fastapi import FastAPI

    from vllm_omni.entrypoints.openai.api_server import router

    app = FastAPI()
    app.include_router(router)

    # Set up app state with diffusion engine
    app.state.diffusion_engine = mock_async_diffusion
    app.state.diffusion_model_name = "Qwen/Qwen-Image"

    return TestClient(app)


@pytest.mark.skip(reason="Async API server uses different health check mechanism")
def test_health_endpoint(test_client):
    """Test health check endpoint - skipped for async server"""
    pass


def test_generate_single_image(test_client):
    """Test generating a single image"""
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

    # Ensure generate() was called exactly once
    mock_async_diffusion.generate.assert_awaited_once()
    call_kwargs = mock_async_diffusion.generate.call_args[1]
    assert call_kwargs["num_inference_steps"] == 100
    assert call_kwargs["guidance_scale"] == 7.5
    assert call_kwargs["true_cfg_scale"] == 3.0
    assert call_kwargs["seed"] == 42


def test_optional_parameters_omitted(test_client, mock_async_diffusion):
    """Verify optional parameters not passed when omitted"""
    response = test_client.post(
        "/v1/images/generations",
        json={
            "prompt": "test",
            "size": "512x512",
        },
    )
    assert response.status_code == 200

    # Ensure generate() was called exactly once
    mock_async_diffusion.generate.assert_awaited_once()
    call_kwargs = mock_async_diffusion.generate.call_args[1]
    assert "num_inference_steps" not in call_kwargs
    assert "guidance_scale" not in call_kwargs
    assert "true_cfg_scale" not in call_kwargs


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
