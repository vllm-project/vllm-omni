# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for OpenAI-compatible video generation endpoints.
"""

import io
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from vllm_omni.entrypoints.openai.api_server import router
from vllm_omni.entrypoints.openai.protocol.videos import VideoGenerationRequest, VideoResponseFormat
from vllm_omni.entrypoints.openai.serving_video import OmniOpenAIServingVideo


class MockVideoResult:
    def __init__(self, videos):
        self.multimodal_output = {"video": videos}


class FakeAsyncOmni:
    def __init__(self):
        self.stage_list = ["diffusion"]
        self.captured_prompt = None
        self.captured_sampling_params_list = None

    async def generate(self, prompt, request_id, sampling_params_list):
        self.captured_prompt = prompt
        self.captured_sampling_params_list = sampling_params_list
        num_outputs = sampling_params_list[0].num_outputs_per_prompt
        videos = [object() for _ in range(num_outputs)]
        yield MockVideoResult(videos)


@pytest.fixture
def test_client():
    app = FastAPI()
    app.include_router(router)
    app.state.openai_serving_video = OmniOpenAIServingVideo.for_diffusion(
        diffusion_engine=FakeAsyncOmni(),
        model_name="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    )
    return TestClient(app)


def _make_test_image_bytes(size=(64, 64)) -> bytes:
    image = Image.new("RGB", size, color="blue")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_t2v_video_generation_form(test_client):
    fps_values = []

    def _fake_encode(video, fps):
        fps_values.append(fps)
        return "Zg=="

    with patch(
        "vllm_omni.entrypoints.openai.serving_video.encode_video_base64",
        side_effect=_fake_encode,
    ):
        response = test_client.post(
            "/v1/videos",
            data={
                "prompt": "A cat runs across the street.",
                "size": "640x360",
                "seconds": "2",
                "fps": "12",
                "n": "2",
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert "data" in data and len(data["data"]) == 2
    assert all(item["b64_json"] == "Zg==" for item in data["data"])

    engine = test_client.app.state.openai_serving_video._engine_client
    captured = engine.captured_sampling_params_list[0]
    assert captured.num_outputs_per_prompt == 2
    assert captured.width == 640
    assert captured.height == 360
    assert captured.num_frames == 24
    assert captured.fps == 12
    assert fps_values == [12, 12]


def test_i2v_video_generation_form(test_client):
    image_bytes = _make_test_image_bytes((48, 32))

    with patch(
        "vllm_omni.entrypoints.openai.serving_video.encode_video_base64",
        return_value="Zg==",
    ):
        response = test_client.post(
            "/v1/videos",
            data={"prompt": "A bear playing with yarn."},
            files={"input_reference": ("input.png", image_bytes, "image/png")},
        )

    assert response.status_code == 200
    data = response.json()
    assert "data" in data and len(data["data"]) == 1
    assert data["data"][0]["b64_json"] == "Zg=="

    engine = test_client.app.state.openai_serving_video._engine_client
    prompt = engine.captured_prompt
    assert "multi_modal_data" in prompt
    assert "image" in prompt["multi_modal_data"]
    input_image = prompt["multi_modal_data"]["image"]
    assert isinstance(input_image, Image.Image)
    assert input_image.size == (48, 32)


def test_seconds_defaults_fps_and_frames(test_client):
    fps_values = []

    def _fake_encode(video, fps):
        fps_values.append(fps)
        return "Zg=="

    with patch(
        "vllm_omni.entrypoints.openai.serving_video.encode_video_base64",
        side_effect=_fake_encode,
    ):
        response = test_client.post(
            "/v1/videos",
            data={
                "prompt": "A bird flying.",
                "seconds": "3",
            },
        )

    assert response.status_code == 200
    engine = test_client.app.state.openai_serving_video._engine_client
    captured = engine.captured_sampling_params_list[0]
    assert captured.num_frames == 72
    assert captured.fps == 24
    assert fps_values == [24]


def test_size_param_sets_width_height(test_client):
    with patch(
        "vllm_omni.entrypoints.openai.serving_video.encode_video_base64",
        return_value="Zg==",
    ):
        response = test_client.post(
            "/v1/videos",
            data={
                "prompt": "size test",
                "size": "320x240",
            },
        )

    assert response.status_code == 200
    engine = test_client.app.state.openai_serving_video._engine_client
    captured = engine.captured_sampling_params_list[0]
    assert captured.width == 320
    assert captured.height == 240


def test_sampling_params_pass_through(test_client):
    with patch(
        "vllm_omni.entrypoints.openai.serving_video.encode_video_base64",
        return_value="Zg==",
    ):
        response = test_client.post(
            "/v1/videos",
            data={
                "prompt": "param pass",
                "num_inference_steps": "30",
                "guidance_scale": "6.5",
                "guidance_scale_2": "8.0",
                "true_cfg_scale": "4.0",
                "boundary_ratio": "0.7",
                "flow_shift": "0.25",
            },
        )

    assert response.status_code == 200
    engine = test_client.app.state.openai_serving_video._engine_client
    captured = engine.captured_sampling_params_list[0]
    assert captured.num_inference_steps == 30
    assert captured.guidance_scale == 6.5
    assert captured.guidance_scale_2 == 8.0
    assert captured.true_cfg_scale == 4.0
    assert captured.boundary_ratio == 0.7
    assert captured.extra_args["flow_shift"] == 0.25


def test_missing_handler_returns_503():
    app = FastAPI()
    app.include_router(router)
    app.state.openai_serving_video = None
    client = TestClient(app)

    response = client.post(
        "/v1/videos",
        data={"prompt": "no handler"},
    )
    assert response.status_code == 503
    assert "not initialized" in response.json()["detail"].lower()


def test_missing_prompt_returns_422(test_client):
    response = test_client.post(
        "/v1/videos",
        data={"size": "320x240"},
    )
    assert response.status_code == 422


def test_invalid_size_format_raises_validation_error(test_client):
    with pytest.raises(ValidationError):
        test_client.post(
            "/v1/videos",
            data={"prompt": "bad size", "size": "invalid"},
        )


def test_invalid_size_parse_returns_500(test_client):
    response = test_client.post(
        "/v1/videos",
        data={"prompt": "bad size", "size": "640x"},
    )
    assert response.status_code == 500
    assert "video generation failed" in response.json()["detail"].lower()


def test_invalid_response_format_raises_validation_error(test_client):
    with pytest.raises(ValidationError):
        test_client.post(
            "/v1/videos",
            data={"prompt": "bad format", "response_format": "url"},
        )


def test_invalid_seconds_returns_422(test_client):
    response = test_client.post(
        "/v1/videos",
        data={"prompt": "bad seconds", "seconds": "abc"},
    )
    assert response.status_code == 422


def test_invalid_n_raises_validation_error(test_client):
    with pytest.raises(ValidationError):
        test_client.post(
            "/v1/videos",
            data={"prompt": "bad n", "n": "0"},
        )

    with pytest.raises(ValidationError):
        test_client.post(
            "/v1/videos",
            data={"prompt": "bad n", "n": "5"},
        )


def test_negative_prompt_and_seed_pass_through(test_client):
    with patch(
        "vllm_omni.entrypoints.openai.serving_video.encode_video_base64",
        return_value="Zg==",
    ):
        response = test_client.post(
            "/v1/videos",
            data={
                "prompt": "snowy mountain",
                "negative_prompt": "blurry",
                "seed": "123",
            },
        )

    assert response.status_code == 200
    engine = test_client.app.state.openai_serving_video._engine_client
    captured_prompt = engine.captured_prompt
    captured_params = engine.captured_sampling_params_list[0]
    assert captured_prompt["negative_prompt"] == "blurry"
    assert captured_params.seed == 123


def test_invalid_lora_returns_400(test_client):
    response = test_client.post(
        "/v1/videos",
        data={
            "prompt": "lora test",
            "lora": '{"name": "bad-lora"}',
        },
    )
    assert response.status_code == 400
    assert "lora" in response.json()["detail"].lower()


def test_video_request_validation():
    req = VideoGenerationRequest(prompt="test")
    assert req.prompt == "test"
    assert req.n == 1
    assert req.response_format == VideoResponseFormat.B64_JSON

    with pytest.raises(ValueError):
        VideoGenerationRequest(prompt="test", response_format="url")

    with pytest.raises(ValueError):
        VideoGenerationRequest(prompt="test", size="invalid")

    with pytest.raises(ValueError):
        VideoGenerationRequest(prompt="test", seconds="abc")

    with pytest.raises(ValueError):
        VideoGenerationRequest(prompt="test", n=0)
