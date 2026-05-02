# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for OpenAI-compatible video generation endpoints.
"""

import asyncio
import base64
import io
import json
import os
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from pytest_mock import MockerFixture

from vllm_omni.entrypoints.openai import api_server
from vllm_omni.entrypoints.openai.api_server import router
from vllm_omni.entrypoints.openai.protocol.videos import (
    VideoGenerationRequest,
    VideoGenerationStatus,
    VideoResponse,
)
from vllm_omni.entrypoints.openai.serving_video import OmniOpenAIServingVideo
from vllm_omni.entrypoints.openai.storage import LocalStorageManager
from vllm_omni.entrypoints.openai.stores import AsyncDictStore, TaskRegistry
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class MockVideoResult:
    def __init__(
        self,
        videos,
        audios=None,
        sample_rate=None,
        custom_output=None,
        stage_durations=None,
        peak_memory_mb=0.0,
    ):
        self.multimodal_output = {"video": videos}
        if audios is not None:
            self.multimodal_output["audio"] = audios
        if sample_rate is not None:
            self.multimodal_output["audio_sample_rate"] = sample_rate
        self._custom_output = custom_output or {}
        self.stage_durations = stage_durations or {}
        self.peak_memory_mb = peak_memory_mb

    @property
    def custom_output(self):
        return self._custom_output


class FakeAsyncOmni:
    def __init__(self):
        self.stage_configs = [SimpleNamespace(stage_type="diffusion")]
        self.default_sampling_params_list = [OmniDiffusionSamplingParams()]
        self.captured_prompt = None
        self.captured_sampling_params_list = None

    async def generate(self, prompt, request_id, sampling_params_list):
        self.captured_prompt = prompt
        self.captured_sampling_params_list = sampling_params_list
        num_outputs = sampling_params_list[0].num_outputs_per_prompt
        videos = [object() for _ in range(num_outputs)]
        yield MockVideoResult(videos)


class BlockingVideoHandler:
    def __init__(self):
        self.model_name = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        self.stage_configs = None
        self.started = threading.Event()
        self.cancelled = threading.Event()

    def set_stage_configs_if_missing(self, stage_configs):
        if self.stage_configs is None:
            self.stage_configs = stage_configs

    async def generate_video_bytes(self, request, reference_id, *, reference_image=None):
        self.started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


@pytest.fixture(autouse=True)
def isolated_video_backends(tmp_path, monkeypatch):
    """Use isolated in-memory metadata and local storage for each test."""
    store: AsyncDictStore[VideoResponse] = AsyncDictStore()
    tasks = TaskRegistry()
    storage = LocalStorageManager(storage_path=str(tmp_path / "storage"))
    monkeypatch.setattr(api_server, "VIDEO_STORE", store)
    monkeypatch.setattr(api_server, "VIDEO_TASKS", tasks)
    monkeypatch.setattr(api_server, "STORAGE_MANAGER", storage)
    return store, tasks, storage


@pytest.fixture
def test_client():
    app = FastAPI()
    app.include_router(router)
    app.state.openai_serving_video = OmniOpenAIServingVideo.for_diffusion(
        diffusion_engine=FakeAsyncOmni(),
        model_name="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    )
    with TestClient(app) as client:
        yield client


def _make_test_image_bytes(size=(64, 64)) -> bytes:
    image = Image.new("RGB", size, color="blue")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _make_test_image_data_url(size=(64, 64)) -> str:
    image_bytes = _make_test_image_bytes(size)
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _wait_for_status(client: TestClient, video_id: str, status: str, timeout_s: float = 2.0):
    deadline = time.time() + timeout_s
    last_payload = None
    while time.time() < deadline:
        response = client.get(f"/v1/videos/{video_id}")
        last_payload = response.json()
        if last_payload["status"] == status:
            return last_payload
        time.sleep(0.02)
    raise AssertionError(f"Timed out waiting for status={status}. Last payload: {last_payload}")


def _wait_until(predicate, timeout_s: float = 2.0, interval_s: float = 0.02):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval_s)
    raise AssertionError("Timed out waiting for condition")


def test_async_video_generation_bypasses_base64(test_client, mocker: MockerFixture):
    """Regression test: Ensure async video generation saves raw bytes directly
    without bouncing through base64 encoding."""
    # We mock _encode_video_bytes (the correct path)
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        return_value=b"raw-mp4-bytes",
    )

    # We assert that encode_video_base64 is never called
    mock_base64 = mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video.encode_video_base64",
        side_effect=RuntimeError("Regression: async video path should not base64 encode"),
    )

    response = test_client.post(
        "/v1/videos",
        data={"prompt": "A base64 test."},
    )
    assert response.status_code == 200
    video_id = response.json()["id"]

    # Wait for completion. If it used base64, the RuntimeError would fail the task
    _wait_for_status(test_client, video_id, VideoGenerationStatus.COMPLETED.value)
    mock_base64.assert_not_called()


def test_async_video_generation_with_audio_bypasses_base64(test_client, mocker: MockerFixture):
    """Regression test: Ensure async video generation passes audio through
    generate_video_bytes without bouncing through base64 encoding."""
    mock_encode = mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        return_value=b"raw-mp4-bytes",
    )

    mock_base64 = mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video.encode_video_base64",
        side_effect=RuntimeError("Regression: async video path should not base64 encode"),
    )

    engine = test_client.app.state.openai_serving_video._engine_client

    async def _generate(prompt, request_id, sampling_params_list):
        engine.captured_prompt = prompt
        engine.captured_sampling_params_list = sampling_params_list
        yield MockVideoResult([object()], audios=[object()], sample_rate=48000)

    engine.generate = _generate

    response = test_client.post(
        "/v1/videos",
        data={"prompt": "A base64 test with audio."},
    )
    assert response.status_code == 200
    video_id = response.json()["id"]

    _wait_for_status(test_client, video_id, VideoGenerationStatus.COMPLETED.value)
    mock_base64.assert_not_called()

    mock_encode.assert_called_once()
    kwargs = mock_encode.call_args.kwargs
    assert "audio" in kwargs
    assert kwargs["audio"] is not None
    assert kwargs["audio_sample_rate"] == 48000


def test_t2v_video_generation_form(test_client, mocker: MockerFixture):
    fps_values = []

    def _fake_encode(video, fps, audio=None, audio_sample_rate=None, **kwargs):
        fps_values.append(fps)
        return b"fake-video"

    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        side_effect=_fake_encode,
    )
    response = test_client.post(
        "/v1/videos",
        data={
            "prompt": "A cat runs across the street.",
            "size": "640x360",
            "seconds": "2",
            "fps": "12",
        },
    )

    assert response.status_code == 200
    video_id = response.json()["id"]
    _wait_for_status(test_client, video_id, VideoGenerationStatus.COMPLETED.value)

    engine = test_client.app.state.openai_serving_video._engine_client
    captured = engine.captured_sampling_params_list[0]
    assert captured.num_outputs_per_prompt == 1
    assert captured.width == 640
    assert captured.height == 360
    assert captured.num_frames == 24
    assert captured.fps == 12
    assert captured.frame_rate == 12.0
    assert fps_values == [12]


def test_i2v_video_generation_form(test_client, mocker: MockerFixture):
    image_bytes = _make_test_image_bytes((48, 32))

    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        return_value=b"fake-video",
    )
    response = test_client.post(
        "/v1/videos",
        data={"prompt": "A bear playing with yarn."},
        files={"input_reference": ("input.png", image_bytes, "image/png")},
    )

    assert response.status_code == 200
    video_id = response.json()["id"]
    _wait_for_status(test_client, video_id, VideoGenerationStatus.COMPLETED.value)

    engine = test_client.app.state.openai_serving_video._engine_client
    prompt = engine.captured_prompt
    assert "multi_modal_data" in prompt
    assert "image" in prompt["multi_modal_data"]
    input_image = prompt["multi_modal_data"]["image"]
    assert isinstance(input_image, Image.Image)
    assert input_image.size == (48, 32)


def test_i2v_video_generation_resizes_input_to_requested_dimensions(test_client, mocker: MockerFixture):
    image_bytes = _make_test_image_bytes((48, 32))

    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        return_value=b"fake-video",
    )
    response = test_client.post(
        "/v1/videos",
        data={
            "prompt": "A bear playing with yarn.",
            "width": "96",
            "height": "64",
        },
        files={"input_reference": ("input.png", image_bytes, "image/png")},
    )

    assert response.status_code == 200
    video_id = response.json()["id"]
    _wait_for_status(test_client, video_id, VideoGenerationStatus.COMPLETED.value)

    engine = test_client.app.state.openai_serving_video._engine_client
    prompt = engine.captured_prompt
    input_image = prompt["multi_modal_data"]["image"]
    assert isinstance(input_image, Image.Image)
    assert input_image.size == (96, 64)


def test_i2v_video_generation_with_image_reference_form(test_client, mocker: MockerFixture):
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        return_value=b"fake-video",
    )
    response = test_client.post(
        "/v1/videos",
        data={
            "prompt": "A fox running through snow.",
            "image_reference": json.dumps({"image_url": _make_test_image_data_url((40, 24))}),
        },
    )

    assert response.status_code == 200
    video_id = response.json()["id"]
    _wait_for_status(test_client, video_id, VideoGenerationStatus.COMPLETED.value)

    engine = test_client.app.state.openai_serving_video._engine_client
    prompt = engine.captured_prompt
    input_image = prompt["multi_modal_data"]["image"]
    assert isinstance(input_image, Image.Image)
    assert input_image.size == (40, 24)


def test_seconds_defaults_fps_and_frames(test_client, mocker: MockerFixture):
    fps_values = []

    def _fake_encode(video, fps, audio=None, audio_sample_rate=None, **kwargs):
        fps_values.append(fps)
        return b"fake-video"

    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        side_effect=_fake_encode,
    )
    response = test_client.post(
        "/v1/videos",
        data={
            "prompt": "A bird flying.",
            "seconds": "3",
        },
    )

    assert response.status_code == 200
    video_id = response.json()["id"]
    _wait_for_status(test_client, video_id, VideoGenerationStatus.COMPLETED.value)
    engine = test_client.app.state.openai_serving_video._engine_client
    captured = engine.captured_sampling_params_list[0]
    assert captured.num_frames == 72
    assert captured.fps == 24
    assert captured.frame_rate == 24.0
    assert fps_values == [24]


def test_size_param_sets_width_height(test_client, mocker: MockerFixture):
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        return_value=b"fake-video",
    )
    response = test_client.post(
        "/v1/videos",
        data={
            "prompt": "size test",
            "size": "320x240",
        },
    )

    assert response.status_code == 200
    video_id = response.json()["id"]
    _wait_for_status(test_client, video_id, VideoGenerationStatus.COMPLETED.value)
    engine = test_client.app.state.openai_serving_video._engine_client
    captured = engine.captured_sampling_params_list[0]
    assert captured.width == 320
    assert captured.height == 240


def test_sampling_params_pass_through(test_client, mocker: MockerFixture):
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        return_value=b"fake-video",
    )
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
    video_id = response.json()["id"]
    _wait_for_status(test_client, video_id, VideoGenerationStatus.COMPLETED.value)
    engine = test_client.app.state.openai_serving_video._engine_client
    captured = engine.captured_sampling_params_list[0]
    assert captured.num_inference_steps == 30
    assert captured.guidance_scale == 6.5
    assert captured.guidance_scale_2 == 8.0
    assert captured.true_cfg_scale == 4.0
    assert captured.boundary_ratio == 0.7
    assert captured.extra_args["flow_shift"] == 0.25


def test_frame_interpolation_params_pass_to_diffusion_sampling_params(test_client, mocker: MockerFixture):
    """Frame interpolation parameters should be forwarded to diffusion worker sampling params."""
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        return_value=b"fake-video",
    )
    response = test_client.post(
        "/v1/videos",
        data={
            "prompt": "smooth motion",
            "fps": "8",
            "enable_frame_interpolation": "true",
            "frame_interpolation_exp": "2",
            "frame_interpolation_scale": "0.5",
            "frame_interpolation_model_path": "local-rife",
        },
    )

    assert response.status_code == 200
    video_id = response.json()["id"]
    _wait_for_status(test_client, video_id, VideoGenerationStatus.COMPLETED.value)

    engine = test_client.app.state.openai_serving_video._engine_client
    captured = engine.captured_sampling_params_list[0]
    assert captured.enable_frame_interpolation is True
    assert captured.frame_interpolation_exp == 2
    assert captured.frame_interpolation_scale == 0.5
    assert captured.frame_interpolation_model_path == "local-rife"


def test_default_sampling_params_apply_to_video_requests(test_client, mocker: MockerFixture):
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        return_value=b"fake-video",
    )
    engine = test_client.app.state.openai_serving_video._engine_client
    engine.default_sampling_params_list = [
        OmniDiffusionSamplingParams(
            num_inference_steps=4,
            guidance_scale=7.5,
            generator_device="cpu",
            enable_frame_interpolation=True,
            frame_interpolation_exp=2,
            frame_interpolation_scale=0.5,
            frame_interpolation_model_path="default-rife",
        )
    ]

    response = test_client.post(
        "/v1/videos",
        data={
            "prompt": "default param pass-through",
        },
    )

    assert response.status_code == 200
    video_id = response.json()["id"]
    _wait_for_status(test_client, video_id, VideoGenerationStatus.COMPLETED.value)

    captured = engine.captured_sampling_params_list[0]
    assert captured.num_inference_steps == 4
    assert captured.guidance_scale == 7.5
    assert captured.generator_device == "cpu"
    assert captured.enable_frame_interpolation is True
    assert captured.frame_interpolation_exp == 2
    assert captured.frame_interpolation_scale == 0.5
    assert captured.frame_interpolation_model_path == "default-rife"


def test_request_params_override_default_video_sampling_params(test_client, mocker: MockerFixture):
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        return_value=b"fake-video",
    )
    engine = test_client.app.state.openai_serving_video._engine_client
    engine.default_sampling_params_list = [
        OmniDiffusionSamplingParams(
            num_inference_steps=4,
            guidance_scale=7.5,
            enable_frame_interpolation=True,
            frame_interpolation_exp=2,
            frame_interpolation_scale=0.5,
            frame_interpolation_model_path="default-rife",
        )
    ]

    response = test_client.post(
        "/v1/videos",
        data={
            "prompt": "explicit override",
            "num_inference_steps": "8",
            "enable_frame_interpolation": "false",
            "frame_interpolation_exp": "1",
            "frame_interpolation_scale": "1.0",
            "frame_interpolation_model_path": "custom-rife",
        },
    )

    assert response.status_code == 200
    video_id = response.json()["id"]
    _wait_for_status(test_client, video_id, VideoGenerationStatus.COMPLETED.value)

    captured = engine.captured_sampling_params_list[0]
    assert captured.num_inference_steps == 8
    assert captured.guidance_scale == 7.5
    assert captured.enable_frame_interpolation is False
    assert captured.frame_interpolation_exp == 1
    assert captured.frame_interpolation_scale == 1.0
    assert captured.frame_interpolation_model_path == "custom-rife"


def test_worker_fps_multiplier_is_applied_to_async_encoding(test_client, mocker: MockerFixture):
    fps_values = []
    engine = test_client.app.state.openai_serving_video._engine_client

    async def _generate(prompt, request_id, sampling_params_list):
        engine.captured_prompt = prompt
        engine.captured_sampling_params_list = sampling_params_list
        import numpy as np

        yield MockVideoResult([np.zeros((1, 64, 64, 3), dtype=np.uint8)], custom_output={"video_fps_multiplier": 2})

    engine.generate = _generate

    def _fake_encode(video, fps, **kwargs):
        del video, kwargs
        fps_values.append(fps)
        return b"fake-video"

    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        side_effect=_fake_encode,
    )

    response = test_client.post("/v1/videos", data={"prompt": "fps multiplier", "fps": "8"})

    assert response.status_code == 200
    video_id = response.json()["id"]
    _wait_for_status(test_client, video_id, VideoGenerationStatus.COMPLETED.value)
    assert fps_values == [16]


def test_audio_sample_rate_comes_from_model_config(test_client, mocker: MockerFixture):
    audio_sample_rates = []

    def _fake_encode(video, fps, audio=None, audio_sample_rate=None, video_codec_options=None):
        del video, fps, audio, video_codec_options
        audio_sample_rates.append(audio_sample_rate)
        return b"fake-video"

    engine = test_client.app.state.openai_serving_video._engine_client
    engine.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            vocoder=SimpleNamespace(
                config=SimpleNamespace(output_sampling_rate=16000),
            ),
        ),
    )

    async def _generate(prompt, request_id, sampling_params_list):
        engine.captured_prompt = prompt
        engine.captured_sampling_params_list = sampling_params_list
        import numpy as np

        yield MockVideoResult([np.zeros((1, 64, 64, 3), dtype=np.uint8)], audios=[object()])

    engine.generate = _generate

    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        side_effect=_fake_encode,
    )
    response = test_client.post(
        "/v1/videos",
        data={"prompt": "video with audio"},
    )

    assert response.status_code == 200
    video_id = response.json()["id"]
    _wait_for_status(test_client, video_id, VideoGenerationStatus.COMPLETED.value)
    assert audio_sample_rates == [16000]


def test_video_job_persists_profiler_metadata(test_client, mocker: MockerFixture):
    engine = test_client.app.state.openai_serving_video._engine_client

    async def _generate(prompt, request_id, sampling_params_list):
        engine.captured_prompt = prompt
        engine.captured_sampling_params_list = sampling_params_list
        yield MockVideoResult(
            [object()],
            stage_durations={"diffuse": 2.5, "vae.decode": 0.3},
            peak_memory_mb=4096.5,
        )

    engine.generate = _generate
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        return_value=b"fake-video",
    )

    response = test_client.post("/v1/videos", data={"prompt": "profile me"})
    assert response.status_code == 200
    video_id = response.json()["id"]
    completed = _wait_for_status(test_client, video_id, VideoGenerationStatus.COMPLETED.value)

    assert completed["stage_durations"] == {"diffuse": 2.5, "vae.decode": 0.3}
    assert completed["peak_memory_mb"] == 4096.5


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


def test_video_generation_rejects_model_mismatch(test_client):
    response = test_client.post(
        "/v1/videos",
        data={
            "prompt": "bad model",
            "model": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        },
    )
    assert response.status_code == 400
    assert "model mismatch" in response.json()["detail"].lower()


def test_invalid_size_parse_returns_422(test_client):
    response = test_client.post(
        "/v1/videos",
        data={"prompt": "bad size", "size": "640x"},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["detail"][0]["loc"] == ["body", "size"]
    assert body["detail"][0]["type"] == "string_pattern_mismatch"
    assert body["detail"][0]["input"] == "640x"


def test_rejects_input_reference_and_image_reference_together(test_client):
    response = test_client.post(
        "/v1/videos",
        data={
            "prompt": "bad refs",
            "image_reference": '{"image_url": "https://example.com/cat.png"}',
        },
        files={"input_reference": ("input.png", _make_test_image_bytes(), "image/png")},
    )
    assert response.status_code == 400
    assert "either input_reference or image_reference" in response.json()["detail"].lower()


def test_invalid_seconds_returns_422(test_client):
    response = test_client.post(
        "/v1/videos",
        data={"prompt": "bad seconds", "seconds": "abc"},
    )
    assert response.status_code == 422


def test_negative_prompt_and_seed_pass_through(test_client, mocker: MockerFixture):
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        return_value=b"fake-video",
    )
    response = test_client.post(
        "/v1/videos",
        data={
            "prompt": "snowy mountain",
            "negative_prompt": "blurry",
            "seed": "123",
        },
    )

    assert response.status_code == 200
    video_id = response.json()["id"]
    _wait_for_status(test_client, video_id, VideoGenerationStatus.COMPLETED.value)
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
    assert response.status_code == 200
    video_id = response.json()["id"]
    failed = _wait_for_status(test_client, video_id, VideoGenerationStatus.FAILED.value)
    assert failed["error"]["code"] == 400
    assert "lora object" in failed["error"]["message"].lower()


def test_unsupported_image_reference_file_id_returns_400(test_client):
    response = test_client.post(
        "/v1/videos",
        data={
            "prompt": "unsupported ref",
            "image_reference": '{"file_id": "file-123"}',
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid image_reference: file_id is not supported yet."


def test_invalid_uploaded_input_reference_returns_400(test_client):
    response = test_client.post(
        "/v1/videos",
        data={"prompt": "bad upload"},
        files={"input_reference": ("input.png", b"not-an-image", "image/png")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid input_reference: provided content is not a valid image."


def test_video_request_validation():
    req = VideoGenerationRequest(prompt="test")
    assert req.prompt == "test"
    with pytest.raises(ValueError):
        VideoGenerationRequest(prompt="test", size="invalid")

    with pytest.raises(ValueError):
        VideoGenerationRequest(prompt="test", seconds="abc")

    with pytest.raises(ValueError):
        VideoGenerationRequest(prompt="test", image_reference={"file_id": "file-1", "image_url": "https://example.com"})
    with pytest.raises(ValueError):
        VideoGenerationRequest(prompt="test", frame_interpolation_exp=0)
    with pytest.raises(ValueError):
        VideoGenerationRequest(prompt="test", frame_interpolation_scale=0)


def test_list_videos_supports_order_after_and_limit(test_client, mocker: MockerFixture):
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        return_value=b"fake-video",
    )
    ids = []
    for i in range(3):
        create_resp = test_client.post("/v1/videos", data={"prompt": f"video-{i}"})
        assert create_resp.status_code == 200
        video_id = create_resp.json()["id"]
        _wait_for_status(test_client, video_id, VideoGenerationStatus.COMPLETED.value)
        ids.append(video_id)

    asyncio.run(api_server.VIDEO_STORE.update_fields(ids[0], {"created_at": 100}))
    asyncio.run(api_server.VIDEO_STORE.update_fields(ids[1], {"created_at": 200}))
    asyncio.run(api_server.VIDEO_STORE.update_fields(ids[2], {"created_at": 300}))

    asc_resp = test_client.get("/v1/videos", params={"order": "asc"})
    assert asc_resp.status_code == 200
    asc_body = asc_resp.json()
    asc_ids = [item["id"] for item in asc_body["data"]]
    assert asc_ids == [ids[0], ids[1], ids[2]]
    assert asc_body["object"] == "list"
    assert asc_body["first_id"] == ids[0]
    assert asc_body["last_id"] == ids[2]
    assert asc_body["has_more"] is False

    desc_resp = test_client.get("/v1/videos", params={"order": "desc", "limit": 2})
    assert desc_resp.status_code == 200
    desc_body = desc_resp.json()
    desc_ids = [item["id"] for item in desc_body["data"]]
    assert desc_ids == [ids[2], ids[1]]
    assert desc_body["object"] == "list"
    assert desc_body["first_id"] == ids[2]
    assert desc_body["last_id"] == ids[1]
    assert desc_body["has_more"] is True

    after_resp = test_client.get("/v1/videos", params={"order": "asc", "after": ids[0]})
    assert after_resp.status_code == 200
    after_body = after_resp.json()
    after_ids = [item["id"] for item in after_body["data"]]
    assert after_ids == [ids[1], ids[2]]
    assert after_body["object"] == "list"
    assert after_body["first_id"] == ids[1]
    assert after_body["last_id"] == ids[2]
    assert after_body["has_more"] is False

    zero_limit_resp = test_client.get("/v1/videos", params={"order": "asc", "limit": 0})
    assert zero_limit_resp.status_code == 200
    zero_limit_body = zero_limit_resp.json()
    assert zero_limit_body["data"] == []
    assert zero_limit_body["object"] == "list"
    assert zero_limit_body["first_id"] is None
    assert zero_limit_body["last_id"] is None
    assert zero_limit_body["has_more"] is True

    zero_limit_after_resp = test_client.get(
        "/v1/videos",
        params={"order": "asc", "after": ids[2], "limit": 0},
    )
    assert zero_limit_after_resp.status_code == 200
    zero_limit_after_body = zero_limit_after_resp.json()
    assert zero_limit_after_body["data"] == []
    assert zero_limit_after_body["object"] == "list"
    assert zero_limit_after_body["first_id"] is None
    assert zero_limit_after_body["last_id"] is None
    assert zero_limit_after_body["has_more"] is False


def test_delete_completed_job_removes_file_and_metadata(test_client, mocker: MockerFixture):
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        return_value=b"fake-video",
    )
    create_resp = test_client.post("/v1/videos", data={"prompt": "Delete this video"})
    assert create_resp.status_code == 200
    video_id = create_resp.json()["id"]

    final = _wait_for_status(test_client, video_id, VideoGenerationStatus.COMPLETED.value)
    file_name = final["file_name"]
    assert file_name is not None
    file_path = os.path.join(api_server.STORAGE_MANAGER.storage_path, file_name)
    assert os.path.exists(file_path)

    delete_resp = test_client.delete(f"/v1/videos/{video_id}")
    assert delete_resp.status_code == 200
    assert delete_resp.json()["id"] == video_id
    assert delete_resp.json()["deleted"] is True
    assert delete_resp.json()["object"] == "video.deleted"
    assert not os.path.exists(file_path)


def test_delete_in_progress_job_cancels_task_and_removes_metadata(test_client):
    handler = BlockingVideoHandler()
    test_client.app.state.openai_serving_video = handler

    create_resp = test_client.post("/v1/videos", data={"prompt": "Cancel this video"})
    assert create_resp.status_code == 200
    video_id = create_resp.json()["id"]

    assert handler.started.wait(timeout=2.0)

    delete_resp = test_client.delete(f"/v1/videos/{video_id}")
    assert delete_resp.status_code == 200
    assert delete_resp.json()["id"] == video_id
    assert delete_resp.json()["deleted"] is True
    assert delete_resp.json()["object"] == "video.deleted"

    assert handler.cancelled.wait(timeout=2.0)
    _wait_until(lambda: asyncio.run(api_server.VIDEO_TASKS.get(video_id)) is None)
    assert asyncio.run(api_server.VIDEO_STORE.get(video_id)) is None

    retrieve_resp = test_client.get(f"/v1/videos/{video_id}")
    assert retrieve_resp.status_code == 404


def test_video_response_file_extension_is_robust():
    response = VideoResponse(model="test-model", prompt="Make something beautiful")
    assert response.file_extension == "mp4"

    with_params = VideoResponse.model_construct(
        model="test-model",
        media_type="video/mp4; charset=binary",
    )
    assert with_params.file_extension == "mp4"

    webm = VideoResponse.model_construct(
        model="test-model",
        media_type="video/webm",
    )
    assert webm.file_extension == "webm"

    with pytest.raises(ValueError):
        unknown = VideoResponse.model_construct(
            model="test-model",
            media_type="application/x-custom-video",
        )
        _ = unknown.file_extension


def test_extra_params_merged_into_extra_args(test_client, mocker: MockerFixture):
    """extra_params JSON object is merged into sampling_params.extra_args."""
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        return_value=b"fake-video",
    )
    extra_params = {
        "is_enable_stage2": True,
        "pyramid_num_stages": 3,
        "pyramid_num_inference_steps_list": [20, 20, 20],
        "use_cfg_zero_star": True,
    }
    response = test_client.post(
        "/v1/videos",
        data={
            "prompt": "A rocket launching.",
            "extra_params": json.dumps(extra_params),
        },
    )

    assert response.status_code == 200
    video_id = response.json()["id"]
    _wait_for_status(test_client, video_id, VideoGenerationStatus.COMPLETED.value)
    engine = test_client.app.state.openai_serving_video._engine_client
    captured = engine.captured_sampling_params_list[0]
    assert captured.extra_args["is_enable_stage2"] is True
    assert captured.extra_args["pyramid_num_stages"] == 3
    assert captured.extra_args["pyramid_num_inference_steps_list"] == [20, 20, 20]
    assert captured.extra_args["use_cfg_zero_star"] is True


def test_extra_params_none_by_default(test_client, mocker: MockerFixture):
    """When extra_params is omitted, extra_args stays empty."""
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        return_value=b"fake-video",
    )
    response = test_client.post(
        "/v1/videos",
        data={"prompt": "A calm river."},
    )

    assert response.status_code == 200
    video_id = response.json()["id"]
    _wait_for_status(test_client, video_id, VideoGenerationStatus.COMPLETED.value)
    engine = test_client.app.state.openai_serving_video._engine_client
    captured = engine.captured_sampling_params_list[0]
    assert "is_enable_stage2" not in captured.extra_args


def test_extra_params_invalid_json(test_client):
    """Malformed JSON for extra_params returns 400."""
    response = test_client.post(
        "/v1/videos",
        data={
            "prompt": "A forest.",
            "extra_params": "{not valid json}",
        },
    )
    assert response.status_code == 400

    """extra_params must be a JSON object, not an array."""
    response = test_client.post(
        "/v1/videos",
        data={
            "prompt": "A desert.",
            "extra_params": json.dumps([1, 2, 3]),
        },
    )
    assert response.status_code == 400


def test_extra_params_merged_with_existing_extra_args(test_client, mocker: MockerFixture):
    """extra_params is merged on top of existing extra_args (e.g. flow_shift)."""
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        return_value=b"fake-video",
    )
    response = test_client.post(
        "/v1/videos",
        data={
            "prompt": "A mountain peak.",
            "flow_shift": "0.5",
            "extra_params": json.dumps({"use_zero_init": True, "zero_steps": 2}),
        },
    )

    assert response.status_code == 200
    video_id = response.json()["id"]
    _wait_for_status(test_client, video_id, VideoGenerationStatus.COMPLETED.value)
    engine = test_client.app.state.openai_serving_video._engine_client
    captured = engine.captured_sampling_params_list[0]
    assert captured.extra_args["flow_shift"] == 0.5
    assert captured.extra_args["use_zero_init"] is True
    assert captured.extra_args["zero_steps"] == 2


def test_sample_solver_forwarded_via_extra_params(test_client, mocker: MockerFixture):
    """sample_solver can be passed through existing extra_params for Wan2.2 online serving."""
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        return_value=b"fake-video",
    )
    response = test_client.post(
        "/v1/videos",
        data={
            "prompt": "A fox running through snow.",
            "extra_params": json.dumps({"sample_solver": "euler"}),
        },
    )

    assert response.status_code == 200
    video_id = response.json()["id"]
    _wait_for_status(test_client, video_id, VideoGenerationStatus.COMPLETED.value)
    engine = test_client.app.state.openai_serving_video._engine_client
    captured = engine.captured_sampling_params_list[0]
    assert captured.extra_args["sample_solver"] == "euler"


# ---------------------------------------------------------------------------
# Sync endpoint tests (POST /v1/videos/sync)
# ---------------------------------------------------------------------------


def _mock_encode_video_bytes(mocker: MockerFixture, return_value: bytes = b"fake-video-bytes"):
    """Mock the raw-bytes encoder used by the sync video path."""
    return mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        return_value=return_value,
    )


def test_sync_t2v_returns_video_bytes(test_client, mocker: MockerFixture):
    """Sync endpoint should block until generation finishes and return raw
    video bytes with metadata headers."""
    _mock_encode_video_bytes(mocker, b"fake-video-bytes")
    response = test_client.post(
        "/v1/videos/sync",
        data={
            "prompt": "A cat running across the street.",
            "size": "640x360",
            "seconds": "2",
            "fps": "12",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert response.content == b"fake-video-bytes"
    assert response.headers["x-request-id"].startswith("video_sync-")
    assert response.headers["x-model"] == "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    assert float(response.headers["x-inference-time-s"]) >= 0
    assert json.loads(response.headers["x-stage-durations"]) == {}
    assert float(response.headers["x-peak-memory-mb"]) == 0.0


def test_sync_t2v_returns_profiler_headers(test_client, mocker: MockerFixture):
    engine = test_client.app.state.openai_serving_video._engine_client

    async def _generate(prompt, request_id, sampling_params_list):
        engine.captured_prompt = prompt
        engine.captured_sampling_params_list = sampling_params_list
        yield MockVideoResult(
            [object()],
            stage_durations={"diffuse": 1.75},
            peak_memory_mb=1234.25,
        )

    engine.generate = _generate
    _mock_encode_video_bytes(mocker, b"profiled-video")

    response = test_client.post("/v1/videos/sync", data={"prompt": "sync profile"})

    assert response.status_code == 200
    assert response.content == b"profiled-video"
    assert json.loads(response.headers["x-stage-durations"]) == {"diffuse": 1.75}
    assert float(response.headers["x-peak-memory-mb"]) == pytest.approx(1234.25, rel=0, abs=1e-3)


def test_sync_i2v_returns_video_bytes(test_client, mocker: MockerFixture):
    """Sync I2V endpoint should accept an uploaded reference image and return
    raw video bytes."""
    image_bytes = _make_test_image_bytes((48, 32))
    _mock_encode_video_bytes(mocker, b"i2v-video-data")
    response = test_client.post(
        "/v1/videos/sync",
        data={"prompt": "A bear playing with yarn."},
        files={"input_reference": ("input.png", image_bytes, "image/png")},
    )

    assert response.status_code == 200
    assert response.content == b"i2v-video-data"
    assert response.headers["content-type"] == "video/mp4"


def test_sync_i2v_with_image_reference(test_client, mocker: MockerFixture):
    """Sync I2V endpoint should accept a JSON image_reference field."""
    _mock_encode_video_bytes(mocker, b"ref-video")
    response = test_client.post(
        "/v1/videos/sync",
        data={
            "prompt": "A fox running through snow.",
            "image_reference": json.dumps({"image_url": _make_test_image_data_url((40, 24))}),
        },
    )

    assert response.status_code == 200
    assert response.content == b"ref-video"


def test_sync_missing_handler_returns_503():
    app = FastAPI()
    app.include_router(router)
    app.state.openai_serving_video = None
    client = TestClient(app)

    response = client.post(
        "/v1/videos/sync",
        data={"prompt": "no handler"},
    )
    assert response.status_code == 503
    assert "not initialized" in response.json()["detail"].lower()


def test_sync_missing_prompt_returns_422(test_client):
    response = test_client.post(
        "/v1/videos/sync",
        data={"size": "320x240"},
    )
    assert response.status_code == 422


def test_sync_rejects_both_references(test_client):
    response = test_client.post(
        "/v1/videos/sync",
        data={
            "prompt": "bad refs",
            "image_reference": '{"image_url": "https://example.com/cat.png"}',
        },
        files={"input_reference": ("input.png", _make_test_image_bytes(), "image/png")},
    )
    assert response.status_code == 400
    assert "either input_reference or image_reference" in response.json()["detail"].lower()


def test_sync_generation_error_returns_500(test_client, mocker: MockerFixture):
    """If the underlying generation raises, the sync endpoint should return 500."""
    mocker.patch.object(
        OmniOpenAIServingVideo,
        "generate_video_bytes",
        side_effect=RuntimeError("GPU exploded"),
    )
    response = test_client.post(
        "/v1/videos/sync",
        data={"prompt": "will fail"},
    )
    assert response.status_code == 500
    assert "GPU exploded" in response.json()["detail"]


def test_sync_does_not_create_store_entry(test_client, mocker: MockerFixture):
    """The sync endpoint should NOT leave any record in VIDEO_STORE — it is
    stateless by design."""
    _mock_encode_video_bytes(mocker)
    response = test_client.post(
        "/v1/videos/sync",
        data={"prompt": "stateless test"},
    )
    assert response.status_code == 200
    loop = asyncio.new_event_loop()
    try:
        stored = loop.run_until_complete(api_server.VIDEO_STORE.list_values())
    finally:
        loop.close()
    assert len(stored) == 0


def test_sync_sampling_params_pass_through(test_client, mocker: MockerFixture):
    """Sampling parameters should propagate to the engine through the sync path."""
    _mock_encode_video_bytes(mocker)
    response = test_client.post(
        "/v1/videos/sync",
        data={
            "prompt": "param pass",
            "num_inference_steps": "30",
            "guidance_scale": "6.5",
            "seed": "42",
        },
    )
    assert response.status_code == 200
    engine = test_client.app.state.openai_serving_video._engine_client
    captured = engine.captured_sampling_params_list[0]
    assert captured.num_inference_steps == 30
    assert captured.guidance_scale == 6.5
    assert captured.seed == 42


def test_sync_frame_interpolation_params_pass_to_sampling_params(test_client, mocker: MockerFixture):
    """Frame interpolation parameters should be forwarded on the sync path."""
    encode_mock = _mock_encode_video_bytes(mocker)
    response = test_client.post(
        "/v1/videos/sync",
        data={
            "prompt": "smooth sync",
            "fps": "8",
            "enable_frame_interpolation": "true",
            "frame_interpolation_exp": "2",
            "frame_interpolation_scale": "0.5",
            "frame_interpolation_model_path": "local-rife",
        },
    )

    assert response.status_code == 200
    engine = test_client.app.state.openai_serving_video._engine_client
    captured = engine.captured_sampling_params_list[0]
    assert captured.enable_frame_interpolation is True
    assert captured.frame_interpolation_exp == 2
    assert captured.frame_interpolation_scale == 0.5
    assert captured.frame_interpolation_model_path == "local-rife"
    _, kwargs = encode_mock.call_args
    assert kwargs["fps"] == 8


def test_sync_default_sampling_params_apply_to_video_requests(test_client, mocker: MockerFixture):
    _mock_encode_video_bytes(mocker)
    engine = test_client.app.state.openai_serving_video._engine_client
    engine.default_sampling_params_list = [
        OmniDiffusionSamplingParams(
            num_inference_steps=4,
            guidance_scale=7.5,
            enable_frame_interpolation=True,
            frame_interpolation_exp=2,
            frame_interpolation_scale=0.5,
            frame_interpolation_model_path="default-rife",
        )
    ]

    response = test_client.post(
        "/v1/videos/sync",
        data={
            "prompt": "sync default param pass-through",
            "fps": "8",
        },
    )

    assert response.status_code == 200
    engine = test_client.app.state.openai_serving_video._engine_client
    captured = engine.captured_sampling_params_list[0]
    assert captured.num_inference_steps == 4
    assert captured.guidance_scale == 7.5
    assert captured.enable_frame_interpolation is True
    assert captured.frame_interpolation_exp == 2
    assert captured.frame_interpolation_scale == 0.5
    assert captured.frame_interpolation_model_path == "default-rife"


def test_worker_fps_multiplier_is_applied_to_sync_encoding(test_client, mocker: MockerFixture):
    engine = test_client.app.state.openai_serving_video._engine_client
    fps_values = []

    async def _generate(prompt, request_id, sampling_params_list):
        engine.captured_prompt = prompt
        engine.captured_sampling_params_list = sampling_params_list
        yield MockVideoResult([object()], custom_output={"video_fps_multiplier": 2})

    engine.generate = _generate

    def _fake_encode(video, fps, **kwargs):
        del video, kwargs
        fps_values.append(fps)
        return b"fps-multiplied"

    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        side_effect=_fake_encode,
    )

    response = test_client.post("/v1/videos/sync", data={"prompt": "fps multiplier", "fps": "8"})

    assert response.status_code == 200
    assert response.content == b"fps-multiplied"
    assert fps_values == [16]
