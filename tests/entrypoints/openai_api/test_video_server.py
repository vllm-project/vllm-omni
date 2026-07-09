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
from contextlib import asynccontextmanager
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from pytest_mock import MockerFixture

from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes
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
from vllm_omni.errors import GuardrailViolationError
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

    async def generate_video_bytes(
        self, request, reference_id, *, reference_image=None, reference_video=None, reference_audio=None
    ):
        del request, reference_id, reference_image, reference_video, reference_audio
        self.started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class FakeServerSocket:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


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


@pytest.mark.asyncio
async def test_server_worker_keeps_engine_alive_until_http_shutdown(monkeypatch):
    events: list[str] = []
    serve_started = asyncio.Event()
    http_shutdown = asyncio.Event()
    engine_context_exited = asyncio.Event()
    sock = FakeServerSocket()

    class FakeEngine:
        stage_configs = []

        async def get_supported_tasks(self):
            return ("generate",)

    @asynccontextmanager
    async def fake_build_async_omni(*args, **kwargs):
        del args, kwargs
        events.append("engine_enter")
        try:
            yield FakeEngine()
        finally:
            events.append("engine_exit")
            engine_context_exited.set()

    async def fake_serve_http(*args, **kwargs):
        del args, kwargs
        events.append("serve_http")
        serve_started.set()

        async def wait_for_shutdown():
            await http_shutdown.wait()
            events.append("http_shutdown")

        return asyncio.create_task(wait_for_shutdown())

    async def fake_storage_start():
        events.append("storage_start")

    async def fake_get_vllm_config(engine_client):
        del engine_client
        return None

    async def fake_init_app_state(engine_client, state, args):
        del engine_client, state, args
        events.append("init_app_state")

    monkeypatch.setattr(api_server, "build_async_omni", fake_build_async_omni)
    monkeypatch.setattr(api_server, "build_openai_app", lambda args, supported_tasks: FastAPI())
    monkeypatch.setattr(api_server, "serve_http", fake_serve_http)
    monkeypatch.setattr(api_server.STORAGE_MANAGER, "start", fake_storage_start)
    monkeypatch.setattr(api_server, "_get_vllm_config", fake_get_vllm_config)
    monkeypatch.setattr(api_server, "omni_init_app_state", fake_init_app_state)
    monkeypatch.setattr(api_server, "get_uvicorn_log_config", lambda args: None)

    args = SimpleNamespace(
        tool_parser_plugin="",
        reasoning_parser_plugin="",
        reasoning_parser=None,
        structured_outputs_config=SimpleNamespace(reasoning_parser=None),
        enable_ssl_refresh=False,
        host="127.0.0.1",
        port=0,
        uvicorn_log_level="info",
        disable_uvicorn_access_log=True,
        ssl_keyfile=None,
        ssl_certfile=None,
        ssl_ca_certs=None,
        ssl_cert_reqs=None,
        ssl_ciphers=None,
        h11_max_incomplete_event_size=None,
        h11_max_header_count=None,
    )

    worker_task = asyncio.create_task(api_server.omni_run_server_worker("127.0.0.1:0", sock, args))
    await asyncio.wait_for(serve_started.wait(), timeout=2)

    assert not engine_context_exited.is_set()

    http_shutdown.set()
    await asyncio.wait_for(worker_task, timeout=2)

    assert sock.closed
    assert events.index("http_shutdown") < events.index("engine_exit")


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


def _make_test_video_bytes(size=(32, 24), num_frames=3) -> bytes:
    width, height = size
    frames = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
    for idx in range(num_frames):
        frames[idx, :, :, 0] = idx * 40
        frames[idx, :, :, 1] = 128
        frames[idx, :, :, 2] = 255 - idx * 40
    return mux_video_audio_bytes(frames, fps=8, video_codec_options={"preset": "ultrafast", "threads": "0"})


def _make_test_video_data_url(size=(32, 24), num_frames=3) -> str:
    encoded = base64.b64encode(_make_test_video_bytes(size, num_frames)).decode("utf-8")
    return f"data:video/mp4;base64,{encoded}"


def _cosmos3_stage_configs():
    return [
        SimpleNamespace(
            stage_type="diffusion",
            engine_args=SimpleNamespace(model_class_name="Cosmos3OmniDiffusersPipeline"),
        )
    ]


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
    assert engine.captured_prompt["modalities"] == ["video"]
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


def test_v2v_video_generation_form(test_client, mocker: MockerFixture):
    video_bytes = _make_test_video_bytes((32, 24), num_frames=3)

    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        return_value=b"fake-video",
    )
    response = test_client.post(
        "/v1/videos",
        data={"prompt": "Continue this motion."},
        files={"input_reference": ("input.mp4", video_bytes, "video/mp4")},
    )

    assert response.status_code == 200
    video_id = response.json()["id"]
    _wait_for_status(test_client, video_id, VideoGenerationStatus.COMPLETED.value)

    engine = test_client.app.state.openai_serving_video._engine_client
    prompt = engine.captured_prompt
    assert "multi_modal_data" in prompt
    assert "video" in prompt["multi_modal_data"]
    input_video = prompt["multi_modal_data"]["video"]
    assert len(input_video) == 3
    assert all(isinstance(frame, Image.Image) for frame in input_video)
    assert input_video[0].size == (32, 24)


def test_v2v_video_generation_with_video_reference_form(test_client, mocker: MockerFixture):
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        return_value=b"fake-video",
    )
    response = test_client.post(
        "/v1/videos",
        data={
            "prompt": "Continue this motion.",
            "video_reference": json.dumps({"video_url": _make_test_video_data_url((32, 24), 2)}),
        },
    )

    assert response.status_code == 200
    video_id = response.json()["id"]
    _wait_for_status(test_client, video_id, VideoGenerationStatus.COMPLETED.value)

    engine = test_client.app.state.openai_serving_video._engine_client
    input_video = engine.captured_prompt["multi_modal_data"]["video"]
    assert len(input_video) == 2
    assert input_video[0].size == (32, 24)


def test_decode_video_bytes_can_keep_last_frames():
    from vllm_omni.entrypoints.openai.video_api_utils import _decode_video_bytes

    frames = _decode_video_bytes(
        _make_test_video_bytes((32, 24), num_frames=6),
        source="input_reference",
        max_frames=2,
        keep="last",
    )

    assert len(frames) == 2
    assert frames.fps == pytest.approx(8.0)
    red_means = [np.asarray(frame)[:, :, 0].mean() for frame in frames]
    assert red_means[0] > 100
    assert red_means[1] > red_means[0]


def test_cosmos3_reference_video_limit_uses_v2v_condition_frames():
    request = VideoGenerationRequest(
        prompt="Continue this motion.",
        num_frames=189,
        extra_params={"condition_frame_indexes_vision": [0, 2]},
    )

    spec = api_server._reference_video_decode_spec(request, _cosmos3_stage_configs())
    assert spec.max_frames == 9
    assert spec.keep == "first"


def test_cosmos3_reference_video_limit_preserves_action_frames():
    request = VideoGenerationRequest(
        prompt="Predict the action.",
        num_frames=17,
        extra_params={"action_mode": "inverse_dynamics", "action_chunk_size": 16},
    )

    assert api_server._reference_video_decode_spec(request, _cosmos3_stage_configs()).max_frames == 17


def test_cosmos3_reference_video_limit_caps_condition_frames_to_output_frames():
    request = VideoGenerationRequest(
        prompt="Continue this motion.",
        num_frames=5,
        extra_params={"condition_frame_indexes_vision": [0, 20]},
    )

    assert api_server._reference_video_decode_spec(request, _cosmos3_stage_configs()).max_frames == 5


def test_s2v_video_generation_with_audio_reference_form(test_client, mocker: MockerFixture):
    """Speech-to-video: image + audio_reference (base64 data URL) passes audio path to multi_modal_data."""
    audio_bytes = b"\xff\xfb\x90\x00" * 50
    audio_b64 = base64.b64encode(audio_bytes).decode()
    audio_ref = json.dumps({"audio_url": f"data:audio/mp3;base64,{audio_b64}"})

    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        return_value=b"fake-video",
    )
    response = test_client.post(
        "/v1/videos",
        data={
            "prompt": "A person singing",
            "audio_reference": audio_ref,
            "width": "832",
            "height": "480",
        },
        files={"input_reference": ("face.png", _make_test_image_bytes((64, 64)), "image/png")},
    )

    assert response.status_code == 200
    video_id = response.json()["id"]
    _wait_for_status(test_client, video_id, VideoGenerationStatus.COMPLETED.value)

    engine = test_client.app.state.openai_serving_video._engine_client
    prompt = engine.captured_prompt
    assert "multi_modal_data" in prompt
    assert "image" in prompt["multi_modal_data"]
    assert "audio" in prompt["multi_modal_data"]
    audio_path = prompt["multi_modal_data"]["audio"]
    assert isinstance(audio_path, str)
    assert audio_path.endswith(".mp3")


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
    # fps omitted -> sampling params carry None (the "not provided" signal); the 24
    # default is applied only at output encoding.
    assert captured.fps is None
    assert captured.frame_rate is None
    assert fps_values == [24]


def test_model_reported_fps_wins_when_request_fps_omitted(test_client, mocker: MockerFixture):
    fps_values = []

    def _fake_encode(video, fps, audio=None, audio_sample_rate=None, **kwargs):
        del video, audio, audio_sample_rate, kwargs
        fps_values.append(fps)
        return b"fake-video"

    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        side_effect=_fake_encode,
    )

    engine = test_client.app.state.openai_serving_video._engine_client

    async def _generate(prompt, request_id, sampling_params_list):
        engine.captured_prompt = prompt
        engine.captured_sampling_params_list = sampling_params_list
        result = MockVideoResult([object()])
        result.multimodal_output["fps"] = 8
        yield result

    engine.generate = _generate

    response = test_client.post("/v1/videos", data={"prompt": "source fps"})

    assert response.status_code == 200
    video_id = response.json()["id"]
    _wait_for_status(test_client, video_id, VideoGenerationStatus.COMPLETED.value)
    captured = engine.captured_sampling_params_list[0]
    # fps omitted -> None on the sampling params; the model-reported fps (8) wins for output.
    assert captured.fps is None
    assert captured.frame_rate is None
    assert fps_values == [8]


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
            "generate_sound": "true",
            "sound_duration": "2.5",
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
    assert captured.extra_args["generate_sound"] is True
    assert captured.extra_args["sound_duration"] == 2.5


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
    assert completed["action"] is None


def test_video_generation_response_exposes_action_payload(mocker: MockerFixture):
    engine = FakeAsyncOmni()
    handler = OmniOpenAIServingVideo.for_diffusion(
        diffusion_engine=engine,
        model_name="Cosmos3-8B-UVA",
    )

    async def _generate(prompt, request_id, sampling_params_list):
        del prompt, request_id, sampling_params_list
        import numpy as np

        yield MockVideoResult(
            [object()],
            custom_output={
                "action": np.array([[[1.5, 2.5], [3.5, 4.5]]], dtype=np.float32),
                "raw_action_dim": 2,
                "action_mode": "policy",
                "domain_id": 7,
            },
        )

    engine.generate = _generate
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video.encode_video_base64",
        return_value="encoded-video",
    )

    response = asyncio.run(
        handler.generate_videos(
            VideoGenerationRequest(prompt="predict actions"),
            "action-json",
        )
    )

    action = response.data[0].action
    assert action is not None
    assert action.data == [[1.5, 2.5], [3.5, 4.5]]
    assert action.shape == [2, 2]
    assert action.dtype == "float32"
    assert action.raw_action_dim == 2
    assert action.action_mode == "policy"
    assert action.domain_id == 7
    assert response.model_dump(mode="json")["data"][0]["action"]["data"] == [[1.5, 2.5], [3.5, 4.5]]


def test_video_job_persists_action_metadata(test_client, mocker: MockerFixture):
    engine = test_client.app.state.openai_serving_video._engine_client

    async def _generate(prompt, request_id, sampling_params_list):
        import numpy as np

        engine.captured_prompt = prompt
        engine.captured_sampling_params_list = sampling_params_list
        yield MockVideoResult(
            [object()],
            custom_output={
                "action": np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32),
                "raw_action_dim": 2,
                "action_mode": "policy",
                "domain_id": 7,
            },
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

    expected_action = {
        "data": [[1.0, 2.0], [3.0, 4.0]],
        "shape": [2, 2],
        "dtype": "float32",
        "raw_action_dim": 2,
        "action_mode": "policy",
        "domain_id": 7,
    }
    assert completed["action"] == expected_action

    listed = test_client.get("/v1/videos").json()
    assert listed["data"][0]["action"] == expected_action


def test_action_extraction_accepts_unbatched_action():
    import numpy as np

    result = MockVideoResult(
        [object()],
        custom_output={
            "action": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "raw_action_dim": 2,
            "action_mode": "policy",
            "domain_id": 7,
        },
    )

    actions = OmniOpenAIServingVideo._extract_action_outputs(result, expected_count=1)

    assert actions[0] is not None
    assert actions[0].data == [[1.0, 2.0], [3.0, 4.0]]
    assert actions[0].shape == [2, 2]


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
    assert "only one of input_reference, image_reference, or video_reference" in response.json()["detail"].lower()


def test_rejects_image_reference_and_video_reference_together(test_client):
    response = test_client.post(
        "/v1/videos",
        data={
            "prompt": "bad refs",
            "image_reference": '{"image_url": "https://example.com/cat.png"}',
            "video_reference": '{"video_url": "https://example.com/cat.mp4"}',
        },
    )
    assert response.status_code == 400
    assert "only one of input_reference, image_reference, or video_reference" in response.json()["detail"].lower()


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


def test_failed_generation_awaits_storage_cleanup(test_client, isolated_video_backends, mocker: MockerFixture):
    """Regression (merge seam): when async generation raises, the failure handler
    must ``await _cleanup_video(video_id)`` (single-arg, async) and still record
    FAILED. Upstream carried a sync ``_cleanup_video(video_id, output_path)`` whose
    stale call in the generic handler sat outside the conflict markers; against the
    PR's async storage manager that raised NameError before the FAILED update,
    wedging the job in IN_PROGRESS and orphaning the artifact."""
    _store, _tasks, storage = isolated_video_backends
    delete_spy = mocker.spy(storage, "delete")
    mocker.patch.object(
        OmniOpenAIServingVideo,
        "generate_video_bytes",
        side_effect=RuntimeError("GPU exploded"),
    )

    response = test_client.post("/v1/videos", data={"prompt": "will fail"})
    assert response.status_code == 200
    video_id = response.json()["id"]

    failed = _wait_for_status(test_client, video_id, VideoGenerationStatus.FAILED.value)
    assert failed["error"]["code"] == 500
    assert "GPU exploded" in failed["error"]["message"]
    delete_spy.assert_called_once_with(video_id)


def test_async_guardrail_error_returns_400_on_retrieve(test_client, mocker: MockerFixture):
    mocker.patch.object(
        OmniOpenAIServingVideo,
        "generate_video_bytes",
        side_effect=GuardrailViolationError("Input was blocked by Cosmos3 guardrails."),
    )
    response = test_client.post("/v1/videos", data={"prompt": "blocked prompt"})
    assert response.status_code == 200

    video_id = response.json()["id"]
    failed = _wait_for_status(test_client, video_id, VideoGenerationStatus.FAILED.value)
    assert failed["error"]["code"] == 400
    assert failed["error"]["message"] == "Input was blocked by Cosmos3 guardrails."

    retrieve = test_client.get(f"/v1/videos/{video_id}")
    assert retrieve.status_code == 400
    assert retrieve.json()["error"]["code"] == 400


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


def test_unsupported_video_reference_file_id_returns_400(test_client):
    response = test_client.post(
        "/v1/videos",
        data={
            "prompt": "unsupported ref",
            "video_reference": '{"file_id": "file-123"}',
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid video_reference: file_id is not supported yet."


def test_invalid_uploaded_input_reference_returns_400(test_client):
    response = test_client.post(
        "/v1/videos",
        data={"prompt": "bad upload"},
        files={"input_reference": ("input.png", b"not-an-image", "image/png")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid input_reference: provided content is not a valid image or video."


def test_video_request_validation():
    req = VideoGenerationRequest(prompt="test")
    assert req.prompt == "test"
    assert req.generate_sound is False
    assert req.sound_duration is None
    assert VideoGenerationRequest(prompt="test", generate_sound=True, sound_duration=1.5).generate_sound is True
    with pytest.raises(ValueError):
        VideoGenerationRequest(prompt="test", size="invalid")

    with pytest.raises(ValueError):
        VideoGenerationRequest(prompt="test", seconds="abc")

    with pytest.raises(ValueError):
        VideoGenerationRequest(prompt="test", image_reference={"file_id": "file-1", "image_url": "https://example.com"})
    with pytest.raises(ValueError):
        VideoGenerationRequest(prompt="test", video_reference={"file_id": "file-1", "video_url": "https://example.com"})
    with pytest.raises(ValueError):
        VideoGenerationRequest(prompt="test", frame_interpolation_exp=0)
    with pytest.raises(ValueError):
        VideoGenerationRequest(prompt="test", frame_interpolation_scale=0)
    with pytest.raises(ValueError):
        VideoGenerationRequest(prompt="test", sound_duration=0)


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
    file_path = os.path.join(api_server.STORAGE_MANAGER.storage_path, video_id)
    assert os.path.exists(file_path)

    delete_resp = test_client.delete(f"/v1/videos/{video_id}")
    assert delete_resp.status_code == 200
    assert delete_resp.json()["id"] == video_id
    assert delete_resp.json()["deleted"] is True
    assert delete_resp.json()["object"] == "video.deleted"
    assert not os.path.exists(file_path)


def test_download_completed_job_uses_storage_open_and_download_name(test_client, mocker: MockerFixture):
    video_bytes = b"stored-video-data"
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        return_value=video_bytes,
    )
    create_resp = test_client.post("/v1/videos", data={"prompt": "Download this video"})
    assert create_resp.status_code == 200
    video_id = create_resp.json()["id"]

    final = _wait_for_status(test_client, video_id, VideoGenerationStatus.COMPLETED.value)
    file_name = final["file_name"]
    assert file_name == f"{video_id}.mp4"

    storage_path = os.path.join(api_server.STORAGE_MANAGER.storage_path, video_id)
    assert os.path.exists(storage_path)

    response = test_client.get(f"/v1/videos/{video_id}/content")
    assert response.status_code == 200
    assert response.content == video_bytes
    assert response.headers["content-type"] == "video/mp4"
    assert file_name in response.headers["content-disposition"]


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
        "pyramid_num_inference_steps_list": [1, 1, 1],
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
    assert captured.extra_args["pyramid_num_inference_steps_list"] == [1, 1, 1]
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


def test_extra_params_allows_inline_action(test_client, mocker: MockerFixture):
    """Inline ``action`` data is accepted and forwarded verbatim to
    ``extra_args`` (the supported way to pass forward-dynamics actions)."""
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        return_value=b"fake-video",
    )
    action = [[0.1, 0.2], [0.3, 0.4]]
    response = test_client.post(
        "/v1/videos",
        data={
            "prompt": "forward dynamics inline",
            "extra_params": json.dumps({"action_mode": "forward_dynamics", "action": action}),
        },
    )

    assert response.status_code == 200
    video_id = response.json()["id"]
    _wait_for_status(test_client, video_id, VideoGenerationStatus.COMPLETED.value)
    engine = test_client.app.state.openai_serving_video._engine_client
    captured = engine.captured_sampling_params_list[0]
    assert captured.extra_args["action"] == action
    assert captured.extra_args["action_mode"] == "forward_dynamics"


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
    engine = test_client.app.state.openai_serving_video._engine_client
    assert engine.captured_prompt["modalities"] == ["video"]


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


def test_sync_v2v_returns_video_bytes(test_client, mocker: MockerFixture):
    video_bytes = _make_test_video_bytes((32, 24), num_frames=3)
    _mock_encode_video_bytes(mocker, b"v2v-video-data")
    response = test_client.post(
        "/v1/videos/sync",
        data={"prompt": "Continue this motion."},
        files={"input_reference": ("input.mp4", video_bytes, "video/mp4")},
    )

    assert response.status_code == 200
    assert response.content == b"v2v-video-data"
    engine = test_client.app.state.openai_serving_video._engine_client
    input_video = engine.captured_prompt["multi_modal_data"]["video"]
    assert len(input_video) == 3
    assert input_video[0].size == (32, 24)


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
    assert "only one of input_reference, image_reference, or video_reference" in response.json()["detail"].lower()


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


def test_sync_guardrail_error_returns_400(test_client, mocker: MockerFixture):
    mocker.patch.object(
        OmniOpenAIServingVideo,
        "generate_video_bytes",
        side_effect=GuardrailViolationError("Input was blocked by Cosmos3 guardrails."),
    )
    response = test_client.post(
        "/v1/videos/sync",
        data={"prompt": "blocked prompt"},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Input was blocked by Cosmos3 guardrails."


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
