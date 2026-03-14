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

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class MockVideoResult:
    def __init__(self, videos, audios=None, sample_rate=None):
        self.multimodal_output = {"video": videos}
        if audios is not None:
            self.multimodal_output["audio"] = audios
        if sample_rate is not None:
            self.multimodal_output["audio_sample_rate"] = sample_rate


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


class BlockingVideoHandler:
    def __init__(self):
        self.model_name = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        self.stage_configs = None
        self.started = threading.Event()
        self.cancelled = threading.Event()

    def set_stage_configs_if_missing(self, stage_configs):
        if self.stage_configs is None:
            self.stage_configs = stage_configs

    async def generate_videos(self, request, reference_id, *, reference_image=None):
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
        assert response.status_code == 200
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


def test_t2v_video_generation_form(test_client, mocker: MockerFixture):
    fps_values = []

    def _fake_encode(video, fps):
        fps_values.append(fps)
        return "Zg=="

    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video.encode_video_base64",
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
        "vllm_omni.entrypoints.openai.serving_video.encode_video_base64",
        return_value="Zg==",
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
        "vllm_omni.entrypoints.openai.serving_video.encode_video_base64",
        return_value="Zg==",
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
        "vllm_omni.entrypoints.openai.serving_video.encode_video_base64",
        return_value="Zg==",
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

    def _fake_encode(video, fps):
        fps_values.append(fps)
        return "Zg=="

    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video.encode_video_base64",
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
        "vllm_omni.entrypoints.openai.serving_video.encode_video_base64",
        return_value="Zg==",
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
        "vllm_omni.entrypoints.openai.serving_video.encode_video_base64",
        return_value="Zg==",
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


def test_audio_sample_rate_comes_from_model_config(test_client, mocker: MockerFixture):
    audio_sample_rates = []

    def _fake_encode(video, fps, audio=None, audio_sample_rate=None):
        del video, fps, audio
        audio_sample_rates.append(audio_sample_rate)
        return "Zg=="

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
        yield MockVideoResult([object()], audios=[object()])

    engine.generate = _generate

    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video.encode_video_base64",
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
        "vllm_omni.entrypoints.openai.serving_video.encode_video_base64",
        return_value="Zg==",
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
    assert failed["error"]["code"] == "HTTPException"
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


def test_list_videos_supports_order_after_and_limit(test_client, mocker: MockerFixture):
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video.encode_video_base64",
        return_value="Zg==",
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
        "vllm_omni.entrypoints.openai.serving_video.encode_video_base64",
        return_value="Zg==",
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
