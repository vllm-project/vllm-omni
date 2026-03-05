# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for the async OpenAI-compatible video generation API endpoints.
"""

import io
from types import SimpleNamespace
import asyncio
import base64
import os
import time
from contextlib import contextmanager
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_omni.entrypoints.openai import api_server
from vllm_omni.entrypoints.openai.api_server import router
from vllm_omni.entrypoints.openai.protocol.videos import (
    VideoData,
    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoGenerationStatus,
    VideoResponse,
    VideoResponseFormat,
)
from vllm_omni.entrypoints.openai.serving_video import OmniOpenAIServingVideo
from vllm_omni.entrypoints.openai.storage import LocalStorageManager
from vllm_omni.entrypoints.openai.stores import AsyncDictStore

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class DeterministicVideoHandler:
    """Test double that behaves like an async video generation handler."""

    def __init__(
        self,
        output_payloads: list[bytes],
        *,
        delay_s: float = 0.0,
        fail_exc: Exception | None = None,
        model_name: str = "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    ) -> None:
        self._output_payloads = output_payloads
        self._delay_s = delay_s
        self._fail_exc = fail_exc
        self._model_name = model_name
        self._stage_configs: list[Any] | None = None

    @property
    def model_name(self) -> str | None:
        return self._model_name

    @property
    def stage_configs(self) -> list[Any] | None:
        return self._stage_configs

    def set_stage_configs_if_missing(self, stage_configs: list[Any] | None) -> None:
        if self._stage_configs is None and stage_configs is not None:
            self._stage_configs = stage_configs

    async def generate_videos(
        self,
        request: VideoGenerationRequest,
        reference_id: str,
        *,
        input_reference_bytes: bytes | None = None,
    ) -> VideoGenerationResponse:
        del request, reference_id, input_reference_bytes
        if self._delay_s > 0:
            await asyncio.sleep(self._delay_s)
        if self._fail_exc is not None:
            raise self._fail_exc

        data = [VideoData(b64_json=base64.b64encode(payload).decode("utf-8")) for payload in self._output_payloads]
        return VideoGenerationResponse(created=int(time.time()), data=data)


class MockVideoResult:
    def __init__(self, videos, audios=None, sample_rate=None):
        self.multimodal_output = {"video": videos}
        if audios is not None:
            self.multimodal_output["audio"] = audios
        if sample_rate is not None:
            self.multimodal_output["audio_sample_rate"] = sample_rate


class FakeAsyncOmni:
    """Minimal async engine used by the real serving handler in tests."""

    def __init__(self) -> None:
        self.stage_list = ["diffusion"]

    async def generate(self, prompt, request_id, sampling_params_list):
        del prompt, request_id
        num_outputs = sampling_params_list[0].num_outputs_per_prompt
        yield MockVideoResult([object() for _ in range(num_outputs)])


@pytest.fixture(autouse=True)
def isolated_video_backends(tmp_path, monkeypatch):
    """Use isolated in-memory metadata and local storage for each test."""
    store: AsyncDictStore[VideoResponse] = AsyncDictStore()
    storage = LocalStorageManager(storage_path=str(tmp_path / "storage"))
    monkeypatch.setattr(api_server, "VIDEO_STORE", store)
    monkeypatch.setattr(api_server, "STORAGE_MANAGER", storage)
    return store, storage


@contextmanager
def make_client(handler: Any):
    app = FastAPI()
    app.include_router(router)
    app.state.openai_serving_video = handler
    app.state.stage_configs = [{"stage_type": "diffusion"}]
    with TestClient(app) as client:
        yield client


def wait_for_status(client: TestClient, video_id: str, status: str, timeout_s: float = 2.0) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last_payload: dict[str, Any] | None = None
    while time.time() < deadline:
        response = client.get(f"/v1/videos/{video_id}")
        assert response.status_code == 200
        last_payload = response.json()
        if last_payload["status"] == status:
            return last_payload
        time.sleep(0.02)
    raise AssertionError(f"Timed out waiting for status={status}. Last payload: {last_payload}")


# E2E: Online serving lifecycle tests for async video API.


def test_create_video_returns_queued_reference() -> None:
    """POST /v1/videos returns a queued reference without payload data."""
    handler = DeterministicVideoHandler([b"video-0"], delay_s=0.1)
    with make_client(handler) as client:
        response = client.post("/v1/videos", data={"prompt": "A cat running."})

    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "video"
    assert data["status"] == VideoGenerationStatus.QUEUED.value
    assert data["id"].startswith("video_gen_")
    assert "data" not in data


def test_job_completes_and_persists_file() -> None:
    """Completed jobs persist a file that can be downloaded via /content."""
    expected_bytes = b"video-content-0"
    handler = DeterministicVideoHandler([expected_bytes])
    with make_client(handler) as client:
        create_resp = client.post("/v1/videos", data={"prompt": "A bear playing with yarn."})
        assert create_resp.status_code == 200
        video_id = create_resp.json()["id"]

        final = wait_for_status(client, video_id, VideoGenerationStatus.COMPLETED.value)
        saved_path = final["file_path"]
        assert os.path.exists(saved_path)

        with open(saved_path, "rb") as f:
            assert f.read() == expected_bytes

        content_resp = client.get(f"/v1/videos/{video_id}/content")
        assert content_resp.status_code == 200
        assert content_resp.content == expected_bytes
        assert content_resp.headers["content-type"].startswith("video/mp4")


def test_job_uses_first_output_when_handler_returns_multiple_payloads() -> None:
    """Single-output API persists the first generated payload."""
    payloads = [b"video-content-0", b"video-content-1"]
    handler = DeterministicVideoHandler(payloads)
    with make_client(handler) as client:
        create_resp = client.post("/v1/videos", data={"prompt": "Two videos"})
        assert create_resp.status_code == 200
        video_id = create_resp.json()["id"]

        final = wait_for_status(client, video_id, VideoGenerationStatus.COMPLETED.value)
        assert final["file_path"].endswith(".mp4")

        content_resp = client.get(f"/v1/videos/{video_id}/content")
        assert content_resp.status_code == 200
        assert content_resp.content == payloads[0]


def test_download_while_in_progress_then_complete() -> None:
    """Content endpoint returns in-progress error before job completion."""
    handler = DeterministicVideoHandler([b"slow-video"], delay_s=0.25)
    with make_client(handler) as client:
        create_resp = client.post("/v1/videos", data={"prompt": "Slow generation"})
        assert create_resp.status_code == 200
        video_id = create_resp.json()["id"]

        early_content = client.get(f"/v1/videos/{video_id}/content")
        assert early_content.status_code == 404
        assert "in-progress" in early_content.json()["detail"].lower()

        wait_for_status(client, video_id, VideoGenerationStatus.COMPLETED.value)
        final_content = client.get(f"/v1/videos/{video_id}/content")
        assert final_content.status_code == 200
        assert final_content.content == b"slow-video"


def test_list_videos_supports_order_after_and_limit() -> None:
    """List endpoint supports deterministic ordering, after cursor, and limit."""
    handler = DeterministicVideoHandler([b"video"])
    with make_client(handler) as client:
        ids = []
        for i in range(3):
            create_resp = client.post("/v1/videos", data={"prompt": f"video-{i}"})
            assert create_resp.status_code == 200
            video_id = create_resp.json()["id"]
            wait_for_status(client, video_id, VideoGenerationStatus.COMPLETED.value)
            ids.append(video_id)

        # Normalize ordering deterministically for pagination assertions.
        asyncio.run(api_server.VIDEO_STORE.update_fields(ids[0], {"created_at": 100}))
        asyncio.run(api_server.VIDEO_STORE.update_fields(ids[1], {"created_at": 200}))
        asyncio.run(api_server.VIDEO_STORE.update_fields(ids[2], {"created_at": 300}))

        asc_resp = client.get("/v1/videos", params={"order": "asc"})
        assert asc_resp.status_code == 200
        asc_body = asc_resp.json()
        assert asc_body["object"] == "list"
        asc_ids = [item["id"] for item in asc_body["data"]]
        assert asc_ids == [ids[0], ids[1], ids[2]]

        desc_resp = client.get("/v1/videos", params={"order": "desc", "limit": 2})
        assert desc_resp.status_code == 200
        desc_body = desc_resp.json()
        assert desc_body["object"] == "list"
        desc_ids = [item["id"] for item in desc_body["data"]]
        assert desc_ids == [ids[2], ids[1]]

        after_resp = client.get("/v1/videos", params={"order": "asc", "after": ids[0]})
        assert after_resp.status_code == 200
        after_body = after_resp.json()
        assert after_body["object"] == "list"
        after_ids = [item["id"] for item in after_body["data"]]
        assert after_ids == [ids[1], ids[2]]


def test_delete_completed_job_removes_file_and_metadata() -> None:
    """Delete removes persisted files and metadata for completed jobs."""
    payload = b"delete-me"
    handler = DeterministicVideoHandler([payload])
    with make_client(handler) as client:
        create_resp = client.post("/v1/videos", data={"prompt": "Delete this video"})
        assert create_resp.status_code == 200
        video_id = create_resp.json()["id"]

        final = wait_for_status(client, video_id, VideoGenerationStatus.COMPLETED.value)
        file_path = final["file_path"]
        assert os.path.exists(file_path)

        delete_resp = client.delete(f"/v1/videos/{video_id}")
        assert delete_resp.status_code == 200
        assert delete_resp.json()["id"] == video_id
        assert delete_resp.json()["deleted"] is True
        assert delete_resp.json()["object"] == "object.deleted"
        assert not os.path.exists(file_path)
        assert client.get(f"/v1/videos/{video_id}").status_code == 404


def test_invalid_size_becomes_failed_job() -> None:
    """Invalid size requests transition the async job to failed state."""
    handler = OmniOpenAIServingVideo.for_diffusion(
        diffusion_engine=FakeAsyncOmni(),
        model_name="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    )
    with make_client(handler) as client:
        create_resp = client.post("/v1/videos", data={"prompt": "bad size", "size": "640x"})
        assert create_resp.status_code == 200
        video_id = create_resp.json()["id"]

        failed = wait_for_status(client, video_id, VideoGenerationStatus.FAILED.value)
        assert failed["error"]["type"] == "ValueError"
        assert "invalid size format" in failed["error"]["message"].lower()


def test_invalid_lora_becomes_failed_job() -> None:
    """Malformed LoRA payloads transition the async job to failed state."""
    handler = OmniOpenAIServingVideo.for_diffusion(
        diffusion_engine=FakeAsyncOmni(),
        model_name="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    )
    with make_client(handler) as client:
        create_resp = client.post(
            "/v1/videos",
            data={"prompt": "lora test", "lora": '{"name": "bad-lora"}'},
        )
        assert create_resp.status_code == 200
        video_id = create_resp.json()["id"]

<<<<<<< HEAD
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
    assert captured.frame_rate == 12.0
    assert fps_values == [12, 12]


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

    engine = test_client.app.state.openai_serving_video._engine_client
    prompt = engine.captured_prompt
    input_image = prompt["multi_modal_data"]["image"]
    assert isinstance(input_image, Image.Image)
    assert input_image.size == (96, 64)


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
    engine = test_client.app.state.openai_serving_video._engine_client
    captured = engine.captured_sampling_params_list[0]
    assert captured.num_frames == 72
    assert captured.fps == 24
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
    engine = test_client.app.state.openai_serving_video._engine_client
    captured = engine.captured_sampling_params_list[0]
    assert captured.width == 320
    assert captured.height == 240


def test_audio_sample_rate_comes_from_model_config(test_client, mocker: MockerFixture):
    audio_sample_rates = []

    def _fake_encode(video, fps, audio=None, audio_sample_rate=None):
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
    assert audio_sample_rates == [16000]


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
    engine = test_client.app.state.openai_serving_video._engine_client
    captured = engine.captured_sampling_params_list[0]
    assert captured.num_inference_steps == 30
    assert captured.guidance_scale == 6.5
    assert captured.guidance_scale_2 == 8.0
    assert captured.true_cfg_scale == 4.0
    assert captured.boundary_ratio == 0.7
    assert captured.extra_args["flow_shift"] == 0.25

    failed = wait_for_status(client, video_id, VideoGenerationStatus.FAILED.value)
    assert failed["error"]["type"] == "HTTPException"
    assert "lora object" in failed["error"]["message"].lower()


def test_missing_handler_returns_503() -> None:
    """POST /v1/videos returns 503 when video serving is uninitialized."""
    app = FastAPI()
    app.include_router(router)
    app.state.openai_serving_video = None
    client = TestClient(app)

    response = client.post("/v1/videos", data={"prompt": "no handler"})
    assert response.status_code == 503
    assert "not initialized" in response.json()["detail"].lower()


def test_missing_prompt_returns_422() -> None:
    """POST /v1/videos validates required prompt input."""
    handler = DeterministicVideoHandler([b"unused"])
    with make_client(handler) as client:
        response = client.post("/v1/videos", data={"size": "320x240"})
    assert response.status_code == 422


# UT: Request schema validation tests.


def test_video_request_validation() -> None:
    """VideoGenerationRequest validates response format and field constraints."""
    req = VideoGenerationRequest(prompt="test")
    assert req.prompt == "test"
    assert req.response_format == VideoResponseFormat.B64_JSON

    with pytest.raises(ValueError):
        VideoGenerationRequest(prompt="test", response_format="url")

    with pytest.raises(ValueError):
        VideoGenerationRequest(prompt="test", size="invalid")

    with pytest.raises(ValueError):
        VideoGenerationRequest(prompt="test", seconds="abc")


def test_video_response_file_extension_is_robust() -> None:
    """VideoResponse.file_extension normalizes MIME and validates unknown types."""
    response = VideoResponse(model="test-model")
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
