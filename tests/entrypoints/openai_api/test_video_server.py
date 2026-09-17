# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
Unit tests for OpenAI-compatible video generation endpoints.
"""

import asyncio
import base64
import io
import json
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import av
import httpx
import numpy as np
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from PIL import Image
from pytest_mock import MockerFixture
from vllm import envs

from vllm_omni.diffusion.data import DIFFUSION_REQUEST_LIFECYCLE_KEY, DIFFUSION_REQUEST_STARTED
from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes
from vllm_omni.entrypoints.openai import api_server, video_api_utils
from vllm_omni.entrypoints.openai.api_server import router
from vllm_omni.entrypoints.openai.protocol.videos import (
    VideoGenerationRequest,
    VideoGenerationStatus,
    VideoParams,
    VideoResponse,
)
from vllm_omni.entrypoints.openai.serving_video import OmniOpenAIServingVideo, ReferenceImage
from vllm_omni.entrypoints.openai.storage import LocalStorageManager
from vllm_omni.entrypoints.openai.stores import AsyncDictStore, TaskRegistry
from vllm_omni.entrypoints.openai.video.generation import helpers as video_generation_helpers
from vllm_omni.entrypoints.openai.video.generation.guided_lifetime import GUIDED_JOBS
from vllm_omni.entrypoints.openai.video.generation.helpers import (
    MINIMAX_H3_MAX_REFERENCE_IMAGE_BYTES,
    _read_upload_limited,
    _reference_video_decode_spec,
)
from vllm_omni.errors import GuardrailViolationError, OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
_requires_guided_runtime = pytest.mark.skipif(
    sys.version_info < (3, 11), reason="Guided ownership requires Python 3.11+"
)


def _delete_request_stub(handler=None):
    """Minimal ``Request`` stand-in for direct ``delete_video`` calls.

    ``delete_video`` resolves the video handler from ``app.state`` to issue the
    bounded engine abort. Guided jobs return before that lookup, so the stub only
    needs to expose the attribute chain.
    """
    if handler is None:
        handler = SimpleNamespace(abort_request=_noop_abort_request)
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(openai_serving_video=handler)))


async def _noop_abort_request(request_id):
    del request_id


class MockVideoResult:
    def __init__(
        self,
        videos,
        audios=None,
        sample_rate=None,
        multimodal_output=None,
        stage_durations=None,
        peak_memory_mb=0.0,
        custom_output=None,
    ):
        self.multimodal_output = dict(multimodal_output or {"video": videos})
        if audios is not None:
            self.multimodal_output["audio"] = audios
        if sample_rate is not None:
            self.multimodal_output["audio_sample_rate"] = sample_rate
        self.stage_durations = stage_durations or {}
        self.peak_memory_mb = peak_memory_mb
        self.custom_output = custom_output or {}


class FakeAsyncOmni:
    def __init__(self):
        self.stage_configs = [
            SimpleNamespace(
                stage_type="diffusion",
                final_output=True,
                final_output_type="video",
            )
        ]
        self.default_sampling_params_list = [OmniDiffusionSamplingParams()]
        self.model_class_name = "WanPipeline"
        self.captured_prompt = None
        self.captured_reference_video_bytes = None
        self.captured_control_reference_bytes = {}
        self.captured_sampling_params_list = None

    def get_diffusion_od_config(self):
        return SimpleNamespace(model_class_name=self.model_class_name)

    async def generate(self, prompt, request_id, sampling_params_list, **kwargs):
        self.captured_prompt = prompt
        self.captured_sampling_params_list = sampling_params_list
        # ``_run_generation`` passes ``on_engine_admitted`` for guided requests.
        on_engine_admitted = kwargs.get("on_engine_admitted")
        if on_engine_admitted is not None:
            on_engine_admitted()
        for control_type in ("edge", "blur", "depth", "seg", "wsm"):
            control_params = sampling_params_list[0].extra_args.get(control_type)
            if isinstance(control_params, dict) and isinstance(control_params.get("control_path"), str):
                self.captured_control_reference_bytes[control_type] = Path(control_params["control_path"]).read_bytes()
        reference_videos = prompt.get("multi_modal_data", {}).get("video")
        if (
            isinstance(reference_videos, list)
            and reference_videos
            and all(isinstance(item, str) for item in reference_videos)
        ):
            self.captured_reference_video_bytes = [Path(item).read_bytes() for item in reference_videos]
        num_outputs = sampling_params_list[0].num_outputs_per_prompt
        if sampling_params_list[0].emit_request_lifecycle:
            yield MockVideoResult(
                [],
                custom_output={DIFFUSION_REQUEST_LIFECYCLE_KEY: DIFFUSION_REQUEST_STARTED},
            )
        videos = [object() for _ in range(num_outputs)]
        yield MockVideoResult(videos)

    async def abort(self, request_id, *, timeout=None):
        del request_id, timeout


def test_raw_and_base64_encoders_receive_persistent_converter(mocker: MockerFixture):
    engine = FakeAsyncOmni()
    handler = OmniOpenAIServingVideo.for_diffusion(
        engine,
        model_name="test-model",
    )
    assert handler._video_frame_converter.max_workers == 8
    raw_encoder = mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        return_value=b"encoded-video",
    )
    base64_encoder = mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video.encode_video_base64",
        return_value="encoded-video",
    )

    async def _generate_both_response_types():
        request = VideoGenerationRequest(prompt="test prompt")
        await handler.generate_video_bytes(request, "raw-request")
        await handler.generate_videos(request, "base64-request")

    asyncio.run(_generate_both_response_types())

    assert raw_encoder.call_args.kwargs["frame_converter"] is handler._video_frame_converter
    assert base64_encoder.call_args.kwargs["frame_converter"] is handler._video_frame_converter
    handler.shutdown()


@pytest.mark.parametrize("batch_frames", [0, -1, True, 1.5, "17", None])
def test_preencode_rejects_invalid_batch_frames_before_generation(batch_frames):
    engine = FakeAsyncOmni()
    handler = OmniOpenAIServingVideo.for_diffusion(engine, model_name="test-model")
    request = VideoGenerationRequest(
        prompt="test", extra_params={"preencode_mp4": True, "preencode_batch_frames": batch_frames}
    )
    try:
        with pytest.raises(HTTPException, match="preencode_batch_frames") as exc:
            asyncio.run(handler.generate_video_bytes(request, "invalid-batch"))
        assert exc.value.status_code == 400
        assert engine.captured_prompt is None
    finally:
        handler.shutdown()


def test_preencoded_video_bytes_preserve_metadata(mocker: MockerFixture):
    from vllm_omni.entrypoints.openai.serving_video import VideoGenerationArtifacts

    handler = OmniOpenAIServingVideo.for_diffusion(FakeAsyncOmni(), model_name="test-model")
    # Resolved frame count differs from anything the request asked for, so the
    # metadata has to come from the encoded stream rather than request defaults.
    preencoded = _make_test_video_bytes((32, 24), num_frames=7)
    artifacts = VideoGenerationArtifacts(
        videos=[preencoded],
        audios=[None],
        actions=[None],
        audio_sample_rate=24000,
        output_fps=24.0,
        stage_durations={"decode": 0.5},
        peak_memory_mb=123.0,
        metrics={"generation_time": 1.25},
    )
    mocker.patch.object(handler, "_run_and_extract", return_value=artifacts)
    encoder = mocker.patch("vllm_omni.entrypoints.openai.serving_video._encode_video_bytes")
    try:
        result = asyncio.run(handler.generate_video_bytes(VideoGenerationRequest(prompt="test"), "preencoded"))
        assert result == (
            preencoded,
            {"decode": 0.5},
            123.0,
            None,
            {
                "fps": 24.0,
                "num_frames": 7,
                "duration_s": 7 / 24.0,
                "metrics": {"generation_time": 1.25},
            },
        )
        encoder.assert_not_called()
    finally:
        handler.shutdown()


def test_resolve_diffusion_od_config_falls_back_to_attribute():
    od_config = SimpleNamespace(model_class_name="WanPipeline")
    handler = OmniOpenAIServingVideo.for_diffusion(
        SimpleNamespace(od_config=od_config),
        model_name="test-model",
    )

    assert handler._resolve_diffusion_od_config() is od_config


def test_resolve_diffusion_od_config_prefers_getter_over_attribute():
    attribute_config = SimpleNamespace(model_class_name="WanPipeline")
    getter_config = SimpleNamespace(model_class_name="MiniMaxH3Pipeline")
    handler = OmniOpenAIServingVideo.for_diffusion(
        SimpleNamespace(
            od_config=attribute_config,
            get_diffusion_od_config=lambda: getter_config,
        ),
        model_name="test-model",
    )

    assert handler._resolve_diffusion_od_config() is getter_config


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
        self,
        request,
        reference_id,
        *,
        reference_image=None,
        reference_video=None,
        reference_audio=None,
        on_started=None,
    ):
        del request, reference_id, reference_image, reference_video, reference_audio
        if on_started is not None:
            await on_started()
        self.started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise

    async def abort_request(self, request_id):
        del request_id


class HangingAbortHandler(BlockingVideoHandler):
    """Engine abort never returns; DELETE must time out and still cancel."""

    async def abort_request(self, request_id):
        del request_id
        await asyncio.sleep(30)


class CompletingDuringAbortHandler(BlockingVideoHandler):
    """Finishes and persists output while DELETE is still awaiting abort."""

    def __init__(self):
        super().__init__()
        self._finish = asyncio.Event()

    async def generate_video_bytes(
        self,
        request,
        reference_id,
        *,
        reference_image=None,
        reference_video=None,
        reference_audio=None,
        on_started=None,
    ):
        del request, reference_image, reference_video, reference_audio
        if on_started is not None:
            await on_started()
        self.started.set()
        await self._finish.wait()
        return b"late-complete", {}, 0.0, None

    async def abort_request(self, request_id):
        self._finish.set()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            job = await api_server.VIDEO_STORE.get(request_id)
            if job is not None and job.status is VideoGenerationStatus.COMPLETED and job.file_name is not None:
                return
            await asyncio.sleep(0.01)
        raise RuntimeError(f"video job {request_id} did not complete during abort")


class SchedulerQueuedVideoHandler(BlockingVideoHandler):
    """Parks after engine submission but before scheduler admission."""

    def __init__(self):
        super().__init__()
        self.admit = threading.Event()
        self.in_progress = threading.Event()

    async def generate_video_bytes(
        self,
        request,
        reference_id,
        *,
        reference_image=None,
        reference_video=None,
        reference_audio=None,
        on_started=None,
    ):
        del request, reference_id, reference_image, reference_video, reference_audio
        self.started.set()
        try:
            while not self.admit.is_set():
                await asyncio.sleep(0.01)
            if on_started is not None:
                await on_started()
            self.in_progress.set()
            await asyncio.Future()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class AbortTrackingOmni(FakeAsyncOmni):
    def __init__(self):
        super().__init__()
        self.aborted: list[str] = []
        self.entered = threading.Event()

    async def generate(self, prompt, request_id, sampling_params_list, **kwargs):
        del prompt, request_id, sampling_params_list, kwargs
        self.entered.set()
        yield MockVideoResult(
            [],
            custom_output={DIFFUSION_REQUEST_LIFECYCLE_KEY: DIFFUSION_REQUEST_STARTED},
        )
        await asyncio.Future()

    async def abort(self, request_id, *, timeout=None):
        assert timeout is not None
        self.aborted.append(request_id)


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
    monkeypatch.setattr(video_generation_helpers, "VIDEO_STORE", store)
    monkeypatch.setattr(video_generation_helpers, "STORAGE_MANAGER", storage)
    return store, tasks, storage


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_drain", [False, True])
@pytest.mark.parametrize("serve_failure", [None, RuntimeError, asyncio.CancelledError])
async def test_server_worker_keeps_engine_alive_until_http_shutdown(monkeypatch, cancel_drain, serve_failure):
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

        if serve_failure is not None:
            await wait_for_shutdown()
            raise serve_failure()
        return asyncio.create_task(wait_for_shutdown())

    async def fake_storage_start():
        events.append("storage_start")

    async def fake_get_vllm_config(engine_client):
        del engine_client
        return None

    async def fake_init_app_state(engine_client, state, args):
        del engine_client, args

        class FakeVideo:
            async def drain_guided_requests(self):
                events.append("video_drain")
                if cancel_drain:
                    raise asyncio.CancelledError

            def shutdown(self):
                events.append("video_shutdown")

        state.openai_serving_video = FakeVideo()
        events.append("init_app_state")

    monkeypatch.setattr(api_server, "build_async_omni", fake_build_async_omni)
    monkeypatch.setattr(api_server, "build_openai_app", lambda args, supported_tasks: FastAPI())
    monkeypatch.setattr(api_server, "serve_http", fake_serve_http)
    monkeypatch.setattr(api_server.STORAGE_MANAGER, "start", fake_storage_start)
    monkeypatch.setattr(api_server.openai_app_state, "_get_vllm_config", fake_get_vllm_config)
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
    expected_error = asyncio.CancelledError if cancel_drain else serve_failure
    if expected_error is not None:
        with pytest.raises(expected_error):
            await asyncio.wait_for(worker_task, timeout=2)
    else:
        await asyncio.wait_for(worker_task, timeout=2)

    assert sock.closed
    assert events.index("http_shutdown") < events.index("video_drain")
    assert events.index("video_drain") < events.index("video_shutdown") < events.index("engine_exit")


@pytest.fixture
def test_client():
    app = FastAPI()
    app.state.api_server_count = 1
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


@pytest.mark.parametrize("endpoint", ["/v1/videos", "/v1/videos/sync"])
@pytest.mark.parametrize(
    "guide",
    [
        {"frame_index": True, "image": {"upload_index": 0}},
        {"frame_index": "0", "image": {"upload_index": 0}},
        {"frame_index": 0.0, "image": {"upload_index": 0}},
        {"frame_index": 0, "image": {"upload_index": False}},
        {"frame_index": 0, "image": {"upload_index": "0"}},
        {"frame_index": 0, "image": {"upload_index": -1}},
        {"frame_index": 0, "image": {"upload_index": 1}},
        {"frame_index": 0, "image": {"path": "/etc/passwd"}},
        {"frame_index": 0, "image": "/etc/passwd"},
        {"frame_index": 0, "image": {"upload_index": 0}, "unknown": 1},
        {"frame_index": 0, "image": {"upload_index": 0, "path": "/etc/passwd"}},
        {"frame_index": 0},
        {"frame_index": 0, "image": {"upload_index": 0}, "video": {"upload_index": 0}},
    ],
)
def test_timeline_guide_manifest_rejections(test_client, endpoint, guide):
    handler = test_client.app.state.openai_serving_video
    handler._engine_client.model_class_name = "MiniMaxH3Pipeline"
    response = test_client.post(
        endpoint,
        data={"prompt": "test", "timeline_guides": json.dumps([guide])},
        files={"guide_files": ("guide.png", _make_test_image_bytes(), "image/png")},
    )
    assert response.status_code == 400
    assert not handler.guided_requests.bundles


@pytest.mark.parametrize("endpoint", ["/v1/videos", "/v1/videos/sync"])
@pytest.mark.parametrize(
    "manifest,files",
    [
        ([], [("image.png", b"image", "image/png")]),
        ([{"frame_index": 0, "image": {"upload_index": 0}}], []),
        ([{"frame_index": 0, "audio": {"upload_index": 0}}], [("guide.png", b"image", "image/png")]),
        ([{"frame_index": 0, "image": {"upload_index": 0}}], [("guide.png", b"image", "audio/wav")]),
        ([{"frame_index": 0, "audio": {"upload_index": 0}}], [("guide.ogg", b"audio", "audio/ogg")]),
        ([{"frame_index": 0, "image": {"upload_index": 0}}], [("guide.png", b"", "image/png")]),
    ],
)
@_requires_guided_runtime
def test_timeline_guide_binding_rejections(test_client, endpoint, manifest, files):
    handler = test_client.app.state.openai_serving_video
    handler._engine_client.model_class_name = "MiniMaxH3Pipeline"
    response = test_client.post(
        endpoint,
        data={"prompt": "test", "timeline_guides": json.dumps(manifest)},
        files=[("guide_files", item) for item in files],
    )
    assert response.status_code == 400
    assert not handler.guided_requests.bundles


@pytest.mark.parametrize("endpoint", ["/v1/videos", "/v1/videos/sync"])
@pytest.mark.parametrize("key", ["_minimax_h3_timeline_guides", "timeline_guides", "guide_files"])
def test_timeline_guide_extra_injection_rejected(test_client, endpoint, key):
    response = test_client.post(endpoint, data={"prompt": "test", "extra_params": json.dumps({key: "/etc/passwd"})})
    assert response.status_code == 400


@pytest.mark.parametrize("endpoint", ["/v1/videos", "/v1/videos/sync"])
@pytest.mark.parametrize("kinds", [("video",), ("audio",), ("video", "audio")])
@_requires_guided_runtime
def test_timeline_guide_clip_flac_and_av_transport(test_client, mocker, monkeypatch, endpoint, kinds):
    handler = test_client.app.state.openai_serving_video
    engine = handler._engine_client
    engine.model_class_name = "MiniMaxH3Pipeline"
    # Opaque sentinel bytes exercise transport/binding only, not codec or model decode.
    uploads = {
        "video": ("guide.mp4", b"transport-only-video", "video/mp4"),
        "audio": ("guide.flac", b"fLaC-transport-only-audio", "audio/flac"),
    }
    manifest = [{"frame_index": 36, **{kind: {"upload_index": index} for index, kind in enumerate(kinds)}}]
    paths = []
    captured = {}

    async def generate(prompt, request_id, sampling_params_list, on_engine_admitted=None):
        descriptors = sampling_params_list[0].extra_args["_minimax_h3_timeline_guides"]
        assert len(descriptors) == 1
        assert descriptors[0]["frame_index"] == 36
        assert set(descriptors[0]) == {"frame_index", *kinds}
        for kind in kinds:
            path = Path(descriptors[0][kind])
            paths.append(path)
            captured[kind] = path.read_bytes()
            assert path.suffix == Path(uploads[kind][0]).suffix
        assert not prompt.get("multi_modal_data")
        on_engine_admitted()
        yield MockVideoResult([object()])

    monkeypatch.setattr(engine, "generate", generate)
    mocker.patch("vllm_omni.entrypoints.openai.serving_video._encode_video_bytes", return_value=b"video")
    response = test_client.post(
        endpoint,
        data={"prompt": "test", "timeline_guides": json.dumps(manifest)},
        files=[("guide_files", uploads[kind]) for kind in kinds],
    )
    assert response.status_code == 200
    if endpoint == "/v1/videos":
        _wait_for_status(test_client, response.json()["id"], "completed")
    else:
        assert response.content == b"video"
    assert captured == {kind: uploads[kind][1] for kind in kinds}
    assert len(set(paths)) == len(kinds)
    assert not any(path.exists() for path in paths)
    assert not handler.guided_requests.bundles


@pytest.mark.parametrize("endpoint", ["/v1/videos", "/v1/videos/sync"])
def test_timeline_guides_reject_unsupported_model_before_persistence(test_client, mocker, endpoint):
    handler = test_client.app.state.openai_serving_video
    assert handler._engine_client.model_class_name == "WanPipeline"
    persist = mocker.patch.object(video_generation_helpers, "_persist_guide_uploads")
    generate = mocker.patch.object(handler, "generate_video_bytes")
    response = test_client.post(
        endpoint,
        data={"prompt": "test", "timeline_guides": '[{"frame_index": 0, "audio": {"upload_index": 0}}]'},
        files={"guide_files": ("guide.flac", b"fLaC-transport-only-audio", "audio/flac")},
    )
    assert response.status_code == 400
    assert "does not support timeline guides" in response.json()["detail"]
    persist.assert_not_called()
    generate.assert_not_called()
    assert not handler.guided_requests.bundles


@pytest.mark.parametrize(
    "model_class_name,supported",
    [
        ("MiniMaxH3Pipeline", True),
        # The modular alias shares the guide contract and must not be rejected
        # by a hardcoded pipeline name.
        ("MiniMaxH3ModularPipeline", True),
        ("WanPipeline", False),
        ("Cosmos3OmniDiffusersPipeline", False),
        (None, False),
    ],
)
def test_timeline_guide_capability_follows_model_metadata(test_client, model_class_name, supported):
    handler = test_client.app.state.openai_serving_video
    handler._engine_client.model_class_name = model_class_name

    assert handler.supports_timeline_guides is supported
    if supported:
        assert handler.timeline_guide_limits().max_entries == 8
    else:
        with pytest.raises(HTTPException) as excinfo:
            handler.timeline_guide_limits()
        assert excinfo.value.status_code == 400


def test_timeline_guide_capability_uses_stage_config_metadata(test_client):
    """Split-stage deployments name the pipeline in a stage config only."""
    handler = test_client.app.state.openai_serving_video
    handler._engine_client.model_class_name = None
    handler._stage_configs = [
        SimpleNamespace(engine_args={"model_class_name": "MiniMaxH3TextEncoder"}),
        SimpleNamespace(engine_args={"model_class_name": "MiniMaxH3ModularPipeline"}),
    ]

    assert handler.supports_timeline_guides


@pytest.mark.parametrize("endpoint", ["/v1/videos", "/v1/videos/sync"])
@pytest.mark.parametrize("with_guides", [False, True])
def test_python310_rejects_guides_but_preserves_legacy_requests(test_client, mocker, endpoint, with_guides):
    handler = test_client.app.state.openai_serving_video
    handler._engine_client.model_class_name = "MiniMaxH3Pipeline"
    mocker.patch("vllm_omni.entrypoints.openai.video.generation.guided_lifetime.version_info", (3, 10, 14))
    persist = mocker.patch.object(video_generation_helpers, "_persist_guide_uploads")
    mocker.patch("vllm_omni.entrypoints.openai.serving_video._encode_video_bytes", return_value=b"video")
    manifest = [{"frame_index": 0, "image": {"upload_index": 0}}] if with_guides else []
    response = test_client.post(
        endpoint,
        data={"prompt": "test", "timeline_guides": json.dumps(manifest)},
        files=[("guide_files", ("guide.png", _make_test_image_bytes(), "image/png"))] if with_guides else [],
    )
    if with_guides:
        assert response.status_code == 503
        assert "Python 3.11" in response.json()["detail"]
        assert "omit timeline_guides" in response.json()["detail"]
    else:
        assert response.status_code == 200
        if endpoint == "/v1/videos":
            _wait_for_status(test_client, response.json()["id"], "completed")
    persist.assert_not_called()
    assert not handler.guided_requests.bundles


@pytest.mark.parametrize("endpoint", ["/v1/videos", "/v1/videos/sync"])
@_requires_guided_runtime
def test_timeline_guide_reuse_order_and_trusted_extras(test_client, mocker, endpoint):
    handler = test_client.app.state.openai_serving_video
    engine = handler._engine_client
    engine.model_class_name = "MiniMaxH3Pipeline"
    mocker.patch("vllm_omni.entrypoints.openai.serving_video._encode_video_bytes", return_value=b"video")
    manifest = [{"frame_index": index, "image": {"upload_index": 0}} for index in (36, -1, 0)]
    response = test_client.post(
        endpoint,
        data={"prompt": "test", "timeline_guides": json.dumps(manifest)},
        files={"guide_files": ("../../guide.png", _make_test_image_bytes(), "image/png")},
    )
    assert response.status_code == 200
    if endpoint == "/v1/videos":
        _wait_for_status(test_client, response.json()["id"], "completed")
    descriptors = engine.captured_sampling_params_list[0].extra_args["_minimax_h3_timeline_guides"]
    assert [item["frame_index"] for item in descriptors] == [36, -1, 0]
    paths = {item["image"] for item in descriptors}
    assert len(paths) == 1
    assert not any(Path(path).exists() for path in paths)
    assert "image" not in engine.captured_prompt.get("multi_modal_data", {})


@pytest.mark.parametrize("endpoint", ["/v1/videos", "/v1/videos/sync"])
@_requires_guided_runtime
def test_terminal_guide_validation_errors_do_not_exhaust_capacity(test_client, monkeypatch, endpoint):
    handler = test_client.app.state.openai_serving_video
    engine = handler._engine_client
    engine.model_class_name = "MiniMaxH3Pipeline"
    paths = []

    async def generate(prompt, request_id, sampling_params_list, on_engine_admitted=None):
        guide = sampling_params_list[0].extra_args["_minimax_h3_timeline_guides"][0]
        assert guide["frame_index"] == 100000
        paths.append(Path(guide["image"]))
        assert paths[-1].exists()
        on_engine_admitted()
        error = OmniClientError("timeline guide frame_index 100000 is outside the output")
        error.worker_finished = True  # Simulate an origin-qualified terminal worker rejection.
        raise error
        yield  # Make the engine double an async iterator; serving methods remain real.

    monkeypatch.setattr(engine, "generate", generate)
    for _ in range(6):  # Exceeds the default four outstanding guided requests.
        response = test_client.post(
            endpoint,
            data={"prompt": "test", "timeline_guides": '[{"frame_index": 100000, "image": {"upload_index": 0}}]'},
            files={"guide_files": ("guide.png", _make_test_image_bytes(), "image/png")},
        )
        if endpoint == "/v1/videos":
            assert response.status_code == 200
            failed = _wait_for_status(test_client, response.json()["id"], "failed")
            assert failed["error"]["code"] == 400
        else:
            assert response.status_code == 400
            assert "100000" in response.json()["detail"]
        _wait_until(lambda: not handler.guided_requests.bundles)
        assert not any(path.exists() for path in paths)
    assert len(paths) == 6


@pytest.mark.parametrize("endpoint", ["/v1/videos", "/v1/videos/sync"])
@_requires_guided_runtime
def test_pre_dispatch_generate_failures_do_not_exhaust_capacity(test_client, monkeypatch, endpoint):
    """Failures before EngineCore accepts must release uploads and capacity.

    ``AsyncOmni.generate()`` rejects an asleep engine, a diffusion list prompt
    and bad sampling params after the generator is entered but before
    ``add_request_async``. Nothing is queued in any worker then, so retaining
    the uploads would leak the files *and* an admission slot on every attempt.
    """
    handler = test_client.app.state.openai_serving_video
    engine = handler._engine_client
    engine.model_class_name = "MiniMaxH3Pipeline"
    paths = []

    async def generate(prompt, request_id, sampling_params_list, on_engine_admitted=None):
        guide = sampling_params_list[0].extra_args["_minimax_h3_timeline_guides"][0]
        paths.append(Path(guide["image"]))
        assert paths[-1].exists()
        # Never acknowledge admission: this models the pre-submission rejects.
        raise RuntimeError("Generation rejected: Engine is partially or fully asleep.")
        yield  # Make the engine double an async iterator; serving methods remain real.

    monkeypatch.setattr(engine, "generate", generate)
    for _ in range(6):  # Exceeds the default four outstanding guided requests.
        response = test_client.post(
            endpoint,
            data={"prompt": "test", "timeline_guides": '[{"frame_index": 0, "image": {"upload_index": 0}}]'},
            files={"guide_files": ("guide.png", _make_test_image_bytes(), "image/png")},
        )
        if endpoint == "/v1/videos":
            assert response.status_code == 200
            failed = _wait_for_status(test_client, response.json()["id"], "failed")
            assert failed["error"]["code"] == 500
        else:
            assert response.status_code == 500
        _wait_until(lambda: not handler.guided_requests.bundles)
        assert not any(path.exists() for path in paths)
    assert len(paths) == 6


@pytest.mark.parametrize("endpoint", ["/v1/videos", "/v1/videos/sync"])
@pytest.mark.parametrize("limit", ["max_image_bytes", "max_total_upload_bytes", "max_source_pixels"])
@_requires_guided_runtime
def test_timeline_guide_upload_budgets(test_client, monkeypatch, endpoint, limit):
    handler = test_client.app.state.openai_serving_video
    monkeypatch.setattr(
        handler._engine_client,
        "get_diffusion_od_config",
        lambda: SimpleNamespace(
            model_class_name="MiniMaxH3Pipeline",
            model_config={"minimax_h3_timeline_guides": {limit: 1}},
        ),
    )
    response = test_client.post(
        endpoint,
        data={
            "prompt": "test",
            "timeline_guides": '[{"frame_index": 0, "image": {"upload_index": 0}}]',
        },
        files={"guide_files": ("guide.png", _make_test_image_bytes(), "image/png")},
    )
    assert response.status_code == 400
    assert not handler.guided_requests.bundles


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["timeout", "cancel", "delete", "error"])
@_requires_guided_runtime
async def test_guided_request_keeps_all_inputs_until_completion(monkeypatch, isolated_video_backends, operation):
    store, _, storage = isolated_video_backends
    engine = FakeAsyncOmni()
    engine.model_class_name = "MiniMaxH3Pipeline"
    handler = OmniOpenAIServingVideo.for_diffusion(engine, model_name="test-model")
    app = FastAPI()
    app.include_router(router)
    app.state.openai_serving_video = handler
    # Async video jobs live in process-local state, so the routes require a
    # declared single-API-worker topology.
    app.state.api_server_count = 1
    entered, finish = asyncio.Event(), asyncio.Event()
    paths = set()

    async def generate(request, reference_id, **kwargs):
        paths.update(request._guide_bundle.paths)
        assert kwargs["reference_video"] is not None
        assert kwargs["reference_audio"] is not None
        entered.set()
        await finish.wait()
        assert len(paths) == 3 and all(Path(path).exists() for path in paths)
        if operation == "error":
            raise HTTPException(400, "invalid guide placement")
        return b"video", {}, 0.0, None

    monkeypatch.setattr(handler, "generate_video_bytes", generate)
    monkeypatch.setattr(api_server, "VIDEO_SYNC_TIMEOUT_S", 0.05)
    endpoint = "/v1/videos" if operation in ("delete", "error") else "/v1/videos/sync"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        outer = asyncio.create_task(
            client.post(
                endpoint,
                data={
                    "prompt": "test",
                    "timeline_guides": '[{"frame_index": 0, "image": {"upload_index": 0}}]',
                },
                files=[
                    ("guide_files", ("guide.png", _make_test_image_bytes(), "image/png")),
                    ("input_references", ("reference.mp4", b"reference-video", "video/mp4")),
                    ("input_references", ("reference.wav", b"reference-audio", "audio/wav")),
                ],
            )
        )
        await asyncio.wait_for(entered.wait(), 2)
        bundle = next(iter(handler.guided_requests.bundles))
        if operation == "timeout":
            assert (await outer).status_code == 504
        elif operation == "cancel":
            outer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await outer
        else:
            response = await outer
            job_id = response.json()["id"]
            if operation == "delete":
                assert (await client.delete(f"/v1/videos/{job_id}")).status_code == 200
                assert await store.get(job_id) is None
        assert not bundle.task.done()
        assert all(Path(path).exists() for path in paths)
        finish.set()
        await bundle.task
        await asyncio.sleep(0)
        assert not handler.guided_requests.bundles
        assert not any(Path(path).exists() for path in paths)
        if operation == "delete":
            assert await store.get(job_id) is None
            assert not storage.exists(job_id)
        if operation == "error":
            job = await store.get(job_id)
            assert job.status == VideoGenerationStatus.FAILED
            assert job.error.code == 400
    handler.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_save", [False, True])
@pytest.mark.parametrize("cancel_delete", [False, True])
@pytest.mark.parametrize("save_fails", [False, True])
@_requires_guided_runtime
async def test_guided_delete_during_storage_save(
    monkeypatch,
    isolated_video_backends,
    cancel_save,
    cancel_delete,
    save_fails,
):
    from vllm_omni.entrypoints.openai.video.generation.guided_lifetime import GuidedRequestLifetime

    store, _, storage = isolated_video_backends
    owner = GuidedRequestLifetime()
    bundle = owner.reserve(4)
    request = VideoGenerationRequest(prompt="test")
    request._guide_bundle = bundle
    job = VideoResponse(id="guided-save-race", model="test", prompt="test")
    await store.upsert(job.id, job)
    bundle.job_id = job.id
    GUIDED_JOBS[job.id] = bundle
    saving, finish_save = asyncio.Event(), asyncio.Event()
    original_save = storage.save

    async def save(*args):
        saving.set()
        await finish_save.wait()
        if save_fails:
            raise OSError("storage failure")
        return await original_save(*args)

    async def generate(*args, **kwargs):
        return b"video", {}, 0.0, None

    monkeypatch.setattr(storage, "save", save)
    task = bundle.submit(
        video_generation_helpers._run_video_generation_job(
            SimpleNamespace(generate_video_bytes=generate),
            request,
            job.id,
        )
    )
    await asyncio.wait_for(saving.wait(), 2)
    if cancel_save:
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    deleting = asyncio.create_task(api_server.delete_video(job.id, _delete_request_stub()))
    await asyncio.sleep(0)
    assert bundle.abandoned
    if cancel_delete:
        deleting.cancel()
        await asyncio.sleep(0)
        assert not deleting.done()
    finish_save.set()
    await asyncio.gather(task, deleting)
    assert await store.get(job.id) is None
    assert not storage.exists(job.id)
    if cancel_save:
        assert bundle in owner.bundles
        bundle.close()  # Fake generation is known to be quiescent.
    assert not owner.bundles


@pytest.mark.asyncio
@_requires_guided_runtime
async def test_cancelled_guided_delete_behind_threaded_save_removes_job_and_artifact(
    tmp_path,
    monkeypatch,
    isolated_video_backends,
):
    from vllm_omni.entrypoints.openai.video.generation.guided_lifetime import GuidedRequestLifetime

    store, _, storage = isolated_video_backends
    owner = GuidedRequestLifetime()
    bundle = owner.reserve(1)
    guide = tmp_path / "guide"
    guide.write_bytes(b"input")
    bundle.paths.add(str(guide))
    job = VideoResponse(id="cancelled-delete-threaded-save", model="test", prompt="test")
    await store.upsert(job.id, job)
    bundle.job_id = job.id
    GUIDED_JOBS[job.id] = bundle
    request = VideoGenerationRequest(prompt="test")
    request._guide_bundle = bundle
    saving, delete_waiting = asyncio.Event(), asyncio.Event()
    release_write, write_finished = threading.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    original_save = storage._save_sync
    original_delete = api_server._delete_guided_video

    def save(*args):
        loop.call_soon_threadsafe(saving.set)
        assert release_write.wait(5)
        assert guide.exists()
        saved = original_save(*args)
        write_finished.set()
        return saved

    async def remove(video_id, owner_bundle):
        delete_waiting.set()
        await original_delete(video_id, owner_bundle)

    async def generate(*args, on_started=None, **kwargs):
        # Guided jobs stay QUEUED until the engine reports inference start.
        await on_started()
        return b"video", {}, 0.0, None

    monkeypatch.setattr(storage, "_save_sync", save)
    monkeypatch.setattr(api_server, "_delete_guided_video", remove)
    task = bundle.submit(
        video_generation_helpers._run_video_generation_job(
            SimpleNamespace(generate_video_bytes=generate),
            request,
            job.id,
        )
    )
    deleting = None
    try:
        await asyncio.wait_for(saving.wait(), 2)
        deleting = asyncio.create_task(api_server.delete_video(job.id, _delete_request_stub()))
        await asyncio.wait_for(delete_waiting.wait(), 2)
        assert bundle.lock.locked()
        assert (await store.get(job.id)).status == VideoGenerationStatus.IN_PROGRESS
        deleting.cancel()
        assert task.cancelling() == 0
        release_write.set()
        response = await deleting
        assert response.deleted
        await task
        assert write_finished.is_set()
        assert await store.get(job.id) is None
        assert not storage.exists(job.id)
        assert not guide.exists() and not owner.bundles
    finally:
        release_write.set()
        await asyncio.gather(task, *([] if deleting is None else [deleting]), return_exceptions=True)


@pytest.mark.asyncio
@_requires_guided_runtime
async def test_guided_cancel_during_metadata_update_discards_saved_output(monkeypatch, isolated_video_backends):
    from vllm_omni.entrypoints.openai.video.generation.guided_lifetime import GuidedRequestLifetime

    store, _, storage = isolated_video_backends
    owner = GuidedRequestLifetime()
    bundle = owner.reserve(1)
    job = VideoResponse(id="guided-metadata-cancel", model="test", prompt="test")
    await store.upsert(job.id, job)
    bundle.job_id = job.id
    GUIDED_JOBS[job.id] = bundle
    request = VideoGenerationRequest(prompt="test")
    request._guide_bundle = bundle
    updating = asyncio.Event()
    original_update = store.update_fields

    async def update(key, fields):
        if fields.get("status") == VideoGenerationStatus.COMPLETED:
            updating.set()
            await asyncio.Future()
        return await original_update(key, fields)

    async def generate(*args, **kwargs):
        return b"video", {}, 0.0, None

    monkeypatch.setattr(store, "update_fields", update)
    task = bundle.submit(
        video_generation_helpers._run_video_generation_job(
            SimpleNamespace(generate_video_bytes=generate),
            request,
            job.id,
        )
    )
    await asyncio.wait_for(updating.wait(), 2)
    assert storage.exists(job.id)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert bundle.abandoned
    assert await store.get(job.id) is None
    assert not storage.exists(job.id)
    assert bundle in owner.bundles
    bundle.close()  # The fake engine has no external readers.


@pytest.mark.parametrize("endpoint", ["/v1/videos", "/v1/videos/sync"])
@_requires_guided_runtime
def test_guided_capacity_reserved_before_persisting_any_inputs(test_client, mocker, endpoint):
    handler = test_client.app.state.openai_serving_video
    handler._engine_client.model_class_name = "MiniMaxH3Pipeline"
    reservations = [handler.guided_requests.reserve(4) for _ in range(4)]
    persist = mocker.patch.object(video_generation_helpers, "_persist_guide_uploads")
    refs = mocker.patch.object(video_generation_helpers, "_persist_uploaded_media_references")
    try:
        response = test_client.post(
            endpoint,
            data={
                "prompt": "test",
                "timeline_guides": '[{"frame_index": 0, "image": {"upload_index": 0}}]',
            },
            files=[
                ("guide_files", ("guide.png", _make_test_image_bytes(), "image/png")),
                ("input_references", ("reference.mp4", b"reference", "video/mp4")),
            ],
        )
        assert response.status_code == 503
        persist.assert_not_called()
        refs.assert_not_called()
    finally:
        for bundle in reservations:
            bundle.close()


@pytest.mark.parametrize("endpoint", ["/v1/videos", "/v1/videos/sync"])
@_requires_guided_runtime
def test_guided_partial_upload_failure_releases_files(test_client, monkeypatch, endpoint):
    handler = test_client.app.state.openai_serving_video
    handler._engine_client.model_class_name = "MiniMaxH3Pipeline"
    original = video_generation_helpers.tempfile.mkstemp
    paths = []

    def record(*args, **kwargs):
        fd, path = original(*args, **kwargs)
        paths.append(path)
        return fd, path

    monkeypatch.setattr(video_generation_helpers.tempfile, "mkstemp", record)
    manifest = [{"frame_index": 0, "image": {"upload_index": index}} for index in range(2)]
    response = test_client.post(
        endpoint,
        data={"prompt": "test", "timeline_guides": json.dumps(manifest)},
        files=[
            ("guide_files", ("first.png", _make_test_image_bytes(), "image/png")),
            ("guide_files", ("second.png", b"broken", "image/png")),
        ],
    )
    assert response.status_code == 400
    assert not handler.guided_requests.bundles


@pytest.mark.parametrize("endpoint", ["/v1/videos", "/v1/videos/sync"])
@_requires_guided_runtime
def test_out_of_process_diffusion_stage_limits_reach_the_http_boundary(test_client, monkeypatch, endpoint):
    """Split deployments must not silently fall back to default guide limits.

    An out-of-process diffusion stage keeps its ``OmniDiffusionConfig`` in the
    worker, so the API process only sees the sanitized ``model_config``
    snapshot the stage client carries. This exercises the real resolution
    chain: stage client -> ``AsyncOmniEngine`` view -> ``AsyncOmni`` ->
    ``timeline_guide_limits()``.
    """
    from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
    from vllm_omni.entrypoints.async_omni import AsyncOmni

    monkeypatch.setattr(
        "vllm_omni.diffusion.data.resolve_model_class_name",
        lambda model: "MiniMaxH3Pipeline",
    )
    handler = test_client.app.state.openai_serving_video
    inner = object.__new__(AsyncOmniEngine)
    inner.model = "MiniMaxAI/MiniMax-H3"
    inner._diffusion_od_config_view = None
    inner.stage_clients = [
        SimpleNamespace(
            stage_type="diffusion",
            diffusion_model_config={"minimax_h3_timeline_guides": {"max_entries": 1, "max_outstanding_requests": 1}},
        )
    ]
    omni = object.__new__(AsyncOmni)
    omni.engine = inner
    monkeypatch.setattr(handler._engine_client, "get_diffusion_od_config", omni.get_diffusion_od_config)

    limits = handler.timeline_guide_limits()
    assert limits.max_entries == 1
    assert limits.max_outstanding_requests == 1

    generate = Mock(side_effect=AssertionError("over-limit guided request must never reach the engine"))
    monkeypatch.setattr(handler._engine_client, "generate", generate)
    response = test_client.post(
        endpoint,
        data={
            "prompt": "test",
            "timeline_guides": (
                '[{"frame_index": 0, "image": {"upload_index": 0}}, {"frame_index": 24, "image": {"upload_index": 1}}]'
            ),
        },
        files=[
            ("guide_files", ("first.png", _make_test_image_bytes(), "image/png")),
            ("guide_files", ("second.png", _make_test_image_bytes(), "image/png")),
        ],
    )
    assert response.status_code == 400
    generate.assert_not_called()
    # Rejected before any bundle reserved a slot, so no upload was persisted.
    assert not handler.guided_requests.bundles


@pytest.mark.parametrize("endpoint", ["/v1/videos", "/v1/videos/sync"])
def test_empty_timeline_guides_leave_legacy_model_unchanged(test_client, mocker, endpoint):
    handler = test_client.app.state.openai_serving_video
    mocker.patch("vllm_omni.entrypoints.openai.serving_video._encode_video_bytes", return_value=b"video")
    response = test_client.post(endpoint, data={"prompt": "test", "timeline_guides": "[]"})
    assert response.status_code == 200
    if endpoint == "/v1/videos":
        _wait_for_status(test_client, response.json()["id"], "completed")
    assert not handler.guided_requests.bundles
    assert "_minimax_h3_timeline_guides" not in handler._engine_client.captured_sampling_params_list[0].extra_args


@pytest.mark.asyncio
@_requires_guided_runtime
async def test_guided_delete_before_generation_starts(tmp_path, isolated_video_backends):
    from vllm_omni.entrypoints.openai.video.generation.guided_lifetime import GuidedRequestLifetime

    store, _, storage = isolated_video_backends
    owner = GuidedRequestLifetime()
    bundle = owner.reserve(1)
    path = tmp_path / "queued-guide"
    path.write_bytes(b"guide")
    bundle.paths.add(str(path))
    job = VideoResponse(id="guided-queued", model="test", prompt="test")
    await store.upsert(job.id, job)
    bundle.job_id = job.id
    GUIDED_JOBS[job.id] = bundle
    request = VideoGenerationRequest(prompt="test")
    request._guide_bundle = bundle

    async def generate(*args, **kwargs):
        pytest.fail("Deleted queued generation must not be submitted to the engine")

    task = bundle.submit(
        video_generation_helpers._run_video_generation_job(
            SimpleNamespace(generate_video_bytes=generate),
            request,
            job.id,
        )
    )
    response = await api_server.delete_video(job.id, _delete_request_stub())
    await task
    assert response.deleted
    assert await store.get(job.id) is None
    assert not storage.exists(job.id)
    assert not path.exists()
    assert not owner.bundles


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "engine_error", "partial_then_error", "empty"])
@_requires_guided_runtime
async def test_guided_engine_boundary_requires_normal_output_completion(tmp_path, monkeypatch, outcome):
    engine = FakeAsyncOmni()
    handler = OmniOpenAIServingVideo.for_diffusion(engine, model_name="test")
    bundle = handler.guided_requests.reserve(1)
    path = tmp_path / "guide"
    path.write_bytes(b"input")
    bundle.paths.add(str(path))

    async def generate(**kwargs):
        # A submitted request is the only thing that makes ``engine_started``
        # true; production sets it from this callback, never on entry.
        kwargs["on_engine_admitted"]()
        if outcome == "engine_error":
            raise RuntimeError("engine transport failed")
        if outcome != "empty":
            yield MockVideoResult([object()])
        if outcome == "partial_then_error":
            raise RuntimeError("engine failed after yielding partial output")

    monkeypatch.setattr(engine, "generate", generate)
    task = bundle.submit(
        handler._run_generation(
            {"prompt": "test"},
            OmniDiffusionSamplingParams(),
            "guided-boundary",
            guide_bundle=bundle,
        )
    )
    await asyncio.gather(task, return_exceptions=True)
    assert bundle.engine_started
    assert bundle.engine_completed is (outcome in {"success", "empty"})
    await handler.drain_guided_requests()
    if outcome in {"success", "empty"}:
        assert not path.exists() and not handler.guided_requests.bundles
    else:
        assert path.exists() and bundle in handler.guided_requests.bundles
        bundle.close()  # This test independently knows that its engine double has no readers.
    handler.shutdown()


@pytest.mark.asyncio
@_requires_guided_runtime
async def test_guided_generation_reports_started_and_engine_boundaries(tmp_path, monkeypatch):
    """Guided requests keep both lifecycle signals: admission and inference start.

    ``on_engine_admitted`` guards input ownership while ``on_started`` drives the
    QUEUED -> IN_PROGRESS transition. The started sentinel must not be mistaken
    for a generation result.
    """
    from vllm_omni.diffusion.data import DIFFUSION_REQUEST_LIFECYCLE_KEY, DIFFUSION_REQUEST_STARTED

    engine = FakeAsyncOmni()
    handler = OmniOpenAIServingVideo.for_diffusion(engine, model_name="test")
    bundle = handler.guided_requests.reserve(1)
    path = tmp_path / "guide"
    path.write_bytes(b"input")
    bundle.paths.add(str(path))
    final = MockVideoResult([object()])
    lifecycle_enabled = []

    async def generate(**kwargs):
        kwargs["on_engine_admitted"]()
        lifecycle_enabled.append(kwargs["sampling_params_list"][0].emit_request_lifecycle)
        yield MockVideoResult([], custom_output={DIFFUSION_REQUEST_LIFECYCLE_KEY: DIFFUSION_REQUEST_STARTED})
        yield final

    started = []

    async def on_started() -> None:
        started.append(bundle.engine_started)

    monkeypatch.setattr(engine, "generate", generate)
    result = await bundle.submit(
        handler._run_generation(
            {"prompt": "test"},
            OmniDiffusionSamplingParams(),
            "guided-started",
            on_started=on_started,
            guide_bundle=bundle,
        )
    )
    assert result is final
    assert lifecycle_enabled == [True]
    assert started == [True]
    assert bundle.engine_started and bundle.engine_completed
    await handler.drain_guided_requests()
    assert not path.exists() and not handler.guided_requests.bundles
    handler.shutdown()


@pytest.mark.asyncio
@_requires_guided_runtime
async def test_guided_job_stays_queued_until_engine_reports_start(isolated_video_backends):
    """A guided async job must not report IN_PROGRESS before inference starts."""
    from vllm_omni.entrypoints.openai.video.generation.guided_lifetime import GuidedRequestLifetime

    store, _, _storage = isolated_video_backends
    owner = GuidedRequestLifetime()
    bundle = owner.reserve(1)
    job = VideoResponse(id="guided-queued-until-start", model="test", prompt="test")
    await store.upsert(job.id, job)
    bundle.job_id = job.id
    GUIDED_JOBS[job.id] = bundle
    request = VideoGenerationRequest(prompt="test")
    request._guide_bundle = bundle
    inferring = asyncio.Event()
    release = asyncio.Event()

    async def generate(*args, on_started=None, **kwargs):
        assert (await store.get(job.id)).status == VideoGenerationStatus.QUEUED
        await on_started()
        inferring.set()
        await release.wait()
        return b"video", {}, 0.0, None

    task = bundle.submit(
        video_generation_helpers._run_video_generation_job(
            SimpleNamespace(generate_video_bytes=generate),
            request,
            job.id,
        )
    )
    await asyncio.wait_for(inferring.wait(), 2)
    assert (await store.get(job.id)).status == VideoGenerationStatus.IN_PROGRESS
    release.set()
    await task
    assert (await store.get(job.id)).status == VideoGenerationStatus.COMPLETED
    assert not owner.bundles


@pytest.mark.asyncio
@_requires_guided_runtime
async def test_guided_started_notification_after_delete_keeps_job_removed(tmp_path, isolated_video_backends):
    """DELETE pops the store entry; a later start notification must not revive it."""
    from vllm_omni.entrypoints.openai.video.generation.guided_lifetime import GuidedRequestLifetime

    store, _, _storage = isolated_video_backends
    owner = GuidedRequestLifetime()
    bundle = owner.reserve(1)
    path = tmp_path / "guide"
    path.write_bytes(b"input")
    bundle.paths.add(str(path))
    job = VideoResponse(id="guided-start-after-delete", model="test", prompt="test")
    await store.upsert(job.id, job)
    bundle.job_id = job.id
    GUIDED_JOBS[job.id] = bundle
    request = VideoGenerationRequest(prompt="test")
    request._guide_bundle = bundle
    entered = asyncio.Event()
    deleted = asyncio.Event()

    async def generate(*args, on_started=None, **kwargs):
        entered.set()
        await deleted.wait()
        await on_started()
        return b"video", {}, 0.0, None

    task = bundle.submit(
        video_generation_helpers._run_video_generation_job(
            SimpleNamespace(generate_video_bytes=generate),
            request,
            job.id,
        )
    )
    await asyncio.wait_for(entered.wait(), 2)
    assert (await api_server.delete_video(job.id, _delete_request_stub())).deleted
    deleted.set()
    await task
    assert await store.get(job.id) is None
    assert not path.exists() and not owner.bundles


@pytest.mark.asyncio
@_requires_guided_runtime
async def test_guided_failure_before_engine_admission_releases_inputs(tmp_path, monkeypatch):
    """A failure before EngineCore accepts must release uploads and capacity.

    ``generate()`` can reject a request after the coroutine is entered but
    before ``add_request_async`` (asleep engine, diffusion list-prompt
    rejection, sampling resolution). Nothing is queued in the worker then, so
    retaining the uploads would leak both the files and an admission slot.
    """
    engine = FakeAsyncOmni()
    handler = OmniOpenAIServingVideo.for_diffusion(engine, model_name="test")
    bundle = handler.guided_requests.reserve(1)
    path = tmp_path / "guide"
    path.write_bytes(b"input")
    bundle.paths.add(str(path))

    async def generate(**kwargs):
        raise RuntimeError("Generation rejected: Engine is partially or fully asleep.")
        yield  # Make the double an async iterator without ever admitting.

    monkeypatch.setattr(engine, "generate", generate)
    task = bundle.submit(
        handler._run_generation(
            {"prompt": "test"},
            OmniDiffusionSamplingParams(),
            "guided-pre-dispatch",
            guide_bundle=bundle,
        )
    )
    await asyncio.gather(task, return_exceptions=True)
    assert not bundle.engine_started
    assert not bundle.engine_completed
    assert bundle.closed
    assert not path.exists()
    assert not handler.guided_requests.bundles
    handler.shutdown()


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


def test_mux_video_audio_marks_aac_priming_timestamp():
    frames = np.zeros((2, 16, 16, 3), dtype=np.uint8)
    audio = np.zeros((2, 2048), dtype=np.float32)

    payload = mux_video_audio_bytes(
        frames,
        fps=24,
        audio_waveform=audio,
        audio_sample_rate=32000,
        video_codec_options={"preset": "ultrafast", "threads": "0"},
    )

    with av.open(io.BytesIO(payload)) as container:
        audio_stream = container.streams.audio[0]
        first_packet = next(packet for packet in container.demux(audio_stream) if packet.pts is not None)

    assert first_packet.pts < 0


def _make_test_video_data_url(size=(32, 24), num_frames=3) -> str:
    encoded = base64.b64encode(_make_test_video_bytes(size, num_frames)).decode("utf-8")
    return f"data:video/mp4;base64,{encoded}"


def _cosmos3_stage_configs():
    return [
        SimpleNamespace(
            stage_type="diffusion",
            final_output=True,
            final_output_type="video",
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

    async def _generate(prompt, request_id, sampling_params_list, **kwargs):
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


def test_i2v_resize_policy_can_defer_to_pipeline(monkeypatch):
    engine = FakeAsyncOmni()
    engine.get_diffusion_od_config = lambda: SimpleNamespace(  # type: ignore[method-assign]
        model="org/model",
        model_class_name="ExamplePipeline",
        revision="pinned-revision",
    )
    captured: dict[str, str | None] = {}

    def fake_policy(model_class_name, *, model, revision=None):
        captured.update(
            model_class_name=model_class_name,
            model=model,
            revision=revision,
        )
        return True

    monkeypatch.setattr(
        "vllm_omni.entrypoints.openai.serving_video.should_preserve_reference_image_size",
        fake_policy,
    )
    handler = OmniOpenAIServingVideo.for_diffusion(
        diffusion_engine=engine,
        model_name="fallback/model",
    )
    image = Image.new("RGB", (48, 32))

    asyncio.run(
        handler._run_and_extract(
            VideoGenerationRequest(prompt="A bear playing with yarn.", width=96, height=64),
            "pipeline-owned-resize",
            reference_image=ReferenceImage(image),
        )
    )

    assert engine.captured_prompt is not None
    input_image = engine.captured_prompt["multi_modal_data"]["image"]
    assert isinstance(input_image, Image.Image)
    assert input_image.size == (48, 32)
    assert captured == {
        "model_class_name": "ExamplePipeline",
        "model": "org/model",
        "revision": "pinned-revision",
    }


def test_i2v_minimax_h3_preserves_reference_geometry():
    engine = FakeAsyncOmni()
    engine.model_class_name = "MiniMaxH3Pipeline"
    handler = OmniOpenAIServingVideo.for_diffusion(
        diffusion_engine=engine,
        model_name="MiniMaxAI/MiniMax-H3",
    )
    image = Image.new("RGB", (48, 32))

    asyncio.run(
        handler._run_and_extract(
            VideoGenerationRequest(prompt="A bear playing with yarn.", width=96, height=64),
            "minimax-h3-reference-geometry",
            reference_image=ReferenceImage(image),
        )
    )

    assert engine.captured_prompt is not None
    assert engine.captured_sampling_params_list is not None
    input_image = engine.captured_prompt["multi_modal_data"]["image"]
    assert isinstance(input_image, Image.Image)
    assert input_image.size == (48, 32)
    sampling_params = engine.captured_sampling_params_list[0]
    assert (sampling_params.width, sampling_params.height) == (96, 64)


def test_i2v_extra_params_dimensions_preserve_input_image_geometry(test_client, mocker: MockerFixture):
    image_bytes = _make_test_image_bytes((48, 48))
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        return_value=b"fake-video",
    )
    response = test_client.post(
        "/v1/videos",
        data={
            "prompt": "A bear playing with yarn.",
            "extra_params": json.dumps({"width": 96, "height": 64}),
        },
        files={"input_reference": ("input.png", image_bytes, "image/png")},
    )

    assert response.status_code == 200
    video_id = response.json()["id"]
    _wait_for_status(test_client, video_id, VideoGenerationStatus.COMPLETED.value)

    engine = test_client.app.state.openai_serving_video._engine_client
    input_image = engine.captured_prompt["multi_modal_data"]["image"]
    assert isinstance(input_image, Image.Image)
    assert input_image.size == (48, 48)
    sampling_params = engine.captured_sampling_params_list[0]
    assert sampling_params.extra_args["width"] == 96
    assert sampling_params.extra_args["height"] == 64


@pytest.mark.parametrize(
    ("generation_request", "expected_num_frames", "expected_duration"),
    [
        (
            VideoGenerationRequest(prompt="top-level frames", seconds="5", num_frames=9),
            9,
            5.0,
        ),
        (
            VideoGenerationRequest(prompt="nested frames", video_params=VideoParams(num_frames=9)),
            9,
            None,
        ),
        (
            VideoGenerationRequest(prompt="seconds only", seconds="5"),
            120,
            5.0,
        ),
    ],
)
def test_video_generation_bridges_request_fields(generation_request, expected_num_frames, expected_duration):
    engine = FakeAsyncOmni()
    handler = OmniOpenAIServingVideo.for_diffusion(
        diffusion_engine=engine,
        model_name="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    )

    asyncio.run(handler._run_and_extract(generation_request, "field-bridge"))

    assert engine.captured_sampling_params_list is not None
    sampling = engine.captured_sampling_params_list[0]
    # Top-level ``seconds`` bridges into extra_args["duration"]; num_frames is
    # passed through (or derived as seconds x fps when omitted). No private
    # provenance channel is injected.
    assert "_vllm_request_context" not in sampling.extra_args
    assert sampling.num_frames == expected_num_frames
    if expected_duration is None:
        assert "duration" not in sampling.extra_args
    else:
        assert sampling.extra_args["duration"] == expected_duration


def test_magi2_i2v_preserves_reference_geometry_for_model_preprocessing(test_client, mocker: MockerFixture):
    image_bytes = _make_test_image_bytes((48, 32))
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        return_value=b"fake-video",
    )
    engine = test_client.app.state.openai_serving_video._engine_client
    engine.model_class_name = "Magi2Pipeline"

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
    input_image = engine.captured_prompt["multi_modal_data"]["image"]
    assert isinstance(input_image, Image.Image)
    assert input_image.size == (48, 32)


def test_magi2_serving_applies_native_defaults_and_rejects_explicit_frame_mismatch():
    engine = FakeAsyncOmni()
    engine.model_class_name = "Magi2Pipeline"
    handler = OmniOpenAIServingVideo.for_diffusion(
        diffusion_engine=engine,
        model_name="sand-ai/MAGI-2-preview",
    )

    asyncio.run(handler._run_and_extract(VideoGenerationRequest(prompt="A fox walks through snow"), "defaults"))

    sampling = engine.captured_sampling_params_list[0]
    assert (sampling.width, sampling.height) == (896, 512)
    assert sampling.num_frames == 125
    assert sampling.fps == sampling.frame_rate == 12.5
    assert sampling.num_inference_steps == 100
    assert "duration" not in sampling.extra_args

    with pytest.raises(HTTPException, match="10-second clips only"):
        asyncio.run(
            handler._run_and_extract(
                VideoGenerationRequest(prompt="A fox walks through snow", seconds="5"),
                "bad-duration",
            )
        )
    with pytest.raises(HTTPException, match="10-second clips only"):
        asyncio.run(
            handler._run_and_extract(
                VideoGenerationRequest(
                    prompt="A fox walks through snow",
                    extra_params={"duration": 5},
                ),
                "bad-duration-extra",
            )
        )

    with pytest.raises(HTTPException, match="requires 125 frames"):
        asyncio.run(
            handler._run_and_extract(
                VideoGenerationRequest(prompt="A fox walks through snow", num_frames=1),
                "bad-frames",
            )
        )


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


def test_i2v_video_generation_follows_allowed_image_redirect(test_client, mocker: MockerFixture, monkeypatch):
    requested_paths = []
    async_client = httpx.AsyncClient

    def _handler(request):
        requested_paths.append(request.url.path)
        if request.url.path == "/redirect.png":
            return httpx.Response(302, headers={"location": "/image.png"})
        return httpx.Response(200, content=_make_test_image_bytes((40, 24)))

    def _client_factory(*args, **kwargs):
        return async_client(*args, transport=httpx.MockTransport(_handler), **kwargs)

    monkeypatch.setattr(envs, "VLLM_MEDIA_URL_ALLOW_REDIRECTS", True)
    monkeypatch.setattr(video_api_utils.httpx, "AsyncClient", _client_factory)
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        return_value=b"fake-video",
    )

    response = test_client.post(
        "/v1/videos",
        data={
            "prompt": "A fox running through snow.",
            "image_reference": json.dumps({"image_url": "https://example.com/redirect.png"}),
        },
    )

    assert response.status_code == 200
    video_id = response.json()["id"]
    _wait_for_status(test_client, video_id, VideoGenerationStatus.COMPLETED.value)
    assert requested_paths == ["/redirect.png", "/image.png"]

    engine = test_client.app.state.openai_serving_video._engine_client
    input_image = engine.captured_prompt["multi_modal_data"]["image"]
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


def test_r9_typed_video_reference_form(test_client, mocker: MockerFixture):
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
    assert len(input_video) == 1
    assert all(isinstance(item, str) for item in input_video)
    assert engine.captured_reference_video_bytes is not None
    assert len(engine.captured_reference_video_bytes) == 1
    assert b"ftyp" in engine.captured_reference_video_bytes[0][:32]


def test_r5_mixed_image_video_audio_references_and_fanout(test_client, mocker: MockerFixture):
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        return_value=b"fake-video",
    )
    test_client.app.state.openai_serving_video._engine_client.model_class_name = "MiniMaxH3Pipeline"
    response = test_client.post(
        "/v1/videos",
        data={
            "prompt": "Use both references.",
            "image_reference": json.dumps(
                [
                    {"image_url": _make_test_image_data_url((40, 24))},
                    {"image_url": _make_test_image_data_url((40, 24))},
                ]
            ),
            "video_reference": json.dumps(
                {"video_url": _make_test_video_data_url((32, 24), 2)},
            ),
            "audio_reference": json.dumps(
                [{"audio_url": "data:audio/mp3;base64,ZmFrZQ=="}],
            ),
            "extra_params": json.dumps({"task": "ref2va", "frame_indices": [0, -1], "start_time_seconds": 1.5}),
            "num_outputs_per_prompt": "2",
        },
    )

    assert response.status_code == 200
    video_id = response.json()["id"]
    _wait_for_status(test_client, video_id, VideoGenerationStatus.COMPLETED.value)
    engine = test_client.app.state.openai_serving_video._engine_client
    multi_modal_data = engine.captured_prompt["multi_modal_data"]
    assert len(multi_modal_data["image"]) == 2
    assert len(multi_modal_data["video"]) == 1
    assert all(isinstance(item, str) for item in multi_modal_data["video"])
    assert engine.captured_reference_video_bytes is not None
    assert len(engine.captured_reference_video_bytes) == 1
    assert b"ftyp" in engine.captured_reference_video_bytes[0][:32]
    assert isinstance(multi_modal_data["audio"], str)
    assert engine.captured_sampling_params_list[0].num_outputs_per_prompt == 2


@pytest.mark.parametrize("endpoint", ["/v1/videos", "/v1/videos/sync"])
def test_multi_video_generation_preserves_uploaded_files_until_generation(
    endpoint,
    test_client,
    mocker: MockerFixture,
):
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        return_value=b"fake-video",
    )
    test_client.app.state.openai_serving_video._engine_client.model_class_name = "MiniMaxH3Pipeline"
    response = test_client.post(
        endpoint,
        data={
            "prompt": "Composite the subject from the first video into the second.",
            "extra_params": json.dumps({"task": "ref2va", "duration": 15.0}),
        },
        files=[
            ("input_references", ("subject.png", _make_test_image_bytes((40, 24)), "image/png")),
            ("input_references", ("subject.mp4", b"subject-video", "video/mp4")),
            ("input_references", ("background.mov", b"background-video", "video/quicktime")),
            ("input_references", ("voice.wav", b"reference-audio", "audio/wav")),
        ],
    )

    assert response.status_code == 200
    if endpoint.endswith("/sync"):
        assert response.content == b"fake-video"
    else:
        video_id = response.json()["id"]
        _wait_for_status(test_client, video_id, VideoGenerationStatus.COMPLETED.value)

    engine = test_client.app.state.openai_serving_video._engine_client
    input_videos = engine.captured_prompt["multi_modal_data"]["video"]
    assert engine.captured_reference_video_bytes == [b"subject-video", b"background-video"]
    assert len(input_videos) == 2
    assert all(not Path(path).exists() for path in input_videos)
    assert isinstance(engine.captured_prompt["multi_modal_data"]["image"], Image.Image)
    assert isinstance(engine.captured_prompt["multi_modal_data"]["audio"], str)
    assert engine.captured_sampling_params_list[0].extra_args["task"] == "ref2va"
    assert engine.captured_sampling_params_list[0].extra_args["duration"] == 15.0


def test_mixed_reference_capability_uses_model_metadata_when_config_defaults_false(test_client):
    handler = test_client.app.state.openai_serving_video
    handler._engine_client.model_class_name = None
    handler._engine_client.stage_configs = [
        SimpleNamespace(engine_args={"model_class_name": "MiniMaxH3TextEncoder"}),
        SimpleNamespace(engine_args={"model_class_name": "MiniMaxH3Pipeline"}),
    ]
    handler._stage_configs = handler._engine_client.stage_configs

    assert handler.supports_mixed_reference_inputs


def test_typed_stage_drives_video_capability_checks():
    from vllm_omni.config.config_factory import StageConfigFactory

    minimax_stage = StageConfigFactory.create_typed_default_diffusion(
        "minimax-h3",
        {"model_class_name": "MiniMaxH3Pipeline"},
    ).stage_configs[0]
    minimax_handler = OmniOpenAIServingVideo.for_diffusion(
        SimpleNamespace(od_config=SimpleNamespace(model_class_name=None)),
        model_name="minimax-h3",
        stage_configs=[minimax_stage],
    )
    assert minimax_handler.supports_mixed_reference_inputs

    cosmos_stage = StageConfigFactory.create_typed_default_diffusion(
        "cosmos3",
        {"model_class_name": "Cosmos3OmniDiffusersPipeline"},
    ).stage_configs[0]
    cosmos_handler = OmniOpenAIServingVideo.for_diffusion(
        SimpleNamespace(od_config=SimpleNamespace(model_class_name=None)),
        model_name="cosmos3",
        stage_configs=[cosmos_stage],
    )
    assert cosmos_handler.supported_control_upload_types == frozenset({"edge", "blur", "depth", "seg", "wsm"})


@pytest.mark.parametrize("model_class_name", ["Cosmos3OmniDiffusersPipeline", "Cosmos3OmniPipeline"])
def test_control_upload_capability_is_declared_only_by_cosmos3(test_client, model_class_name):
    handler = test_client.app.state.openai_serving_video
    handler._engine_client.model_class_name = model_class_name

    assert handler.supported_control_upload_types == frozenset({"edge", "blur", "depth", "seg", "wsm"})

    handler._engine_client.model_class_name = "WanPipeline"
    assert handler.supported_control_upload_types == frozenset()


def test_decode_video_bytes_can_keep_first_frames():
    from vllm_omni.entrypoints.openai.video_api_utils import _decode_video_bytes

    frames = _decode_video_bytes(
        _make_test_video_bytes((32, 24), num_frames=6),
        source="input_reference",
        max_frames=2,
        keep="first",
    )

    assert len(frames) == 2
    assert frames.fps == pytest.approx(8.0)
    red_means = [np.asarray(frame)[:, :, 0].mean() for frame in frames]
    assert red_means[0] < red_means[1]
    assert red_means[1] < 100


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

    spec = _reference_video_decode_spec(request, _cosmos3_stage_configs())
    assert spec.max_frames == 9
    assert spec.keep == "first"


@pytest.mark.parametrize("typed", [False, True], ids=["legacy", "typed"])
@pytest.mark.parametrize(
    ("num_frames", "extra_params", "expected"),
    [
        (189, {"condition_frame_indexes_vision": [0, 2], "condition_video_keep": "last"}, (9, "last")),
        (189, {"condition_frame_indexes_vision": [0, 2]}, (9, "first")),
        (5, {"condition_frame_indexes_vision": [0, 20]}, (5, "first")),
        (None, {"action_mode": "inverse_dynamics", "action_chunk_size": 16}, (17, "first")),
    ],
)
def test_cosmos3_reference_video_decode_policy_with_runtime_configs(typed, num_frames, extra_params, expected):
    from vllm_omni.config.config_factory import StageConfigFactory

    if typed:
        stages = list(
            StageConfigFactory.create_typed_default_diffusion(
                "cosmos3",
                {"model_class_name": "Cosmos3OmniDiffusersPipeline"},
            ).stage_configs
        )
    else:
        stages = _cosmos3_stage_configs()
    request = VideoGenerationRequest(prompt="Continue this motion.", num_frames=num_frames, extra_params=extra_params)

    spec = _reference_video_decode_spec(request, stages)

    assert (spec.max_frames, spec.keep) == expected


def test_cosmos3_reference_video_limit_preserves_action_frames():
    request = VideoGenerationRequest(
        prompt="Predict the action.",
        num_frames=17,
        extra_params={"action_mode": "inverse_dynamics", "action_chunk_size": 16},
    )

    assert _reference_video_decode_spec(request, _cosmos3_stage_configs()).max_frames == 17


def test_cosmos3_reference_video_limit_caps_condition_frames_to_output_frames():
    request = VideoGenerationRequest(
        prompt="Continue this motion.",
        num_frames=5,
        extra_params={"condition_frame_indexes_vision": [0, 20]},
    )

    assert _reference_video_decode_spec(request, _cosmos3_stage_configs()).max_frames == 5


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

    async def _generate(prompt, request_id, sampling_params_list, **kwargs):
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
            quality="high",
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
    assert captured.quality == "high"
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

    async def _generate(prompt, request_id, sampling_params_list, **kwargs):
        engine.captured_prompt = prompt
        engine.captured_sampling_params_list = sampling_params_list
        import numpy as np

        yield MockVideoResult(
            [np.zeros((1, 64, 64, 3), dtype=np.uint8)],
            multimodal_output={
                "video": [np.zeros((1, 64, 64, 3), dtype=np.uint8)],
                "metadata": {"video": {"video_fps_multiplier": 2}},
            },
        )

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

    def _fake_encode(
        video,
        fps,
        audio=None,
        audio_sample_rate=None,
        video_codec_options=None,
        frame_converter=None,
    ):
        del video, fps, audio, video_codec_options, frame_converter
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

    async def _generate(prompt, request_id, sampling_params_list, **kwargs):
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

    async def _generate(prompt, request_id, sampling_params_list, **kwargs):
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

    async def _generate(prompt, request_id, sampling_params_list, **kwargs):
        del prompt, request_id, sampling_params_list
        import numpy as np

        yield MockVideoResult(
            [object()],
            multimodal_output={
                "video": [object()],
                "actions": np.array([[[1.5, 2.5], [3.5, 4.5]]], dtype=np.float32),
                "metadata": {
                    "actions": {
                        "raw_action_dim": 2,
                        "action_mode": "policy",
                        "domain_id": 7,
                    },
                },
            },
        )

    engine.generate = _generate  # type: ignore[method-assign]
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

    async def _generate(prompt, request_id, sampling_params_list, **kwargs):
        import numpy as np

        engine.captured_prompt = prompt
        engine.captured_sampling_params_list = sampling_params_list
        yield MockVideoResult(
            [object()],
            multimodal_output={
                "video": [object()],
                "actions": np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32),
                "metadata": {
                    "actions": {
                        "raw_action_dim": 2,
                        "action_mode": "policy",
                        "domain_id": 7,
                    },
                },
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
        multimodal_output={
            "video": [object()],
            "actions": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "metadata": {
                "actions": {
                    "raw_action_dim": 2,
                    "action_mode": "policy",
                    "domain_id": 7,
                },
            },
        },
    )

    actions = OmniOpenAIServingVideo._extract_action_outputs(result, expected_count=1)

    assert actions and actions[0] is not None
    assert actions[0].data == [[1.0, 2.0], [3.0, 4.0]]
    assert actions[0].shape == [2, 2]


def test_action_extraction_accepts_multimodal_actions_payload():
    import numpy as np

    result = MockVideoResult([object()])
    result.multimodal_output.update(
        {
            "actions": np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32),
            "metadata": {
                "actions": {
                    "raw_action_dim": 2,
                    "action_mode": "policy",
                    "domain_id": 7,
                },
            },
        }
    )

    actions = OmniOpenAIServingVideo._extract_action_outputs(result, expected_count=1)

    assert actions[0] is not None
    assert actions[0].data == [[1.0, 2.0], [3.0, 4.0]]
    assert actions[0].shape == [2, 2]
    assert actions[0].raw_action_dim == 2
    assert actions[0].action_mode == "policy"
    assert actions[0].domain_id == 7


def test_missing_handler_returns_503():
    app = FastAPI()
    app.state.api_server_count = 1
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


def test_r10_typed_image_and_video_reference_form(test_client, mocker: MockerFixture):
    mocker.patch(
        "vllm_omni.entrypoints.openai.serving_video._encode_video_bytes",
        return_value=b"fake-video",
    )
    test_client.app.state.openai_serving_video._engine_client.model_class_name = "MiniMaxH3Pipeline"
    response = test_client.post(
        "/v1/videos",
        data={
            "prompt": "use both references",
            "image_reference": json.dumps({"image_url": _make_test_image_data_url((40, 24))}),
            "video_reference": json.dumps({"video_url": _make_test_video_data_url((32, 24), 2)}),
        },
    )
    assert response.status_code == 200
    _wait_for_status(test_client, response.json()["id"], VideoGenerationStatus.COMPLETED.value)


def test_generic_video_model_rejects_mixed_image_and_video_references(test_client):
    response = test_client.post(
        "/v1/videos/sync",
        data={
            "prompt": "use both references",
            "image_reference": json.dumps({"image_url": _make_test_image_data_url((40, 24))}),
            "video_reference": json.dumps({"video_url": _make_test_video_data_url((32, 24), 2)}),
        },
    )

    assert response.status_code == 400
    assert "does not support mixed image and video" in response.json()["detail"].lower()


def test_h3_multipart_rejects_bmp_image_reference(test_client, monkeypatch):
    monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", 100)
    image = Image.new("RGB", (64, 64), color="blue")
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="BMP")

    test_client.app.state.openai_serving_video._engine_client.model_class_name = "MiniMaxH3Pipeline"
    response = test_client.post(
        "/v1/videos/sync",
        data={"prompt": "reject unsupported image container", "extra_params": '{"task":"ref2va"}'},
        files=[("input_references", ("reference.bmp", image_buffer.getvalue(), "image/bmp"))],
    )

    assert response.status_code == 400
    assert "must use jpg" in response.json()["detail"].lower()


@pytest.mark.parametrize("field", ["input_reference", "input_references"])
def test_h3_multipart_rejects_image_over_pixel_limit(field, test_client, monkeypatch):
    monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", 100)
    test_client.app.state.openai_serving_video._engine_client.model_class_name = "MiniMaxH3Pipeline"

    response = test_client.post(
        "/v1/videos/sync",
        data={"prompt": "reject oversized image", "extra_params": '{"task":"ref2va"}'},
        files=[(field, ("reference.png", _make_test_image_bytes((20, 20)), "image/png"))],
    )

    assert response.status_code == 400
    assert "VLLM_MAX_IMAGE_PIXELS" in response.json()["detail"]


@pytest.mark.parametrize("field", ["input_reference", "input_references"])
def test_h3_multipart_maps_pillow_pixel_limit_error(field, test_client, monkeypatch):
    monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", 0)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    test_client.app.state.openai_serving_video._engine_client.model_class_name = "MiniMaxH3Pipeline"

    response = test_client.post(
        "/v1/videos/sync",
        data={"prompt": "reject decoder bomb", "extra_params": '{"task":"ref2va"}'},
        files=[(field, ("reference.png", _make_test_image_bytes((20, 20)), "image/png"))],
    )

    assert response.status_code == 400
    assert "decoder pixel limit" in response.json()["detail"]


@pytest.mark.asyncio
async def test_h3_upload_limit_checks_declared_size_before_read():
    class OversizedUpload:
        size = MINIMAX_H3_MAX_REFERENCE_IMAGE_BYTES + 1

        async def read(self, _size):
            raise AssertionError("the oversized upload must be rejected before reading")

    with pytest.raises(HTTPException, match="size limit"):
        await _read_upload_limited(
            OversizedUpload(),
            max_bytes=MINIMAX_H3_MAX_REFERENCE_IMAGE_BYTES,
        )


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
    assert req.quality is None
    assert req.generate_sound is False
    assert req.sound_duration is None
    assert VideoGenerationRequest(prompt="test", fps=12.5).resolve_video_params().fps == 12.5
    with pytest.raises(ValueError):
        VideoGenerationRequest(prompt="test", fps=float("inf"))
    with pytest.raises(ValueError):
        VideoGenerationRequest(prompt="test", video_params={"fps": float("nan")})
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
    with pytest.raises(ValueError):
        VideoGenerationRequest(prompt="test", quality="medium")


def test_async_create_accepts_fractional_fps(test_client, mocker: MockerFixture):
    """Queued VideoResponse must accept fractional fps from the request path."""
    _mock_encode_video_bytes(mocker)
    response = test_client.post(
        "/v1/videos",
        data={"prompt": "fractional fps", "fps": "12.5", "num_frames": "5"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["fps"] == 12.5
    assert body["num_frames"] == 5
    video_id = body["id"]
    _wait_for_status(test_client, video_id, VideoGenerationStatus.COMPLETED.value)
    engine = test_client.app.state.openai_serving_video._engine_client
    assert engine.captured_sampling_params_list[0].fps == 12.5
    assert engine.captured_sampling_params_list[0].frame_rate == 12.5


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


def test_async_video_stays_queued_until_scheduler_admission(test_client):
    handler = SchedulerQueuedVideoHandler()
    test_client.app.state.openai_serving_video = handler

    create_resp = test_client.post("/v1/videos", data={"prompt": "Queue this video"})
    assert create_resp.status_code == 200
    video_id = create_resp.json()["id"]
    assert handler.started.wait(timeout=2.0)

    queued = test_client.get(f"/v1/videos/{video_id}")
    assert queued.status_code == 200
    assert queued.json()["status"] == VideoGenerationStatus.QUEUED.value

    handler.admit.set()
    assert handler.in_progress.wait(timeout=2.0)
    in_progress = test_client.get(f"/v1/videos/{video_id}")
    assert in_progress.status_code == 200
    assert in_progress.json()["status"] == VideoGenerationStatus.IN_PROGRESS.value

    assert test_client.delete(f"/v1/videos/{video_id}").status_code == 200


def test_delete_times_out_engine_abort_and_still_cancels(test_client, monkeypatch):
    monkeypatch.setattr(api_server, "VIDEO_ABORT_TIMEOUT_S", 0.05)
    handler = HangingAbortHandler()
    test_client.app.state.openai_serving_video = handler

    create_resp = test_client.post("/v1/videos", data={"prompt": "Hang abort"})
    assert create_resp.status_code == 200
    video_id = create_resp.json()["id"]
    assert handler.started.wait(timeout=2.0)

    started = time.monotonic()
    delete_resp = test_client.delete(f"/v1/videos/{video_id}")
    assert time.monotonic() - started < 2.0
    assert delete_resp.status_code == 200
    assert handler.cancelled.wait(timeout=2.0)


def test_delete_removes_artifact_if_job_completes_during_abort(test_client):
    handler = CompletingDuringAbortHandler()
    test_client.app.state.openai_serving_video = handler

    create_resp = test_client.post("/v1/videos", data={"prompt": "Complete during delete"})
    assert create_resp.status_code == 200
    video_id = create_resp.json()["id"]
    assert handler.started.wait(timeout=2.0)

    assert test_client.delete(f"/v1/videos/{video_id}").status_code == 200
    file_path = os.path.join(api_server.STORAGE_MANAGER.storage_path, video_id)
    assert not os.path.exists(file_path)
    assert asyncio.run(api_server.VIDEO_STORE.get(video_id)) is None


def test_delete_aborts_engine_request_before_cancelling_task(test_client):
    engine = AbortTrackingOmni()
    test_client.app.state.openai_serving_video = OmniOpenAIServingVideo.for_diffusion(
        engine,
        model_name="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    )

    create_resp = test_client.post("/v1/videos", data={"prompt": "Abort this video"})
    assert create_resp.status_code == 200
    video_id = create_resp.json()["id"]
    assert engine.entered.wait(timeout=2.0)
    _wait_for_status(test_client, video_id, VideoGenerationStatus.IN_PROGRESS.value)

    delete_resp = test_client.delete(f"/v1/videos/{video_id}")
    assert delete_resp.status_code == 200
    assert delete_resp.json()["deleted"] is True
    _wait_until(lambda: engine.aborted == [video_id])
    assert asyncio.run(api_server.VIDEO_STORE.get(video_id)) is None


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

    async def _generate(prompt, request_id, sampling_params_list, **kwargs):
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


@pytest.mark.parametrize("endpoint", ["/v1/videos", "/v1/videos/sync"])
@pytest.mark.parametrize("control_type", ["wsm", "depth"])
def test_cosmos3_accepts_optional_uploaded_control(
    endpoint,
    control_type,
    test_client,
    mocker: MockerFixture,
):
    control_bytes = f"{control_type}-control".encode()
    _mock_encode_video_bytes(mocker, b"controlled-video")
    engine = test_client.app.state.openai_serving_video._engine_client
    engine.model_class_name = "Cosmos3OmniDiffusersPipeline"

    response = test_client.post(
        endpoint,
        data={
            "prompt": "Follow the uploaded control.",
            "control_type": control_type,
            "extra_params": json.dumps({control_type: {"control_weight": 0.75}}),
        },
        files=[
            ("input_reference", ("input.mp4", _make_test_video_bytes(), "video/mp4")),
            ("control_reference", (f"{control_type}.mp4", control_bytes, "video/mp4")),
        ],
    )

    assert response.status_code == 200
    if endpoint.endswith("/sync"):
        assert response.content == b"controlled-video"
    else:
        video_id = response.json()["id"]
        _wait_for_status(test_client, video_id, VideoGenerationStatus.COMPLETED.value)

    captured = engine.captured_sampling_params_list[0].extra_args[control_type]
    assert captured["control_weight"] == 0.75
    assert engine.captured_control_reference_bytes[control_type] == control_bytes
    assert len(engine.captured_prompt["multi_modal_data"]["video"]) == 3
    assert not Path(captured["control_path"]).exists()


@pytest.mark.parametrize("endpoint", ["/v1/videos", "/v1/videos/sync"])
def test_cosmos3_uploaded_control_selects_transfer_reference_video_decode_policy(
    endpoint,
    test_client,
    mocker: MockerFixture,
):
    _mock_encode_video_bytes(mocker, b"controlled-video")
    test_client.app.state.stage_configs = _cosmos3_stage_configs()

    response = test_client.post(
        endpoint,
        data={
            "prompt": "Preserve all conditioning motion.",
            "control_type": "wsm",
            "num_frames": "9",
            "extra_params": json.dumps({"num_first_chunk_conditional_frames": 9}),
        },
        files=[
            (
                "input_reference",
                ("input.mp4", _make_test_video_bytes(num_frames=6), "video/mp4"),
            ),
            ("control_reference", ("wsm.mp4", b"control", "video/mp4")),
        ],
    )

    assert response.status_code == 200
    if not endpoint.endswith("/sync"):
        video_id = response.json()["id"]
        _wait_for_status(test_client, video_id, VideoGenerationStatus.COMPLETED.value)

    engine = test_client.app.state.openai_serving_video._engine_client
    assert len(engine.captured_prompt["multi_modal_data"]["video"]) == 6


def test_cosmos3_control_upload_is_optional(test_client, mocker: MockerFixture):
    _mock_encode_video_bytes(mocker)
    engine = test_client.app.state.openai_serving_video._engine_client
    engine.model_class_name = "Cosmos3OmniDiffusersPipeline"

    response = test_client.post("/v1/videos/sync", data={"prompt": "No control for this request."})

    assert response.status_code == 200
    assert "wsm" not in engine.captured_sampling_params_list[0].extra_args


def test_control_upload_does_not_affect_models_without_capability(test_client):
    response = test_client.post(
        "/v1/videos/sync",
        data={"prompt": "Unsupported control.", "control_type": "wsm"},
        files={"control_reference": ("wsm.mp4", b"control", "video/mp4")},
    )

    assert response.status_code == 400
    assert "not supported by this model" in response.json()["detail"]
    assert test_client.app.state.openai_serving_video._engine_client.captured_prompt is None


@pytest.mark.parametrize(
    ("data", "files", "message"),
    [
        (
            {"prompt": "Missing type."},
            {"control_reference": ("wsm.mp4", b"control", "video/mp4")},
            "requires control_type",
        ),
        (
            {"prompt": "Missing file.", "control_type": "wsm"},
            None,
            "requires a control_reference",
        ),
        (
            {"prompt": "Unknown type.", "control_type": "unknown"},
            {"control_reference": ("control.mp4", b"control", "video/mp4")},
            "not supported by this model",
        ),
    ],
)
def test_cosmos3_control_upload_validates_contract(data, files, message, test_client):
    test_client.app.state.openai_serving_video._engine_client.model_class_name = "Cosmos3OmniDiffusersPipeline"

    response = test_client.post("/v1/videos/sync", data=data, files=files)

    assert response.status_code == 400
    assert message in response.json()["detail"]


def test_cosmos3_control_upload_rejects_existing_control_source(test_client):
    test_client.app.state.openai_serving_video._engine_client.model_class_name = "Cosmos3OmniDiffusersPipeline"
    response = test_client.post(
        "/v1/videos/sync",
        data={
            "prompt": "Ambiguous control.",
            "control_type": "wsm",
            "extra_params": json.dumps({"wsm": {"control_path": "/already/present.mp4"}}),
        },
        files={"control_reference": ("wsm.mp4", b"control", "video/mp4")},
    )

    assert response.status_code == 400
    assert "not both" in response.json()["detail"]


@pytest.mark.parametrize(
    ("control_bytes", "message"),
    [
        (b"", "must not be empty"),
        (b"control", "size limit"),
    ],
)
def test_cosmos3_control_upload_rejects_invalid_size(control_bytes, message, test_client, monkeypatch):
    test_client.app.state.openai_serving_video._engine_client.model_class_name = "Cosmos3OmniDiffusersPipeline"
    monkeypatch.setattr(video_generation_helpers, "CONTROL_REFERENCE_MAX_BYTES", 3)

    response = test_client.post(
        "/v1/videos/sync",
        data={"prompt": "Invalid control.", "control_type": "wsm"},
        files={"control_reference": ("wsm.mp4", control_bytes, "video/mp4")},
    )

    assert response.status_code == 400
    assert message in response.json()["detail"]
    assert test_client.app.state.openai_serving_video._engine_client.captured_prompt is None


def test_sync_missing_handler_returns_503():
    app = FastAPI()
    app.state.api_server_count = 1
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
            "seconds": "10",
            "fps": "12.5",
            "num_inference_steps": "30",
            "guidance_scale": "6.5",
            "seed": "42",
            "quality": "high",
        },
    )
    assert response.status_code == 200
    engine = test_client.app.state.openai_serving_video._engine_client
    captured = engine.captured_sampling_params_list[0]
    assert captured.num_inference_steps == 30
    assert captured.guidance_scale == 6.5
    assert captured.seed == 42
    assert captured.quality == "high"
    assert captured.num_frames == 125
    assert captured.fps == 12.5
    assert captured.frame_rate == 12.5
    assert captured.extra_args["duration"] == 10.0


def test_sync_sana_wm_extra_params_payload_passes_to_engine_prompt(test_client, mocker: MockerFixture):
    _mock_encode_video_bytes(mocker)
    response = test_client.post(
        "/v1/videos/sync",
        data={
            "prompt": "drive forward",
            "extra_params": json.dumps(
                {
                    "sana_wm": {"action": "d-4", "rotation_speed_deg": 1.5},
                    "sana_wm_native_max_tokens": 30000,
                }
            ),
        },
    )

    assert response.status_code == 200
    engine = test_client.app.state.openai_serving_video._engine_client
    captured_params = engine.captured_sampling_params_list[0]
    # The camera block reaches the model through extra_args; the Sana preprocess
    # hook is what lifts it onto the prompt.
    assert captured_params.extra_args["sana_wm"]["action"] == "d-4"
    assert captured_params.extra_args["sana_wm"]["rotation_speed_deg"] == 1.5
    assert captured_params.extra_args["sana_wm_native_max_tokens"] == 30000


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
            quality="high",
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
    assert captured.quality == "high"
    assert captured.enable_frame_interpolation is True
    assert captured.frame_interpolation_exp == 2
    assert captured.frame_interpolation_scale == 0.5
    assert captured.frame_interpolation_model_path == "default-rife"


def test_worker_fps_multiplier_is_applied_to_sync_encoding(test_client, mocker: MockerFixture):
    engine = test_client.app.state.openai_serving_video._engine_client
    fps_values = []

    async def _generate(prompt, request_id, sampling_params_list, **kwargs):
        engine.captured_prompt = prompt
        engine.captured_sampling_params_list = sampling_params_list
        yield MockVideoResult(
            [object()],
            multimodal_output={
                "video": [object()],
                "metadata": {"video": {"video_fps_multiplier": 2}},
            },
        )

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
