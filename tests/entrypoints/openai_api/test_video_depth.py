# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the ``/v1/videos/depth`` endpoint with a fake engine."""

import io
import os
from argparse import Namespace
from http import HTTPStatus
from types import SimpleNamespace

import av
import numpy as np
import pybase64 as base64
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from vllm.entrypoints.openai.models.protocol import BaseModelPath

from vllm_omni.entrypoints.openai.api_server import router
from vllm_omni.entrypoints.openai.models.serving import _DiffusionServingModels

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

FRAMES, HEIGHT, WIDTH = 3, 5, 7


def _depth() -> np.ndarray:
    """A left-to-right ramp from 1 to 3 that brightens from frame to frame; odd sides exercise padding."""
    ramp = np.linspace(1.0, 2.0, WIDTH, dtype=np.float32)[None, None, :]
    frames = np.arange(FRAMES, dtype=np.float32)[:, None, None] * 0.5
    return np.broadcast_to(ramp + frames, (FRAMES, HEIGHT, WIDTH)).astype(np.float16)


class FakeVideoDepthEngine:
    def __init__(self, model_class_name: str = "VideoDepthAnythingPipeline"):
        self.od_config = SimpleNamespace(model_class_name=model_class_name, supports_multimodal_inputs=False)
        self.captured_prompt: dict | None = None
        self.captured_sampling_params_list: list | None = None
        self.video_bytes: bytes | None = None

    def get_diffusion_od_config(self):
        return self.od_config

    async def generate(self, prompt, request_id, sampling_params_list=None, **kwargs):
        self.captured_prompt = prompt
        self.captured_sampling_params_list = sampling_params_list
        with open(prompt["multi_modal_data"]["video"], "rb") as f:
            self.video_bytes = f.read()
        payload = {"depth": _depth(), "fps": 24.0, "width": WIDTH, "height": HEIGHT, "num_frames": FRAMES}
        yield SimpleNamespace(multimodal_output={"video_depth": {**payload, "metric": False}})


def _client(engine) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.state.engine_client = engine
    app.state.diffusion_engine = engine
    app.state.stage_configs = [SimpleNamespace(stage_type="diffusion")]
    app.state.openai_serving_models = _DiffusionServingModels(
        [BaseModelPath(name="depth-anything/Video-Depth-Anything-Small", model_path="vda")]
    )
    app.state.args = Namespace(default_sampling_params=None, max_generated_image_size=None)
    return TestClient(app)


def _decode(video: str, format: str) -> list[np.ndarray]:
    with av.open(io.BytesIO(base64.b64decode(video))) as container:
        return [frame.to_ndarray(format=format) for frame in container.decode(video=0)]


def test_video_depth_returns_a_normalized_depth_video_and_cleans_up_the_upload():
    engine = FakeVideoDepthEngine()
    response = _client(engine).post(
        "/v1/videos/depth",
        files={"video": ("clip.mp4", b"fake video bytes", "video/mp4")},
        data={"input_size": "392", "max_res": "960"},
    )
    assert response.status_code == HTTPStatus.OK, response.text
    body = response.json()
    assert (body["width"], body["height"], body["num_frames"], body["fps"]) == (WIDTH, HEIGHT, FRAMES, 24.0)
    assert body["metric"] is False and (body["depth_min"], body["depth_max"]) == (1.0, 3.0)
    assert body["depth"] is None
    # Normalized over the whole video: frame 0 starts black, frame 2 ends white; the odd edge is cropped back.
    frames = [frame[:HEIGHT, :WIDTH] for frame in _decode(body["video"], "gray")]
    assert len(frames) == FRAMES
    assert frames[0][:, 0].max() < 40 and frames[2][:, -1].min() > 215
    assert frames[1].mean() > frames[0].mean()

    assert engine.captured_sampling_params_list is not None and engine.captured_prompt is not None
    assert engine.captured_sampling_params_list[0].extra_args == {"input_size": 392, "max_res": 960}
    assert engine.video_bytes == b"fake video bytes"
    assert not os.path.exists(engine.captured_prompt["multi_modal_data"]["video"])


def test_video_depth_returns_raw_depth_and_a_colored_video_on_request():
    response = _client(FakeVideoDepthEngine()).post(
        "/v1/videos/depth",
        files={"video": ("clip.mp4", b"x", "video/mp4")},
        data={"colormap": "inferno", "return_depth": "true"},
    )
    assert response.status_code == HTTPStatus.OK, response.text
    body = response.json()
    with np.load(io.BytesIO(base64.b64decode(body["depth"]))) as archive:
        np.testing.assert_array_equal(archive["depth"], _depth())
    # Inferno runs from near-black to pale yellow: the far end is dark, the near end bright and warm.
    # Chroma is subsampled on this tiny frame, so compare column means.
    first, last = (frame[:HEIGHT, :WIDTH].astype(float) for frame in _decode(body["video"], "rgb24")[::2])
    assert first[:, 0].sum(axis=-1).mean() < 60 and last[:, -1].sum(axis=-1).mean() > 500
    assert last[:, -1, 0].mean() > last[:, -1, 2].mean()


@pytest.mark.parametrize(
    ("files", "data", "message"),
    [
        (None, {}, "Upload the input 'video'"),
        ({"video": ("clip.mp4", b"x", "video/mp4")}, {"colormap": "jet"}, "colormap must be one of"),
    ],
)
def test_video_depth_rejects_bad_requests(files, data, message):
    response = _client(FakeVideoDepthEngine()).post("/v1/videos/depth", files=files, data=data)
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert message in response.text


def test_video_depth_is_rejected_for_models_without_support():
    response = _client(FakeVideoDepthEngine(model_class_name="Sam3VideoModel")).post(
        "/v1/videos/depth", files={"video": ("clip.mp4", b"x", "video/mp4")}
    )
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert "does not support video depth" in response.text
