# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the ``/v1/videos/segmentations`` endpoint with a fake engine."""

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

WIDTH, HEIGHT = 6, 4


def _rle(mask: np.ndarray) -> list[int]:
    flat = mask.flatten(order="F").astype(np.uint8)
    bounds = np.concatenate([[0], np.flatnonzero(np.diff(flat)) + 1, [flat.size]])
    counts = np.diff(bounds).tolist()
    return [0, *counts] if flat[0] else counts


def _payload() -> dict:
    """Object 0 (person) on frames 0-2 in the top row; object 1 (dog) appears on frame 1 as a 2x2 square."""
    top = np.zeros((HEIGHT, WIDTH), dtype=bool)
    top[0] = True
    square = np.zeros((HEIGHT, WIDTH), dtype=bool)
    square[2:4, 2:4] = True
    frames = [
        {"object_ids": [0], "scores": [0.9], "boxes": [[0.0, 0.0, 5.0, 0.0]], "rle": [_rle(top)]},
        {
            "object_ids": [0, 1],
            "scores": [0.9, 0.8],
            "boxes": [[0.0, 0.0, 5.0, 0.0], [2.0, 2.0, 3.0, 3.0]],
            "rle": [_rle(top), _rle(square)],
        },
        {"object_ids": [], "scores": [], "boxes": [], "rle": []},
    ]
    return {
        "width": WIDTH,
        "height": HEIGHT,
        "num_frames": 3,
        "fps": 24.0,
        "objects": {0: "person", 1: "dog"},
        "frames": frames,
    }


class FakeVideoSegmentationEngine:
    def __init__(self, model_class_name: str = "Sam3VideoModel"):
        self.od_config = SimpleNamespace(model_class_name=model_class_name, supports_multimodal_inputs=True)
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
        yield SimpleNamespace(multimodal_output={"video_segmentation": _payload()})


def _client(engine) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.state.engine_client = engine
    app.state.diffusion_engine = engine
    app.state.stage_configs = [SimpleNamespace(stage_type="diffusion")]
    app.state.openai_serving_models = _DiffusionServingModels([BaseModelPath(name="facebook/sam3", model_path="sam3")])
    app.state.args = Namespace(default_sampling_params=None, max_generated_image_size=None)
    return TestClient(app)


def test_video_segmentation_tracks_objects_and_cleans_up_the_upload():
    engine = FakeVideoSegmentationEngine()
    response = _client(engine).post(
        "/v1/videos/segmentations",
        files={"video": ("clip.mp4", b"fake video bytes", "video/mp4")},
        data={"prompt": " person ", "prompts": '["dog"]'},
    )
    assert response.status_code == HTTPStatus.OK, response.text
    body = response.json()
    assert (body["width"], body["height"], body["num_frames"], body["fps"]) == (WIDTH, HEIGHT, 3, 24.0)
    assert body["objects"] == [
        {"id": 0, "prompt": "person", "mask_video": None},
        {"id": 1, "prompt": "dog", "mask_video": None},
    ]
    frame = body["frames"][1]
    assert frame["frame_index"] == 1 and [item["id"] for item in frame["instances"]] == [0, 1]
    assert frame["instances"][1]["box"] == [2.0, 2.0, 3.0, 3.0]
    assert frame["instances"][1]["mask"] == {"size": [HEIGHT, WIDTH], "counts": _payload()["frames"][1]["rle"][1]}
    assert body["frames"][2]["instances"] == []

    assert engine.captured_prompt is not None and engine.captured_sampling_params_list is not None
    assert engine.captured_prompt["prompt"] == "person"
    assert engine.captured_sampling_params_list[0].extra_args == {"prompts": ["dog"]}
    assert engine.video_bytes == b"fake video bytes"
    assert not os.path.exists(engine.captured_prompt["multi_modal_data"]["video"])


def test_video_segmentation_mask_videos_follow_each_object():
    response = _client(FakeVideoSegmentationEngine()).post(
        "/v1/videos/segmentations",
        files={"video": ("clip.mp4", b"x", "video/mp4")},
        data={"prompt": "person", "mask_video": "true"},
    )
    assert response.status_code == HTTPStatus.OK, response.text
    dog = response.json()["objects"][1]
    with av.open(io.BytesIO(base64.b64decode(dog["mask_video"]))) as container:
        frames = [frame.to_ndarray(format="gray") for frame in container.decode(video=0)]
    assert len(frames) == 3 and frames[0].shape == (HEIGHT, WIDTH)
    # Absent on frame 0, a white square on frame 1 (lossy codec: compare against mid-gray).
    assert frames[0].max() < 128
    assert (frames[1][2:4, 2:4] > 128).all() and (frames[1][:2] < 128).all()


@pytest.mark.parametrize(
    ("files", "data", "message"),
    [
        (None, {"prompt": "x"}, "Upload the input 'video'"),
        ({"video": ("clip.mp4", b"x", "video/mp4")}, {"prompt": " "}, "text 'prompt'"),
        ({"video": ("clip.mp4", b"x", "video/mp4")}, {"prompt": "x", "prompts": "{}"}, "JSON array"),
    ],
)
def test_video_segmentation_rejects_bad_requests(files, data, message):
    response = _client(FakeVideoSegmentationEngine()).post("/v1/videos/segmentations", files=files, data=data)
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert message in response.text


def test_video_segmentation_is_rejected_for_models_without_support():
    engine = FakeVideoSegmentationEngine(model_class_name="QwenImagePipeline")
    response = _client(engine).post(
        "/v1/videos/segmentations",
        files={"video": ("clip.mp4", b"x", "video/mp4")},
        data={"prompt": "x"},
    )
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert "does not support video segmentation" in response.text
