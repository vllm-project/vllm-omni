# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the ``/v1/images/segmentations`` endpoint with a fake engine."""

import io
from argparse import Namespace
from http import HTTPStatus
from types import SimpleNamespace

import numpy as np
import pybase64 as base64
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from vllm.entrypoints.openai.models.protocol import BaseModelPath

from vllm_omni.entrypoints.openai.api_server import router
from vllm_omni.entrypoints.openai.models.serving import _DiffusionServingModels

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

WIDTH, HEIGHT = 6, 4


def _payload() -> dict:
    """Two instances: a 2x2 square (score 0.6) and the full top row (score 0.9)."""
    masks = np.zeros((2, HEIGHT, WIDTH), dtype=bool)
    masks[0, 1:3, 1:3] = True
    masks[1, 0, :] = True
    return {
        "masks": np.packbits(masks.reshape(2, -1), axis=1),
        "mask_shape": np.array([2, HEIGHT, WIDTH]),
        "boxes": np.array([[1, 1, 3, 3], [0, 0, 6, 1]], dtype=np.float32),
        "scores": np.array([0.6, 0.9], dtype=np.float32),
    }


def _empty_payload() -> dict:
    """No instance cleared the threshold, packed the way ``Sam3Pipeline`` packs it."""
    return {
        "masks": np.packbits(np.zeros((0, HEIGHT * WIDTH), dtype=bool), axis=1),
        "mask_shape": np.array([0, HEIGHT, WIDTH], dtype=np.int64),
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "scores": np.zeros((0,), dtype=np.float32),
    }


class FakeSegmentationEngine:
    def __init__(self, model_class_name: str = "Sam3VideoModel", payload=_payload):
        self.od_config = SimpleNamespace(model_class_name=model_class_name, supports_multimodal_inputs=True)
        self.payload = payload
        self.captured_prompt: dict | None = None
        self.captured_sampling_params_list: list | None = None

    def get_diffusion_od_config(self):
        return self.od_config

    async def generate(self, prompt, request_id, sampling_params_list=None, **kwargs):
        self.captured_prompt = prompt
        self.captured_sampling_params_list = sampling_params_list
        yield SimpleNamespace(multimodal_output={"segmentation": self.payload()})


def _client(engine: FakeSegmentationEngine) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.state.engine_client = engine
    app.state.diffusion_engine = engine
    app.state.stage_configs = [SimpleNamespace(stage_type="diffusion")]
    app.state.openai_serving_models = _DiffusionServingModels([BaseModelPath(name="facebook/sam3", model_path="sam3")])
    app.state.args = Namespace(default_sampling_params=None, max_generated_image_size=None)
    return TestClient(app)


def _image_file() -> tuple[str, bytes, str]:
    buffer = io.BytesIO()
    Image.new("RGB", (WIDTH, HEIGHT), color="red").save(buffer, format="PNG")
    return ("image.png", buffer.getvalue(), "image/png")


def _decode_rle(rle: dict) -> np.ndarray:
    height, width = rle["size"]
    flat = np.zeros(height * width, dtype=bool)
    position, value = 0, False
    for count in rle["counts"]:
        flat[position : position + count] = value
        position, value = position + count, not value
    return flat.reshape((height, width), order="F")


def test_segmentation_returns_png_masks_ordered_by_score_and_forwards_prompts():
    engine = FakeSegmentationEngine()
    response = _client(engine).post(
        "/v1/images/segmentations",
        files={"image": _image_file()},
        data={
            "prompt": " red square ",
            "boxes": "[[1, 1, 3, 3]]",
            "box_labels": "[0]",
            "score_threshold": "0.25",
        },
    )
    assert response.status_code == HTTPStatus.OK, response.text
    body = response.json()
    assert (body["width"], body["height"], body["mask_format"]) == (WIDTH, HEIGHT, "png")
    assert [item["score"] for item in body["data"]] == pytest.approx([0.9, 0.6])
    assert body["data"][0]["box"] == [0, 0, 6, 1]
    top_row = np.array(Image.open(io.BytesIO(base64.b64decode(body["data"][0]["mask"]))))
    assert top_row.shape == (HEIGHT, WIDTH) and top_row[0].all() and not top_row[1:].any()

    assert engine.captured_prompt is not None and engine.captured_sampling_params_list is not None
    assert engine.captured_prompt["prompt"] == "red square"
    assert engine.captured_prompt["multi_modal_data"]["image"].size == (WIDTH, HEIGHT)
    extra_args = engine.captured_sampling_params_list[0].extra_args
    assert extra_args == {"boxes": [[1, 1, 3, 3]], "box_labels": [0], "score_threshold": 0.25}


def test_segmentation_rle_masks_round_trip():
    response = _client(FakeSegmentationEngine()).post(
        "/v1/images/segmentations",
        files={"image": _image_file()},
        data={"prompt": "square", "mask_format": "rle"},
    )
    assert response.status_code == HTTPStatus.OK, response.text
    square = response.json()["data"][1]["mask"]
    assert square["size"] == [HEIGHT, WIDTH] and sum(square["counts"]) == HEIGHT * WIDTH
    expected = np.zeros((HEIGHT, WIDTH), dtype=bool)
    expected[1:3, 1:3] = True
    np.testing.assert_array_equal(_decode_rle(square), expected)


@pytest.mark.parametrize("mask_format", ["png", "rle"])
def test_segmentation_without_matches_returns_empty_data(mask_format):
    response = _client(FakeSegmentationEngine(payload=_empty_payload)).post(
        "/v1/images/segmentations",
        files={"image": _image_file()},
        data={"prompt": "absent concept", "mask_format": mask_format},
    )
    assert response.status_code == HTTPStatus.OK, response.text
    body = response.json()
    assert body["data"] == []
    assert (body["width"], body["height"], body["mask_format"]) == (WIDTH, HEIGHT, mask_format)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"prompt": " "}, "prompt"),
        ({"prompt": "x", "boxes": "not json"}, "JSON array"),
        ({"prompt": "x", "box_labels": "{}"}, "JSON array"),
    ],
)
def test_segmentation_rejects_bad_prompts(data, message):
    response = _client(FakeSegmentationEngine()).post(
        "/v1/images/segmentations", files={"image": _image_file()}, data=data
    )
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert message in response.text


def test_segmentation_requires_an_uploaded_image():
    response = _client(FakeSegmentationEngine()).post("/v1/images/segmentations", data={"prompt": "x"})
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert "Upload the input 'image'" in response.text


def test_segmentation_rejects_undecodable_upload():
    engine = FakeSegmentationEngine()
    response = _client(engine).post(
        "/v1/images/segmentations", files={"image": ("image.png", b"not an image", "image/png")}, data={"prompt": "x"}
    )
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert "Failed to open uploaded file" in response.text
    assert engine.captured_prompt is None


def test_segmentation_is_rejected_for_models_without_support():
    engine = FakeSegmentationEngine(model_class_name="QwenImagePipeline")
    response = _client(engine).post("/v1/images/segmentations", files={"image": _image_file()}, data={"prompt": "x"})
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert "does not support image segmentation" in response.text
    assert engine.captured_prompt is None
