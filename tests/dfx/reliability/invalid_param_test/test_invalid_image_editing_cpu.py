# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU-only abnormal-input tests for ``POST /v1/images/edits`` (part of #3408).

The GPU-backed suite in ``test_invalid_image_editing.py`` exercises the same
route against a live model. These tests cover the validation matrix with a
mocked engine, so malformed requests are rejected before any generation work:
no model weights and no GPU are required.
"""

from __future__ import annotations

import io
import json
from argparse import Namespace
from http import HTTPStatus
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from vllm_omni.entrypoints.openai.api_server import router

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _MockGenerationResult:
    def __init__(self, images):
        self.images = images
        self.stage_durations = {}
        self.peak_memory_mb = 0.0


class _MockEngine:
    """Minimal diffusion engine: captures the request and yields one image."""

    is_running = True

    def __init__(self) -> None:
        self.generate_calls = 0

    async def generate(self, **kwargs):
        self.generate_calls += 1
        yield _MockGenerationResult([Image.new("RGB", (64, 64), color="blue")])


@pytest.fixture
def edits_client() -> TestClient:
    """FastAPI TestClient with a mocked single-stage diffusion engine."""
    from vllm.entrypoints.openai.models.protocol import BaseModelPath

    from vllm_omni.entrypoints.openai.api_server import _DiffusionServingModels

    app = FastAPI()
    app.include_router(router)
    app.state.engine_client = _MockEngine()
    app.state.diffusion_engine = app.state.engine_client
    app.state.stage_configs = [SimpleNamespace(stage_type="diffusion")]
    app.state.openai_serving_models = _DiffusionServingModels(
        [BaseModelPath(name="test/edit-model", model_path="test/edit-model")]
    )
    app.state.args = Namespace(
        default_sampling_params=None,
        max_generated_image_size=1024 * 1792,
    )
    return TestClient(app)


def _tiny_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color="gray").save(buf, format="PNG")
    return buf.getvalue()


def _post_edits(
    client: TestClient,
    *,
    data: dict | None = None,
    with_image: bool = True,
) -> Any:
    files = {"image": ("edit.png", _tiny_png(), "image/png")} if with_image else None
    return client.post("/v1/images/edits", data=data or {}, files=files)


def _assert_error_response(response, status_code: int, fragment: str) -> None:
    assert response.status_code == status_code
    body = response.json()
    assert isinstance(body, dict)
    assert "detail" in body
    assert fragment in json.dumps(body)


@pytest.mark.parametrize(
    ("field", "value", "fragment"),
    [
        # pydantic form validation: non-int values and out-of-range
        # output_compression must 422. (num_inference_steps / guidance_scale
        # are intentionally not range-constrained at this route; bounds are
        # enforced engine-side.)
        ("seed", "", "seed"),
        ("seed", "abc", "seed"),
        ("output_compression", "101", "output_compression"),
        ("output_compression", "abc", "output_compression"),
    ],
)
def test_edit_rejects_invalid_form_field(edits_client, field, value, fragment) -> None:
    response = _post_edits(edits_client, data={"prompt": "make it brighter", field: value})
    _assert_error_response(response, HTTPStatus.UNPROCESSABLE_ENTITY, fragment)


def test_edit_rejects_missing_prompt(edits_client) -> None:
    response = _post_edits(edits_client, data={})
    _assert_error_response(response, HTTPStatus.UNPROCESSABLE_ENTITY, "prompt")


def test_edit_rejects_missing_image(edits_client) -> None:
    response = _post_edits(edits_client, data={"prompt": "make it brighter"}, with_image=False)
    _assert_error_response(response, HTTPStatus.UNPROCESSABLE_ENTITY, "image")


def test_edit_rejects_unsupported_response_format(edits_client) -> None:
    response = _post_edits(edits_client, data={"prompt": "make it brighter", "response_format": "url"})
    _assert_error_response(response, HTTPStatus.BAD_REQUEST, "response_format")


def test_edit_rejects_model_mismatch(edits_client) -> None:
    response = _post_edits(edits_client, data={"prompt": "make it brighter", "model": "wrong-model-id"})
    _assert_error_response(response, HTTPStatus.BAD_REQUEST, "Model mismatch")


def test_edit_rejects_invalid_layers(edits_client) -> None:
    response = _post_edits(edits_client, data={"prompt": "make it brighter", "layers": "1"})
    _assert_error_response(response, HTTPStatus.BAD_REQUEST, "Invalid layers")


def test_edit_rejects_invalid_resolution(edits_client) -> None:
    response = _post_edits(edits_client, data={"prompt": "make it brighter", "resolution": "512"})
    _assert_error_response(response, HTTPStatus.BAD_REQUEST, "Invalid resolution")


def test_edit_rejects_resolution_with_explicit_size(edits_client) -> None:
    response = _post_edits(
        edits_client,
        data={"prompt": "make it brighter", "resolution": "1024", "size": "1024x1024"},
    )
    _assert_error_response(response, HTTPStatus.BAD_REQUEST, "Cannot specify both")


@pytest.mark.parametrize(
    ("size", "fragment"),
    [
        ("abc", "Invalid size format"),
        ("0x1024", "positive integers"),
    ],
)
def test_edit_rejects_invalid_size(edits_client, size, fragment) -> None:
    response = _post_edits(edits_client, data={"prompt": "make it brighter", "size": size})
    _assert_error_response(response, HTTPStatus.BAD_REQUEST, fragment)


def test_edit_accepts_valid_request(edits_client) -> None:
    """Positive control: the mocked route completes and returns an image."""
    response = _post_edits(edits_client, data={"prompt": "make it brighter"})
    assert response.status_code == HTTPStatus.OK
    body = response.json()
    assert body["data"][0]["b64_json"]
    assert edits_client.app.state.engine_client.generate_calls == 1
