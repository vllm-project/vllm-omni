# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""E2E conformance tests for ``POST /v1/images/edits``.

Validates that a valid image edit request returns the expected response
schema (``created``, ``data[].b64_json``) and that the generated image
is decodable.

Error-case tests already exist at
``tests/dfx/reliability/invalid_param_test/test_invalid_image_editing.py``.

From ``tests/``::

    pytest -s -v e2e/online_serving/test_images_edits_conformance.py
"""

import base64
import io
import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest
from PIL import Image

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OnlineOmniClient

pytestmark = [pytest.mark.slow, pytest.mark.diffusion]

MODEL = "Qwen/Qwen-Image-Edit"

_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            server_args=["--trust-remote-code"],
        ),
        id="qwen_image_edit",
        marks=hardware_marks(res={"cuda": "H100"}),
    ),
]


def _tiny_png_bytes() -> bytes:
    img = Image.new("RGB", (32, 32), color="gray")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---- Schema validation ----


@pytest.mark.parametrize("omni_server", _server_params, indirect=True)
def test_images_edits_response_schema(
    omni_server: OmniServer,
    online_client: OnlineOmniClient,
) -> None:
    """Valid image + prompt returns HTTP 200 with ``created`` (int) and ``data`` (list)."""
    png = _tiny_png_bytes()
    cfg = {
        "data": {
            "prompt": "make it brighter",
            "model": omni_server.model,
            "num_inference_steps": "2",
            "seed": "42",
        },
        "files": {"image": ("test.png", png, "image/png")},
        "timeout": 300,
    }
    responses = online_client.send_images_edits_http_request(cfg)
    resp = responses[0]
    body = resp.json_body

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {body}"
    assert "created" in body, f"Missing 'created' in response: {body}"
    assert isinstance(body["created"], int), f"'created' is not int: {type(body['created'])}"
    assert "data" in body, f"Missing 'data' in response: {body}"
    assert isinstance(body["data"], list), f"'data' is not a list: {type(body['data'])}"
    assert len(body["data"]) >= 1, f"Empty 'data' array: {body}"


@pytest.mark.parametrize("omni_server", _server_params, indirect=True)
def test_images_edits_valid_b64_image(
    omni_server: OmniServer,
    online_client: OnlineOmniClient,
) -> None:
    """Response ``data[0].b64_json`` decodes to a valid image."""
    png = _tiny_png_bytes()
    cfg = {
        "data": {
            "prompt": "Transform into a watercolor painting",
            "model": omni_server.model,
            "num_inference_steps": "2",
            "seed": "42",
        },
        "files": {"image": ("test.png", png, "image/png")},
        "timeout": 300,
    }
    responses = online_client.send_images_edits_http_request(cfg)
    body = responses[0].json_body

    assert responses[0].status_code == 200, f"Request failed: {body}"
    entry = body["data"][0]
    assert "b64_json" in entry, f"Missing 'b64_json' in data entry: {entry}"

    img_bytes = base64.b64decode(entry["b64_json"])
    img = Image.open(io.BytesIO(img_bytes))
    assert img.size[0] > 0 and img.size[1] > 0, f"Decoded image has zero dimensions: {img.size}"
