# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""H3-Image native-4K output transfer against a separately launched SP4 server."""

import base64
import hashlib
import io
import os
from urllib.parse import urlsplit

import pytest
from PIL import Image

from tests.helpers.client import OnlineOmniClient
from tests.helpers.mark import hardware_marks

pytestmark = [
    pytest.mark.advanced_model,
    pytest.mark.diffusion,
    pytest.mark.parallel,
    *hardware_marks(res={"cuda": "B200"}, num_cards=4),
]


def test_repeated_native_4k_outputs_complete_with_identical_pixels() -> None:
    """A 2K smoke test misses the repeated 16MP transfer stall on Rubin."""
    base_url = os.environ.get("H3_IMAGE_TEST_BASE_URL")
    if not base_url:
        pytest.skip("Set H3_IMAGE_TEST_BASE_URL to a resident four-GPU SP4 H3-Image server")
    payload = {
        "model": "MiniMaxAI/MiniMax-H3-Image",
        "prompt": "A red bicycle leaning against a pale stone wall in warm afternoon sunlight.",
        "size": "4096x4096",
        "num_inference_steps": 6,
        "seed": 42,
        "generator_device": "cpu",
        "n": 1,
        "response_format": "b64_json",
        "output_format": "png",
    }
    address = urlsplit(base_url)
    assert address.scheme == "http" and address.hostname and address.port
    client = OnlineOmniClient(host=address.hostname, port=address.port, run_level="core_model")
    pixel_hashes = []
    try:
        for _ in range(3):
            response = client.send_images_generations_http_request({"json": payload, "timeout": 120})[0]
            images = response.json_body["data"]
            png = base64.b64decode(images[0]["b64_json"], validate=True)
            with Image.open(io.BytesIO(png)) as image:
                image.load()
                assert image.format == "PNG"
                assert image.size == (4096, 4096)
                rgb = image.convert("RGB")
                assert any(low != high for low, high in rgb.getextrema())
                pixel_hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())
    finally:
        client.client.close()
    assert len(set(pixel_hashes)) == 1
