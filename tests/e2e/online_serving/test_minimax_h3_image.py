# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""H3-Image serving: 4K output transfer on Rubin, and T2I smoke on Ascend NPUs."""

import base64
import hashlib
import io
import json
import os
from urllib.parse import urlsplit

import pytest
from PIL import Image

from tests.helpers.client import OnlineOmniClient
from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams

MODEL = os.environ.get("VLLM_TEST_MINIMAX_H3_IMAGE_MODEL", "MiniMaxAI/MiniMax-H3-Image")

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


# --- Ascend NPU A3/A5 T2I smoke ---

NPU_SERVER_ARGS = [
    "--enforce-eager",
    "--num-gpus",
    "2",
    "--usp",
    "2",
    "--text-encoder-tp-size",
    "2",
    "--diffusion-offload-config",
    json.dumps(
        {
            "mode": "layer",
            "components": ["dit"],
            "layer_options": {"dit": {"weight_transfer": "allgather"}},
        },
        separators=(",", ":"),
    ),
]
A3_TWO_CARD_MARKS = hardware_marks(res={"npu": "A3"}, num_cards=2)
A5_TWO_CARD_MARKS = hardware_marks(res={"npu": "A5"}, num_cards=2)


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parallel
@pytest.mark.parametrize(
    "omni_server",
    [
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=NPU_SERVER_ARGS,
                stage_init_timeout=1800,
                init_timeout=1800,
            ),
            id="minimax_h3_image_a3_dlo_sp2_t2i",
            marks=A3_TWO_CARD_MARKS,
        ),
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=NPU_SERVER_ARGS,
                stage_init_timeout=1800,
                init_timeout=1800,
            ),
            id="minimax_h3_image_a5_dlo_sp2_t2i",
            marks=A5_TWO_CARD_MARKS,
        ),
    ],
    indirect=True,
)
def test_npu_text_to_image(omni_server: OmniServer, online_client: OnlineOmniClient) -> None:
    """Validate T2I with SP2, text-encoder TP2, and DiT AllGather layer offload."""
    online_client.send_images_generations_http_request(
        {
            "json": {
                "model": omni_server.model,
                "prompt": "A red bicycle leaning against a pale stone wall in warm afternoon sunlight.",
                "size": "1024x1024",
                "num_inference_steps": 6,
                "seed": 42,
                "n": 1,
                "response_format": "b64_json",
                "output_format": "png",
            },
            "timeout": 300,
        }
    )
