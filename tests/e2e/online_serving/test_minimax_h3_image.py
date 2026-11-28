# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real-weight H3-Image generation/edit gates and optional external 4K regression."""

import base64
import hashlib
import io
import os
from urllib.parse import urlsplit

import pytest
from PIL import Image

from tests.helpers.client import OnlineOmniClient
from tests.helpers.mark import hardware_marks, hardware_test
from tests.helpers.media import get_asset_path
from tests.helpers.runtime import OmniServerParams

pytestmark = pytest.mark.diffusion


@pytest.mark.advanced_model
@pytest.mark.parallel
@hardware_test(res={"cuda": "B200"}, num_cards=4)
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


# H3-Image has no tiny checkpoint builder: even core_model uses release weights.
MODEL = os.environ.get("VLLM_TEST_MINIMAX_H3_IMAGE_MODEL", "MiniMaxAI/MiniMax-H3-Image")
DLO_ARGS = [
    "--num-gpus",
    "1",
    "--diffusion-offload-config",
    '{"mode":"layer","components":["dit","text_encoder"]}',
]
DLO_CUDA_SERVER = OmniServerParams(model=MODEL, server_args=DLO_ARGS, startup_timeout=1200)
DLO_NPU_SERVER = OmniServerParams(
    model=MODEL,
    server_args=[*DLO_ARGS, "--enforce-eager"],
    startup_timeout=1200,
)
DLO_CASES = [
    pytest.param(
        DLO_CUDA_SERVER,
        id="single-gpu-dlo",
        marks=[pytest.mark.core_model, *hardware_marks(res={"cuda": ["H100", "B200"]}, num_cards=1)],
    ),
    pytest.param(
        DLO_NPU_SERVER,
        id="single-a3-die-dlo",
        marks=[pytest.mark.core_model, *hardware_marks(res={"npu": "A3"}, num_cards=1)],
    ),
]


TP2_CASE = pytest.param(
    OmniServerParams(
        model=MODEL,
        server_args=["--num-gpus", "2", "--tensor-parallel-size", "2", "--text-encoder-tp-size", "2"],
        startup_timeout=1200,
    ),
    id="two-h100-tp2",
    marks=[pytest.mark.advanced_model, *hardware_marks(res={"cuda": "H100"}, num_cards=2)],
)


def _assert_image_response(response, output_path, size):
    assert response.status_code == 200, response.error_message
    assert len(response.json_body["data"]) == 1
    png = base64.b64decode(response.json_body["data"][0]["b64_json"], validate=True)
    with Image.open(io.BytesIO(png)) as image:
        image.load()
        assert image.format == "PNG"
        assert image.size == size
        assert any(low != high for low, high in image.convert("RGB").getextrema())
        image.save(output_path)


@pytest.mark.parametrize("omni_server", [*DLO_CASES, TP2_CASE], indirect=True)
def test_text_to_image(omni_server, online_client, tmp_path):
    """Default 2K-area generation followed by the requested 1K output resize."""
    response = online_client.send_images_generations_http_request(
        {
            "json": {
                "model": omni_server.model,
                "prompt": "A red bicycle leaning against a pale stone wall in warm afternoon sunlight.",
                "size": "1024x1024",
                "seed": 42,
                "generator_device": "cpu",
                "num_inference_steps": 6,
                "response_format": "b64_json",
                "output_format": "png",
            },
            "timeout": 600,
        }
    )[0]
    _assert_image_response(response, tmp_path / "t2i.png", (1024, 1024))


@pytest.mark.parametrize("omni_server", [*DLO_CASES, TP2_CASE], indirect=True)
def test_image_edit(omni_server, online_client, tmp_path):
    _edit_request(omni_server, online_client, tmp_path, 1)


@pytest.mark.advanced_model
@pytest.mark.parametrize(
    "omni_server",
    [
        pytest.param(
            DLO_CUDA_SERVER,
            id="single-gpu-dlo",
            marks=hardware_marks(res={"cuda": ["H100", "B200"]}, num_cards=1),
        ),
        pytest.param(
            DLO_NPU_SERVER,
            id="single-a3-die-dlo",
            marks=hardware_marks(res={"npu": "A3"}, num_cards=1),
        ),
    ],
    indirect=True,
)
def test_multi_image_edit(omni_server, online_client, tmp_path):
    _edit_request(omni_server, online_client, tmp_path, 2)


def _edit_request(server, client, output_dir, count):
    references = [get_asset_path(f"qwen_image_edit/qwen_image_edit_2511_test{i}.png") for i in range(1, count + 1)]
    prompt = "Restyle <Picture 1> into a warm oil painting while preserving the subject and composition."
    if count == 2:
        prompt = "Combine the subjects from <Picture 1> and <Picture 2> into one warm oil painting."
    response = client.send_images_edits_http_request(
        {
            "data": {
                "model": server.model,
                "prompt": prompt,
                "size": "2048x2048",
                "seed": "42",
                "generator_device": "cpu",
                "num_inference_steps": "6",
                "response_format": "b64_json",
                "output_format": "png",
            },
            "files": [
                ("image", (f"reference-{i}.png", path.read_bytes(), "image/png")) for i, path in enumerate(references)
            ],
            "timeout": 600,
        }
    )[0]
    _assert_image_response(response, output_dir / f"edit-{count}.png", (2048, 2048))
