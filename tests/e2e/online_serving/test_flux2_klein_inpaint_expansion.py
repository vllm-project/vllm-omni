# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
End-to-end tests for Flux2 Klein inpainting in online serving mode.

Uses /v1/images/edits endpoint which is the correct API for image inpainting.
"""

import base64
from io import BytesIO

import httpx
import pytest
from PIL import Image, ImageDraw

from tests.conftest import OmniServer, OmniServerParams

MODEL = "black-forest-labs/FLUX.2-klein-4B"

_HEIGHT = 512
_WIDTH = 512
_NUM_INFERENCE_STEPS = 4


def _get_diffusion_feature_cases(model: str):
    return [
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=["--tensor-parallel-size", "2"],
            ),
            id="tp2_basic",
        ),
    ]


def _image_to_base64_jpeg(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def _create_test_mask_base64(width: int = _WIDTH, height: int = _HEIGHT) -> str:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([width // 4, height // 4, width * 3 // 4, height * 3 // 4], fill=255)
    return _image_to_base64_jpeg(mask)


def _compare_images(img1: Image.Image, img2: Image.Image) -> bool:
    return list(img1.getdata()) == list(img2.getdata())


def _send_edit_request(host: str, port: int, model: str, image_b64: str, mask_b64: str, prompt: str, **kwargs):
    url = f"http://{host}:{port}/v1/images/edits"
    files = {
        "image": ("image.jpg", base64.b64decode(image_b64), "image/jpeg"),
        "mask_image": ("mask.jpg", base64.b64decode(mask_b64), "image/jpeg"),
    }
    data = {"prompt": prompt, "model": model, **kwargs}
    with httpx.Client(timeout=60.0) as client:
        response = client.post(url, files=files, data=data)
        response.raise_for_status()
        return response.json()


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", _get_diffusion_feature_cases(MODEL), indirect=True)
def test_flux2_klein_inpaint_basic(omni_server: OmniServer):
    input_image_b64 = _image_to_base64_jpeg(Image.new("RGB", (_WIDTH, _HEIGHT), (128, 128, 128)))
    mask_b64 = _create_test_mask_base64()

    result = _send_edit_request(
        host=omni_server.host,
        port=omni_server.port,
        model=MODEL,
        image_b64=input_image_b64,
        mask_b64=mask_b64,
        prompt="Fill in the masked area with a beautiful garden",
        guidance_scale=1.0,
        num_inference_steps=_NUM_INFERENCE_STEPS,
        n=1,
        seed=42,
    )

    assert "data" in result and len(result["data"]) == 1
    img_data = result["data"][0].get("b64_json") or result["data"][0].get("url", "").split(",")[-1]
    img = Image.open(BytesIO(base64.b64decode(img_data)))
    assert img.size == (_WIDTH, _HEIGHT)


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", _get_diffusion_feature_cases(MODEL), indirect=True)
def test_flux2_klein_inpaint_deterministic(omni_server: OmniServer):
    input_image_b64 = _image_to_base64_jpeg(Image.new("RGB", (_WIDTH, _HEIGHT), (128, 128, 128)))
    mask_b64 = _create_test_mask_base64()
    prompt = "A red flower in a field"

    result1 = _send_edit_request(
        host=omni_server.host,
        port=omni_server.port,
        model=MODEL,
        image_b64=input_image_b64,
        mask_b64=mask_b64,
        prompt=prompt,
        guidance_scale=1.0,
        num_inference_steps=_NUM_INFERENCE_STEPS,
        n=1,
        seed=12345,
    )

    result2 = _send_edit_request(
        host=omni_server.host,
        port=omni_server.port,
        model=MODEL,
        image_b64=input_image_b64,
        mask_b64=mask_b64,
        prompt=prompt,
        guidance_scale=1.0,
        num_inference_steps=_NUM_INFERENCE_STEPS,
        n=1,
        seed=12345,
    )

    img1_data = result1["data"][0].get("b64_json") or result1["data"][0].get("url", "").split(",")[-1]
    img2_data = result2["data"][0].get("b64_json") or result2["data"][0].get("url", "").split(",")[-1]

    img1 = Image.open(BytesIO(base64.b64decode(img1_data)))
    img2 = Image.open(BytesIO(base64.b64decode(img2_data)))

    assert _compare_images(img1, img2), (
        "Same input with same seed should produce identical output. This is critical for offline/online consistency."
    )


@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", _get_diffusion_feature_cases(MODEL), indirect=True)
def test_flux2_klein_inpaint_multiple_outputs(omni_server: OmniServer):
    input_image_b64 = _image_to_base64_jpeg(Image.new("RGB", (_WIDTH, _HEIGHT), (128, 128, 128)))
    mask_b64 = _create_test_mask_base64()

    result = _send_edit_request(
        host=omni_server.host,
        port=omni_server.port,
        model=MODEL,
        image_b64=input_image_b64,
        mask_b64=mask_b64,
        prompt="A beautiful landscape",
        guidance_scale=1.0,
        num_inference_steps=_NUM_INFERENCE_STEPS,
        n=2,
        seed=42,
    )

    assert "data" in result and len(result["data"]) == 2
