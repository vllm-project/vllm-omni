# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
End-to-end online serving test for Bagel text2img and img2img generation.

This test validates that the Bagel model can serve image generation requests
via the OpenAI-compatible chat completions API.

Equivalent to running:
    vllm-omni serve "ByteDance-Seed/BAGEL-7B-MoT" --omni --port 8091

    # text2img
    python3 examples/online_serving/bagel/openai_chat_client.py \\
        --prompt "A cute cat" --modality text2img

    # img2img
    python3 examples/online_serving/bagel/openai_chat_client.py \\
        --prompt "Let the woman wear a blue dress" --modality img2img \\
        --image-url women.jpg
"""

import base64
import os
from io import BytesIO

import pytest
from PIL import Image
from vllm.assets.image import ImageAsset

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "ByteDance-Seed/BAGEL-7B-MoT"
STAGE_CONFIGS_PATH = get_deploy_config_path("ci/bagel.yaml")

TEXT2IMG_PROMPT = "A cute cat"
IMG2IMG_PROMPT = "Change the grass color to red"

# Create parameter combinations for model and stage config
test_params = [
    OmniServerParams(
        model=MODEL,
        stage_config_path=STAGE_CONFIGS_PATH,
        stage_init_timeout=300,
    ),
]


def _build_text2img_messages(prompt: str) -> list[dict]:
    """Build OpenAI-format messages for text2img generation."""
    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": f"<|im_start|>{prompt}<|im_end|>"}],
        }
    ]


def _build_img2img_messages(prompt: str, image_b64: str) -> list[dict]:
    """Build OpenAI-format messages for img2img generation."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"<|im_start|>{prompt}<|im_end|>"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
            ],
        }
    ]


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"})
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_bagel_text2img_online(omni_server, online_client) -> None:
    """Test Bagel text2img via OpenAI-compatible chat completions API."""
    request_config = {
        "model": omni_server.model,
        "messages": _build_text2img_messages(TEXT2IMG_PROMPT),
        "modalities": ["image"],
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 2,
            "guidance_scale": 0.0,
            "seed": 42,
        },
    }

    online_client.send_diffusion_request(request_config)


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"})
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_bagel_img2img_online(omni_server, online_client) -> None:
    """Test Bagel img2img via OpenAI-compatible chat completions API."""
    input_image = ImageAsset("2560px-Gfp-wisconsin-madison-the-nature-boardwalk").pil_image.convert("RGB")
    buffer = BytesIO()
    input_image.save(buffer, format="JPEG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    request_config = {
        "model": omni_server.model,
        "messages": _build_img2img_messages(IMG2IMG_PROMPT, image_b64),
        "modalities": ["image"],
        "extra_body": {
            "num_inference_steps": 2,
            "guidance_scale": 0.0,
            "seed": 42,
        },
    }

    online_client.send_diffusion_request(request_config)


EDIT_SOURCE_SIZE = (1280, 720)
EDIT_REQUESTED_SIZE = "1024x1024"


def _edit_source_jpeg() -> tuple[str, bytes, str]:
    synthetic = generate_synthetic_image(*EDIT_SOURCE_SIZE, seed=7287)
    return ("source.jpg", base64.b64decode(synthetic["base64"]), "image/jpeg")


def _build_edit_request(size: str) -> dict:
    return {
        "files": [("image", _edit_source_jpeg())],
        "data": {
            "prompt": IMG2IMG_PROMPT,
            "size": size,
            "num_inference_steps": 2,
            "seed": 42,
            "response_format": "b64_json",
        },
    }


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"})
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_bagel_image_edit_honors_explicit_size_two_stage(omni_server, online_client) -> None:
    """An explicit ``size`` wins over the AR stage's source-derived KV image shape (#7283)."""
    (response,) = online_client.send_images_edits_http_request(_build_edit_request(EDIT_REQUESTED_SIZE))

    payload = response.json_body
    assert isinstance(payload, dict)
    assert payload["size"] == EDIT_REQUESTED_SIZE
    image = Image.open(BytesIO(base64.b64decode(payload["data"][0]["b64_json"])))
    assert image.size == (1024, 1024)


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"})
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_bagel_image_edit_floors_unaligned_size_two_stage(omni_server, online_client) -> None:
    """A side that is not a multiple of the latent stride is floored to one (1000x700 -> 992x688)."""
    (response,) = online_client.send_images_edits_http_request(_build_edit_request("1000x700"))

    payload = response.json_body
    assert isinstance(payload, dict)
    assert payload["size"] == "992x688"
    image = Image.open(BytesIO(base64.b64decode(payload["data"][0]["b64_json"])))
    assert image.size == (992, 688)
