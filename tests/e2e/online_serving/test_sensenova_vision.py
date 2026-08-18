# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
End-to-end online serving test for SenseNova-Vision-7B-MoT.

Validates that the SenseNova-Vision multistage pipeline can serve image
generation (text2img / img2img), image understanding (img2text), and the
mixed ``caption_generate`` mode via the OpenAI-compatible chat completions API
exposed by ``vllm-omni serve``.

Equivalent to running:
    vllm serve sensenova/SenseNova-Vision-7B-MoT --omni \\
        --port 8092 --deploy-config <ci/sensenova.yaml>

    python examples/online_serving/sensenova/openai_chat_client.py \\
        --modality text2img --prompt "A cute cat"

    python examples/online_serving/sensenova/openai_chat_client.py \\
        --modality img2text --image-url <image>

    python examples/online_serving/sensenova/openai_chat_client.py \\
        --modality mixed --image-url <image>
"""

import base64
import os
from io import BytesIO

import pytest
from vllm.assets.image import ImageAsset

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "sensenova/SenseNova-Vision-7B-MoT"
STAGE_CONFIG_PATH = get_deploy_config_path("ci/sensenova.yaml")

TEXT2TEXT_PROMPT = "What is the capital of France?"
TEXT2IMG_PROMPT = "A cute corgi astronaut on the moon, cinematic"
IMG2TEXT_PROMPT = "What are the main objects in this scene and their relationships?"
IMG2IMG_PROMPT = "Turn this image into a vibrant cartoon-style illustration."
MIXED_PROMPT = (
    "<image> Please briefly describe the contents of the image. Please respond "
    "with interleaved segmentation masks for the corresponding parts of the answer."
)

# Create parameter combinations for model and stage config
test_params = [
    OmniServerParams(
        model=MODEL,
        stage_config_path=STAGE_CONFIG_PATH,
        stage_init_timeout=300,
    ),
]


def _build_text_messages(prompt: str) -> list[dict]:
    """Build OpenAI-format messages for a text-only request (text2text)."""
    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
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


def _build_img2text_messages(prompt: str, image_b64: str) -> list[dict]:
    """Build OpenAI-format messages for img2text understanding."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
            ],
        }
    ]


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"})
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_sensenova_text2img_online(omni_server, openai_client) -> None:
    """Test SenseNova text2img via OpenAI-compatible chat completions API."""
    request_config = {
        "model": omni_server.model,
        "messages": _build_text2img_messages(TEXT2IMG_PROMPT),
        "modalities": ["image"],
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 2,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"})
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_sensenova_img2img_online(omni_server, openai_client) -> None:
    """Test SenseNova img2img via OpenAI-compatible chat completions API."""
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
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"})
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_sensenova_img2text_online(omni_server, openai_client) -> None:
    """Test SenseNova img2text via OpenAI-compatible chat completions API."""
    input_image = ImageAsset("2560px-Gfp-wisconsin-madison-the-nature-boardwalk").pil_image.convert("RGB")
    buffer = BytesIO()
    input_image.save(buffer, format="JPEG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    request_config = {
        "model": omni_server.model,
        "messages": _build_img2text_messages(IMG2TEXT_PROMPT, image_b64),
        "modalities": ["text"],
        "extra_body": {
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100", "rocm": "MI325"})
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_sensenova_mixed_online(omni_server, openai_client) -> None:
    """Test SenseNova mixed text+image (caption_generate) via chat API."""
    input_image = ImageAsset("2560px-Gfp-wisconsin-madison-the-nature-boardwalk").pil_image.convert("RGB")
    buffer = BytesIO()
    input_image.save(buffer, format="JPEG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    request_config = {
        "model": omni_server.model,
        "messages": _build_img2text_messages(MIXED_PROMPT, image_b64),
        "modalities": ["image", "text"],
        "extra_body": {
            "height": 512,
            "width": 512,
            "num_inference_steps": 2,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)
