# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import os
from io import BytesIO

import pytest
from PIL import Image
from vllm.assets.image import ImageAsset

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from vllm_omni.model_extras.lance import (
    build_image_to_image_prompt,
    build_text_to_image_prompt,
)

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "bytedance-research/Lance"

TEXT2IMG_PROMPT = "A cute corgi astronaut on the moon, cinematic"
IMG2IMG_PROMPT = "Convert this into a vibrant cartoon-style illustration"

_LANCE_SERVE_ARGS = [
    "--pipeline",
    "lance",
    "--max-num-batched-tokens",
    "32768",
    "--max-num-seqs",
    "1",
    "--enforce-eager",
    "--trust-remote-code",
    "--no-enable-prefix-caching",
    "--no-async-chunk",
]

test_params = [
    OmniServerParams(
        model=MODEL,
        server_args=_LANCE_SERVE_ARGS,
        stage_init_timeout=300,
    ),
]


def _build_text2img_messages(prompt: str) -> list[dict]:
    rendered = build_text_to_image_prompt(prompt, negative_prompt=None)["prompt"]
    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": rendered}],
        }
    ]


def _build_img2img_messages(prompt: str, image_b64: str, input_image: Image.Image) -> list[dict]:
    rendered = build_image_to_image_prompt(prompt, negative_prompt=None, input_image=input_image)["prompt"]
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": rendered},
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
def test_lance_text2img_online(omni_server, openai_client) -> None:
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

    openai_client.send_diffusion_request(request_config)


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"})
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_lance_img2img_online(omni_server, openai_client) -> None:
    input_image = ImageAsset("2560px-Gfp-wisconsin-madison-the-nature-boardwalk").pil_image.convert("RGB")
    buffer = BytesIO()
    input_image.save(buffer, format="JPEG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    request_config = {
        "model": omni_server.model,
        "messages": _build_img2img_messages(IMG2IMG_PROMPT, image_b64, input_image),
        "modalities": ["image"],
        "extra_body": {
            "num_inference_steps": 2,
            "guidance_scale": 0.0,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)
