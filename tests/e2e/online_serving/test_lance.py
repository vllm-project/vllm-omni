# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end online serving test for Lance (single-stage).

Verifies that the Lance pipeline can serve text-to-image and image-edit
requests via the OpenAI-compatible chat completions API exposed by
``vllm-omni serve``.

Equivalent to running:

    vllm-omni serve "bytedance-research/Lance" --omni \\
        --pipeline lance --enforce-eager --trust-remote-code --port 8091

    # text2img
    python3 examples/online_serving/lance/openai_chat_client.py \\
        --prompt "A cute corgi astronaut" --modality text2img
"""

import base64
import os
from io import BytesIO

import pytest
from vllm.assets.image import ImageAsset

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from vllm_omni.diffusion.models.lance.prompts import (
    VIDEO_PAD,
    VISION_END,
    VISION_START,
    render_lance_prompt,
)

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "bytedance-research/Lance"

TEXT2IMG_PROMPT = "A cute corgi astronaut on the moon, cinematic"
IMG2IMG_PROMPT = "Convert this into a vibrant cartoon-style illustration"

# Lance vision block: <|vision_start|><|video_pad|><|vision_end|>
_VISION_BLOCK = f"{VISION_START}{VIDEO_PAD}{VISION_END}"

# Lance is single-stage diffusion — no deploy YAML.  Pass the engine
# knobs that used to live in ``vllm_omni/deploy/lance.yaml`` via CLI
# flags so ``create_default_diffusion`` builds the stage config.
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
    """Build OpenAI-format messages for text2img generation."""
    rendered = render_lance_prompt("t2i", prompt)
    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": rendered}],
        }
    ]


def _build_img2img_messages(prompt: str, image_b64: str) -> list[dict]:
    """Build OpenAI-format messages for img2img generation."""
    rendered = render_lance_prompt("image_edit", prompt, vision_token=_VISION_BLOCK)
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
    """Lance text2img via the OpenAI-compatible chat completions API."""
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
    """Lance image_edit via the OpenAI-compatible chat completions API."""
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

    openai_client.send_diffusion_request(request_config)
