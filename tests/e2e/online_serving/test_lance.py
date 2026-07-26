# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end online serving test for Lance (single-stage).

Verifies that the Lance pipeline can serve text-to-image and image-edit
requests via the OpenAI-compatible chat completions API exposed by
``vllm-omni serve``.

Prompts are built through the shared ``model_extras`` registry (the same path
the standard task examples use), so a broken or missing ``LancePipeline``
declaration fails this test rather than silently degrading output.

Equivalent to running:

    vllm-omni serve "bytedance-research/Lance" --omni \\
        --deploy-config vllm_omni/deploy/lance.yaml --port 8091

    # text2img
    python3 examples/offline_inference/text_to_image/text_to_image.py \\
        --model bytedance-research/Lance --deploy-config vllm_omni/deploy/lance.yaml \\
        --prompt "A cute corgi astronaut" \\
        --extra-body '{"cfg_text_scale": 4.0, "timestep_shift": 3.5}'
"""

import base64
import os
from io import BytesIO

import pytest
from vllm.assets.image import ImageAsset

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from vllm_omni.model_extras import (
    build_image_to_image_prompt,
    build_text_to_image_prompt,
)

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "bytedance-research/Lance"

TEXT2IMG_PROMPT = "A cute corgi astronaut on the moon, cinematic"
IMG2IMG_PROMPT = "Convert this into a vibrant cartoon-style illustration"

# Lance-declared knobs (vllm_omni/model_extras/lance.py). Passing them here
# exercises the declaration -> extra_body -> extra_args route end to end; an
# undeclared key would be filtered out before reaching the pipeline.
_LANCE_EXTRA_BODY = {
    "cfg_text_scale": 4.0,
    "timestep_shift": 3.5,
}

# Select the Lance pipeline via its deploy YAML. The YAML already carries the
# engine knobs (pipeline: lance, max_num_seqs: 1, enforce_eager, trust_remote_code,
# enable_prefix_caching: false, async_chunk: false), so no other flags are needed.
# NB: a bare ``--pipeline lance`` flag no longer exists — argparse abbreviates it
# to ``--pipeline-parallel-size`` and fails on the non-int value.
_LANCE_SERVE_ARGS = [
    "--deploy-config",
    "vllm_omni/deploy/lance.yaml",
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
    rendered = build_text_to_image_prompt("LancePipeline", prompt=prompt, negative_prompt=None)["prompt"]
    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": rendered}],
        }
    ]


def _build_img2img_messages(prompt: str, image_b64: str, input_image) -> list[dict]:
    """Build OpenAI-format messages for img2img generation."""
    rendered = build_image_to_image_prompt(
        "LancePipeline",
        prompt=prompt,
        negative_prompt=None,
        input_image=input_image,
    )["prompt"]
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
            **_LANCE_EXTRA_BODY,
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
        "messages": _build_img2img_messages(IMG2IMG_PROMPT, image_b64, input_image),
        "modalities": ["image"],
        "extra_body": {
            "num_inference_steps": 2,
            "guidance_scale": 0.0,
            "seed": 42,
            **_LANCE_EXTRA_BODY,
        },
    }

    openai_client.send_diffusion_request(request_config)
