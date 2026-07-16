# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
E2E online serving smoke for nvidia/Cosmos3-Nano: text-to-image (/v1/images/generations)
and text-to-video (/v1/videos).

Runs the real model at small parameters (256x256, 2 steps) with guardrails disabled
(--no-guardrails) so CI needs no gated repo or cosmos-guardrail package.
"""

import os

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "nvidia/Cosmos3-Nano"
PROMPT = "A small warehouse robot moves a blue box across a clean floor."
NEGATIVE_PROMPT = "blurry, distorted, low quality"
WIDTH = HEIGHT = 256
NUM_INFERENCE_STEPS = 2
SEED = 42

COSMOS3_SERVER_ARGS = [
    "--num-gpus",
    "1",
    "--no-guardrails",
]

SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})


def _get_diffusion_feature_cases(model: str):
    """Return a single default ``OmniServerParams`` row for Cosmos3-Nano."""
    return [
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=COSMOS3_SERVER_ARGS,
                init_timeout=1200,
                stage_init_timeout=900,
            ),
            id="default",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
    ]


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", _get_diffusion_feature_cases(MODEL), indirect=True)
def test_text_to_image_001(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    """Default Cosmos3-Nano T2I smoke: ``/v1/images/generations`` returns one 256x256 image."""
    body = {
        "model": omni_server.model,
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "size": f"{WIDTH}x{HEIGHT}",
        "n": 1,
        "response_format": "b64_json",
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": 1.0,
        "seed": SEED,
    }
    [resp] = openai_client.send_images_generations_http_request({"json": body, "timeout": 1800})
    assert resp.success, f"image generation failed: {resp.status_code} {resp.error_message}"
    assert resp.json_body is not None
    assert len(resp.json_body["data"]) == 1
    assert resp.json_body["data"][0]["b64_json"]


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", _get_diffusion_feature_cases(MODEL), indirect=True)
def test_text_to_video_001(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    """Default Cosmos3-Nano T2V smoke: async ``/v1/videos`` job completes and returns video bytes."""
    request_config = {
        "model": omni_server.model,
        "form_data": {
            "prompt": PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "height": HEIGHT,
            "width": WIDTH,
            "num_frames": 5,
            "fps": 1,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "guidance_scale": 1.0,
            "seed": SEED,
        },
    }
    openai_client.send_video_diffusion_request(request_config)
