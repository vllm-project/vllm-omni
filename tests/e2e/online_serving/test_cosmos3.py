# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Online serving L2 smoke tests for ``nvidia/Cosmos3-Nano``.

These keep Cosmos3 in the regular ``core_model`` diffusion lane while the
broader image/video similarity checks stay in ``tests/e2e/accuracy``.
"""

import os

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_image, generate_synthetic_video
from tests.helpers.runtime import (
    OmniServer,
    OmniServerParams,
    OpenAIClientHandler,
)

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "nvidia/Cosmos3-Nano"
PROMPT = "A small warehouse robot moves a blue box across a clean floor."
NEGATIVE_PROMPT = "blurry, distorted, low quality"
VIDEO_SHAPE = {
    "height": 256,
    "width": 256,
    "num_frames": 5,
    "fps": 1,
    "num_inference_steps": 2,
    "guidance_scale": 1.0,
    "flow_shift": 3.0,
    "seed": 42,
}
SERVER_ARGS = [
    "--model-class-name",
    "Cosmos3OmniDiffusersPipeline",
    "--no-guardrails",
]
SINGLE_CARD_MARKS = hardware_marks(res={"cuda": "H100"})


def _get_diffusion_feature_cases(model: str):
    """Return a single default Cosmos3 server row for L2 coverage."""
    return [
        pytest.param(
            OmniServerParams(model=model, server_args=SERVER_ARGS),
            id="default",
            marks=SINGLE_CARD_MARKS,
        ),
    ]


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", _get_diffusion_feature_cases(MODEL), indirect=True)
def test_text_to_image_001(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    """Default Cosmos3 T2I smoke through ``/v1/images/generations``."""
    responses = openai_client.send_images_generations_http_request(
        {
            "json": {
                "model": omni_server.model,
                "prompt": PROMPT,
                "negative_prompt": NEGATIVE_PROMPT,
                "size": "256x256",
                "n": 1,
                "response_format": "b64_json",
                "num_inference_steps": 2,
                "guidance_scale": 1.0,
                "flow_shift": 3.0,
                "seed": 42,
            }
        }
    )
    response = responses[0]
    assert response.success, response.error_message
    payload = response.json_body
    assert isinstance(payload, dict)
    assert len(payload["data"]) == 1
    assert payload["data"][0]["b64_json"]


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", _get_diffusion_feature_cases(MODEL), indirect=True)
def test_video_generation_modes_001(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    """Default Cosmos3 T2V, I2V, and V2V smoke through the async ``/v1/videos`` endpoint."""
    image_reference = f"data:image/jpeg;base64,{generate_synthetic_image(256, 256, seed=42)['base64']}"
    video_reference = f"data:video/mp4;base64,{generate_synthetic_video(256, 256, 5)['base64']}"

    video_cases = [
        {
            "form_data": {
                **VIDEO_SHAPE,
                "prompt": PROMPT,
                "negative_prompt": NEGATIVE_PROMPT,
            },
        },
        {
            "form_data": {
                **VIDEO_SHAPE,
                "prompt": "The blue box moves slowly forward from the reference image.",
                "negative_prompt": NEGATIVE_PROMPT,
            },
            "image_reference": image_reference,
        },
        {
            "form_data": {
                **VIDEO_SHAPE,
                "prompt": "Continue the same synthetic motion with consistent shapes.",
                "negative_prompt": NEGATIVE_PROMPT,
                "extra_params": '{"condition_frame_indexes_vision":[0,1],"condition_video_keep":"first"}',
            },
            "video_reference": video_reference,
        },
    ]

    for case in video_cases:
        request_config = {
            "model": omni_server.model,
            **case,
        }
        openai_client.send_video_diffusion_request(request_config)
