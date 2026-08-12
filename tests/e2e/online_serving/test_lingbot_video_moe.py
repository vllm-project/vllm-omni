# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Single-GPU serving smoke for ``robbyant/lingbot-video-moe-30b-a3b``."""

import json
import os

import pytest

from tests.helpers.assertions import assert_video_first_frame_matches
from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "robbyant/lingbot-video-moe-30b-a3b"
PROMPT = "a robotic arm picks up a red block"
NEGATIVE_PROMPT = "low quality, blurry, watermark, text"
SMOKE_DEFAULT_SAMPLING_PARAMS = '{"0":{"num_frames":81,"num_inference_steps":2,"guidance_scale":3.0}}'

SINGLE_CARD_MARKS = hardware_marks(res={"cuda": "H100"})


def _get_server_cases(model: str):
    return [
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--model-class-name",
                    "LingBotVideoPipeline",
                    "--default-sampling-params",
                    SMOKE_DEFAULT_SAMPLING_PARAMS,
                ],
            ),
            id="default",
            marks=SINGLE_CARD_MARKS,
        ),
    ]


@pytest.mark.slow
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", _get_server_cases(MODEL), indirect=True)
def test_text_to_image_moe(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    openai_client.send_images_generations_http_request(
        {
            "json": {
                "model": omni_server.model,
                "prompt": PROMPT,
                "negative_prompt": NEGATIVE_PROMPT,
                "size": "320x192",
                "n": 1,
                "response_format": "b64_json",
                "num_inference_steps": 2,
                "guidance_scale": 3.0,
                "flow_shift": 3.0,
                "seed": 42,
            }
        }
    )


@pytest.mark.full_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", _get_server_cases(MODEL), indirect=True)
def test_video_generation_modes_moe(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    request_config = {
        "model": omni_server.model,
        "form_data": {
            "model": omni_server.model,
            "prompt": PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "height": 192,
            "width": 320,
            "num_frames": 9,
            "fps": 24,
            "flow_shift": 3.0,
            "seed": 42,
        },
    }
    openai_client.send_video_diffusion_request(request_config)

    synthetic_image = generate_synthetic_image(320, 192, force_regenerate=True, seed=42)
    request_config["form_data"]["prompt"] = "the red block moves slowly while the camera remains fixed"
    # Keep TI2V dimensions model-scoped until /v1/videos forwards raw reference images.
    request_config["form_data"].pop("height")
    request_config["form_data"].pop("width")
    request_config["form_data"]["extra_params"] = json.dumps({"size": "320x192"}, separators=(",", ":"))
    request_config["image_reference"] = f"data:image/jpeg;base64,{synthetic_image['base64']}"
    responses = openai_client.send_video_diffusion_request(request_config)
    assert responses[0].videos
    assert_video_first_frame_matches(
        responses[0].videos[0],
        synthetic_image["np_array"],
        max_mean_absolute_error=0.25,
    )
