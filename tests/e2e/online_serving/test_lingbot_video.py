# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Online serving smoke coverage for the dense LingBot-Video checkpoint."""

import io
import os

import av
import numpy as np
import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "robbyant/lingbot-video-dense-1.3b"
PROMPT = "a robotic arm picks up a red block"
NEGATIVE_PROMPT = "low quality, blurry, watermark, text"
DEFAULT_SAMPLING_PARAMS = '{"0":{"num_frames":81,"num_inference_steps":40,"guidance_scale":6.0}}'

SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})


def _get_diffusion_feature_cases(model: str):
    return [
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=[
                    "--model-class-name",
                    "LingBotVideoPipeline",
                    "--default-sampling-params",
                    DEFAULT_SAMPLING_PARAMS,
                ],
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
    responses = openai_client.send_images_generations_http_request(
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
    response = responses[0]
    assert response.success, response.error_message
    payload = response.json_body
    assert isinstance(payload, dict)
    assert len(payload["data"]) == 1
    assert payload["data"][0]["b64_json"]


def _assert_first_frame_conditioned(video_bytes: bytes, expected: np.ndarray) -> None:
    with av.open(io.BytesIO(video_bytes)) as container:
        first_frame = next(container.decode(video=0)).to_ndarray(format="rgb24")
    assert first_frame.shape == expected.shape
    mean_absolute_error = float(np.abs(first_frame.astype(np.float32) - expected).mean() / 255.0)
    assert mean_absolute_error < 0.25


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", _get_diffusion_feature_cases(MODEL), indirect=True)
def test_video_generation_modes_001(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
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
            "num_inference_steps": 2,
            "guidance_scale": 3.0,
            "flow_shift": 3.0,
            "seed": 42,
        },
    }
    openai_client.send_video_diffusion_request(request_config)

    synthetic_image = generate_synthetic_image(320, 192, force_regenerate=True, seed=42)
    request_config["form_data"]["prompt"] = "the red block moves slowly while the camera remains fixed"
    request_config["image_reference"] = f"data:image/jpeg;base64,{synthetic_image['base64']}"
    responses = openai_client.send_video_diffusion_request(request_config)
    assert responses[0].videos
    _assert_first_frame_conditioned(responses[0].videos[0], synthetic_image["np_array"])
