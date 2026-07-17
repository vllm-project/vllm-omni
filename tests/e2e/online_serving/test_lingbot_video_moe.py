# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Single-GPU serving smoke for ``robbyant/lingbot-video-moe-30b-a3b``."""

import os

import pytest

from tests.helpers.mark import hardware_marks
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


@pytest.mark.full_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", _get_server_cases(MODEL), indirect=True)
def test_text_to_video_moe(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
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
