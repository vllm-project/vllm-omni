# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online-serving smoke tests for the public LTX-2 pipeline entries."""

import os

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]

DISTILLED_MODEL = os.getenv("VLLM_TEST_LTX23_DISTILLED_MODEL", "diffusers/LTX-2.3-Distilled-Diffusers")
BASE_MODEL = os.getenv("VLLM_TEST_LTX23_MODEL", "diffusers/LTX-2.3-Diffusers")
PROMPT = "A serene lake at sunset with mountains in the background."
SINGLE_CARD_MARKS = hardware_marks(res={"cuda": "H100"})
HSDP_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)


def _server(
    model: str,
    model_class_name: str,
    *extra_args: str,
) -> OmniServerParams:
    return OmniServerParams(
        model=model,
        server_args=[
            "--enforce-eager",
            *extra_args,
            "--model-class-name",
            model_class_name,
        ],
    )


def _cases():
    return [
        pytest.param(
            _server(BASE_MODEL, "LTX2Pipeline"),
            3,
            id="one_stage",
            marks=SINGLE_CARD_MARKS,
        ),
        pytest.param(
            _server(DISTILLED_MODEL, "LTX2DistilledPipeline"),
            8,
            id="distilled",
            marks=SINGLE_CARD_MARKS,
        ),
        pytest.param(
            _server(
                DISTILLED_MODEL,
                "LTX2DistilledPipeline",
                "--use-hsdp",
                "--hsdp-shard-size",
                "2",
            ),
            8,
            id="distilled_hsdp2",
            marks=HSDP_MARKS,
        ),
        pytest.param(
            _server(
                BASE_MODEL,
                "LTX2TwoStagePipeline",
                "--enable-layerwise-offload",
            ),
            3,
            id="two_stage",
            marks=SINGLE_CARD_MARKS,
        ),
    ]


@pytest.mark.parametrize(
    ("omni_server", "num_inference_steps"),
    _cases(),
    indirect=["omni_server"],
)
def test_ltx_pipeline_entries(
    omni_server: OmniServer,
    num_inference_steps: int,
    openai_client: OpenAIClientHandler,
    subtests,
):
    for task in ("t2v", "i2v"):
        with subtests.test(task=task):
            form_data = {
                "prompt": PROMPT,
                "model": omni_server.model,
                "height": 128,
                "width": 128,
                "num_frames": 9,
                "fps": 8,
                "num_inference_steps": num_inference_steps,
                "seed": 42,
            }
            request_config = {
                "model": omni_server.model,
                "form_data": form_data,
            }
            if task == "i2v":
                request_config["image_reference"] = (
                    f"data:image/jpeg;base64,{generate_synthetic_image(512, 512)['base64']}"
                )

            openai_client.send_video_diffusion_request(request_config)
