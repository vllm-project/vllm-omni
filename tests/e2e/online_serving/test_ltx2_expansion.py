# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Expansion tests for LTX-2 two-stage pipelines.

Coverage:
- HSDP with LTX2DistilledPipeline (text-to-video and image-to-video)
"""

import os

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]

MODEL = os.getenv("VLLM_TEST_LTX2_MODEL", "rootonchair/LTX-2-19b-distilled")
PROMPT = "A serene lake at sunset with mountains in the background."
PARALLEL_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)
HSDP_ARGS = ["--use-hsdp", "--hsdp-shard-size", "2"]


def _cases():
    cases = []

    cases.append(
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=[
                    *HSDP_ARGS,
                    "--model-class-name",
                    "LTX2DistilledPipeline",
                ],
            ),
            False,
            id="t2v_hsdp",
            marks=PARALLEL_MARKS,
        )
    )

    cases.append(
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=[
                    *HSDP_ARGS,
                    "--model-class-name",
                    "LTX2DistilledPipeline",
                ],
            ),
            True,
            id="i2v_hsdp",
            marks=PARALLEL_MARKS,
        )
    )

    return cases


@pytest.mark.parametrize(("omni_server", "is_i2v"), _cases(), indirect=["omni_server"])
def test_ltx2_two_stage_hsdp(
    omni_server: OmniServer,
    is_i2v: bool,
    openai_client: OpenAIClientHandler,
):
    # Keep CI small while exercising the fixed 8+3-step distilled recipe.
    # Height and width describe the final output; Stage 1 runs at half size.
    form_data = {
        "prompt": PROMPT,
        "model": omni_server.model,
        "height": 128,
        "width": 128,
        "num_frames": 9,
        "fps": 8,
        "num_inference_steps": 8,
        "seed": 42,
    }

    request_config = {
        "model": omni_server.model,
        "form_data": form_data,
    }

    if is_i2v:
        request_config["image_reference"] = f"data:image/jpeg;base64,{generate_synthetic_image(512, 512)['base64']}"

    openai_client.send_video_diffusion_request(request_config)
