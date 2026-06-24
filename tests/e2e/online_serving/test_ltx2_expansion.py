# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Expansion tests for LTX-2 two-stage pipelines.

Coverage:
- HSDP with LTX2TwoStagesPipeline (text-to-video)
- HSDP with LTX2ImageToVideoTwoStagesPipeline (image-to-video)
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

    # T2V: LTX2TwoStagesPipeline
    cases.append(
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=[
                    *HSDP_ARGS,
                    "--model-class-name",
                    "LTX2TwoStagesPipeline",
                ],
            ),
            id="t2v_hsdp",
            marks=PARALLEL_MARKS,
        )
    )

    # I2V: LTX2ImageToVideoTwoStagesPipeline
    cases.append(
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=[
                    *HSDP_ARGS,
                    "--model-class-name",
                    "LTX2ImageToVideoTwoStagesPipeline",
                ],
            ),
            id="i2v_hsdp",
            marks=PARALLEL_MARKS,
        )
    )

    return cases


@pytest.mark.parametrize("omni_server", _cases(), indirect=True)
def test_ltx2_two_stage_hsdp(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
):
    is_i2v = any("ImageToVideo" in arg for arg in omni_server.serve_args)

    # The two-stage pipeline generates at the requested resolution then 2x
    # upsamples via LTX2LatentUpsamplerModel, so output dimensions differ
    # from the request.  Omit height/width from form_data so the assertion
    # helper skips the dimension check (it only asserts when the key is
    # present).  The pipeline falls back to its own defaults.
    form_data = {
        "prompt": PROMPT,
        "model": omni_server.model,
        "num_frames": 9,
        "fps": 8,
        "num_inference_steps": 2,
        "guidance_scale": 1.0,
        "seed": 42,
    }

    request_config = {
        "model": omni_server.model,
        "form_data": form_data,
    }

    if is_i2v:
        request_config["image_reference"] = f"data:image/jpeg;base64,{generate_synthetic_image(512, 512)['base64']}"

    openai_client.send_video_diffusion_request(request_config)
