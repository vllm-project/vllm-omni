# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L4 expansion coverage for SkyReels V3 V2V (single-shot video extension)."""

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_video
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]

MODEL = "Skywork/SkyReels-V3-V2V-14B"

CUDA_SINGLE_CARD_MARKS = hardware_marks(res={"cuda": "H100"})


def _get_v2v_feature_cases():
    return [
        pytest.param(
            OmniServerParams(model=MODEL),
            id="cuda_v2v_baseline",
            marks=CUDA_SINGLE_CARD_MARKS,
        ),
    ]


@pytest.mark.parametrize(
    "omni_server",
    _get_v2v_feature_cases(),
    indirect=True,
)
def test_skyreels_v3_v2v_features(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
):
    video_asset = generate_synthetic_video(num_frames=72, height=720, width=1280, fps=24)

    form_data = {
        "prompt": "A cinematic continuation of the scene.",
        "num_inference_steps": 4,
        "seed": 42,
        "extra_params": '{"resolution": "720P", "duration": 3, "condition_frames": 25}',
    }

    request_config = {
        "model": MODEL,
        "form_data": form_data,
        "video_reference": f"data:video/mp4;base64,{video_asset['base64']}",
    }

    openai_client.send_video_diffusion_request(request_config)
