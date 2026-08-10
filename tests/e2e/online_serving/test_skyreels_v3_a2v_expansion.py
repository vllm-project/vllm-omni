# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L4 expansion coverage for SkyReels V3 A2V (talking-avatar diffusion)."""

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_audio, generate_synthetic_image
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]

MODEL = "Skywork/SkyReels-V3-A2V-19B"

CUDA_SINGLE_CARD_MARKS = hardware_marks(res={"cuda": "H100"})
CUDA_PARALLEL_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)


def _get_a2v_feature_cases():
    return [
        pytest.param(
            OmniServerParams(model=MODEL),
            id="cuda_a2v_baseline",
            marks=CUDA_SINGLE_CARD_MARKS,
        ),
        pytest.param(
            OmniServerParams(model=MODEL, server_args=["--cfg-parallel-size", "2"]),
            id="cuda_a2v_cfg_parallel",
            marks=CUDA_PARALLEL_MARKS,
        ),
    ]


@pytest.mark.parametrize(
    "omni_server",
    _get_a2v_feature_cases(),
    indirect=True,
)
def test_skyreels_v3_a2v_features(
    omni_server: OmniServer,
    openai_client: OpenAIClientHandler,
):
    audio_asset = generate_synthetic_audio(
        duration=1,
        num_channels=1,
        sample_rate=16000,
        phrase_text="a person is talking",
    )
    portrait_asset = generate_synthetic_image(480, 480)

    form_data = {
        "prompt": "a person is talking",
        "num_frames": 25,
        "fps": 25,
        "num_inference_steps": 4,
        "seed": 42,
        "extra_params": '{"resolution": "480P"}',
    }

    request_config = {
        "model": MODEL,
        "form_data": form_data,
        "image_reference": f"data:image/jpeg;base64,{portrait_asset['base64']}",
        "audio_reference": f"data:audio/wav;base64,{audio_asset['base64']}",
    }

    openai_client.send_video_diffusion_request(request_config)
