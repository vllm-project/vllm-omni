# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import os

import pytest

from tests.helpers.assertions import assert_video_valid
from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OnlineOmniClient

pytestmark = [pytest.mark.local_model, pytest.mark.diffusion]

MODEL = os.environ.get("HY_NOMAS_MODEL", "tencent/Hy-Nomas")


@pytest.mark.parametrize(
    "omni_server",
    [
        pytest.param(
            OmniServerParams(model=MODEL, server_args=["--enforce-eager"]),
            id="single_device",
            marks=hardware_marks(res={"cuda": "H200"}, num_cards=1),
        ),
        pytest.param(
            OmniServerParams(model=MODEL, server_args=["--enforce-eager", "--ulysses-degree", "2"]),
            id="sp2",
            marks=hardware_marks(res={"cuda": "H200"}, num_cards=2),
        ),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "prompt,extra,num_frames",
    [
        pytest.param("A forest beside a river.", {}, 9, id="text"),
        pytest.param(
            "A forest beside a river. ||| The camera moves along the river.",
            {
                "segment_frame_boundaries": [0, 4, 9],
                "camera_trajectory": "pan_left",
                "segment_has_camera": [True, True],
            },
            9,
            id="camera_segments",
        ),
        pytest.param("A beach at sunrise.", {"duration": 1}, None, id="duration_without_audio"),
    ],
)
def test_video_job(
    omni_server: OmniServer, online_client: OnlineOmniClient, prompt: str, extra: dict, num_frames: int | None
):
    responses = online_client.send_video_diffusion_request(
        {
            "model": omni_server.model,
            "form_data": {
                "prompt": prompt,
                "size": "256x256",
                "num_frames": num_frames,
                "generate_sound": False,
                "fps": 24,
                "num_inference_steps": 4,
                "guidance_scale": 1.0,
                "seed": 42,
                "extra_params": json.dumps(extra),
            },
        }
    )
    assert responses[0].videos
    assert_video_valid(
        responses[0].videos[0], num_frames=25 if num_frames is None else num_frames, width=256, height=256, fps=24
    )
