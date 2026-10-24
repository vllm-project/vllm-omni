# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServerParams, OnlineOmniClient

pytestmark = [pytest.mark.local_model, pytest.mark.diffusion, *hardware_marks(res={"cuda": "H200"}, num_cards=1)]

MODEL = os.environ.get("HY_NOMAS_MODEL", "tencent/Hy-Nomas")


@pytest.mark.parametrize("omni_server", [OmniServerParams(model=MODEL, server_args=["--enforce-eager"])], indirect=True)
def test_video_job(openai_client: OnlineOmniClient):
    openai_client.send_video_diffusion_request(
        {
            "model": MODEL,
            "form_data": {
                "prompt": "A forest beside a river.",
                "size": "256x256",
                "num_frames": 9,
                "fps": 24,
                "num_inference_steps": 4,
                "guidance_scale": 1.0,
                "seed": 42,
            },
        }
    )
