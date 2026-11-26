# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA weekly online serving smoke test for ``nvidia/Cosmos3-Nano-Transfer-Auto``."""

import json
import os

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_video
from tests.helpers.runtime import (
    OmniServer,
    OmniServerParams,
    OnlineOmniClient,
)

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

TRANSFER_AUTO_MODEL = "nvidia/Cosmos3-Nano-Transfer-Auto"
TRANSFER_AUTO_SERVER_ARGS = [
    "--model-class-name",
    "Cosmos3MultiviewPipeline",
    "--no-guardrails",
    # The pipeline skips engine warmup, so compilation would otherwise run inside the first request.
    "--enforce-eager",
]
# Multiview attention is CUDA-only.
TRANSFER_AUTO_MARKS = hardware_marks(res={"cuda": "H100", "rocm": "MI300"})


@pytest.mark.slow
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_server",
    [
        pytest.param(
            OmniServerParams(model=TRANSFER_AUTO_MODEL, server_args=TRANSFER_AUTO_SERVER_ARGS),
            id="transfer_auto",
            marks=TRANSFER_AUTO_MARKS,
        ),
    ],
    indirect=True,
)
def test_transfer_auto_uploaded_control_001(omni_server: OmniServer, online_client: OnlineOmniClient) -> None:
    """Single-camera Transfer-Auto WSM transfer through ``/v1/videos``, with the control uploaded by index.

    A 16:9 control selects the 480p 832x480 bucket. One camera keeps the camera-major MP4 at ``num_frames``.
    """
    control_reference = f"data:video/mp4;base64,{generate_synthetic_video(832, 480, 5)['base64']}"
    extra_params = {
        "multiview": {
            "views": [
                {
                    "camera_key": "camera_front_wide_120fov",
                    "prompt": "A car drives along a tree-lined suburban street.",
                    "control_reference_index": 0,
                }
            ]
        },
        "wsm": {},
    }
    online_client.send_video_diffusion_request(
        {
            "model": omni_server.model,
            "form_data": {
                "prompt": "Driving through a suburban street.",
                "width": 832,
                "height": 480,
                "num_frames": 5,
                "fps": 30,
                "num_inference_steps": 2,
                "seed": 42,
                "extra_params": json.dumps(extra_params),
            },
            "input_references": [control_reference],
        }
    )
