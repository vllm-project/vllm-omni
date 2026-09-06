# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Released-checkpoint MAGI-2 Preview serving smoke on four H100 GPUs.

This is a one-step 272p load/dispatch test, not a quality or performance test.
It exercises the recommended resident SP4 topology through ``/v1/videos``.

From ``tests/``::

    pytest -s -v e2e/online_serving/test_magi2.py -m "advanced_model and diffusion" --run-level=advanced_model
"""

import json
import os

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OnlineOmniClient

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "sand-ai/MAGI-2-preview"
PROMPT = "A red fox walks through fresh snow while wind moves the pine branches."

FOUR_CARD_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=4)


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize(
    "omni_server",
    [
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=[
                    "--model-class-name",
                    "Magi2Pipeline",
                    "--num-gpus",
                    "4",
                    "--tensor-parallel-size",
                    "1",
                    "--ulysses-degree",
                    "4",
                ],
                init_timeout=1800,
            ),
            id="resident_sp4_272p_one_step",
            marks=FOUR_CARD_MARKS,
        )
    ],
    indirect=True,
)
def test_magi2_preview_serving_smoke(
    omni_server: OmniServer,
    online_client: OnlineOmniClient,
) -> None:
    request_config = {
        "model": omni_server.model,
        "form_data": {
            "prompt": PROMPT,
            "height": 256,
            "width": 448,
            "num_frames": 125,
            "fps": 12.5,
            "num_inference_steps": 1,
            "seed": 42,
            "extra_params": json.dumps({"resolution": "272p"}),
        },
        "expected_audio": {"sample_rate": 44100, "channels": 2},
        "fps_tolerance": 0.01,
    }

    online_client.send_video_diffusion_request(request_config)
