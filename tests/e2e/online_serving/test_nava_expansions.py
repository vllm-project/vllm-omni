# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Online serving smoke test for NAVA through the async ``/v1/videos`` API."""

import json
import os

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = os.getenv("VLLM_TEST_NAVA_MODEL", "baidu/NAVA")
SERVED_MODEL_NAME = "nava"
PROMPT = "A person speaks while standing near the sea."
SINGLE_CARD_MARKS = hardware_marks(res={"cuda": "H100"})


def _get_nava_cases():
    return [
        pytest.param(
            OmniServerParams(
                model=MODEL,
                server_args=[
                    "--model-class-name",
                    "NAVAPipeline",
                    "--served-model-name",
                    SERVED_MODEL_NAME,
                ],
                init_timeout=1200,
                stage_init_timeout=1200,
                env_dict={"VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
            ),
            id="default",
            marks=SINGLE_CARD_MARKS,
        ),
    ]


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", _get_nava_cases(), indirect=True)
def test_nava_text_to_audio_video(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    """NAVA T2AV smoke: create an async video job, wait for completion, and download MP4 content."""
    request_config = {
        "model": SERVED_MODEL_NAME,
        "form_data": {
            "model": SERVED_MODEL_NAME,
            "prompt": PROMPT,
            "extra_params": json.dumps(
                {
                    "height": 192,
                    "width": 336,
                    "num_frames": 5,
                    "fps": 8,
                    "num_inference_steps": 1,
                }
            ),
        },
    }
    openai_client.send_video_diffusion_request(request_config)
