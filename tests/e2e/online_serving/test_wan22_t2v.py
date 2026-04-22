# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Online serving smoke for ``Wan-AI/Wan2.2-T2V-A14B-Diffusers`` (text-to-video via ``/v1/videos``).

Uses a single ``default`` ``OmniServerParams`` row via ``_get_diffusion_feature_cases`` (no extra
``server_args``). Multi-variant / parallel coverage lives in ``test_wan22_expansion.py`` (L4).

From ``tests/``::

    pytest -s -v e2e/online_serving/test_wan22_t2v.py -m "core_model and diffusion" --run-level=core_model
    pytest -s -v e2e/online_serving/test_wan22_t2v.py -m "advanced_model and diffusion" --run-level=advanced_model
"""

import os

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OpenAIClientHandler

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
PROMPT = "Two anthropomorphic cats in boxing gear on a spotlighted stage."
NEGATIVE_PROMPT = "low quality, blurry, watermark, text"

SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})


def _get_diffusion_feature_cases(model: str):
    """Return a single default ``OmniServerParams`` row (no extra ``server_args``)."""
    return [
        pytest.param(
            OmniServerParams(model=model),
            id="default",
            marks=SINGLE_CARD_FEATURE_MARKS,
        ),
    ]


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", _get_diffusion_feature_cases(MODEL), indirect=True)
def test_text_to_video_001(omni_server: OmniServer, openai_client: OpenAIClientHandler) -> None:
    """Default Wan2.2 T2V smoke: async ``/v1/videos`` job completes and returns video bytes."""
    request_config = {
        "model": omni_server.model,
        "form_data": {
            "prompt": PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "height": 512,
            "width": 512,
            "num_frames": 8,
            "fps": 8,
            "num_inference_steps": 2,
            "guidance_scale": 4.0,
            "seed": 42,
        },
    }
    openai_client.send_video_diffusion_request(request_config)
