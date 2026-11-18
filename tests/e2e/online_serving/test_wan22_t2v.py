# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Online serving smoke for ``Wan-AI/Wan2.2-T2V-A14B-Diffusers`` (text-to-video via ``/v1/videos``).

Uses a single ``default`` ``OmniServerParams`` row via ``_get_diffusion_feature_cases`` (no extra
``server_args``), with explicit startup budgets for loading both experts from slow storage.
Multi-variant / parallel coverage lives in ``test_wan22_expansion.py`` (L4).

From ``tests/``::

    pytest -s -v e2e/online_serving/test_wan22_t2v.py -m "core_model and diffusion" --run-level=core_model
    pytest -s -v e2e/online_serving/test_wan22_t2v.py -m "advanced_model and diffusion" --run-level=advanced_model
"""

import os

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OnlineOmniClient
from vllm_omni.platforms import current_omni_platform

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
PROMPT = "Two anthropomorphic cats in boxing gear on a spotlighted stage."
NEGATIVE_PROMPT = "low quality, blurry, watermark, text"
# The online and offline tests run in separate Buildkite jobs, so either can
# encounter the cold ROCm load. Keep the test deadline five minutes below the
# merge job's 40-minute process deadline for cleanup.
WAN22_INIT_TIMEOUT_S = 35 * 60

# CUDA / ROCm: single card, no extra server_args — behavior unchanged.
# Skip on NPU, where a single A3 (64 GB HBM) cannot hold Wan2.2-T2V-A14B.
CUDA_SINGLE_CARD_FEATURE_MARKS = [
    *hardware_marks(res={"cuda": "H100"}, num_cards=1),
    pytest.mark.skipif(
        current_omni_platform.is_npu(),
        reason="CUDA/ROCm single-card path; skip on NPU",
    ),
]

# NPU: TP=2 across two A3 cards.
# Skip on any non-NPU platform.
NPU_TP2_FEATURE_MARKS = [
    *hardware_marks(res={"npu": "A3"}, num_cards=2),
    pytest.mark.skipif(
        not current_omni_platform.is_npu(),
        reason="Requires Ascend NPU platform",
    ),
]


def _get_diffusion_feature_cases(model: str):
    """Return one param per platform with explicit platform skip conditions."""
    return [
        # CUDA: single card, no extra server_args
        pytest.param(
            OmniServerParams(
                model=model,
                init_timeout=WAN22_INIT_TIMEOUT_S,
                stage_init_timeout=WAN22_INIT_TIMEOUT_S,
                startup_timeout=WAN22_INIT_TIMEOUT_S,
            ),
            id="default",
            marks=CUDA_SINGLE_CARD_FEATURE_MARKS,
        ),
        # NPU: TP=2 across two cards
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=["--tensor-parallel-size", "2"],
            ),
            id="default",
            marks=NPU_TP2_FEATURE_MARKS,
        ),
    ]


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.diffusion
@pytest.mark.parametrize("omni_server", _get_diffusion_feature_cases(MODEL), indirect=True)
def test_text_to_video_001(omni_server: OmniServer, online_client: OnlineOmniClient) -> None:
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
    online_client.send_video_diffusion_request(request_config)
