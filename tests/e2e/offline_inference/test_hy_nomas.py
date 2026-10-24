# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OfflineOmniClient
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.local_model, pytest.mark.diffusion, *hardware_marks(res={"cuda": "H200"}, num_cards=1)]

MODEL = os.environ.get("HY_NOMAS_MODEL", "tencent/Hy-Nomas")


@pytest.mark.parametrize("omni_runner", [(MODEL, None, {"enforce_eager": True})], indirect=True)
def test_two_segments_then_independent_request(offline_client: OfflineOmniClient):
    for prompt, extra in (
        ("A forest beside a river. ||| The camera moves along the river.", {"segment_frame_boundaries": [0, 4, 9]}),
        ("A beach at sunrise.", {}),
    ):
        offline_client.send_diffusion_request(
            {
                "model": MODEL,
                "prompt": prompt,
                "sampling_params": OmniDiffusionSamplingParams(
                    height=256,
                    width=256,
                    num_frames=9,
                    fps=24,
                    num_inference_steps=4,
                    guidance_scale=1.0,
                    seed=42,
                    extra_args=extra,
                ),
            }
        )
