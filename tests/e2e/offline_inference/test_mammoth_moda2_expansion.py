# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
End-to-end test for MammothModa2 text-to-image generation.

Model Hub repo id: ``bytedance-research/MammothModa2-Preview``.
Deploy config: ``get_deploy_config_path("mammoth_moda2.yaml")`` -> ``vllm_omni/deploy/mammoth_moda2.yaml``
"""

from __future__ import annotations

import pytest
from vllm.sampling_params import SamplingParams

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs
from vllm_omni.entrypoints.openai.stage_params import clone_sampling_params
from vllm_omni.model_extras.mammothmodal2_preview import build_text_to_image_prompt

MODEL_PATH = "bytedance-research/MammothModa2-Preview"
T2I_DEPLOY_CONFIG = get_deploy_config_path("mammoth_moda2.yaml")

_OMNI_RUNNER_PARAM = (MODEL_PATH, T2I_DEPLOY_CONFIG)

PROMPT = "A cat sitting on a laptop keyboard"
# Small for CI speed, and not square so the size assertion catches a H/W swap.
HEIGHT, WIDTH = 256, 320
NUM_INFERENCE_STEPS = 2

pytestmark = [
    pytest.mark.slow,
    pytest.mark.diffusion,
    pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True),
]


@hardware_test(res={"cuda": "H100"})
def test_mammothmoda2_t2i_e2e(omni_runner: OmniRunner):
    """The two-stage AR -> DiT pipeline returns an image at the requested size."""
    prompt = build_text_to_image_prompt(PROMPT, None, HEIGHT, WIDTH)
    prompt["modalities"] = ["image"]

    # The DiT stage takes its knobs through extra_args, so keep the deploy
    # config's guidance settings and only shorten the schedule for CI.
    *_, dit_sampling = (clone_sampling_params(p) for p in omni_runner.omni.default_sampling_params_list)
    dit_sampling.extra_args["num_inference_steps"] = NUM_INFERENCE_STEPS

    grid = prompt["additional_information"]
    ar_width, ar_height = grid["ar_width"][0], grid["ar_height"][0]
    ar_sampling = SamplingParams(
        temperature=0.0,
        top_k=1,
        # One visual token per grid cell, one EOL per row, one final look-ahead token.
        max_tokens=ar_height * (ar_width + 1) + 1,
        detokenize=False,
    )

    outputs = omni_runner.omni.generate([prompt], [ar_sampling, dit_sampling])

    images = extract_images_from_outputs(outputs)
    assert images, "Pipeline produced no image"
    assert images[0].size == (WIDTH, HEIGHT)
