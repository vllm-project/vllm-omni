# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunnerHandler
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = "nvidia/Cosmos3-Nano"
PROMPT = "A small warehouse robot moves a blue box across a clean floor."
NEGATIVE_PROMPT = "blurry, distorted, low quality"

# (model, stage_config_path, extra_omni_kwargs). Offline has no --no-guardrails CLI flag, so
# set its engine-side equivalent: serve.py maps --no-guardrails to model_config["guardrails"]=False.
OMNI_RUNNER_PARAM = (MODEL, None, {"model_config": {"guardrails": False}})


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards={"cuda": 1})
@pytest.mark.parametrize("omni_runner", [OMNI_RUNNER_PARAM], indirect=True)
def test_text_to_video_001(omni_runner_handler: OmniRunnerHandler) -> None:
    """Default Cosmos3-Nano T2V offline smoke: in-process generation returns video bytes."""
    sampling = OmniDiffusionSamplingParams(
        height=256,
        width=256,
        num_frames=5,
        fps=1,
        num_inference_steps=2,
        guidance_scale=1.0,
        seed=42,
    )
    request_config = {
        "model": MODEL,
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "sampling_params": sampling,
    }
    omni_runner_handler.send_diffusion_request(request_config)
