# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import MagicMock

import pytest

from vllm_omni.diffusion.models.flux2.pipeline_flux2 import Flux2Pipeline
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _pipeline():
    pipeline = object.__new__(Flux2Pipeline)
    pipeline.vae_scale_factor = 8
    return pipeline


def _request(prompt, num_inference_steps):
    req = MagicMock()
    req.prompts = [prompt]
    req.sampling_params = OmniDiffusionSamplingParams(
        height=512,
        width=512,
        num_inference_steps=num_inference_steps,
    )
    return req


@pytest.mark.parametrize(
    "prompt",
    [
        "",
        "   \t  ",
    ],
)
def test_forward_rejects_empty_or_whitespace_prompt(prompt):
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        _pipeline().forward(_request(prompt, 4))


@pytest.mark.parametrize("num_inference_steps", [0, -1])
def test_forward_rejects_non_positive_steps(num_inference_steps):
    with pytest.raises(ValueError, match="num_inference_steps must be positive"):
        _pipeline().forward(_request("valid prompt", num_inference_steps))


@pytest.mark.parametrize(
    "prompt",
    [
        ["valid prompt", ""],
        ["valid prompt", "   "],
        ["   "],
    ],
)
def test_check_inputs_rejects_list_with_empty_or_whitespace_element(prompt):
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        _pipeline().check_inputs(
            prompt=prompt,
            height=512,
            width=512,
            num_inference_steps=4,
        )


def test_forward_keeps_kwarg_steps_when_sampling_is_none():
    pipeline = _pipeline()
    seen: dict[str, int | None] = {}

    def _capture_check_inputs(*args, **kwargs):
        seen["num_inference_steps"] = kwargs.get("num_inference_steps")
        raise RuntimeError("stopped_after_check")

    pipeline.check_inputs = _capture_check_inputs  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="stopped_after_check"):
        pipeline.forward(_request("valid prompt", None), num_inference_steps=7)
    assert seen["num_inference_steps"] == 7
