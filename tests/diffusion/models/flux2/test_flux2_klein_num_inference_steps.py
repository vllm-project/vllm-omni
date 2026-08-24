# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest

from vllm_omni.diffusion.models.flux2_klein.pipeline_flux2_klein import (
    Flux2KleinPipeline,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _StopAfterCheckInputsError(Exception):
    pass


def _make_pipeline():
    pipeline = object.__new__(Flux2KleinPipeline)
    pipeline.vae_scale_factor = 8
    pipeline.is_distilled = True
    pipeline._guidance_scale = 0.0
    return pipeline


def _make_minimal_request(
    prompt="valid prompt",
    *,
    num_inference_steps=None,
):
    """Build a minimal OmniDiffusionRequest-like object that forward() reads from."""
    params = OmniDiffusionSamplingParams(
        height=512,
        width=512,
        num_inference_steps=num_inference_steps,
        seed=42,
    )
    req = MagicMock()
    req.sampling_params = params
    req.prompt = prompt
    req.multi_modal_data = {}
    return req


def _capture_num_inference_steps(pipe, steps):
    """Run pipe.forward() and capture the num_inference_steps passed to check_inputs."""
    captured = {}

    def fake_check_inputs(**kwargs):
        captured["num_inference_steps"] = kwargs.get("num_inference_steps")
        raise _StopAfterCheckInputsError

    original = pipe.check_inputs
    pipe.check_inputs = fake_check_inputs
    try:
        req = _make_minimal_request(num_inference_steps=steps)
        pipe.forward(req)
    except _StopAfterCheckInputsError:
        pass
    finally:
        pipe.check_inputs = original
    return captured.get("num_inference_steps")


# --- Forward resolution tests (the actual regression) ---


def test_forward_preserves_zero_not_default():
    """#3703: num_inference_steps=0 must reach validation as 0, not as default 50."""
    pipe = _make_pipeline()
    captured = _capture_num_inference_steps(pipe, 0)
    assert captured == 0, f"Expected 0, got {captured}"


def test_forward_preserves_negative():
    pipe = _make_pipeline()
    captured = _capture_num_inference_steps(pipe, -1)
    assert captured == -1


def test_forward_none_uses_default():
    """None means unset — forward() should substitute the pipeline default."""
    pipe = _make_pipeline()
    captured = _capture_num_inference_steps(pipe, None)
    assert captured == 50


def test_forward_preserves_positive():
    pipe = _make_pipeline()
    captured = _capture_num_inference_steps(pipe, 9)
    assert captured == 9


# --- check_inputs validation tests ---


@pytest.mark.parametrize("steps", [0, -1])
def test_check_inputs_rejects_non_positive(steps):
    pipe = _make_pipeline()
    with pytest.raises(ValueError):
        pipe.check_inputs(
            prompt="valid prompt",
            height=512,
            width=512,
            num_inference_steps=steps,
        )


def test_check_inputs_accepts_none():
    pipe = _make_pipeline()
    pipe.check_inputs(prompt="valid prompt", height=512, width=512, num_inference_steps=None)


@pytest.mark.parametrize("steps", [1, 2])
def test_check_inputs_accepts_positive(steps):
    pipe = _make_pipeline()
    pipe.check_inputs(prompt="valid prompt", height=512, width=512, num_inference_steps=steps)
