# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

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


def test_forward_preserves_strength_schedule_at_first_prediction(monkeypatch):
    from diffusers import FlowMatchEulerDiscreteScheduler

    class StopAfterFirstPredictionError(Exception):
        pass

    pipe = _make_pipeline()
    torch.nn.Module.__init__(pipe)
    pipe.latent_channels = 4
    pipe._execution_device = torch.device("cpu")
    pipe.transformer = SimpleNamespace(config=SimpleNamespace(in_channels=16), dtype=torch.float32)
    pipe.scheduler = FlowMatchEulerDiscreteScheduler()
    pipe.vae = SimpleNamespace(dtype=torch.float32)
    pipe.check_inputs = lambda **kwargs: None
    pipe.encode_prompt = lambda **kwargs: (torch.zeros(1, 2, 4), torch.zeros(1, 2, 4))
    pipe.prepare_latents = lambda **kwargs: (torch.zeros(1, 4, 4), torch.zeros(1, 4, 4))
    pipe.prepare_image_latents = pipe.prepare_latents
    pipe._cache_context_factory = lambda name: nullcontext()

    def check_first_prediction(**kwargs):
        assert pipe.scheduler.begin_index == 2
        assert pipe.num_timesteps == 2
        assert pipe.current_step_index == 0
        assert float(pipe.current_sigma) == pytest.approx(0.6)
        raise StopAfterFirstPredictionError

    monkeypatch.setattr(pipe, "predict_noise", check_first_prediction)

    sampling = OmniDiffusionSamplingParams(
        height=32,
        width=32,
        num_inference_steps=4,
        sigmas=[1.0, 0.8, 0.6, 0.4],
        strength=0.5,
        output_type="latent",
    )
    request = SimpleNamespace(
        sampling_params=sampling,
        prompts=[{"prompt": "edit", "multi_modal_data": {"reference_image": torch.zeros(1, 4, 2, 2)}}],
    )
    with pytest.raises(StopAfterFirstPredictionError):
        pipe.forward(request)
