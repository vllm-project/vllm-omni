# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
from unittest.mock import MagicMock, patch

import pytest
from diffusers import FlowMatchEulerDiscreteScheduler

from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import WanT2VDMD2Pipeline
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_i2v import WanI2VDMD2Pipeline
from vllm_omni.diffusion.models.schedulers import FlowUniPCMultistepScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

def _make_pipeline(cls):
    """
    Instantiate a DMD2 pipeline letting __init__ run — but mocking the
    parent's __init__ so no model weights are loaded.

    This verifies that the DMD2 class itself (not the test helper) sets up
    the correct scheduler.
    """
    import torch
    parent = cls.__bases__[0]  # Wan22Pipeline or Wan22I2VPipeline
    with patch.object(parent, "__init__", lambda *a, **kw: None):
        pipeline = object.__new__(cls)
        torch.nn.Module.__init__(pipeline)
        cls.__init__(pipeline, od_config=MagicMock())
    return pipeline


@pytest.fixture(params=[WanT2VDMD2Pipeline, WanI2VDMD2Pipeline], ids=["t2v", "i2v"])
def pipeline(request):
    return _make_pipeline(request.param)



from vllm_omni.diffusion.request import OmniDiffusionRequest, OmniDiffusionSamplingParams


def _make_request(**sp_kwargs) -> OmniDiffusionRequest:
    sp = OmniDiffusionSamplingParams(**sp_kwargs)
    return OmniDiffusionRequest(
        prompts=[{"prompt": "a cat"}],
        sampling_params=sp,
    )



def test_scheduler_is_euler(pipeline):
    """DMD2 __init__ must replace the parent's UniPC with Euler."""
    assert isinstance(pipeline.scheduler, FlowMatchEulerDiscreteScheduler)



def _fake_parent_forward(self, req, *args, num_inference_steps=40, **kwargs):
    """Minimal parent forward() stub: calls set_timesteps exactly as the real parent does."""
    self.scheduler.set_timesteps(num_inference_steps, device="cpu")
    return MagicMock()

def test_forward_timesteps_match_dmd2_schedule(pipeline):
    """
    After forward() runs, scheduler.timesteps must equal DMD2_TIMESTEPS.
    """
    parent = type(pipeline).__bases__[0]

    # Baseline: Euler scheduler with num_steps=40 gives a different schedule
    pipeline.scheduler.set_timesteps(40, device="cpu")
    default_timesteps = pipeline.scheduler.timesteps.long().tolist()
    assert default_timesteps != pipeline.DMD2_TIMESTEPS, (
        "Euler scheduler default 40-step schedule unexpectedly matches DMD2_TIMESTEPS — "
        "this test would be vacuous."
    )

    # After DMD2 forward() — scheduler.timesteps must be DMD2_TIMESTEPS
    # regardless of the num_steps the caller passed (40 here).
    with patch.object(parent, "forward", _fake_parent_forward):
        pipeline.forward(_make_request(), num_inference_steps=40)

    assert pipeline.scheduler.timesteps.long().tolist() == pipeline.DMD2_TIMESTEPS


def test_forward_timesteps_fixed_across_num_steps(pipeline):
    """DMD2 timesteps are always the same regardless of what num_steps the caller passes."""
    parent = type(pipeline).__bases__[0]

    for num_steps in [1, 4, 10, 40, 100]:
        with patch.object(parent, "forward", _fake_parent_forward):
            pipeline.forward(_make_request(), num_inference_steps=num_steps)

        assert pipeline.scheduler.timesteps.long().tolist() == pipeline.DMD2_TIMESTEPS, (
            f"num_steps={num_steps}: scheduler.timesteps {pipeline.scheduler.timesteps.tolist()} "
            f"!= DMD2_TIMESTEPS {pipeline.DMD2_TIMESTEPS}"
        )


