# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import Wan22Pipeline, WanT2VDMD2Pipeline
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_i2v import Wan22I2VPipeline, WanI2VDMD2Pipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest, OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DMD2_TIMESTEPS = [999, 937, 833, 624]

# Wan base pipeline whose __init__ loads model weights — mocked in tests.
_WAN_BASE = {
    WanT2VDMD2Pipeline: Wan22Pipeline,
    WanI2VDMD2Pipeline: Wan22I2VPipeline,
}


def _make_pipeline(cls):
    """Run the DMD2 __init__ (including __init_dmd2__) with the Wan base mocked."""

    base = _WAN_BASE[cls]
    od_config = MagicMock()
    od_config.model = "/nonexistent"  # _load_model_index returns {} → uses inline defaults

    def _mock_base_init(self, *a, **kw):
        self.od_config = od_config  # __init_dmd2__ needs this

    with patch.object(base, "__init__", _mock_base_init):
        pipeline = object.__new__(cls)
        torch.nn.Module.__init__(pipeline)
        cls.__init__(pipeline, od_config=od_config)
    return pipeline


def _make_request(**sp_kwargs) -> OmniDiffusionRequest:
    sp = OmniDiffusionSamplingParams(**sp_kwargs)
    return OmniDiffusionRequest(prompts=[{"prompt": "a cat"}], sampling_params=sp)


@pytest.fixture(params=[WanT2VDMD2Pipeline, WanI2VDMD2Pipeline], ids=["t2v", "i2v"])
def pipeline(request):
    return _make_pipeline(request.param)


# ---------------------------------------------------------------------------
# forward() timestep injection
# ---------------------------------------------------------------------------


def _fake_parent_forward(self, req, *args, num_inference_steps=40, **kwargs):
    """Stub that calls set_timesteps as the real parent does."""
    self.scheduler.set_timesteps(num_inference_steps, device="cpu")
    return MagicMock()


def test_forward_timesteps_match_dmd2_schedule(pipeline):
    """After forward() runs, scheduler.timesteps must equal the DMD2 training schedule."""
    parent = _WAN_BASE[type(pipeline)]

    # Baseline: calling set_timesteps(40) without the DMD2 override gives a different schedule
    pipeline.scheduler.set_timesteps(40, device="cpu")
    default_timesteps = pipeline.scheduler.timesteps.long().tolist()
    assert default_timesteps == _DMD2_TIMESTEPS, (
        "DMD2EulerScheduler should always return DMD2 timesteps regardless of num_steps"
    )

    with patch.object(parent, "forward", _fake_parent_forward):
        pipeline.forward(_make_request())

    assert pipeline.scheduler.timesteps.long().tolist() == _DMD2_TIMESTEPS


def test_forward_timesteps_fixed_across_num_steps(pipeline):
    """scheduler.timesteps is always the DMD2 schedule regardless of num_steps passed."""
    parent = _WAN_BASE[type(pipeline)]

    for num_steps in [1, 4, 10, 40, 100]:
        with patch.object(parent, "forward", _fake_parent_forward):
            pipeline.forward(_make_request())

        assert pipeline.scheduler.timesteps.long().tolist() == _DMD2_TIMESTEPS, (
            f"num_steps={num_steps}: got {pipeline.scheduler.timesteps.tolist()}"
        )
