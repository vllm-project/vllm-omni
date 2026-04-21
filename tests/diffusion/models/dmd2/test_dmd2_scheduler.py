# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline, LTX2T2VDMD2Pipeline
from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_image2video import LTX2I2VDMD2Pipeline, LTX2ImageToVideoPipeline
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import Wan22Pipeline, WanT2VDMD2Pipeline
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_i2v import Wan22I2VPipeline, WanI2VDMD2Pipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest, OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DMD2_TIMESTEPS = [999, 937, 833, 624]

# DMD2 subclass → immediate base pipeline whose __init__ loads model weights (mocked in tests).
_DMD2_BASE = {
    WanT2VDMD2Pipeline: Wan22Pipeline,
    WanI2VDMD2Pipeline: Wan22I2VPipeline,
    LTX2T2VDMD2Pipeline: LTX2Pipeline,
    LTX2I2VDMD2Pipeline: LTX2ImageToVideoPipeline,
}


def _make_pipeline(cls):
    """Run the DMD2 __init__ (including __init_dmd2__) with the base pipeline mocked."""

    base = _DMD2_BASE[cls]
    od_config = MagicMock()
    od_config.model = "/nonexistent"

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


@pytest.fixture(
    params=list(_DMD2_BASE.keys()),
    ids=["wan_t2v", "wan_i2v", "ltx2_t2v", "ltx2_i2v"],
)
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
    parent = _DMD2_BASE[type(pipeline)]

    # Baseline: calling set_timesteps(40) without the DMD2 override gives a different schedule
    pipeline.scheduler.set_timesteps(40, device="cpu")
    default_timesteps = pipeline.scheduler.timesteps.long().tolist()
    assert default_timesteps == _DMD2_TIMESTEPS, (
        "DMD2EulerScheduler should always return DMD2 timesteps regardless of num_steps"
    )

    with patch.object(parent, "forward", _fake_parent_forward):
        pipeline.forward(_make_request())

    assert pipeline.scheduler.timesteps.long().tolist() == _DMD2_TIMESTEPS


def test_forward_timesteps_idempotent_across_calls(pipeline):
    """Successive forward() calls must not cause scheduler state to drift."""
    parent = _DMD2_BASE[type(pipeline)]

    with patch.object(parent, "forward", _fake_parent_forward):
        pipeline.forward(_make_request())
        pipeline.forward(_make_request())

    assert pipeline.scheduler.timesteps.long().tolist() == _DMD2_TIMESTEPS
