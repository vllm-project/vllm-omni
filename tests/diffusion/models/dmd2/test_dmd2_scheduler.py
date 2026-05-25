# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_omni.diffusion.models.flux.pipeline_flux import FluxDMD2Pipeline, FluxPipeline
from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline, LTX2T2VDMD2Pipeline
from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_image2video import LTX2I2VDMD2Pipeline, LTX2ImageToVideoPipeline
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import QwenImageDMD2Pipeline, QwenImagePipeline
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import Wan22Pipeline, WanT2VDMD2Pipeline
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_i2v import Wan22I2VPipeline, WanI2VDMD2Pipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest, OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Linspace fallback timesteps for num_inference_steps=4 (the mixin default when model_index is empty).
_DMD2_TIMESTEPS = [999, 749, 499, 249]

# DMD2 subclass → immediate base pipeline whose __init__ loads model weights (mocked in tests).
_DMD2_BASE = {
    WanT2VDMD2Pipeline: Wan22Pipeline,
    WanI2VDMD2Pipeline: Wan22I2VPipeline,
    LTX2T2VDMD2Pipeline: LTX2Pipeline,
    LTX2I2VDMD2Pipeline: LTX2ImageToVideoPipeline,
    FluxDMD2Pipeline: FluxPipeline,
    QwenImageDMD2Pipeline: QwenImagePipeline,
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
    ids=["wan_t2v", "wan_i2v", "ltx2_t2v", "ltx2_i2v", "flux", "qwen_image"],
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


def test_default_solver_is_ode(pipeline):
    """Default dmd2_config.solver is 'ode' → scheduler.stochastic_sampling is False."""
    assert pipeline.dmd2_config.solver == "ode"
    assert pipeline.scheduler.config.stochastic_sampling is False


def test_sde_solver_plumbed_to_scheduler():
    """solver='sde' in model_index → scheduler.stochastic_sampling is True."""
    from vllm_omni.diffusion.models.dmd2 import DMD2Config
    from vllm_omni.diffusion.models.schedulers import DMD2EulerScheduler

    cfg = DMD2Config.from_model_index({"dmd2_config": {"solver": "sde"}})
    scheduler = DMD2EulerScheduler(
        num_train_timesteps=1000,
        shift=1.0,
        dmd2_timesteps=cfg.resolve_timesteps(),
        stochastic_sampling=(cfg.solver == "sde"),
    )
    assert scheduler.config.stochastic_sampling is True


def test_solver_case_insensitive():
    """'SDE', 'Sde', ' sde ' all normalize to 'sde'."""
    from vllm_omni.diffusion.models.dmd2 import DMD2Config

    for raw in ("SDE", "Sde", " sde ", "sde"):
        cfg = DMD2Config.from_model_index({"dmd2_config": {"solver": raw}})
        assert cfg.solver == "sde"


def test_solver_invalid_raises():
    """Unknown solver strings raise ValueError with a clear message."""
    import pytest

    from vllm_omni.diffusion.models.dmd2 import DMD2Config

    with pytest.raises(ValueError, match="solver must be one of"):
        DMD2Config.from_model_index({"dmd2_config": {"solver": "euler"}})
    with pytest.raises(ValueError, match="solver must be one of"):
        DMD2Config(solver="dpmpp")  # type: ignore[arg-type]


def test_forward_timesteps_idempotent_across_calls(pipeline):
    """Successive forward() calls must not cause scheduler state to drift."""
    parent = _DMD2_BASE[type(pipeline)]

    with patch.object(parent, "forward", _fake_parent_forward):
        pipeline.forward(_make_request())
        pipeline.forward(_make_request())

    assert pipeline.scheduler.timesteps.long().tolist() == _DMD2_TIMESTEPS
