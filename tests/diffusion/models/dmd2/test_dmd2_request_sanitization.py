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

# DMD2 subclass → immediate base pipeline whose __init__ loads model weights (mocked in tests).
_DMD2_BASE = {
    WanT2VDMD2Pipeline: Wan22Pipeline,
    WanI2VDMD2Pipeline: Wan22I2VPipeline,
    LTX2T2VDMD2Pipeline: LTX2Pipeline,
    LTX2I2VDMD2Pipeline: LTX2ImageToVideoPipeline,
}


def _make_pipeline(cls):
    """Run the DMD2 __init__ with the base pipeline mocked out (no model weights loaded)."""

    base = _DMD2_BASE[cls]
    od_config = MagicMock()
    od_config.model = "/nonexistent"

    def _mock_base_init(self, *a, **kw):
        self.od_config = od_config

    with patch.object(base, "__init__", _mock_base_init):
        pipeline = object.__new__(cls)
        torch.nn.Module.__init__(pipeline)
        cls.__init__(pipeline, od_config=od_config)
    return pipeline


def _make_request(prompts=None, **sp_kwargs) -> OmniDiffusionRequest:
    sp = OmniDiffusionSamplingParams(**sp_kwargs)
    return OmniDiffusionRequest(
        prompts=prompts or [{"prompt": "a cat dancing"}],
        sampling_params=sp,
    )


@pytest.fixture(
    params=list(_DMD2_BASE.keys()),
    ids=["wan_t2v", "wan_i2v", "ltx2_t2v", "ltx2_i2v"],
)
def pipeline(request):
    return _make_pipeline(request.param)


# ---------------------------------------------------------------------------
# num_inference_steps
# ---------------------------------------------------------------------------


def test_num_inference_steps_forced_to_dmd2_value(pipeline):
    req = _make_request(num_inference_steps=40)
    pipeline._sanitize_dmd2_request(req)
    assert req.sampling_params.num_inference_steps == pipeline.num_inference_steps


def test_num_inference_steps_already_correct(pipeline):
    req = _make_request(num_inference_steps=pipeline.num_inference_steps)
    pipeline._sanitize_dmd2_request(req)
    assert req.sampling_params.num_inference_steps == pipeline.num_inference_steps


# ---------------------------------------------------------------------------
# guidance_scale
# ---------------------------------------------------------------------------


def test_guidance_scale_forced_to_one(pipeline):
    req = _make_request(guidance_scale=5.0, guidance_scale_provided=True)
    pipeline._sanitize_dmd2_request(req)
    assert req.sampling_params.guidance_scale == pipeline.dmd2_guidance_scale
    assert req.sampling_params.guidance_scale_provided is False


def test_guidance_scale_already_correct(pipeline):
    req = _make_request(guidance_scale=pipeline.dmd2_guidance_scale, guidance_scale_provided=False)
    pipeline._sanitize_dmd2_request(req)
    assert req.sampling_params.guidance_scale == pipeline.dmd2_guidance_scale


def test_guidance_scale_provided_flag_cleared(pipeline):
    """guidance_scale_provided=True must be cleared even if scale is already dmd2_guidance_scale."""
    req = _make_request(guidance_scale=pipeline.dmd2_guidance_scale, guidance_scale_provided=True)
    pipeline._sanitize_dmd2_request(req)
    assert req.sampling_params.guidance_scale_provided is False


def test_guidance_scale_2_cleared(pipeline):
    req = _make_request(guidance_scale_2=3.0)
    pipeline._sanitize_dmd2_request(req)
    assert req.sampling_params.guidance_scale_2 is None


def test_guidance_scale_2_unset_unchanged(pipeline):
    req = _make_request()
    pipeline._sanitize_dmd2_request(req)
    assert req.sampling_params.guidance_scale_2 is None


def test_true_cfg_scale_cleared(pipeline):
    req = _make_request(true_cfg_scale=2.0)
    pipeline._sanitize_dmd2_request(req)
    assert req.sampling_params.true_cfg_scale is None


def test_do_classifier_free_guidance_forced_false(pipeline):
    req = _make_request(do_classifier_free_guidance=True)
    pipeline._sanitize_dmd2_request(req)
    assert req.sampling_params.do_classifier_free_guidance is False


def test_is_cfg_negative_forced_false(pipeline):
    req = _make_request(is_cfg_negative=True)
    pipeline._sanitize_dmd2_request(req)
    assert req.sampling_params.is_cfg_negative is False


def test_negative_prompt_stripped_from_prompt_dict(pipeline):
    req = _make_request(prompts=[{"prompt": "a cat", "negative_prompt": "blurry"}])
    pipeline._sanitize_dmd2_request(req)
    assert "negative_prompt" not in req.prompts[0]
    assert req.prompts[0]["prompt"] == "a cat"


def test_no_negative_prompt_unchanged(pipeline):
    req = _make_request(prompts=[{"prompt": "a cat"}])
    pipeline._sanitize_dmd2_request(req)
    assert req.prompts[0] == {"prompt": "a cat"}


def test_string_prompt_not_mutated(pipeline):
    """String prompts (not dicts) must pass through unchanged."""
    req = _make_request(prompts=["a cat dancing"])
    pipeline._sanitize_dmd2_request(req)
    assert req.prompts == ["a cat dancing"]


def test_multiple_prompts_all_sanitized(pipeline):
    req = _make_request(
        prompts=[
            {"prompt": "a cat", "negative_prompt": "blurry"},
            {"prompt": "a dog", "negative_prompt": "ugly"},
        ]
    )
    pipeline._sanitize_dmd2_request(req)
    for p in req.prompts:
        assert "negative_prompt" not in p


# ---------------------------------------------------------------------------
# Clean request — nothing changes
# ---------------------------------------------------------------------------


def test_clean_request_no_changes(pipeline):
    req = _make_request(
        guidance_scale=pipeline.dmd2_guidance_scale,
        guidance_scale_provided=False,
        do_classifier_free_guidance=False,
        is_cfg_negative=False,
    )
    pipeline._sanitize_dmd2_request(req)
    assert req.sampling_params.guidance_scale == pipeline.dmd2_guidance_scale
    assert req.sampling_params.guidance_scale_provided is False
    assert req.sampling_params.guidance_scale_2 is None
    assert req.sampling_params.true_cfg_scale is None
    assert req.sampling_params.do_classifier_free_guidance is False
    assert req.sampling_params.is_cfg_negative is False
