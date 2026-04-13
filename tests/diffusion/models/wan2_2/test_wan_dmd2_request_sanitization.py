# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import WanT2VDMD2Pipeline
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_i2v import WanI2VDMD2Pipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest, OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline(cls):
    """Instantiate a DMD2 pipeline without loading any model weights."""
    import torch
    pipeline = object.__new__(cls)
    torch.nn.Module.__init__(pipeline)
    return pipeline


def _make_request(prompts=None, **sp_kwargs) -> OmniDiffusionRequest:
    """Build a minimal OmniDiffusionRequest with given sampling params."""
    sp = OmniDiffusionSamplingParams(**sp_kwargs)
    return OmniDiffusionRequest(
        prompts=prompts or [{"prompt": "a cat dancing"}],
        sampling_params=sp,
    )


@pytest.fixture(params=[WanT2VDMD2Pipeline, WanI2VDMD2Pipeline], ids=["t2v", "i2v"])
def pipeline(request):
    return _make_pipeline(request.param)


# ---------------------------------------------------------------------------
# guidance_scale
# ---------------------------------------------------------------------------

def test_guidance_scale_forced_to_one(pipeline):
    req = _make_request(guidance_scale=5.0, guidance_scale_provided=True)
    pipeline._verify_dmd2_request(req)
    assert req.sampling_params.guidance_scale == 1.0
    assert req.sampling_params.guidance_scale_provided is False


def test_guidance_scale_already_correct(pipeline):
    req = _make_request(guidance_scale=1.0, guidance_scale_provided=False)
    pipeline._verify_dmd2_request(req)
    assert req.sampling_params.guidance_scale == 1.0


def test_guidance_scale_provided_flag_cleared(pipeline):
    """guidance_scale_provided=True must be cleared even if scale is already 1.0."""
    req = _make_request(guidance_scale=1.0, guidance_scale_provided=True)
    pipeline._verify_dmd2_request(req)
    assert req.sampling_params.guidance_scale_provided is False

def test_guidance_scale_2_cleared(pipeline):
    req = _make_request(guidance_scale_2=3.0)
    pipeline._verify_dmd2_request(req)
    assert req.sampling_params.guidance_scale_2 is None


def test_guidance_scale_2_unset_unchanged(pipeline):
    req = _make_request()
    pipeline._verify_dmd2_request(req)
    assert req.sampling_params.guidance_scale_2 is None


# ---------------------------------------------------------------------------
# CFG flags
# ---------------------------------------------------------------------------

def test_true_cfg_scale_cleared(pipeline):
    req = _make_request(true_cfg_scale=2.0)
    pipeline._verify_dmd2_request(req)
    assert req.sampling_params.true_cfg_scale is None


def test_do_classifier_free_guidance_forced_false(pipeline):
    req = _make_request(do_classifier_free_guidance=True)
    pipeline._verify_dmd2_request(req)
    assert req.sampling_params.do_classifier_free_guidance is False


def test_is_cfg_negative_forced_false(pipeline):
    req = _make_request(is_cfg_negative=True)
    pipeline._verify_dmd2_request(req)
    assert req.sampling_params.is_cfg_negative is False


# ---------------------------------------------------------------------------
# negative_prompt in prompt dict
# ---------------------------------------------------------------------------

def test_negative_prompt_stripped_from_prompt_dict(pipeline):
    req = _make_request(prompts=[{"prompt": "a cat", "negative_prompt": "blurry"}])
    pipeline._verify_dmd2_request(req)
    assert "negative_prompt" not in req.prompts[0]
    assert req.prompts[0]["prompt"] == "a cat"


def test_no_negative_prompt_unchanged(pipeline):
    req = _make_request(prompts=[{"prompt": "a cat"}])
    pipeline._verify_dmd2_request(req)
    assert req.prompts[0] == {"prompt": "a cat"}


def test_string_prompt_not_mutated(pipeline):
    """String prompts (not dicts) must pass through unchanged."""
    req = _make_request(prompts=["a cat dancing"])
    pipeline._verify_dmd2_request(req)
    assert req.prompts == ["a cat dancing"]


def test_multiple_prompts_all_sanitized(pipeline):
    req = _make_request(prompts=[
        {"prompt": "a cat", "negative_prompt": "blurry"},
        {"prompt": "a dog", "negative_prompt": "ugly"},
    ])
    pipeline._verify_dmd2_request(req)
    for p in req.prompts:
        assert "negative_prompt" not in p


# ---------------------------------------------------------------------------
# Clean request — nothing changes
# ---------------------------------------------------------------------------

def test_clean_request_no_changes(pipeline):
    req = _make_request(
        guidance_scale=1.0,
        guidance_scale_provided=False,
        do_classifier_free_guidance=False,
        is_cfg_negative=False,
    )
    pipeline._verify_dmd2_request(req)
    assert req.sampling_params.guidance_scale == 1.0
    assert req.sampling_params.guidance_scale_provided is False
    assert req.sampling_params.guidance_scale_2 is None
    assert req.sampling_params.true_cfg_scale is None
    assert req.sampling_params.do_classifier_free_guidance is False
    assert req.sampling_params.is_cfg_negative is False
