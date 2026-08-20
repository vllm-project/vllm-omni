# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import pytest

from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID, OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _make_request() -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompt={"prompt": "a cup of coffee on a table"},
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
        request_id="request-test",
    )


def test_request_id_is_required():
    with pytest.raises(TypeError, match="request_id"):
        OmniDiffusionRequest(
            prompt={"prompt": "a cup of coffee on a table"},
            sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
        )


def test_request_ids_identity_list_is_removed():
    req = _make_request()

    assert req.request_id == "request-test"
    assert not hasattr(req, "request_ids")


def test_request_id_must_be_non_empty():
    with pytest.raises(ValueError, match="request_id must be a non-empty string"):
        OmniDiffusionRequest(
            prompt={"prompt": "a cup of coffee on a table"},
            sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
            request_id="",
        )


def test_dummy_run_request_is_identified_by_reserved_request_id():
    req = OmniDiffusionRequest(
        prompt={"prompt": "dummy run"},
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
        request_id=DUMMY_DIFFUSION_REQUEST_ID,
    )

    assert req.is_dummy_run()
    assert OmniDiffusionRequest.is_dummy_run_request_id(DUMMY_DIFFUSION_REQUEST_ID)
    assert OmniDiffusionRequest.is_dummy_run_request_id(f"{DUMMY_DIFFUSION_REQUEST_ID}/kv-profile-1")


def test_non_dummy_request_is_not_identified_as_dummy_run():
    req = _make_request()

    assert not req.is_dummy_run()
    assert not OmniDiffusionRequest.is_dummy_run_request_id(req.request_id)
    assert not OmniDiffusionRequest.is_dummy_run_request_id(None)


def test_tp_seed_same_across_ranks_and_varies_across_requests():
    random.seed(0)
    n_requests = 5
    seeds = [_make_request().sampling_params.seed for _ in range(n_requests)]

    # Seed must be auto-assigned (not None) so every TP rank can use it.
    assert all(s is not None for s in seeds)

    # Seeds must vary across requests (non-determinism preserved).
    assert len(set(seeds)) == n_requests, f"Expected {n_requests} unique seeds but got {len(set(seeds))}: {seeds}"


def _request_with_guidance(**sampling_kwargs) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompt={"prompt": "a cup of coffee on a table"},
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1, **sampling_kwargs),
        request_id="request-test",
    )


def test_omitted_guidance_scale_is_unset_and_resolved_to_one():
    # No guidance_scale passed: the caller did not set it, so it must NOT count as
    # "provided" (each pipeline applies its own default), and __post_init__ resolves the
    # None "unset" sentinel to a concrete float so no consumer ever observes None.
    req = _request_with_guidance()

    assert req.sampling_params.guidance_scale_provided is False
    assert req.sampling_params.guidance_scale == 1.0


def test_explicit_zero_guidance_scale_is_honored():
    # Regression (#4998): an explicit guidance_scale=0 (disable CFG) must be honored,
    # not mistaken for "unset" and overridden by the pipeline's model default. This is
    # the sentinel collision that the None default + ``is not None`` check fixes.
    req = _request_with_guidance(guidance_scale=0.0)

    assert req.sampling_params.guidance_scale_provided is True
    assert req.sampling_params.guidance_scale == 0.0


@pytest.mark.parametrize("value", [0.999, 1.0, 5.0])
def test_explicit_nonzero_guidance_scale_is_honored(value):
    # Any explicitly-set value (truthy or not) is honored as provided and preserved.
    req = _request_with_guidance(guidance_scale=value)

    assert req.sampling_params.guidance_scale_provided is True
    assert req.sampling_params.guidance_scale == value
