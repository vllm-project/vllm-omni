# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import pytest

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _make_request() -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompts=[{"prompt": "a cup of coffee on a table"}],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
    )


def test_tp_seed_same_across_ranks_and_varies_across_requests():
    """Auto-assigned seeds should be stable per request and differ across requests."""
    random.seed(0)
    n_requests = 5
    seeds = [_make_request().sampling_params.seed for _ in range(n_requests)]

    # Seed must be auto-assigned (not None) so every TP rank can use it.
    assert all(s is not None for s in seeds)

    # Seeds must vary across requests (non-determinism preserved).
    assert len(set(seeds)) == n_requests, f"Expected {n_requests} unique seeds but got {len(set(seeds))}: {seeds}"


def test_canonical_prompts_normalize_none_values():
    """Prompt dicts should be normalized once in the shared request layer."""
    req = OmniDiffusionRequest(
        prompts=[
            {"prompt": None, "negative_prompt": None},
            {"prompt": "a cat", "negative_prompt": "blurry"},
        ],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1, guidance_scale=7.5),
    )

    assert req.canonical_prompts == [
        {"prompt": "", "negative_prompt": None},
        {"prompt": "a cat", "negative_prompt": "blurry"},
    ]
    assert req.get_prompt_texts() == ["", "a cat"]
    assert req.get_negative_prompt_texts() == ["", "blurry"]
    assert req.sampling_params.do_classifier_free_guidance is True


def test_canonical_prompts_preserve_extra_fields():
    """Canonicalization should keep embeds and auxiliary metadata intact."""
    embeds = object()
    metadata = {"preprocessed_image": object(), "calculated_height": 512}
    req = OmniDiffusionRequest(
        prompts=[
            {
                "prompt": "edit this image",
                "negative_prompt": None,
                "prompt_embeds": embeds,
                "additional_information": metadata,
            }
        ],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
    )

    prompt = req.canonical_prompts[0]
    assert prompt["prompt"] == "edit this image"
    assert prompt["negative_prompt"] is None
    assert prompt["prompt_embeds"] is embeds
    assert prompt["additional_information"] is metadata


def test_refresh_canonical_prompts_after_prompt_mutation():
    """Preprocessors that mutate prompts should be able to refresh canonical prompts."""
    metadata = {"preprocessed_image": object()}
    req = OmniDiffusionRequest(
        prompts=[{"prompt": "edit this image", "negative_prompt": None}],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
    )

    req.prompts[0]["additional_information"] = metadata

    assert "additional_information" not in req.canonical_prompts[0]
    req.refresh_canonical_prompts()
    assert req.canonical_prompts[0]["additional_information"] is metadata


def test_get_negative_prompt_texts_returns_none_when_absent():
    """All-None negative prompts should normalize to a disabled CFG payload."""
    req = OmniDiffusionRequest(
        prompts=["a mountain at sunset", {"prompt": "a city skyline", "negative_prompt": None}],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1, guidance_scale=7.5),
    )

    assert req.get_prompt_texts() == ["a mountain at sunset", "a city skyline"]
    assert req.get_negative_prompt_texts() is None
    assert req.sampling_params.do_classifier_free_guidance is False


def test_canonical_prompts_accept_prompts_alias():
    """Online payload drift using `prompts` should still produce a canonical prompt."""
    req = OmniDiffusionRequest(
        prompts=[{"prompts": "fallback field", "negative_prompt": None}],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
    )

    assert req.canonical_prompts[0]["prompt"] == "fallback field"


def test_canonical_prompts_reject_unsupported_prompt_types():
    """Unexpected prompt types should fail early with a clear error."""
    with pytest.raises(TypeError, match="Diffusion prompts must be strings or mapping-like prompt objects"):
        OmniDiffusionRequest(
            prompts=[123],  # type: ignore[list-item]
            sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
        )
