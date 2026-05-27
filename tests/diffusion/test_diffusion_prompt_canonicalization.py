# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Regression tests for diffusion prompt canonicalization helpers."""

import pytest

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_flux_style_batched_negative_prompts_match_previous_hack_behavior():
    """Flux-style batched pipelines should see stable prompt and negative prompt lists."""
    req = OmniDiffusionRequest(
        prompts=[
            {"prompt": "a cat", "negative_prompt": None},
            {"prompt": "a dog", "negative_prompt": "blurry"},
        ],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
    )

    assert req.get_prompt_texts() == ["a cat", "a dog"]
    assert req.get_negative_prompt_texts() == ["", "blurry"]


def test_stable_audio_style_negative_prompts_disable_cfg_when_all_absent():
    """Stable-audio style pipelines should get ``None`` when all negatives are absent."""
    req = OmniDiffusionRequest(
        prompts=[
            {"prompt": "lofi beats", "negative_prompt": None},
            {"prompt": "ambient synth"},
        ],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
    )

    assert req.get_prompt_texts() == ["lofi beats", "ambient synth"]
    assert req.get_negative_prompt_texts() is None


def test_qwen_image_edit_style_first_prompt_keeps_preprocessed_metadata():
    """Single-prompt image edit pipelines should retain preprocessing metadata."""
    prompt_image = object()
    preprocessed_image = object()
    req = OmniDiffusionRequest(
        prompts=[
            {
                "prompt": "replace the sky",
                "negative_prompt": None,
                "additional_information": {
                    "prompt_image": prompt_image,
                    "preprocessed_image": preprocessed_image,
                    "calculated_height": 768,
                    "calculated_width": 512,
                },
            }
        ],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
    )

    first_prompt = req.canonical_prompts[0]
    assert first_prompt["prompt"] == "replace the sky"
    assert first_prompt["negative_prompt"] is None
    assert first_prompt["additional_information"]["prompt_image"] is prompt_image
    assert first_prompt["additional_information"]["preprocessed_image"] is preprocessed_image


def test_sd3_style_negative_prompts_keep_empty_string_fallback():
    """SD3-style pipelines can preserve their historical empty-string fallback."""
    req = OmniDiffusionRequest(
        prompts=[
            {"prompt": "a castle at dusk", "negative_prompt": None},
            {"prompt": "a forest trail"},
        ],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
    )

    prompts = req.get_prompt_texts()
    negative_prompts = req.get_negative_prompt_texts() or [""] * len(prompts)

    assert prompts == ["a castle at dusk", "a forest trail"]
    assert negative_prompts == ["", ""]


def test_longcat_style_prompt_embeds_are_read_from_canonical_prompts():
    """Batched prompt embeds should still be retrievable after canonicalization."""
    first_embed = object()
    second_embed = object()
    req = OmniDiffusionRequest(
        prompts=[
            {"prompt": "scene one", "prompt_embeds": first_embed},
            {"prompt": "scene two", "prompt_embeds": second_embed},
        ],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
    )

    prompt_embeds = [prompt.get("prompt_embeds") for prompt in req.canonical_prompts]

    assert req.get_prompt_texts() == ["scene one", "scene two"]
    assert prompt_embeds == [first_embed, second_embed]


def test_single_prompt_pipelines_accept_prompts_alias_and_keep_metadata():
    """Single-prompt pipelines should accept the `prompts` alias used by online payloads."""
    metadata = {"layers": 4, "use_en_prompt": True}
    req = OmniDiffusionRequest(
        prompts=[{"prompts": "redraw this scene", "additional_information": metadata}],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
    )

    first_prompt = req.canonical_prompts[0]

    assert first_prompt["prompt"] == "redraw this scene"
    assert first_prompt["negative_prompt"] is None
    assert first_prompt["additional_information"] is metadata
