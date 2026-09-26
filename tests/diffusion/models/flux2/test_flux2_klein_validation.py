# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.diffusion.models.flux2_klein.pipeline_flux2_klein import (
    Flux2KleinPipeline,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _make_pipeline():
    pipeline = object.__new__(Flux2KleinPipeline)
    pipeline.vae_scale_factor = 8
    pipeline.is_distilled = True
    pipeline._guidance_scale = 0.0
    return pipeline


def _check(pipeline, prompt, prompt_embeds=None):
    pipeline.check_inputs(
        prompt=prompt,
        height=512,
        width=512,
        prompt_embeds=prompt_embeds,
        num_inference_steps=4,
        guidance_scale=0.0,
    )


@pytest.mark.parametrize(
    "prompt",
    [
        "",
        "   ",
    ],
)
def test_rejects_empty_or_whitespace_prompt(prompt):
    pipe = _make_pipeline()
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        _check(pipe, prompt)


@pytest.mark.parametrize(
    "prompt",
    [
        ["valid prompt", ""],
        ["valid prompt", "   "],
        ["   "],
    ],
)
def test_rejects_list_with_empty_or_whitespace_element(prompt):
    pipe = _make_pipeline()
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        _check(pipe, prompt)


@pytest.mark.parametrize(
    "prompt",
    [
        "valid prompt",
        ["valid prompt", "another valid prompt"],
    ],
)
def test_accepts_valid_prompts(prompt):
    pipe = _make_pipeline()
    _check(pipe, prompt)


def test_rejects_none_without_embeds():
    pipe = _make_pipeline()
    with pytest.raises(ValueError):
        _check(pipe, None)


def test_accepts_none_with_prompt_embeds():
    pipe = _make_pipeline()
    _check(pipe, None, prompt_embeds=torch.randn(1, 256, 4096))
