# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import types

import pytest
import torch

from vllm_omni.diffusion.models.z_image.pipeline_z_image import ZImagePipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _pipeline_with_fake_encoder():
    pipeline = object.__new__(ZImagePipeline)
    object.__setattr__(pipeline, "_execution_device", torch.device("cpu"))
    object.__setattr__(pipeline, "_negative_prompt_cache", {})
    calls = []

    def encode(self, prompt, prompt_embeds=None, **_kwargs):
        calls.append((list(prompt), prompt_embeds))
        if prompt_embeds is not None:
            return prompt_embeds
        return [torch.tensor(len(calls)) for _ in prompt]

    object.__setattr__(pipeline, "_encode_prompt", types.MethodType(encode, pipeline))
    return pipeline, calls


def test_exact_negative_condition_is_reused_across_positive_prompts():
    pipeline, calls = _pipeline_with_fake_encoder()

    _, first_negative = pipeline.encode_prompt(
        ["first"],
        negative_prompt=["same negative"],
        do_classifier_free_guidance=True,
    )
    _, second_negative = pipeline.encode_prompt(
        ["second"],
        negative_prompt=["same negative"],
        do_classifier_free_guidance=True,
    )

    assert [prompt for prompt, _ in calls] == [["first"], ["same negative"], ["second"]]
    assert second_negative is first_negative


def test_precomputed_negative_condition_bypasses_cache():
    pipeline, calls = _pipeline_with_fake_encoder()
    negative_input = [torch.tensor(7)]

    _, negative = pipeline.encode_prompt(
        ["first"],
        negative_prompt=["same negative"],
        negative_prompt_embeds=negative_input,
        do_classifier_free_guidance=True,
    )

    assert calls == [(["first"], None)]
    assert negative is negative_input
    assert pipeline._negative_prompt_cache == {}
