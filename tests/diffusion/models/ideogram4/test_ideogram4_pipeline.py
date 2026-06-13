# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Ideogram4Pipeline."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_omni.diffusion.models.ideogram4.pipeline_ideogram4 import (
    DEFAULT_GUIDANCE_HI,
    DEFAULT_GUIDANCE_LO,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_POLISH_STEPS,
    Ideogram4Pipeline,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _make_pipeline(*, in_channels=128, patch_size=2, ae_scale_factor=16):
    pipe = object.__new__(Ideogram4Pipeline)
    torch.nn.Module.__init__(pipe)
    pipe.device = torch.device("cpu")
    pipe.patch_size = patch_size
    pipe.ae_scale_factor = ae_scale_factor
    pipe.vae_scale_factor = ae_scale_factor
    pipe.max_text_tokens = 2048
    pipe.scheduler = MagicMock()
    pipe.od_config = SimpleNamespace(parallel_config=SimpleNamespace(sequence_parallel_size=1))
    pipe.conditional_transformer = MagicMock()
    pipe.conditional_transformer.config = SimpleNamespace(in_channels=in_channels)
    pipe.unconditional_transformer = MagicMock()
    pipe.unconditional_transformer.config = SimpleNamespace(in_channels=in_channels)
    return pipe


@pytest.mark.parametrize("num_steps", [3, 8, 30, 48, 50, 100, 200])
def test_default_guidance_schedule_starts_hi_and_ends_lo(num_steps):
    polish = min(DEFAULT_POLISH_STEPS, num_steps)
    expected_hi = [DEFAULT_GUIDANCE_HI] * (num_steps - polish)
    expected_lo = [DEFAULT_GUIDANCE_LO] * polish
    schedule = _resolve_schedule(num_steps=num_steps)
    assert schedule == expected_hi + expected_lo


def test_default_schedule_falls_back_to_steps_when_smaller_than_polish():
    schedule = _resolve_schedule(num_steps=2)
    assert schedule == [DEFAULT_GUIDANCE_HI] * 2


def test_constant_guidance_scale_overrides_schedule():
    schedule = _resolve_schedule(num_steps=4, guidance_scale=4.5)
    assert schedule == [4.5, 4.5, 4.5, 4.5]


def test_explicit_schedule_is_truncated_when_too_long():
    schedule = _resolve_schedule(num_steps=3, guidance_schedule=(1.0, 2.0, 3.0, 4.0, 5.0))
    assert schedule == [1.0, 2.0, 3.0]


def test_explicit_schedule_is_extended_with_last_value_when_too_short():
    schedule = _resolve_schedule(num_steps=5, guidance_schedule=(1.0, 2.0, 3.0))
    assert schedule == [1.0, 2.0, 3.0, 3.0, 3.0]


def test_num_inference_steps_none_uses_default():
    schedule = _resolve_schedule(num_steps=None)
    assert len(schedule) == DEFAULT_NUM_INFERENCE_STEPS


def test_height_not_divisible_by_patch_size_raises():
    pipe = _make_pipeline(patch_size=2, ae_scale_factor=16)
    with pytest.raises(ValueError, match="divisible by patch_size"):
        pipe._build_inputs(["a photo of a cat"], height=1025, width=1024)


def test_predict_noise_runs_only_one_branch_and_slices_text_tokens():
    pipe = _make_pipeline()
    head_dim = 4
    num_heads = 3
    max_text_tokens = 2
    num_image_tokens = 5
    transformer = MagicMock()
    transformer.return_value = torch.arange(
        (max_text_tokens + num_image_tokens) * num_heads * head_dim,
        dtype=torch.float32,
    ).view(1, max_text_tokens + num_image_tokens, num_heads * head_dim)

    out = pipe.predict_noise(
        transformer=transformer,
        x=torch.zeros(1, max_text_tokens + num_image_tokens, head_dim),
        t=torch.zeros(1),
        llm_features=torch.zeros(1, max_text_tokens, 4),
        position_ids=torch.zeros(1, max_text_tokens + num_image_tokens, 3, dtype=torch.long),
        segment_ids=torch.zeros(1, max_text_tokens + num_image_tokens, dtype=torch.long),
        indicator=torch.zeros(1, max_text_tokens + num_image_tokens, dtype=torch.long),
        max_text_tokens=max_text_tokens,
    )

    assert transformer.call_count == 1
    assert out.shape == (1, num_image_tokens, num_heads * head_dim)


def test_predict_noise_negative_branch_does_not_slice():
    pipe = _make_pipeline()
    transformer = MagicMock()
    transformer.return_value = torch.arange(15, dtype=torch.float32).view(1, 5, 3)

    out = pipe.predict_noise(
        transformer=transformer,
        x=torch.zeros(1, 5, 3),
        t=torch.zeros(1),
        llm_features=torch.zeros(1, 5, 4),
        position_ids=torch.zeros(1, 5, 3, dtype=torch.long),
        segment_ids=torch.zeros(1, 5, dtype=torch.long),
        indicator=torch.zeros(1, 5, dtype=torch.long),
        max_text_tokens=0,
    )
    assert out.shape == (1, 5, 3)
    assert torch.equal(out, torch.arange(15, dtype=torch.float32).view(1, 5, 3))


def _resolve_schedule(*, num_steps, guidance_scale=None, guidance_schedule=None):
    num_steps = num_steps or DEFAULT_NUM_INFERENCE_STEPS
    if guidance_scale is not None:
        return [float(guidance_scale)] * num_steps
    if guidance_schedule is not None:
        sched = list(guidance_schedule)
        if len(sched) > num_steps:
            return sched[:num_steps]
        return sched + [sched[-1]] * (num_steps - len(sched))
    polish = min(DEFAULT_POLISH_STEPS, num_steps)
    return [DEFAULT_GUIDANCE_HI] * (num_steps - polish) + [DEFAULT_GUIDANCE_LO] * polish
