# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the MixFusion batch-key shape relaxation (CPU).

``enable_mixfusion`` packs requests of different resolutions into one denoise step
batch, so ``height``/``width``/``resolution`` must stop splitting the batch key.
Requests that carry per-prompt ``additional_information`` cannot share a key at all,
which both schedulers express by returning ``None``.
"""

from __future__ import annotations

from dataclasses import fields
from types import SimpleNamespace
from typing import Any

import pytest

from vllm_omni.diffusion.sched.base_scheduler import BaseScheduler, _apply_mixfusion_shape_relaxation
from vllm_omni.diffusion.sched.interface import RequestBatchSamplingParamsKey, StepBatchSamplingParamsKey
from vllm_omni.diffusion.sched.request_scheduler import RequestScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SHAPE: dict[str, Any] = {"height": 1024, "width": 768, "resolution": 1024}


class _Scheduler(BaseScheduler):
    """Minimal concrete scheduler; only the step-batch key builder is exercised."""

    def update_from_output(self, sched_output, output) -> set[str]:
        return set()


def _request(prompts: list[Any], **sampling_overrides: Any) -> SimpleNamespace:
    sampling: dict[str, Any] = {field.name: field.default for field in fields(StepBatchSamplingParamsKey)}
    sampling.update({field.name: field.default for field in fields(RequestBatchSamplingParamsKey)})
    sampling.update(SHAPE, extra_args={}, lora_request=None)
    sampling.update(sampling_overrides)
    return SimpleNamespace(
        sampling_params=SimpleNamespace(**sampling),
        prompts=prompts,
        batch_compatibility_key=None,
        use_step_execution=True,
    )


@pytest.mark.parametrize("scheduler_cls", [_Scheduler, RequestScheduler])
def test_shape_still_splits_the_key_without_mixfusion(scheduler_cls):
    request = _request(["a cat"])
    values = dict(SHAPE)

    assert _apply_mixfusion_shape_relaxation(request, values) is values
    key = scheduler_cls()._build_sampling_params_key(request)
    assert (key.height, key.width, key.resolution) == (1024, 768, 1024)


@pytest.mark.parametrize("scheduler_cls", [_Scheduler, RequestScheduler])
def test_mixfusion_drops_shape_from_the_key(scheduler_cls):
    request = _request(["a cat"], extra_args={"enable_mixfusion": True})
    values = dict(SHAPE)

    assert _apply_mixfusion_shape_relaxation(request, values) is values
    assert values == {"height": None, "width": None, "resolution": None}
    key = scheduler_cls()._build_sampling_params_key(request)
    assert (key.height, key.width, key.resolution) == (None, None, None)
    # Other shape fields still participate, so mixfusion only relaxes resolution.
    assert key.num_frames == 1


@pytest.mark.parametrize("scheduler_cls", [_Scheduler, RequestScheduler])
def test_mixfusion_leaves_per_prompt_requests_without_a_key(scheduler_cls):
    request = _request(
        [{"prompt": "a cat", "additional_information": {"image": "x"}}],
        extra_args={"enable_mixfusion": True},
    )

    assert _apply_mixfusion_shape_relaxation(request, dict(SHAPE)) is None
    assert scheduler_cls()._build_sampling_params_key(request) is None


def test_requests_without_mixfusion_keep_their_key_despite_additional_information():
    request = _request([{"prompt": "a cat", "additional_information": {"image": "x"}}])

    assert _Scheduler()._build_sampling_params_key(request) is not None


def test_empty_additional_information_still_batches():
    request = _request(
        [{"prompt": "a cat", "additional_information": {}}],
        extra_args={"enable_mixfusion": True},
    )

    key = _Scheduler()._build_sampling_params_key(request)
    assert key is not None
    assert key.height is None
