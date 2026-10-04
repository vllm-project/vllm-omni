# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.models.ming_flash_omni.pipeline_ming_imagegen import (
    get_ming_image_pre_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.request_scheduler import RequestScheduler
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _request(request_id: str, *, width: int = 16) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompt={"prompt": "", "extra": {"thinker_hidden_states": torch.ones(2, 3)}},
        sampling_params=OmniDiffusionSamplingParams(height=16, width=width, num_inference_steps=2, seed=1),
        request_id=request_id,
    )


def _scheduler(max_wait: float) -> RequestScheduler:
    scheduler = RequestScheduler()
    scheduler.initialize(
        SimpleNamespace(
            max_num_seqs=4,
            request_batch_max_wait_ms=max_wait,
            omni_kv_config=None,
            diffusion_kv_mode=DiffusionKVCacheMode.DENSE_LEGACY,
        )
    )
    return scheduler


def test_ming_compatible_requests_form_one_wave():
    pre = get_ming_image_pre_process_func(SimpleNamespace())
    scheduler = _scheduler(0.0)
    scheduler.add_request(pre(_request("A")))
    scheduler.add_request(pre(_request("B")))

    assert scheduler.schedule().scheduled_request_ids == ["A", "B"]


def test_ming_incompatible_requests_are_not_merged():
    pre = get_ming_image_pre_process_func(SimpleNamespace())
    scheduler = _scheduler(0.0)
    scheduler.add_request(pre(_request("A", width=16)))
    scheduler.add_request(pre(_request("B", width=32)))

    assert scheduler.schedule().scheduled_request_ids == ["A"]


@pytest.mark.parametrize("max_wait, should_wait", [(0.0, False), (20.0, True)])
def test_request_batch_admission_wait_respects_config(max_wait, should_wait):
    scheduler = _scheduler(max_wait)
    decision = scheduler.get_admission_wait_decision(now=0.0)
    assert decision.should_wait is should_wait
