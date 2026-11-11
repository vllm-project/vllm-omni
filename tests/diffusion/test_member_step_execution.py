# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared batch mapping is exercised through the real Stepwise runner."""

import pytest
import torch

from tests.diffusion.test_diffusion_step_pipeline import (
    _make_batch_scheduler_output,
    _make_runner,
    _make_step_request,
    _noop_forward_context,
)
from vllm_omni.diffusion.worker import diffusion_model_runner as runner_module

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("short_output", [False, True])
def test_step_runner_maps_each_member_before_updating_states(monkeypatch, short_output):
    runner = _make_runner()
    updated = []

    class Pipeline:
        supports_step_execution = True

        def prepare_encode(self, state, **_kwargs):
            value = 2.0 if state.request_id == "a" else 7.0
            state.latents = torch.tensor([[value]])
            state.prompt_embeds = torch.tensor([[[value]]])
            state.timesteps = torch.tensor([10.0, 5.0])
            return state

        def denoise_step(self, batch, **_kwargs):
            assert batch.execution_batch.request_ids == ("a", "b")
            assert batch.execution_batch.kv_views == (None, None)
            result = batch.latents + batch.prompt_embeds.flatten().reshape(-1, 1)
            return result[:1] if short_output else result

        def step_scheduler(self, state, noise):
            updated.append((state.request_id, noise.item()))
            state.latents.copy_(noise)
            state.step_index += 1

        def post_decode(self, state):
            pytest.fail("The first of two steps must not decode")

    runner.pipeline = Pipeline()
    monkeypatch.setattr(runner_module, "set_forward_context", _noop_forward_context)
    requests = [_make_step_request(2), _make_step_request(2)]
    for request, identity in zip(requests, ("a", "b"), strict=True):
        request.request_id = identity
    if short_output:
        with pytest.raises(ValueError, match="requires 2 result rows"):
            runner.execute_stepwise(_make_batch_scheduler_output(requests))
        assert updated == [], "No member may update before the entire row mapping is validated"
    else:
        result = runner.execute_stepwise(_make_batch_scheduler_output(requests))
        assert updated == [("a", 4.0), ("b", 14.0)]
        assert not result.get_request_output("a").finished
        assert not result.get_request_output("b").finished
        assert runner.state_cache["a"].latents.item() == 4.0
        assert runner.state_cache["b"].latents.item() == 14.0
