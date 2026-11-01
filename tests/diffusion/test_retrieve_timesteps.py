# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.models.utils import retrieve_timesteps

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeScheduler:
    def __init__(self):
        self.timesteps = None

    def set_timesteps(self, num_inference_steps=None, *, device=None, timesteps=None, sigmas=None, **kwargs):
        if timesteps is not None:
            self.timesteps = torch.tensor(timesteps, dtype=torch.float32)
        elif sigmas is not None:
            self.timesteps = torch.tensor(sigmas, dtype=torch.float32)
        else:
            self.timesteps = torch.linspace(1.0, 0.0, num_inference_steps)


class FakeSchedulerNoTimesteps(FakeScheduler):
    def set_timesteps(self, num_inference_steps=None, *, device=None, **kwargs):
        self.timesteps = torch.linspace(1.0, 0.0, num_inference_steps)


class FakeSchedulerNoSigmas(FakeScheduler):
    def set_timesteps(self, num_inference_steps=None, *, device=None, timesteps=None, **kwargs):
        if timesteps is not None:
            self.timesteps = torch.tensor(timesteps, dtype=torch.float32)
        else:
            self.timesteps = torch.linspace(1.0, 0.0, num_inference_steps)


def test_num_inference_steps():
    scheduler = FakeScheduler()
    timesteps, n = retrieve_timesteps(scheduler, num_inference_steps=20)
    assert n == 20
    assert torch.equal(timesteps, scheduler.timesteps)


def test_custom_timesteps():
    scheduler = FakeScheduler()
    timesteps, n = retrieve_timesteps(scheduler, timesteps=[999, 500, 0])
    assert n == 3
    assert torch.equal(timesteps, scheduler.timesteps)


def test_custom_sigmas():
    scheduler = FakeScheduler()
    timesteps, n = retrieve_timesteps(scheduler, sigmas=[1.0, 0.5, 0.0])
    assert n == 3
    assert torch.equal(timesteps, scheduler.timesteps)


def test_both_timesteps_and_sigmas_raises():
    with pytest.raises(ValueError, match="Only one of"):
        retrieve_timesteps(FakeScheduler(), timesteps=[1], sigmas=[0.5])


def test_unsupported_timesteps_raises():
    with pytest.raises(ValueError, match="does not support custom"):
        retrieve_timesteps(FakeSchedulerNoTimesteps(), timesteps=[999, 500])


def test_unsupported_sigmas_raises():
    with pytest.raises(ValueError, match="does not support custom"):
        retrieve_timesteps(FakeSchedulerNoSigmas(), sigmas=[1.0, 0.5])
