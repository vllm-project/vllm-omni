# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for Pi-family shared flow-matching math."""

import pytest
import torch

from vllm_omni.diffusion.models.pi.common import flow_matching
from vllm_omni.diffusion.models.pi.pi0 import modeling_pi0
from vllm_omni.diffusion.models.pi.pi05 import modeling_pi05

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_variants_export_the_common_timestep_embedding():
    assert modeling_pi0.create_sinusoidal_pos_embedding is flow_matching.create_sinusoidal_pos_embedding
    assert modeling_pi05.create_sinusoidal_pos_embedding is flow_matching.create_sinusoidal_pos_embedding


def test_sinusoidal_embedding_reference_values_and_dtype():
    embedding = flow_matching.create_sinusoidal_pos_embedding(torch.tensor([0.0]), dimension=4)

    assert embedding.dtype == torch.float64
    assert torch.allclose(embedding, torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float64))


@pytest.mark.parametrize(
    "time,dimension,error",
    [
        (torch.tensor([1.0]), 3, "must be divisible by 2"),
        (torch.tensor([[0.0, 0.5]]), 4, "must be 1-D"),
    ],
)
def test_sinusoidal_embedding_rejects_invalid_inputs(time, dimension, error):
    with pytest.raises(ValueError, match=error):
        flow_matching.create_sinusoidal_pos_embedding(time, dimension)


def test_euler_schedule_stops_before_zero_and_lands_on_zero():
    schedule = flow_matching.make_euler_schedule(4)

    assert schedule == ((1.0, -0.25), (0.75, -0.25), (0.5, -0.25), (0.25, -0.25))
    assert schedule[-1][0] + schedule[-1][1] == 0.0


def test_euler_step_follows_predicted_velocity():
    sample = torch.tensor([1.0, -1.0])
    velocity = torch.tensor([0.5, 2.0])

    assert torch.equal(flow_matching.euler_step(sample, velocity, -0.25), torch.tensor([0.875, -1.5]))


def test_zero_step_schedule_preserves_previous_error():
    with pytest.raises(ZeroDivisionError):
        flow_matching.make_euler_schedule(0)
