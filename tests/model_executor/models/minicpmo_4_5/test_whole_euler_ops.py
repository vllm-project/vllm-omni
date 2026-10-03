# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Fused Euler/CFG step and estimator-input staging match the eager ops."""

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.cuda_graph_wrapper import _euler_step
from vllm_omni.model_executor.models.minicpmo_4_5.whole_euler_ops import euler_cfg_step, stage_estimator_input

pytestmark = [pytest.mark.core_model]

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", _DEVICES)
def test_stage_estimator_input_is_the_transposed_cat(device: str) -> None:
    torch.manual_seed(0)
    batch, frames = 3, 50
    x = torch.randn(batch, 80, frames, device=device)
    mu = torch.randn(2 * batch, 80, frames, device=device)
    speakers = torch.randn(2 * batch, 80, device=device)
    cond = torch.randn(2 * batch, 80, frames, device=device)
    expected = torch.cat(
        (torch.cat((x, x), dim=0), mu, speakers.unsqueeze(-1).expand(-1, -1, frames), cond), dim=1
    ).transpose(1, 2)
    out = torch.empty((2 * batch, frames, 320), device=device)
    stage_estimator_input(out, x, mu, speakers, cond)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("device", _DEVICES)
def test_euler_cfg_step_is_bitwise_the_eager_step(device: str) -> None:
    torch.manual_seed(1)
    batch, frames = 4, 150
    x = torch.randn(batch, 80, frames, device=device)
    estimate = torch.randn(2 * batch, frames, 80, device=device).transpose(1, 2)
    dt = float(torch.tensor(0.0123456789))
    expected = _euler_step(x, estimate, dt, 0.7, batch)
    staged = torch.full((2 * batch, frames, 320), float("nan"), device=device)
    out = torch.empty_like(x)
    euler_cfg_step(out, x, estimate, dt, 0.7, staged)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    torch.testing.assert_close(staged[:batch, :, :80], expected.transpose(1, 2), rtol=0, atol=0)
    torch.testing.assert_close(staged[batch:, :, :80], expected.transpose(1, 2), rtol=0, atol=0)
    assert torch.isnan(staged[:, :, 80:]).all()
    inplace = x.clone()
    euler_cfg_step(inplace, inplace, estimate, dt, 0.7)
    torch.testing.assert_close(inplace, expected, rtol=0, atol=0)
