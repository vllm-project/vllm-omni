# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3.pdd import MiniMaxH3PDDOutput, pdd_head_weights, pdd_sigmas

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("shift", [3.0, 12.0])
def test_pdd_schedule_fixes_training_endpoint(shift):
    sigmas = pdd_sigmas(shift)
    assert len(sigmas) == 9
    assert sigmas[0] == pytest.approx(0.999, abs=3e-8)
    assert sigmas[-1] == 0
    assert all(a > b for a, b in zip(sigmas, sigmas[1:]))
    normalized = torch.linspace(1, 0, 9, dtype=torch.float64)
    expected = (0.999 * shift * normalized / (1 + (shift - 1) * normalized)).float()
    torch.testing.assert_close(torch.tensor(sigmas), expected, rtol=0, atol=0)


@pytest.mark.parametrize("shift", [3.0, 12.0])
@pytest.mark.parametrize("step", range(8))
def test_pdd_projection_matches_explicit_integration_of_all_heads(shift, step):
    torch.manual_seed(43)
    layer = MiniMaxH3PDDOutput(7, 3, shift=shift)
    with torch.no_grad():
        layer.weight.normal_()
        layer.bias.normal_()
    x = torch.randn(5, 7)
    all_predictions = torch.nn.functional.linear(x, layer.weight, layer.bias).reshape(5, 32, 3)
    grid = torch.linspace(0.999, 0, 33, dtype=torch.float64)
    sigmas = 0.999 * shift * grid / (0.999 + (shift - 1) * grid)
    start, end = 4 * step, 4 * step + 4
    increments = sigmas[start:end] - sigmas[start + 1 : end + 1]
    expected_delta = (all_predictions[:, start:end].double() * increments[None, :, None]).sum(1)
    actual_delta = layer(x, step).double() * (sigmas[start] - sigmas[end])
    torch.testing.assert_close(actual_delta, expected_delta, rtol=2e-5, atol=3e-7)
    torch.testing.assert_close(pdd_head_weights(shift, step), (increments / increments.sum()).float())


def test_pdd_interleaved_requests_do_not_share_step_state():
    layer = MiniMaxH3PDDOutput(2, 1, shift=12)
    with torch.no_grad():
        layer.weight.copy_(torch.arange(64).reshape(32, 2))
        layer.bias.zero_()
    x = torch.ones(1, 2)
    first = layer(x, 0)
    assert not torch.equal(first, layer(x, 7))
    torch.testing.assert_close(first, layer(x, 0), rtol=0, atol=0)
    assert set(layer.state_dict()) == {"weight", "bias"}


def test_pdd_forward_reuses_coefficients_without_rebuilding_grid(monkeypatch):
    layer = MiniMaxH3PDDOutput(2, 1, shift=12)
    with torch.no_grad():
        layer.weight.fill_(1)
        layer.bias.zero_()
    x = torch.ones(1, 2)
    expected = [layer(x, step) for step in range(8)]

    def unexpected_grid(*args, **kwargs):
        pytest.fail("PDD forward must not rebuild or transfer its integration grid")

    monkeypatch.setattr(torch, "linspace", unexpected_grid)
    for step in range(8):
        torch.testing.assert_close(layer(x, step), expected[step], rtol=0, atol=0)
    assert set(layer.state_dict()) == {"weight", "bias"}


@pytest.mark.parametrize("step", [-1, 8, None, True, 1.5, "0"])
def test_pdd_rejects_missing_or_invalid_step(step):
    with pytest.raises(ValueError, match="pdd_step"):
        pdd_head_weights(12, step)
    layer = MiniMaxH3PDDOutput(2, 1, shift=12)
    with pytest.raises(ValueError, match="pdd_step"):
        layer(torch.ones(1, 2), step)
