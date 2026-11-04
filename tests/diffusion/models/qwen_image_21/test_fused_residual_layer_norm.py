# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The fused residual+LayerNorm path must stay opt-in, and stay within two ulp."""

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.layers import fused_residual_layer_norm as fused

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.cpu
def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv(fused.FUSED_RESIDUAL_NORM_ENV, raising=False)
    assert fused.fused_residual_norm_enabled() is False


@pytest.mark.cpu
@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " On "])
def test_gate_opens_on_truthy_values(monkeypatch, value):
    monkeypatch.setenv(fused.FUSED_RESIDUAL_NORM_ENV, value)
    assert fused.fused_residual_norm_enabled() is True


@pytest.mark.cpu
@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "2"])
def test_gate_stays_closed_on_other_values(monkeypatch, value):
    monkeypatch.setenv(fused.FUSED_RESIDUAL_NORM_ENV, value)
    assert fused.fused_residual_norm_enabled() is False


@pytest.mark.cpu
def test_closed_gate_returns_none(monkeypatch):
    monkeypatch.delenv(fused.FUSED_RESIDUAL_NORM_ENV, raising=False)
    hidden = torch.zeros(1, 8, 16, dtype=torch.bfloat16)
    update = torch.zeros_like(hidden)
    gate = torch.zeros(1, 8, 16, dtype=torch.bfloat16)
    scale = torch.zeros(1, 8, 16, dtype=torch.bfloat16)
    assert fused.fused_residual_layer_norm(hidden, update, gate, scale, 1e-6) is None


@pytest.mark.cpu
def test_eligible_rejects_unqualified_inputs():
    hidden = torch.zeros(1, 8, 16, dtype=torch.bfloat16)
    fine = torch.zeros_like(hidden)
    # CPU device, float32 dtype, wrong rank, and a gate that cannot broadcast.
    assert not fused.eligible(hidden, fine, fine, fine)
    assert not fused.eligible(hidden.float(), fine.float(), fine, fine)
    assert not fused.eligible(hidden[0], fine[0], fine, fine)
    bad_gate = torch.zeros(3, dtype=torch.bfloat16)
    assert not fused.eligible(hidden, fine, bad_gate, fine)


@hardware_test(res={"musa": "S5000"}, num_cards=1)
def test_kernel_matches_reference_within_two_ulp(monkeypatch):
    monkeypatch.setenv(fused.FUSED_RESIDUAL_NORM_ENV, "1")
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    rows, cols = 128, 4096
    device, dtype = "musa:0", torch.bfloat16
    hidden = torch.randn(1, rows, cols, dtype=dtype, device=device)
    update = torch.randn(1, rows, cols, dtype=dtype, device=device)
    gate = torch.randn(1, rows, cols, dtype=dtype, device=device)
    scale = 1 + torch.rand(1, 1, cols, dtype=dtype, device=device)

    got_residual, got_output = fused.fused_residual_layer_norm(hidden, update, gate, scale, 1e-6)

    reference_residual = hidden + gate * update
    reference_output = torch.nn.functional.layer_norm(reference_residual, (cols,), eps=1e-6) * scale

    # The residual sums are the same two bf16 roundings as the reference.
    assert torch.equal(got_residual, reference_residual)

    # The norm reduces in fp32 in a different order, so values sitting on a rounding
    # boundary can land one grid step away. Measured at T=5120, H=4096: at most two
    # ulp, on 0.0001% of the elements.
    difference = (got_output.float() - reference_output.float()).abs()
    exponent = torch.frexp(reference_output.float().abs()).exponent
    ratio = (difference / torch.pow(2.0, (exponent - 8).float())).nan_to_num(0.0)
    assert ratio.max().item() <= 2.0
    assert (ratio > 1.0).float().mean().item() <= 5e-6


@pytest.mark.cpu
def test_norm_scale_closed_gate_returns_none(monkeypatch):
    monkeypatch.delenv(fused.FUSED_RESIDUAL_NORM_ENV, raising=False)
    x = torch.zeros(1, 8, 16, dtype=torch.bfloat16)
    assert fused.fused_layer_norm_scale(x, x, 1e-6) is None


@hardware_test(res={"musa": "S5000"}, num_cards=1)
def test_norm_scale_matches_reference_within_two_ulp(monkeypatch):
    monkeypatch.setenv(fused.FUSED_RESIDUAL_NORM_ENV, "1")
    device, rows, cols = "musa:0", 5120, 4096
    x = torch.randn(1, rows, cols, dtype=torch.bfloat16, device=device)
    multiplier = 1 + torch.rand(1, 1, cols, dtype=torch.bfloat16, device=device)

    with torch.set_grad_enabled(False):
        got = fused.fused_layer_norm_scale(x, multiplier, 1e-6)

    assert got is not None
    reference = torch.nn.functional.layer_norm(x, (cols,), eps=1e-6) * multiplier
    difference = (got.float() - reference.float()).abs()
    exponent = torch.frexp(reference.float().abs()).exponent
    ratio = (difference / torch.pow(2.0, (exponent - 8).float())).nan_to_num(0.0)
    assert ratio.max().item() <= 2.0
    assert (ratio > 1.0).float().mean().item() <= 5e-6
