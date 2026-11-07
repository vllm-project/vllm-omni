# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The fused residual+LayerNorm path must stay opt-in, and stay within two ulp."""

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.layers import fused_residual_layer_norm as fused

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _ulp_gap(got: torch.Tensor, reference: torch.Tensor) -> float:
    """Largest bf16 grid-step distance between two tensors.

    Non-finite values are asserted away rather than folded to zero: a NaN produced
    by the kernel must fail this test, not pass it.
    """
    assert torch.isfinite(got).all(), "kernel produced non-finite values"
    assert torch.isfinite(reference).all()
    difference = (got.float() - reference.float()).abs()
    exponent = torch.frexp(reference.float().abs()).exponent
    ratio = difference / torch.pow(2.0, (exponent - 8).float())
    assert torch.isfinite(ratio).all()
    return ratio.max().item()


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
@torch.no_grad()
def test_kernel_matches_reference_within_two_ulp(monkeypatch):
    monkeypatch.setenv(fused.FUSED_RESIDUAL_NORM_ENV, "1")
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
    assert _ulp_gap(got_output, reference_output) <= 2.0
    difference = (got_output.float() - reference_output.float()).abs()
    exponent = torch.frexp(reference_output.float().abs()).exponent
    ratio = difference / torch.pow(2.0, (exponent - 8).float())
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
    assert _ulp_gap(got, reference) <= 2.0
    difference = (got.float() - reference.float()).abs()
    exponent = torch.frexp(reference.float().abs()).exponent
    ratio = difference / torch.pow(2.0, (exponent - 8).float())
    assert (ratio > 1.0).float().mean().item() <= 5e-6


@hardware_test(res={"musa": "S5000"}, num_cards=1)
@torch.no_grad()
def test_eligible_accepts_a_qualified_input():
    """The negative-only CPU case cannot show which predicate fired.

    Here the baseline qualifies, so each deviation below isolates exactly one check.
    """
    device, rows, cols = "musa:0", 64, 4096
    torch.manual_seed(1)
    hidden = torch.randn(1, rows, cols, dtype=torch.bfloat16, device=device)
    update = torch.randn_like(hidden)
    gate = torch.randn_like(hidden)
    scale = 1 + torch.rand(1, 1, cols, dtype=torch.bfloat16, device=device)
    assert fused.eligible(hidden, update, gate, scale)

    assert not fused.eligible(hidden.float(), update, gate, scale)
    assert not fused.eligible(hidden[0], update[0], gate, scale)
    assert not fused.eligible(hidden, update, torch.zeros(3, dtype=torch.bfloat16, device=device), scale)
    padded = torch.zeros(1, rows, cols * 2, dtype=torch.bfloat16, device=device)[:, :, :cols]
    assert not fused.eligible(hidden, padded, gate, scale)
    with torch.set_grad_enabled(True):
        assert not fused.eligible(hidden, update, gate, scale)


@hardware_test(res={"musa": "S5000"}, num_cards=1)
@pytest.mark.parametrize("broadcast", ["per-channel", "per-token"])
@torch.no_grad()
def test_kernel_accepts_both_scale_broadcasts(monkeypatch, broadcast):
    """The kernel reads ``scale`` flat, so both broadcast forms must agree with ATen."""
    monkeypatch.setenv(fused.FUSED_RESIDUAL_NORM_ENV, "1")
    device, rows, cols = "musa:0", 128, 4096
    torch.manual_seed(2)
    hidden, update, gate = (torch.randn(1, rows, cols, dtype=torch.bfloat16, device=device) for _ in range(3))
    if broadcast == "per-channel":
        scale = 1 + torch.rand(1, 1, cols, dtype=torch.bfloat16, device=device)
    else:
        scale = 1 + torch.rand(1, rows, cols, dtype=torch.bfloat16, device=device)

    assert fused.eligible(hidden, update, gate, scale)
    got_residual, got_output = fused.fused_residual_layer_norm(hidden, update, gate, scale, 1e-6)

    reference_residual = hidden + gate * update
    reference_output = torch.nn.functional.layer_norm(reference_residual, (cols,), eps=1e-6) * scale
    assert torch.equal(got_residual, reference_residual)
    assert _ulp_gap(got_output, reference_output) <= 2.0


@hardware_test(res={"musa": "S5000"}, num_cards=1)
@pytest.mark.parametrize("broadcast", ["per-channel", "per-token"])
@torch.no_grad()
def test_norm_scale_accepts_both_multiplier_broadcasts(monkeypatch, broadcast):
    monkeypatch.setenv(fused.FUSED_RESIDUAL_NORM_ENV, "1")
    device, rows, cols = "musa:0", 128, 4096
    torch.manual_seed(3)
    x = torch.randn(1, rows, cols, dtype=torch.bfloat16, device=device)
    shape = (1, 1, cols) if broadcast == "per-channel" else (1, rows, cols)
    multiplier = 1 + torch.rand(*shape, dtype=torch.bfloat16, device=device)

    assert fused.eligible_norm_scale(x, multiplier)
    got = fused.fused_layer_norm_scale(x, multiplier, 1e-6)
    assert got is not None
    reference = torch.nn.functional.layer_norm(x, (cols,), eps=1e-6) * multiplier
    assert _ulp_gap(got, reference) <= 2.0


@hardware_test(res={"musa": "S5000"}, num_cards=1)
@torch.no_grad()
def test_fused_norm_compiles_fullgraph(monkeypatch):
    """A regional compile must see the kernel as one opaque op, not a graph break."""
    monkeypatch.setenv(fused.FUSED_RESIDUAL_NORM_ENV, "1")
    device, cols = "musa:0", 4096
    torch.manual_seed(4)
    hidden = torch.randn(1, 128, cols, dtype=torch.bfloat16, device=device)
    update = torch.randn_like(hidden)
    gate = torch.randn_like(hidden)
    scale = 1 + torch.rand(1, 1, cols, dtype=torch.bfloat16, device=device)

    def block(hidden, update, gate, scale):
        return fused.fused_residual_layer_norm(hidden, update, gate, scale, 1e-6)

    compiled = torch.compile(block, backend="inductor", fullgraph=True)
    got = compiled(hidden, update, gate, scale)
    eager = block(hidden, update, gate, scale)
    assert got is not None and eager is not None
    assert torch.equal(got[0], eager[0])
    assert torch.equal(got[1], eager[1])
