# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA round-trip tests for the H3 int8 all-to-all packet codec.

The speed argument for the wire is "0.5625x the bytes"; the price is quantisation error, and
the number that matters to a render is that the error stays inside the tolerance the arm was
accepted at. ``round_trip_error`` encodes and decodes locally (no collective), so this runs on
one GPU and cannot be confused with a distributed test.

These exercise the real Triton kernels, so they need CUDA. They are deliberately small (a few MB)
so they can run beside a loaded lane without disturbing it.
"""

from __future__ import annotations

import math

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3 import a2a_wire as h3_a2a_wire

if not torch.cuda.is_available():  # pragma: no cover - hardware gate
    pytest.skip("CUDA is not available", allow_module_level=True)

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]

# The acceptance bound. The module docstring quotes the measurement that justified the arm
# (relative error ~4-6e-3 against bf16 for UE5M3 packets); 2e-2 is a deliberately loose
# assertion that still fails loudly if the codec regresses by an order of magnitude.
REL_ERROR_BOUND = 2e-2


@pytest.mark.parametrize("rows", [64, 1024, 4096])
def test_round_trip_error_stays_inside_the_accepted_tolerance(rows):
    torch.manual_seed(0)
    x = torch.randn(rows, 128, dtype=torch.bfloat16, device="cuda")
    abs_err, rel_err = h3_a2a_wire.round_trip_error(x)
    assert math.isfinite(abs_err) and abs_err >= 0.0
    assert rel_err < REL_ERROR_BOUND, (
        f"int8 wire round-trip relative error {rel_err:.4f} exceeds {REL_ERROR_BOUND} "
        f"for {rows}x128 (max abs {abs_err:.4f})"
    )


def test_round_trip_error_is_non_zero_for_random_data():
    """A codec that is a no-op would pass the tolerance test trivially."""
    torch.manual_seed(1)
    x = torch.randn(256, 128, dtype=torch.bfloat16, device="cuda")
    _, rel_err = h3_a2a_wire.round_trip_error(x)
    assert rel_err > 0.0


def test_round_trip_preserves_shape_and_dtype():
    torch.manual_seed(2)
    x = torch.randn(512, 128, dtype=torch.bfloat16, device="cuda")
    packet = h3_a2a_wire.encode_pack(x)
    back = h3_a2a_wire.decode_pack(packet, x)
    assert back.shape == x.shape
    assert back.dtype == torch.bfloat16
    assert packet.dtype == torch.uint8


def test_encode_pack_emits_144_bytes_per_128_values():
    torch.manual_seed(3)
    rows = 300
    x = torch.randn(rows, 128, dtype=torch.bfloat16, device="cuda")
    packet = h3_a2a_wire.encode_pack(x)
    assert packet.numel() == rows * 144
    assert packet.numel() / (x.numel() * x.element_size()) == 0.5625


def test_round_trip_is_deterministic_for_a_fixed_input():
    """Same input, same output: the codec must not depend on pooled-buffer history."""
    torch.manual_seed(4)
    x = torch.randn(256, 128, dtype=torch.bfloat16, device="cuda")
    first = h3_a2a_wire.decode_pack(h3_a2a_wire.encode_pack(x), x).clone()
    second = h3_a2a_wire.decode_pack(h3_a2a_wire.encode_pack(x), x).clone()
    assert torch.equal(first, second)
