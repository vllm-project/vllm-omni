# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Correctness tests for OmniVoice Triton kernels.

Verifies that RMSNorm, SwiGLU, and fused-add+RMSNorm Triton kernels
produce results numerically equivalent to the PyTorch reference.
All computations follow the kernel's internal float32 promotion rules.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

DEVICE = torch.device("cuda:0")
EPS = 1e-6

try:
    from vllm_omni.model_executor.models.omnivoice.omnivoice_generator import (
        _TRITON_AVAILABLE,
        triton_fused_add_rms_norm,
        triton_rms_norm,
        triton_swiglu,
    )
except ImportError:
    _TRITON_AVAILABLE = False

triton_available = pytest.mark.skipif(not _TRITON_AVAILABLE, reason="Triton not available on this platform")


# ---------------------------------------------------------------------------
# PyTorch reference implementations (mirror the kernel's internal arithmetic)
# ---------------------------------------------------------------------------


def _ref_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # Kernel promotes x to float32, computes RMS, casts back to x.dtype, then mul weight
    x_f32 = x.float()
    ms = x_f32.pow(2).mean(-1, keepdim=True)
    normed = (x_f32 * torch.rsqrt(ms + eps)).to(x.dtype)
    return normed * weight.to(normed.dtype)


def _ref_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    # Kernel: gate → float32, silu, cast to up.dtype, mul up
    gate_f32 = gate.float()
    silu = gate_f32 * torch.sigmoid(gate_f32)
    return silu.to(up.dtype) * up


def _ref_fused_add_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    # Kernel: s_f32 = x+r in float32, s stored as x.dtype; y = norm(s_f32)*weight
    s_f32 = x.float() + residual.float()
    s = s_f32.to(x.dtype)
    ms = s_f32.pow(2).mean(-1, keepdim=True)
    normed = (s_f32 * torch.rsqrt(ms + eps)).to(x.dtype)
    y = normed * weight.to(normed.dtype)
    return y, s


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


@triton_available
@pytest.mark.parametrize("rows,cols", [(1, 64), (4, 512), (16, 1024), (32, 3072)])
def test_rms_norm_float32(rows, cols):
    torch.manual_seed(0)
    x = torch.randn(rows, cols, device=DEVICE)
    w = torch.ones(cols, device=DEVICE)

    ref = _ref_rms_norm(x, w, EPS)
    out = triton_rms_norm(x, w, EPS)

    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@triton_available
@pytest.mark.parametrize("shape", [(2, 16, 64), (4, 32, 512)])
def test_rms_norm_3d_input(shape):
    """Triton wrapper must handle 3-D [B, S, H] inputs by flattening and restoring shape."""
    torch.manual_seed(1)
    hidden = shape[-1]
    x = torch.randn(*shape, device=DEVICE)
    w = torch.ones(hidden, device=DEVICE)

    ref = _ref_rms_norm(x, w, EPS)
    out = triton_rms_norm(x, w, EPS)

    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    assert out.shape == x.shape


@triton_available
def test_rms_norm_no_nan_on_zero_input():
    """RMSNorm of the zero vector must not produce NaN (eps prevents divide-by-zero)."""
    x = torch.zeros(4, 256, device=DEVICE)
    w = torch.ones(256, device=DEVICE)

    out = triton_rms_norm(x, w, EPS)

    assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# SwiGLU
# ---------------------------------------------------------------------------


@triton_available
@pytest.mark.parametrize("rows,cols", [(1, 64), (4, 512), (16, 3072)])
def test_swiglu_float32(rows, cols):
    torch.manual_seed(2)
    gate = torch.randn(rows, cols, device=DEVICE)
    up = torch.randn(rows, cols, device=DEVICE)

    ref = _ref_swiglu(gate, up)
    out = triton_swiglu(gate, up)

    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


@triton_available
def test_swiglu_negative_gate():
    """silu(x) = x*sigmoid(x) is defined for negative x; result must not be NaN."""
    gate = torch.full((4, 64), -10.0, device=DEVICE)
    up = torch.ones(4, 64, device=DEVICE)

    ref = _ref_swiglu(gate, up)
    out = triton_swiglu(gate, up)

    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    assert not torch.isnan(out).any()


@triton_available
def test_swiglu_output_shape_preserved():
    gate = torch.randn(3, 8, 512, device=DEVICE)
    up = torch.randn(3, 8, 512, device=DEVICE)
    out = triton_swiglu(gate, up)
    assert out.shape == gate.shape


# ---------------------------------------------------------------------------
# Fused Add + RMSNorm
# ---------------------------------------------------------------------------


@triton_available
@pytest.mark.parametrize("rows,cols", [(1, 64), (4, 512), (16, 1024)])
def test_fused_add_rms_norm_float32(rows, cols):
    torch.manual_seed(3)
    x = torch.randn(rows, cols, device=DEVICE)
    residual = torch.randn(rows, cols, device=DEVICE)
    weight = torch.ones(cols, device=DEVICE)

    ref_y, ref_s = _ref_fused_add_rms_norm(x, residual, weight, EPS)
    out_y, out_s = triton_fused_add_rms_norm(x, residual, weight, EPS)

    torch.testing.assert_close(out_y, ref_y, atol=1e-5, rtol=1e-5)
    # Sum output s = x + r must be bit-exact (no numerics involved, just addition)
    torch.testing.assert_close(out_s, ref_s, atol=0, rtol=0)


@triton_available
def test_fused_add_rms_norm_sum_is_bitexact():
    """The residual sum s = x + r must be bit-exact between Triton and PyTorch."""
    torch.manual_seed(4)
    x = torch.randn(8, 512, device=DEVICE)
    r = torch.randn(8, 512, device=DEVICE)
    w = torch.ones(512, device=DEVICE)

    _, ref_s = _ref_fused_add_rms_norm(x, r, w, EPS)
    _, out_s = triton_fused_add_rms_norm(x, r, w, EPS)

    torch.testing.assert_close(out_s, ref_s, atol=0, rtol=0)


@triton_available
def test_fused_add_rms_norm_returns_two_tensors():
    """Wrapper must return exactly two tensors: (normed_output, sum)."""
    x = torch.randn(4, 256, device=DEVICE)
    r = torch.randn(4, 256, device=DEVICE)
    w = torch.ones(256, device=DEVICE)

    result = triton_fused_add_rms_norm(x, r, w, EPS)

    assert len(result) == 2
    y, s = result
    assert y.shape == x.shape
    assert s.shape == x.shape
