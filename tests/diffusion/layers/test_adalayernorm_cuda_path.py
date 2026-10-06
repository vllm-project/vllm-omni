# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Candidate-added tests for the fused Triton AdaLayerNorm CUDA path.

Complements (never replaces) the frozen suite: exercises the fused fast path
and the forward_native fallback branches of forward_cuda. Uses the same
tolerance scheme as the frozen suite; golden = forward_native real behavior.
"""

import pytest
import torch

from vllm_omni.diffusion.layers.adalayernorm import (
    AdaLayerNorm,
    _adaln_fused_forward,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cuda]

TOL_STRICT = {"bf16": (2e-2, 2e-2), "fp32": (1e-3, 1e-3)}
# Matches the frozen suite: the double-rounded golden chain deviates from a
# pure-fp32 reference more than the strict band, so the fp32-reference check
# uses the same loose band as the frozen tests.
TOL_LOOSE = {"bf16": (5e-2, 2e-2), "fp32": (5e-3, 5e-3)}
DTYPES = [torch.bfloat16, torch.float32]


def fp32_reference(x, scale, shift, eps):
    xf = x.float()
    xn = torch.nn.functional.layer_norm(xf, (x.shape[-1],), None, None, eps)
    return xn * (1 + scale.float().reshape(1, 1, -1)) + shift.float().reshape(1, 1, -1)


def make_module(hidden, affine, eps, device, dtype):
    m = AdaLayerNorm(hidden, elementwise_affine=affine, eps=eps)
    if affine:
        m = m.to(dtype)
    return m.to(device)


def assert_close(a, b, dtype, loose=False):
    atol, rtol = (TOL_LOOSE if loose else TOL_STRICT)["bf16" if dtype == torch.bfloat16 else "fp32"]
    torch.testing.assert_close(a.float(), b.float(), atol=atol, rtol=rtol)


def make_inputs(bs, seq, hidden, dtype, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(bs, seq, hidden, generator=g, device=device, dtype=dtype)
    scale = torch.randn(1, hidden, generator=g, device=device, dtype=dtype)
    shift = torch.randn(1, hidden, generator=g, device=device, dtype=dtype)
    return x, scale, shift


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("bs,seq,hidden", [(1, 512, 3072), (2, 128, 1536), (1, 3, 1000)])
@torch.no_grad()
def test_fused_fast_path_matches_native(dtype, affine, bs, seq, hidden):
    # hidden=1000 exercises BLOCK_C masking (next_pow2(1000)=1024 > 1000)
    device = "cuda"
    m = make_module(hidden, affine, 1e-6, device, dtype)
    x, scale, shift = make_inputs(bs, seq, hidden, dtype, device)
    fused = _adaln_fused_forward(m, x, scale, shift)
    assert fused is not None, "supported inputs must take the fused path so BLOCK_C masking is exercised"
    out = fused
    assert out.shape == x.shape and out.dtype == dtype and out.device == x.device
    assert_close(out, m.forward_native(x, scale, shift), dtype)
    assert_close(out, fp32_reference(x, scale, shift, 1e-6), dtype, loose=True)


@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_noncontiguous_x(dtype):
    # Non-contiguous x routes to the forward_native fallback (the fused kernel
    # requires contiguous x); result must still match native exactly.
    device = "cuda"
    hidden = 1536
    m = make_module(hidden, False, 1e-6, device, dtype)
    g = torch.Generator(device=device).manual_seed(3)
    x_big = torch.randn(1, 1024, hidden, generator=g, device=device, dtype=dtype)
    x = x_big[:, ::2, :]
    scale = torch.randn(1, hidden, generator=g, device=device, dtype=dtype)
    shift = torch.randn(1, hidden, generator=g, device=device, dtype=dtype)
    out = m.forward_cuda(x, scale, shift)
    assert_close(out, m.forward_native(x, scale, shift), dtype)
    assert_close(out, fp32_reference(x.contiguous(), scale, shift, 1e-6), dtype, loose=True)


def test_normalized_shape_mismatch_preserves_native_error():
    m = AdaLayerNorm(16, elementwise_affine=False).cuda()
    x = torch.randn(1, 2, 32, device="cuda")
    scale = torch.zeros(32, device="cuda")
    shift = torch.zeros_like(scale)
    assert _adaln_fused_forward(m, x, scale, shift) is None
    with pytest.raises(RuntimeError) as native_error:
        m.forward_native(x, scale, shift)
    with pytest.raises(RuntimeError) as cuda_error:
        m.forward_cuda(x, scale, shift)
    assert str(cuda_error.value) == str(native_error.value)


@pytest.mark.parametrize("grad_target", ["x", "scale", "shift", "weight", "bias"])
def test_grad_fallback_matches_native_backward(grad_target):
    m = make_module(32, grad_target in ("weight", "bias"), 1e-6, "cuda", torch.float32)
    m.requires_grad_(False)
    x, scale, shift = make_inputs(1, 2, 32, torch.float32, "cuda", seed=12)
    targets = {"x": x, "scale": scale, "shift": shift, "weight": m.layernorm.weight, "bias": m.layernorm.bias}
    target = targets[grad_target]
    target.requires_grad_(True)
    with torch.enable_grad():
        assert _adaln_fused_forward(m, x, scale, shift) is None
        native = m.forward_native(x, scale, shift)
        out = m.forward_cuda(x, scale, shift)
        assert out.requires_grad
        torch.testing.assert_close(out, native)
        upstream = torch.randn_like(out)
        native_grad = torch.autograd.grad(native, target, upstream)[0]
        cuda_grad = torch.autograd.grad(out, target, upstream)[0]
        torch.testing.assert_close(cuda_grad, native_grad)


def test_no_grad_keeps_fused_path_for_grad_requiring_tensors():
    m = make_module(32, True, 1e-6, "cuda", torch.float32)
    x, scale, shift = make_inputs(1, 2, 32, torch.float32, "cuda", seed=14)
    for tensor in (x, scale, shift):
        tensor.requires_grad_(True)
    with torch.no_grad():
        fused = _adaln_fused_forward(m, x, scale, shift)
        assert fused is not None
        assert not fused.requires_grad
        torch.testing.assert_close(fused, m.forward_native(x, scale, shift), atol=1e-3, rtol=1e-3)


def test_1d_scale_shift_fast_path():
    # (C,) scale/shift broadcast per-channel exactly like (1, C)
    device = "cuda"
    hidden = 3072
    m = make_module(hidden, False, 1e-6, device, torch.bfloat16)
    g = torch.Generator(device=device).manual_seed(10)
    x = torch.randn(1, 256, hidden, generator=g, device=device, dtype=torch.bfloat16)
    scale = torch.randn(hidden, generator=g, device=device, dtype=torch.bfloat16)
    shift = torch.randn(hidden, generator=g, device=device, dtype=torch.bfloat16)
    out = _adaln_fused_forward(m, x, scale, shift)
    assert out is not None, "(C,) scale/shift must take the fused path"
    assert_close(out, m.forward_native(x, scale, shift), torch.bfloat16)
    assert_close(out, fp32_reference(x, scale, shift, 1e-6), torch.bfloat16, loose=True)


@pytest.mark.parametrize("dtype", DTYPES)
def test_per_row_scale_falls_back_to_native(dtype):
    # (L, C) scale with B=1 is a valid per-row torch broadcast; it is not
    # per-channel, so forward_cuda must route it to forward_native.
    device = "cuda"
    hidden = 512
    m = make_module(hidden, False, 1e-6, device, dtype)
    g = torch.Generator(device=device).manual_seed(4)
    x = torch.randn(1, 64, hidden, generator=g, device=device, dtype=dtype)
    scale = torch.randn(64, hidden, generator=g, device=device, dtype=dtype)
    shift = torch.randn(64, hidden, generator=g, device=device, dtype=dtype)
    out = m.forward_cuda(x, scale, shift)
    ref = m.forward_native(x, scale, shift)
    assert out.dtype == ref.dtype
    assert_close(out, ref, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_mixed_dtype_falls_back_to_native(dtype):
    # scale/shift dtype != x.dtype: native type-promotes; fused path requires
    # matching dtypes, so this must go through forward_native.
    other = torch.float32 if dtype == torch.bfloat16 else torch.bfloat16
    device = "cuda"
    hidden = 256
    m = make_module(hidden, False, 1e-6, device, dtype)
    g = torch.Generator(device=device).manual_seed(5)
    x = torch.randn(1, 32, hidden, generator=g, device=device, dtype=dtype)
    scale = torch.randn(1, hidden, generator=g, device=device, dtype=other)
    shift = torch.randn(1, hidden, generator=g, device=device, dtype=other)
    out = m.forward_cuda(x, scale, shift)
    ref = m.forward_native(x, scale, shift)
    assert out.dtype == ref.dtype
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("dtype", DTYPES)
def test_zero_dim_scale_falls_back_to_native(dtype):
    # 0-d scalar modulation is a valid torch broadcast; not per-channel.
    device = "cuda"
    hidden = 256
    m = make_module(hidden, False, 1e-6, device, dtype)
    g = torch.Generator(device=device).manual_seed(6)
    x = torch.randn(1, 32, hidden, generator=g, device=device, dtype=dtype)
    scale = torch.tensor(0.5, device=device, dtype=dtype)
    shift = torch.tensor(-0.25, device=device, dtype=dtype)
    out = m.forward_cuda(x, scale, shift)
    assert_close(out, m.forward_native(x, scale, shift), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_per_sample_scale_raises_like_native(dtype):
    # Frozen behavior: B>1 with (B, C) scale/shift -> RuntimeError.
    device = "cuda"
    hidden = 3072
    m = make_module(hidden, False, 1e-6, device, dtype)
    g = torch.Generator(device=device).manual_seed(7)
    x = torch.randn(4, 64, hidden, generator=g, device=device, dtype=dtype)
    scale = torch.randn(4, hidden, generator=g, device=device, dtype=dtype)
    shift = torch.randn(4, hidden, generator=g, device=device, dtype=dtype)
    with pytest.raises(RuntimeError):
        m.forward_native(x, scale, shift)
    with pytest.raises(RuntimeError):
        m.forward_cuda(x, scale, shift)


@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_determinism_repeat_calls(dtype):
    device = "cuda"
    hidden = 3072
    m = make_module(hidden, False, 1e-6, device, dtype)
    x, scale, shift = make_inputs(1, 1024, hidden, dtype, device, seed=8)
    out1 = _adaln_fused_forward(m, x, scale, shift)
    out2 = _adaln_fused_forward(m, x, scale, shift)
    assert out1 is not None and out2 is not None, "determinism must exercise the fused path"
    assert torch.equal(out1, out2)


@pytest.mark.parametrize("eps", [1e-5, 1e-3])
def test_fused_eps_variants(eps):
    device = "cuda"
    hidden = 3072
    m = make_module(hidden, False, eps, device, torch.bfloat16)
    x, scale, shift = make_inputs(1, 256, hidden, torch.bfloat16, device, seed=9)
    out = _adaln_fused_forward(m, x, scale, shift)
    assert out is not None, "supported eps variants must take the fused path"
    assert_close(out, m.forward_native(x, scale, shift), torch.bfloat16)
    assert_close(out, fp32_reference(x, scale, shift, eps), torch.bfloat16, loose=True)
