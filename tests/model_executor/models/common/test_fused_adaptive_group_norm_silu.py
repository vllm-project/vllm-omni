"""Unit tests for fused_adaptive_group_norm_silu operator.

Tests numeric correctness against PyTorch native implementation:
    F.silu(F.group_norm(x, num_groups, weight, bias, eps) * (1 + scale) + shift)
"""

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.model_executor.models.common.ops import fused_adaptive_group_norm_silu

# Skip tests if CUDA not available
pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernels"),
]


def _reference(x, weight, bias, scale, shift, num_groups, eps):
    """AdaGN reference, computed in fp32.

    As with fused_group_norm_silu, a same-dtype reference is unusable: the
    GroupNorm affine can cancel to near zero, and rounding its O(1) terms to
    bf16 wipes out the remaining significant digits.
    """
    B, C = x.shape[:2]
    normed = F.group_norm(x.float(), num_groups, weight.float(), bias.float(), eps)
    scale_b = scale.float().view(B, C, *([1] * (x.ndim - 2)))
    shift_b = shift.float().view(B, C, *([1] * (x.ndim - 2)))
    return F.silu(normed * (1 + scale_b) + shift_b)


def _tolerance(dtype):
    if dtype == torch.float32:
        return 1e-5, 1e-5
    if dtype == torch.float16:
        return 2e-3, 2e-3
    return 1e-2, 1e-2  # bfloat16


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("channels", [32, 64, 128])
@pytest.mark.parametrize("spatial_size", [(16, 16), (32, 32)])
@pytest.mark.parametrize("num_groups", [8, 32])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fused_adaptive_group_norm_silu_correctness(batch_size, channels, spatial_size, num_groups, dtype):
    """Test numeric correctness against PyTorch native ops."""
    if channels % num_groups != 0:
        pytest.skip(f"channels {channels} not divisible by num_groups {num_groups}")

    H, W = spatial_size
    eps = 1e-6
    torch.manual_seed(42)
    kw = dict(device="cuda", dtype=dtype)

    x = torch.randn(batch_size, channels, H, W, **kw)
    weight = torch.randn(channels, **kw)
    bias = torch.randn(channels, **kw)
    scale = torch.randn(batch_size, channels, **kw)
    shift = torch.randn(batch_size, channels, **kw)

    ref_out = _reference(x, weight, bias, scale, shift, num_groups, eps).to(dtype)
    fused_out = fused_adaptive_group_norm_silu(x, weight, bias, scale, shift, num_groups, eps)

    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(
        fused_out,
        ref_out,
        rtol=rtol,
        atol=atol,
        msg=f"Mismatch for batch={batch_size}, C={channels}, HW={spatial_size}, groups={num_groups}, dtype={dtype}",
    )


def test_broadcast_scale_shift_shapes():
    """scale/shift may arrive as (B, C) or already broadcast to (B, C, 1, 1)."""
    B, C, H, W = 2, 64, 16, 16
    kw = dict(device="cuda", dtype=torch.float32)
    x = torch.randn(B, C, H, W, **kw)
    weight = torch.randn(C, **kw)
    bias = torch.randn(C, **kw)
    scale = torch.randn(B, C, **kw)
    shift = torch.randn(B, C, **kw)

    flat = fused_adaptive_group_norm_silu(x, weight, bias, scale, shift, 32, 1e-6)
    expanded = fused_adaptive_group_norm_silu(x, weight, bias, scale.view(B, C, 1, 1), shift.view(B, C, 1, 1), 32, 1e-6)
    torch.testing.assert_close(flat, expanded, rtol=0, atol=0)


def test_output_dtype_matches_eager():
    """Output dtype must match what eager GroupNorm would produce."""
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        B, C = 2, 64
        kw = dict(device="cuda", dtype=dtype)
        x = torch.randn(B, C, 16, 16, **kw)
        weight = torch.randn(C, **kw)
        bias = torch.randn(C, **kw)
        scale = torch.randn(B, C, **kw)
        shift = torch.randn(B, C, **kw)

        eager_dtype = F.silu(F.group_norm(x, 32, weight, bias, 1e-6)).dtype
        out = fused_adaptive_group_norm_silu(x, weight, bias, scale, shift, 32, 1e-6)

        assert out.dtype == eager_dtype, (
            f"Output dtype {out.dtype} != eager dtype {eager_dtype} for input dtype {dtype}"
        )
        assert out.shape == x.shape


@pytest.mark.parametrize(
    "autocast_dtype, input_dtype",
    [
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
    ],
)
def test_autocast_matches_eager(autocast_dtype, input_dtype):
    """Inside autocast the fused op must stay a drop-in replacement.

    Also pins the fp32-activation corner so the inlined ``is_autocast_enabled
    -> fp32`` branch cannot silently downcast an already-fp32 activation.
    """
    B, C = 2, 64
    x = torch.randn(B, C, 16, 16, device="cuda", dtype=input_dtype)
    weight = torch.randn(C, device="cuda", dtype=torch.float32)
    bias = torch.randn(C, device="cuda", dtype=torch.float32)
    scale = torch.randn(B, C, device="cuda", dtype=torch.float32)
    shift = torch.randn(B, C, device="cuda", dtype=torch.float32)

    with torch.autocast("cuda", dtype=autocast_dtype):
        eager_out = F.silu(
            F.group_norm(x, 32, weight, bias, 1e-6) * (1 + scale.view(B, C, 1, 1)) + shift.view(B, C, 1, 1)
        )
        fused_out = fused_adaptive_group_norm_silu(x, weight, bias, scale, shift, 32, 1e-6)

    assert eager_out.dtype == torch.float32, "sanity check: eager group_norm is expected to return fp32 under autocast"
    assert fused_out.dtype == eager_out.dtype, (
        f"Output dtype {fused_out.dtype} != eager dtype {eager_out.dtype} "
        f"under autocast({autocast_dtype}) for input {input_dtype}"
    )
    torch.testing.assert_close(fused_out, eager_out, rtol=1e-5, atol=1e-5)


def test_invalid_channels_not_divisible():
    """Test that assertion fires when channels not divisible by num_groups."""
    B, C = 1, 33
    kw = dict(device="cuda", dtype=torch.float32)
    x = torch.randn(B, C, 16, 16, **kw)
    weight = torch.randn(C, **kw)
    bias = torch.randn(C, **kw)
    scale = torch.randn(B, C, **kw)
    shift = torch.randn(B, C, **kw)

    with pytest.raises(AssertionError, match="must be divisible by num_groups"):
        fused_adaptive_group_norm_silu(x, weight, bias, scale, shift, 32, 1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(2, 64, 3, 16, 16), (1, 32, 8, 8), (2, 64, 128)])
def test_non_2d_spatial_matches_eager(dtype, shape):
    """Any spatial rank must work; the wrapper collapses the spatial axes.

    GroupNorm reduces over the whole channel group and every spatial position,
    so flattening is exact rather than an approximation.
    """
    torch.manual_seed(0)
    B, C = shape[:2]
    kw = dict(device="cuda", dtype=dtype)
    x = torch.randn(*shape, **kw)
    weight = torch.randn(C, **kw)
    bias = torch.randn(C, **kw)
    scale = torch.randn(B, C, **kw)
    shift = torch.randn(B, C, **kw)

    fused_out = fused_adaptive_group_norm_silu(x, weight, bias, scale, shift, 32, 1e-6)
    ref_out = _reference(x, weight, bias, scale, shift, 32, 1e-6).to(dtype)

    assert fused_out.shape == x.shape
    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(fused_out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_scale_shift_from_chunk_are_not_contiguous(batch_size):
    """Regression: scale/shift arrive as halves of a torch.chunk.

    ResBlock builds them as ``torch.chunk(emb_out, 2, dim=1)`` where emb_out is
    (B, 2C, 1, 1). For B > 1 each half is a strided view, so the wrapper must
    ``reshape`` (which copies) instead of ``view`` (which raises).
    """
    C, H, W = 64, 16, 16
    kw = dict(device="cuda", dtype=torch.float32)
    x = torch.randn(batch_size, C, H, W, **kw)
    weight = torch.randn(C, **kw)
    bias = torch.randn(C, **kw)

    emb_out = torch.randn(batch_size, 2 * C, 1, 1, **kw)
    scale, shift = torch.chunk(emb_out, 2, dim=1)
    if batch_size > 1:
        assert not scale.is_contiguous(), "test premise: chunk half is strided"

    fused_out = fused_adaptive_group_norm_silu(x, weight, bias, scale, shift, 32, 1e-6)
    ref_out = F.silu(F.group_norm(x, 32, weight, bias, 1e-6) * (1.0 + scale) + shift)

    torch.testing.assert_close(fused_out, ref_out, rtol=1e-5, atol=1e-5)


def test_non_contiguous_input_matches_eager():
    """A channels-last / permuted activation must still be handled correctly."""
    B, C, H, W = 2, 64, 16, 16
    kw = dict(device="cuda", dtype=torch.float32)
    x = torch.randn(B, H, W, C, **kw).permute(0, 3, 1, 2)
    assert not x.is_contiguous(), "test premise: input is strided"
    weight = torch.randn(C, **kw)
    bias = torch.randn(C, **kw)
    scale = torch.randn(B, C, **kw)
    shift = torch.randn(B, C, **kw)

    fused_out = fused_adaptive_group_norm_silu(x, weight, bias, scale, shift, 32, 1e-6)
    ref_out = F.silu(F.group_norm(x, 32, weight, bias, 1e-6) * (1.0 + scale.view(B, C, 1, 1)) + shift.view(B, C, 1, 1))

    torch.testing.assert_close(fused_out, ref_out, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("spatial", [32, 64])
def test_dit_sized_activation(spatial):
    """Shapes the HunyuanImage3 DiT actually sees.

    The earlier kernel walked the spatial axis one position at a time with
    ``spatial_size`` as a ``tl.constexpr``; at 64x64 that is 4096 serial steps
    per pass. This test pins the realistic size so a regression to that layout
    shows up as a timeout rather than silently costing throughput.
    """
    B, C = 2, 256
    kw = dict(device="cuda", dtype=torch.bfloat16)
    x = torch.randn(B, C, spatial, spatial, **kw)
    weight = torch.randn(C, **kw)
    bias = torch.randn(C, **kw)
    scale = torch.randn(B, C, **kw)
    shift = torch.randn(B, C, **kw)

    fused_out = fused_adaptive_group_norm_silu(x, weight, bias, scale, shift, 32, 1e-5)
    ref_out = _reference(x, weight, bias, scale, shift, 32, 1e-5).to(torch.bfloat16)

    rtol, atol = _tolerance(torch.bfloat16)
    torch.testing.assert_close(fused_out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_large_offset_small_variance(dtype):
    """A large constant offset with a small variance must stay stable.

    Mirrors the fused_group_norm_silu case: ``x ~ 10000 ± 0.1`` has mean ~1e4
    and variance ~1e-2, so ``E[x^2] - E[x]^2`` cancels catastrophically in fp32
    (rounding the variance to ~8.0). The Welford reduction keeps it accurate.
    fp32 tolerance is 0.1 (not 1e-6) because ``x``'s fp32 representation noise
    (~0.001 ULP at 1e4) is amplified by ``rstd ~ 10`` to ~0.01-0.02 — the same
    floor PyTorch's own reference hits; the old naive implementation would be
    off by ~1.0 here. fp16/bf16 quantize the perturbation away, so for those
    dtypes the meaningful assertion is just "no NaN".
    """
    torch.manual_seed(0)
    B, C, H, W = 2, 64, 16, 16
    num_groups = 32
    eps = 1e-6
    device = torch.device("cuda")

    x = torch.full((B, C, H, W), 10000.0, device=device, dtype=dtype)
    x = x + torch.randn(B, C, H, W, device=device, dtype=dtype) * 0.1
    weight = torch.randn(C, device=device, dtype=dtype)
    bias = torch.randn(C, device=device, dtype=dtype)
    scale = torch.randn(B, C, device=device, dtype=dtype)
    shift = torch.randn(B, C, device=device, dtype=dtype)

    fused_out = fused_adaptive_group_norm_silu(x, weight, bias, scale, shift, num_groups, eps)
    ref_out = _reference(x, weight, bias, scale, shift, num_groups, eps).to(dtype)

    assert not torch.isnan(fused_out).any(), "fused output must not contain NaN"

    if dtype == torch.float32:
        # ~0.02 is the representation-noise floor at offset 1e4; old impl ~1.0.
        torch.testing.assert_close(fused_out, ref_out, rtol=0.05, atol=0.1)
    # fp16/bf16: perturbation is swallowed by the dtype — no meaningful close,
    # the no-NaN assertion above is the regression guard.


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
