"""Unit tests for fused_group_norm_silu operator.

Tests numeric correctness against PyTorch native implementation:
    F.silu(F.group_norm(x, num_groups, weight, bias, eps))
"""

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.model_executor.models.common.ops import fused_group_norm_silu

# Skip tests if CUDA not available
pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Triton kernels"),
]


def _tolerance(dtype):
    """Tolerance against an fp32 reference, rounded once to ``dtype``."""
    if dtype == torch.float32:
        return 1e-5, 1e-6
    if dtype == torch.float16:
        return 2e-3, 2e-3
    return 1e-2, 1e-2  # bfloat16


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("channels", [32, 64, 128])
@pytest.mark.parametrize("spatial_size", [(16, 16), (32, 32), (64, 64)])
@pytest.mark.parametrize("num_groups", [8, 16, 32])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fused_group_norm_silu_correctness(batch_size, channels, spatial_size, num_groups, dtype):
    """Test numeric correctness against PyTorch native ops."""
    # Skip invalid configurations
    if channels % num_groups != 0:
        pytest.skip(f"channels {channels} not divisible by num_groups {num_groups}")

    H, W = spatial_size
    device = torch.device("cuda")
    eps = 1e-6

    # Create inputs
    torch.manual_seed(42)
    x = torch.randn(batch_size, channels, H, W, device=device, dtype=dtype)
    weight = torch.randn(channels, device=device, dtype=dtype)
    bias = torch.randn(channels, device=device, dtype=dtype)

    # Reference implementation, computed in fp32.
    #
    # A same-dtype reference is NOT usable here: GroupNorm's affine step
    # ``(x - mean) * rstd * weight + bias`` adds two O(1) terms that can cancel
    # down to ~1e-3. Rounding those terms to bf16 (ULP ~8e-3 near 1.0) destroys
    # every remaining significant digit -- measured cases flip sign, e.g. a true
    # +0.006136 comes out as -0.000471. The low-precision "reference" is then
    # far less accurate than the kernel it is supposed to validate: the fused
    # output matches the correctly-rounded fp32 result for 99.997% of elements,
    # a bf16 reference for only 65.2%.
    ref_out = F.silu(F.group_norm(x.float(), num_groups, weight.float(), bias.float(), eps)).to(dtype)

    # Fused implementation
    fused_out = fused_group_norm_silu(x, weight, bias, num_groups, eps)

    # Check correctness
    # Tolerances are set from the measured worst case across this parameter
    # sweep, leaving roughly 2x headroom for each dtype.
    if dtype == torch.float32:
        rtol, atol = 1e-5, 1e-6
    elif dtype == torch.float16:
        rtol, atol = 2e-3, 2e-3
    else:  # bfloat16
        rtol, atol = 1e-2, 1e-2

    torch.testing.assert_close(
        fused_out,
        ref_out,
        rtol=rtol,
        atol=atol,
        msg=f"Mismatch for batch={batch_size}, C={channels}, HW={spatial_size}, groups={num_groups}, dtype={dtype}",
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fp32_accumulation_precision(dtype):
    """Test that fp32 accumulation matches PyTorch's behavior.

    GroupNorm uses fp32 for mean/variance computation regardless of input dtype.
    Our kernel should match this behavior.
    """
    batch_size, channels, H, W = 2, 64, 32, 32
    num_groups = 32
    eps = 1e-6
    device = torch.device("cuda")

    # Create inputs with values that expose precision differences
    torch.manual_seed(123)
    x = torch.randn(batch_size, channels, H, W, device=device, dtype=dtype) * 100
    weight = torch.ones(channels, device=device, dtype=dtype)
    bias = torch.zeros(channels, device=device, dtype=dtype)

    # Reference: fp32 accumulation, then a single round back to ``dtype``.
    # (Same discipline as the main correctness test -- a same-dtype reference
    # wouldn't isolate the accumulation path this test is named after.)
    ref_out = F.silu(F.group_norm(x.float(), num_groups, weight.float(), bias.float(), eps)).to(dtype)

    # Fused
    fused_out = fused_group_norm_silu(x, weight, bias, num_groups, eps)

    # Should match closely even with large values
    if dtype == torch.float32:
        rtol, atol = 1e-5, 1e-5
    else:
        rtol, atol = 1e-2, 1e-2

    torch.testing.assert_close(fused_out, ref_out, rtol=rtol, atol=atol)


def test_edge_case_zero_input():
    """Test with zero input."""
    x = torch.zeros(1, 32, 16, 16, device="cuda")
    weight = torch.ones(32, device="cuda")
    bias = torch.zeros(32, device="cuda")

    ref_out = F.silu(F.group_norm(x, 32, weight, bias, 1e-6))
    fused_out = fused_group_norm_silu(x, weight, bias, 32, 1e-6)

    torch.testing.assert_close(fused_out, ref_out, rtol=1e-5, atol=1e-6)


def test_edge_case_large_eps():
    """Test with large epsilon value."""
    x = torch.randn(1, 32, 16, 16, device="cuda")
    weight = torch.randn(32, device="cuda")
    bias = torch.randn(32, device="cuda")
    eps = 1e-2  # Larger than typical

    ref_out = F.silu(F.group_norm(x, 32, weight, bias, eps))
    fused_out = fused_group_norm_silu(x, weight, bias, 32, eps)

    torch.testing.assert_close(fused_out, ref_out, rtol=1e-4, atol=1e-5)


def test_output_dtype_matches_eager():
    """Output dtype must match what eager GroupNorm+SiLU would produce.

    Outside autocast that means the input dtype; inside autocast it means fp32,
    because group_norm is on autocast's fp32 cast policy. HunyuanImage3's VAE
    runs fp32 weights under fp16 autocast, so getting this wrong would silently
    downcast the post-norm activation.
    """
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        x = torch.randn(2, 64, 32, 32, device="cuda", dtype=dtype)
        weight = torch.randn(64, device="cuda", dtype=dtype)
        bias = torch.randn(64, device="cuda", dtype=dtype)

        eager_out = F.silu(F.group_norm(x, 32, weight, bias, 1e-6))
        out = fused_group_norm_silu(x, weight, bias, 32, 1e-6)

        assert out.dtype == eager_out.dtype, (
            f"Output dtype {out.dtype} != eager dtype {eager_out.dtype} for input dtype {dtype}"
        )
        assert out.shape == x.shape, f"Output shape {out.shape} != input shape {x.shape}"


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

    Mirrors the HunyuanImage3 VAE setup: fp32 module weights, activations
    arriving as fp16/bf16 from an upstream conv, all under autocast. Also pins
    the fp32-activation corner (some VAE stages) so the inlined
    ``is_autocast_enabled -> fp32`` branch cannot silently downcast it.
    """
    x = torch.randn(2, 64, 32, 32, device="cuda", dtype=input_dtype)
    weight = torch.randn(64, device="cuda", dtype=torch.float32)
    bias = torch.randn(64, device="cuda", dtype=torch.float32)

    with torch.autocast("cuda", dtype=autocast_dtype):
        eager_out = F.silu(F.group_norm(x, 32, weight, bias, 1e-6))
        fused_out = fused_group_norm_silu(x, weight, bias, 32, 1e-6)

    assert eager_out.dtype == torch.float32, "sanity check: eager group_norm is expected to return fp32 under autocast"
    assert fused_out.dtype == eager_out.dtype, (
        f"Output dtype {fused_out.dtype} != eager dtype {eager_out.dtype} "
        f"under autocast({autocast_dtype}) for input {input_dtype}"
    )
    torch.testing.assert_close(fused_out, eager_out, rtol=1e-5, atol=1e-5)


def test_invalid_channels_not_divisible():
    """Test that assertion fires when channels not divisible by num_groups."""
    x = torch.randn(1, 33, 16, 16, device="cuda")  # 33 not divisible by 32
    weight = torch.randn(33, device="cuda")
    bias = torch.randn(33, device="cuda")

    with pytest.raises(AssertionError, match="must be divisible by num_groups"):
        fused_group_norm_silu(x, weight, bias, num_groups=32, eps=1e-6)


def test_invalid_weight_shape():
    """Test that assertion fires with wrong weight shape."""
    x = torch.randn(1, 64, 16, 16, device="cuda")
    weight = torch.randn(32, device="cuda")  # Wrong: should be 64
    bias = torch.randn(64, device="cuda")

    with pytest.raises(AssertionError, match="doesn't match channels"):
        fused_group_norm_silu(x, weight, bias, num_groups=32, eps=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(2, 64, 3, 16, 16), (1, 32, 1, 8, 8), (2, 64, 128)])
def test_non_2d_spatial_matches_eager(dtype, shape):
    """Spatial ranks other than 2 must work: the 3D VAE feeds (N, C, T, H, W).

    GroupNorm reduces over the whole channel group and every spatial position,
    so flattening the spatial axes inside the wrapper has to be exact.
    """
    torch.manual_seed(0)
    C = shape[1]
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    weight = torch.randn(C, device="cuda", dtype=dtype)
    bias = torch.randn(C, device="cuda", dtype=dtype)

    fused_out = fused_group_norm_silu(x, weight, bias, num_groups=32, eps=1e-6)
    ref_out = F.silu(F.group_norm(x.float(), 32, weight.float(), bias.float(), 1e-6)).to(dtype)

    assert fused_out.shape == x.shape
    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(fused_out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_permuted_input_matches_eager(dtype):
    """A permuted (channel-stride-1) activation must still be correct.

    This is not a hypothetical layout: HunyuanImage3's ``UNetUp`` calls this op
    on the output of ``rearrange(x, "b (h w) c -> b c h w")``. An earlier
    version of the kernel indexed such inputs through their strides, which was
    correct but uncoalesced enough to run at 0.44x of eager, so the wrapper now
    materializes them first.
    """
    torch.manual_seed(0)
    B, C, H, W = 2, 64, 16, 16
    kw = dict(device="cuda", dtype=dtype)
    x = torch.randn(B, H * W, C, **kw).permute(0, 2, 1).reshape(B, C, H, W)
    assert not x.is_contiguous(), "test premise: input is strided"

    weight = torch.randn(C, **kw)
    bias = torch.randn(C, **kw)

    fused_out = fused_group_norm_silu(x, weight, bias, num_groups=32, eps=1e-6)
    ref_out = F.silu(F.group_norm(x.float(), 32, weight.float(), bias.float(), 1e-6)).to(dtype)

    rtol, atol = _tolerance(dtype)
    torch.testing.assert_close(fused_out, ref_out, rtol=rtol, atol=atol)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("channels", [64, 128])
def test_hunyuan_vae_config(batch_size, channels):
    """Test with HunyuanImage3 VAE actual configuration.

    HunyuanImage3 ResnetBlock uses:
    - num_groups=32
    - eps=1e-6
    - Typical spatial sizes: 16x16 to 128x128 (after tiling)
    """
    spatial_sizes = [(16, 16), (32, 32), (64, 64), (128, 128)]

    for H, W in spatial_sizes:
        x = torch.randn(batch_size, channels, H, W, device="cuda", dtype=torch.float32)
        weight = torch.randn(channels, device="cuda")
        bias = torch.randn(channels, device="cuda")

        ref_out = F.silu(F.group_norm(x, 32, weight, bias, 1e-6))
        fused_out = fused_group_norm_silu(x, weight, bias, 32, 1e-6)

        torch.testing.assert_close(
            fused_out,
            ref_out,
            rtol=1e-5,
            atol=1e-6,
            msg=f"HunyuanVAE config: batch={batch_size}, C={channels}, HW=({H},{W})",
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_large_offset_small_variance(dtype):
    """A large constant offset with a small variance must stay stable.

    ``x ~ 10000 ± 0.1`` has mean ~1e4 and variance ~1e-2, so the naive
    ``E[x^2] - E[x]^2`` subtracts two ~1e8 quantities and rounds the variance to
    ~8.0 (the ULP of 1e8) — catastrophic cancellation, producing garbage
    normalization. The Welford reduction keeps the variance accurate.

    Why fp32 tolerance is 0.1, not the usual 1e-6: at this offset, ``x`` is stored
    in fp32 with ULP ~0.001, so ``x - mean`` carries ~0.001 of representation
    noise, and ``rstd ~ 10`` amplifies it to ~0.01-0.02. Even PyTorch's own
    GroupNorm reference hits this floor (the old naive implementation would be
    off by ~1.0 here). fp16/bf16 quantize the 0.1 perturbation away entirely, so
    for those dtypes the meaningful assertion is just "no NaN".
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

    fused_out = fused_group_norm_silu(x, weight, bias, num_groups, eps)
    # fp32 reference = PyTorch's Welford path, the ground truth.
    ref_out = F.silu(F.group_norm(x.float(), num_groups, weight.float(), bias.float(), eps)).to(dtype)

    assert not torch.isnan(fused_out).any(), "fused output must not contain NaN"

    if dtype == torch.float32:
        # ~0.02 is the representation-noise floor at offset 1e4; old impl ~1.0.
        torch.testing.assert_close(fused_out, ref_out, rtol=0.05, atol=0.1)
    # fp16/bf16: perturbation is swallowed by the dtype — no meaningful close,
    # the no-NaN assertion above is the regression guard.


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
