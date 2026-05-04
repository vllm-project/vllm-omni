# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for LayerNorm and RMSNorm custom ops in diffusion layers."""

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


# ── Import tests ──


def test_layernorm_import():
    """Verify LayerNorm can be imported from the norm module."""
    from vllm_omni.diffusion.layers.norm import LayerNorm  # noqa: F401


def test_rmsnorm_import():
    """Verify RMSNorm can be imported from the norm module."""
    from vllm_omni.diffusion.layers.norm import RMSNorm  # noqa: F401


# ── LayerNorm tests ──


def test_layernorm_forward_shape():
    """LayerNorm produces correct output shapes."""
    from vllm_omni.diffusion.layers.norm import LayerNorm

    dim = 64
    batch = 2
    seq_len = 4
    norm = LayerNorm(dim)

    x = torch.randn(batch, seq_len, dim)
    out = norm(x)

    assert out.shape == (batch, seq_len, dim)


def test_layernorm_forward_shape_2d():
    """LayerNorm works with 2D input tensors."""
    from vllm_omni.diffusion.layers.norm import LayerNorm

    dim = 64
    batch = 2
    norm = LayerNorm(dim)

    x = torch.randn(batch, dim)
    out = norm(x)

    assert out.shape == (batch, dim)


def test_layernorm_preserves_dtype_fp32():
    """LayerNorm preserves float32 dtype."""
    from vllm_omni.diffusion.layers.norm import LayerNorm

    dim = 64
    norm = LayerNorm(dim)

    x = torch.randn(2, 4, dim, dtype=torch.float32)
    out = norm(x)

    assert out.dtype == torch.float32


def test_layernorm_preserves_dtype_fp16():
    """LayerNorm preserves float16 dtype."""
    from vllm_omni.diffusion.layers.norm import LayerNorm

    dim = 64
    norm = LayerNorm(dim)

    x = torch.randn(2, 4, dim, dtype=torch.float16)
    out = norm(x)

    assert out.dtype == torch.float16


def test_layernorm_preserves_dtype_bf16():
    """LayerNorm preserves bfloat16 dtype."""
    from vllm_omni.diffusion.layers.norm import LayerNorm

    dim = 64
    norm = LayerNorm(dim)

    x = torch.randn(2, 4, dim, dtype=torch.bfloat16)
    out = norm(x)

    assert out.dtype == torch.bfloat16


def test_layernorm_without_elementwise_affine():
    """LayerNorm works without elementwise_affine (no learned parameters)."""
    from vllm_omni.diffusion.layers.norm import LayerNorm

    dim = 64
    norm = LayerNorm(dim, elementwise_affine=False)

    assert norm.weight is None
    assert norm.bias is None

    x = torch.randn(2, 4, dim)
    out = norm(x)

    assert out.shape == (2, 4, dim)


def test_layernorm_custom_eps():
    """LayerNorm accepts custom epsilon value."""
    from vllm_omni.diffusion.layers.norm import LayerNorm

    dim = 64
    eps = 1e-5
    norm = LayerNorm(dim, eps=eps)

    assert norm.eps == eps


def test_layernorm_has_learnable_parameters():
    """LayerNorm has learnable weight and bias by default."""
    from vllm_omni.diffusion.layers.norm import LayerNorm

    dim = 64
    norm = LayerNorm(dim)

    assert norm.weight is not None
    assert norm.bias is not None
    assert norm.weight.shape == (dim,)
    assert norm.bias.shape == (dim,)


def test_layernorm_matches_fp32_reference():
    """Verify LayerNorm produces identical output to FP32 nn.LayerNorm."""
    from vllm_omni.diffusion.layers.norm import LayerNorm

    dim = 64
    eps = 1e-6
    torch.manual_seed(42)

    ours = LayerNorm(dim, eps=eps)
    ref = torch.nn.LayerNorm(dim, eps=eps)

    # Copy weights
    ref.weight.data.copy_(ours.weight.data)
    ref.bias.data.copy_(ours.bias.data)

    x = torch.randn(2, 4, dim)

    out_ours = ours(x)
    out_ref = ref(x.float()).to(x.dtype)

    torch.testing.assert_close(out_ours, out_ref, atol=1e-5, rtol=1e-5)


def test_layernorm_matches_diffusers_fp32layernorm():
    """Verify LayerNorm produces identical output to diffusers FP32LayerNorm."""
    from diffusers.models.normalization import FP32LayerNorm

    from vllm_omni.diffusion.layers.norm import LayerNorm

    dim = 64
    eps = 1e-6
    torch.manual_seed(42)

    ours = LayerNorm(dim, eps=eps)
    ref = FP32LayerNorm(dim, eps=eps)

    # Copy weights
    ref.weight.data.copy_(ours.weight.data)
    ref.bias.data.copy_(ours.bias.data)

    # Test with fp16 input to verify FP32 computation
    x = torch.randn(2, 4, dim, dtype=torch.float16)

    out_ours = ours(x)
    out_ref = ref(x)

    torch.testing.assert_close(out_ours, out_ref, atol=1e-3, rtol=1e-3)


# ── RMSNorm tests ──


def test_rmsnorm_forward_shape():
    """RMSNorm produces correct output shapes."""
    from vllm_omni.diffusion.layers.norm import RMSNorm

    hidden_size = 64
    batch = 2
    seq_len = 4
    norm = RMSNorm(hidden_size)

    x = torch.randn(batch, seq_len, hidden_size)
    out = norm(x)

    assert out.shape == (batch, seq_len, hidden_size)


def test_rmsnorm_forward_shape_2d():
    """RMSNorm works with 2D input tensors."""
    from vllm_omni.diffusion.layers.norm import RMSNorm

    hidden_size = 64
    batch = 2
    norm = RMSNorm(hidden_size)

    x = torch.randn(batch, hidden_size)
    out = norm(x)

    assert out.shape == (batch, hidden_size)


def test_rmsnorm_preserves_dtype_fp32():
    """RMSNorm preserves float32 dtype."""
    from vllm_omni.diffusion.layers.norm import RMSNorm

    hidden_size = 64
    norm = RMSNorm(hidden_size)

    x = torch.randn(2, 4, hidden_size, dtype=torch.float32)
    out = norm(x)

    assert out.dtype == torch.float32


def test_rmsnorm_preserves_dtype_fp16():
    """RMSNorm preserves float16 dtype."""
    from vllm_omni.diffusion.layers.norm import RMSNorm

    hidden_size = 64
    norm = RMSNorm(hidden_size)

    x = torch.randn(2, 4, hidden_size, dtype=torch.float16)
    out = norm(x)

    assert out.dtype == torch.float16


def test_rmsnorm_preserves_dtype_bf16():
    """RMSNorm preserves bfloat16 dtype."""
    from vllm_omni.diffusion.layers.norm import RMSNorm

    hidden_size = 64
    norm = RMSNorm(hidden_size)

    x = torch.randn(2, 4, hidden_size, dtype=torch.bfloat16)
    out = norm(x)

    assert out.dtype == torch.bfloat16


def test_rmsnorm_custom_eps():
    """RMSNorm accepts custom epsilon value."""
    from vllm_omni.diffusion.layers.norm import RMSNorm

    hidden_size = 64
    eps = 1e-5
    norm = RMSNorm(hidden_size, eps=eps)

    assert norm.variance_epsilon == eps


def test_rmsnorm_has_weight_parameter():
    """RMSNorm has learnable weight parameter initialized to ones."""
    from vllm_omni.diffusion.layers.norm import RMSNorm

    hidden_size = 64
    norm = RMSNorm(hidden_size)

    assert norm.weight is not None
    assert norm.weight.shape == (hidden_size,)
    torch.testing.assert_close(norm.weight, torch.ones(hidden_size))


def test_rmsnorm_numerical_correctness():
    """Verify RMSNorm produces numerically correct output."""
    from vllm_omni.diffusion.layers.norm import RMSNorm

    hidden_size = 64
    eps = 1e-6
    torch.manual_seed(42)

    norm = RMSNorm(hidden_size, eps=eps)
    x = torch.randn(2, 4, hidden_size)

    # Compute expected output manually
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(-1, keepdim=True)
    expected = x_fp32 * torch.rsqrt(variance + eps)
    expected = norm.weight.to(torch.float32) * expected
    expected = expected.to(x.dtype)

    out = norm(x)

    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_rmsnorm_matches_reference_implementation():
    """Verify RMSNorm matches a reference implementation."""
    from vllm_omni.diffusion.layers.norm import RMSNorm

    def reference_rmsnorm(x, weight, eps):
        """Reference RMSNorm implementation."""
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        out = x * torch.rsqrt(variance + eps)
        out = weight.to(torch.float32) * out
        return out.to(input_dtype)

    hidden_size = 128
    eps = 1e-6
    torch.manual_seed(123)

    norm = RMSNorm(hidden_size, eps=eps)

    # Test with various dtypes
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        x = torch.randn(4, 8, hidden_size, dtype=dtype)
        expected = reference_rmsnorm(x, norm.weight, eps)
        out = norm(x)
        torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-3)


# ── CustomOp dispatch tests ──


def test_layernorm_inherits_from_customop():
    """LayerNorm inherits from CustomOp for platform dispatch."""
    from vllm_omni.diffusion.layers.custom_op import CustomOp
    from vllm_omni.diffusion.layers.norm import LayerNorm

    norm = LayerNorm(64)
    assert isinstance(norm, CustomOp)


def test_rmsnorm_inherits_from_customop():
    """RMSNorm inherits from CustomOp for platform dispatch."""
    from vllm_omni.diffusion.layers.custom_op import CustomOp
    from vllm_omni.diffusion.layers.norm import RMSNorm

    norm = RMSNorm(64)
    assert isinstance(norm, CustomOp)


def test_layernorm_has_platform_methods():
    """LayerNorm has forward methods for each platform."""
    from vllm_omni.diffusion.layers.norm import LayerNorm

    norm = LayerNorm(64)

    assert hasattr(norm, "forward_cuda")
    assert hasattr(norm, "forward_hip")
    assert hasattr(norm, "forward_xpu")
    assert hasattr(norm, "forward_npu")
    assert hasattr(norm, "forward_native")


def test_rmsnorm_has_platform_methods():
    """RMSNorm has forward methods for each platform."""
    from vllm_omni.diffusion.layers.norm import RMSNorm

    norm = RMSNorm(64)

    assert hasattr(norm, "forward_cuda")
    assert hasattr(norm, "forward_hip")
    assert hasattr(norm, "forward_xpu")
    assert hasattr(norm, "forward_npu")
    assert hasattr(norm, "forward_native")


def test_layernorm_forward_native_directly():
    """LayerNorm.forward_native can be called directly."""
    from vllm_omni.diffusion.layers.norm import LayerNorm

    dim = 64
    norm = LayerNorm(dim)
    x = torch.randn(2, 4, dim)

    out = norm.forward_native(x)

    assert out.shape == (2, 4, dim)


def test_rmsnorm_forward_native_directly():
    """RMSNorm.forward_native can be called directly."""
    from vllm_omni.diffusion.layers.norm import RMSNorm

    hidden_size = 64
    norm = RMSNorm(hidden_size)
    x = torch.randn(2, 4, hidden_size)

    out = norm.forward_native(x)

    assert out.shape == (2, 4, hidden_size)


# ── Edge case tests ──


def test_layernorm_with_large_dim():
    """LayerNorm works with large hidden dimensions."""
    from vllm_omni.diffusion.layers.norm import LayerNorm

    dim = 4096
    norm = LayerNorm(dim)
    x = torch.randn(1, 16, dim)

    out = norm(x)

    assert out.shape == (1, 16, dim)


def test_rmsnorm_with_large_dim():
    """RMSNorm works with large hidden dimensions."""
    from vllm_omni.diffusion.layers.norm import RMSNorm

    hidden_size = 4096
    norm = RMSNorm(hidden_size)
    x = torch.randn(1, 16, hidden_size)

    out = norm(x)

    assert out.shape == (1, 16, hidden_size)


def test_layernorm_with_single_element_batch():
    """LayerNorm works with batch size of 1."""
    from vllm_omni.diffusion.layers.norm import LayerNorm

    dim = 64
    norm = LayerNorm(dim)
    x = torch.randn(1, 1, dim)

    out = norm(x)

    assert out.shape == (1, 1, dim)


def test_rmsnorm_with_single_element_batch():
    """RMSNorm works with batch size of 1."""
    from vllm_omni.diffusion.layers.norm import RMSNorm

    hidden_size = 64
    norm = RMSNorm(hidden_size)
    x = torch.randn(1, 1, hidden_size)

    out = norm(x)

    assert out.shape == (1, 1, hidden_size)


# ── Fused vs Native correctness tests ──


def test_rmsnorm_fused_native_parity_cuda():
    """Verify fused and native RMSNorm produce identical results on CUDA.

    The fused kernel uses vllm._custom_ops.rms_norm while native uses pure PyTorch.
    Both should produce numerically equivalent results.
    """
    from vllm_omni.diffusion.layers.norm import RMSNorm

    hidden_size = 128
    eps = 1e-6
    torch.manual_seed(42)

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        torch.manual_seed(42)
        norm = RMSNorm(hidden_size, eps=eps).cuda()
        norm.weight.data = norm.weight.data.to(dtype)

        x = torch.randn(4, 8, hidden_size, dtype=dtype, device="cuda")

        out_fused = norm._forward_fused(x)
        out_native = norm.forward_native(x)

        torch.testing.assert_close(out_fused, out_native, atol=1e-3, rtol=1e-3)


def test_rmsnorm_fused_native_parity_large_tensor():
    """Verify fused and native RMSNorm parity with large tensors.

    Tests realistic tensor sizes used in diffusion models.
    """
    from vllm_omni.diffusion.layers.norm import RMSNorm

    hidden_size = 4096
    batch_size = 16
    seq_len = 1024
    eps = 1e-6

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        torch.manual_seed(123)
        norm = RMSNorm(hidden_size, eps=eps).cuda()
        norm.weight.data = norm.weight.data.to(dtype)

        x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device="cuda")

        out_fused = norm._forward_fused(x)
        out_native = norm.forward_native(x)

        torch.testing.assert_close(out_fused, out_native, atol=1e-2, rtol=1e-2)


def test_rmsnorm_fused_native_parity_2d_input():
    """Verify fused and native RMSNorm parity with 2D input tensors."""
    from vllm_omni.diffusion.layers.norm import RMSNorm

    hidden_size = 256
    batch_size = 32
    eps = 1e-6

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        torch.manual_seed(456)
        norm = RMSNorm(hidden_size, eps=eps).cuda()
        norm.weight.data = norm.weight.data.to(dtype)

        x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")

        out_fused = norm._forward_fused(x)
        out_native = norm.forward_native(x)

        torch.testing.assert_close(out_fused, out_native, atol=1e-3, rtol=1e-3)


def test_rmsnorm_fused_native_parity_custom_weight():
    """Verify fused and native RMSNorm parity with non-default weights."""
    from vllm_omni.diffusion.layers.norm import RMSNorm

    hidden_size = 64
    eps = 1e-6

    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        torch.manual_seed(789)
        norm = RMSNorm(hidden_size, eps=eps).cuda()
        norm.weight.data = torch.randn(hidden_size, dtype=dtype, device="cuda")

        x = torch.randn(2, 4, hidden_size, dtype=dtype, device="cuda")

        out_fused = norm._forward_fused(x)
        out_native = norm.forward_native(x)

        torch.testing.assert_close(out_fused, out_native, atol=1e-3, rtol=1e-3)
