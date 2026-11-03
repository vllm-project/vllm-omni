# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz import time_major_conv as tmc

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _snake_nct(x, exp_alpha, inv_beta):
    return x + inv_beta[None, :, None] * torch.sin(x * exp_alpha[None, :, None]).square()


@pytest.mark.parametrize(("kernel_size", "dilation"), [(1, 1), (7, 1), (7, 3), (3, 9)])
@pytest.mark.parametrize("with_bias_and_snake", [False, True])
def test_im2col_gemm_matches_causal_conv1d(kernel_size, dilation, with_bias_and_snake):
    torch.manual_seed(0)
    bsz, t, c_in, c_out = 2, 11, 6, 5
    conv = torch.nn.Conv1d(c_in, c_out, kernel_size, dilation=dilation)
    x = torch.randn(bsz, c_in, t)
    in_bias = torch.randn(c_in) if with_bias_and_snake else None
    snake = (torch.rand(c_in) + 0.5, torch.rand(c_in) + 0.5) if with_bias_and_snake else None

    ref_in = x if in_bias is None else x + in_bias[None, :, None]
    if snake is not None:
        ref_in = _snake_nct(ref_in, *snake)
    expected = conv(F.pad(ref_in, ((kernel_size - 1) * dilation, 0)))

    cols = tmc.snake_im2col(x.transpose(1, 2).contiguous(), kernel_size, dilation, in_bias, snake)
    actual = torch.addmm(conv.bias, cols, tmc.conv_gemm_weight(conv.weight)).view(bsz, t, c_out)
    torch.testing.assert_close(actual.transpose(1, 2), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(("stride", "kernel_size"), [(2, 2), (3, 6), (8, 16)])
def test_transposed_overlap_add_matches_trimmed_conv_transpose(stride, kernel_size):
    torch.manual_seed(0)
    bsz, t, c_in, c_out = 2, 5, 4, 3
    conv = torch.nn.ConvTranspose1d(c_in, c_out, kernel_size, stride=stride)
    x = torch.randn(bsz, c_in, t)
    expected = conv(x)[..., : t * stride]  # causal: drop the kernel - stride right tail

    z = (x.transpose(1, 2).reshape(bsz * t, c_in) @ tmc.transposed_gemm_weight(conv.weight)).view(
        bsz, t, kernel_size, c_out
    )
    actual = tmc.transposed_overlap_add(z, stride, conv.bias)
    torch.testing.assert_close(actual.transpose(1, 2), expected, rtol=1e-5, atol=1e-5)


def test_transposed_overlap_add_rejects_other_kernels():
    with pytest.raises(ValueError, match="kernel stride or 2"):
        tmc.transposed_overlap_add(torch.zeros(1, 2, 5, 3), 2, torch.zeros(3))


@pytest.mark.parametrize(("kernel_size", "dilation"), [(1, 1), (7, 3)])
@pytest.mark.parametrize("with_residual", [False, True])
def test_causal_conv_matches_conv1d(kernel_size, dilation, with_residual):
    torch.manual_seed(0)
    bsz, t, c_in, c_out = 2, 13, 6, 5
    conv = torch.nn.Conv1d(c_in, c_out, kernel_size, dilation=dilation)
    x = torch.randn(bsz, c_in, t)
    residual = torch.randn(bsz, t, c_out) if with_residual else None
    expected = conv(F.pad(x, ((kernel_size - 1) * dilation, 0))).transpose(1, 2)
    if residual is not None:
        expected = expected + residual
    actual = tmc.causal_conv(
        x.transpose(1, 2).contiguous(), tmc.conv_gemm_weight(conv.weight), conv.bias, kernel_size, dilation, residual
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def _gpu_tolerances(dtype):
    # bf16 results may differ by one ulp (sin rounding, fp32 vs bf16 accumulation order).
    return {"rtol": 1.6e-2, "atol": 1e-2} if dtype == torch.bfloat16 else {"rtol": 1e-4, "atol": 1e-4}


@pytest.mark.skipif(not torch.cuda.is_available() or not tmc.HAS_TRITON, reason="needs CUDA and Triton")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_triton_kernels_match_reference(dtype):
    torch.manual_seed(0)
    tol = _gpu_tolerances(dtype)
    x = torch.randn(3, 37, 96, device="cuda", dtype=dtype)
    bias = torch.randn(96, device="cuda", dtype=dtype)
    snake = (torch.rand(96, device="cuda", dtype=dtype) + 0.5, torch.rand(96, device="cuda", dtype=dtype) + 0.5)
    for kernel_size, dilation in ((1, 1), (7, 9)):
        torch.testing.assert_close(
            tmc.snake_im2col(x, kernel_size, dilation, bias, snake),
            tmc.snake_im2col_reference(x, kernel_size, dilation, bias, snake),
            **tol,
        )
    z = torch.randn(3, 37, 10, 48, device="cuda", dtype=dtype)
    torch.testing.assert_close(
        tmc.transposed_overlap_add(z, 5, bias[:48]), tmc.transposed_overlap_add_reference(z, 5, bias[:48]), **tol
    )


@pytest.mark.skipif(not torch.cuda.is_available() or not tmc.HAS_TRITON, reason="needs CUDA and Triton")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(("c_in", "c_out", "kernel_size", "dilation"), [(96, 96, 7, 9), (192, 96, 1, 1), (96, 1, 7, 1)])
def test_triton_causal_conv_matches_reference(dtype, c_in, c_out, kernel_size, dilation):
    torch.manual_seed(0)
    x = torch.randn(3, 101, c_in, device="cuda", dtype=dtype)
    weight = torch.randn(kernel_size * c_in, c_out, device="cuda", dtype=dtype) / (kernel_size * c_in) ** 0.5
    bias = torch.randn(c_out, device="cuda", dtype=dtype)
    residual = torch.randn(3, 101, c_out, device="cuda", dtype=dtype)
    expected = tmc.causal_conv_reference(
        x.float(), weight.float(), bias.float(), kernel_size, dilation, residual.float()
    )
    actual = tmc.causal_conv(x, weight, bias, kernel_size, dilation, residual)
    torch.testing.assert_close(actual.float(), expected, **_gpu_tolerances(dtype))
