# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.layers.mxfp8 import (
    _scale_numel,
    mxfp8_quantize_project,
    mxfp8_quantize_swizzled,
    mxfp8_scaled_mm,
    silu_mul_mxfp8_cuda,
)

pytestmark = [pytest.mark.local_model, pytest.mark.cuda, pytest.mark.diffusion]


@pytest.fixture(autouse=True)
def sm120_only():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    if not hasattr(torch.nn.functional, "scaled_mm"):
        pytest.skip("requires the public PyTorch block-scaled GEMM API")


@pytest.mark.parametrize("rows,hidden", [(1, 32), (127, 96), (137, 128), (256, 512), (11992, 14336)])
def test_swiglu_quantization_preserves_cuda_activation_bytes(rows, hidden):
    from vllm.model_executor.layers.activation import SiluAndMul

    generator = torch.Generator(device="cuda").manual_seed(1101)
    x = torch.randn(rows, 2 * hidden, dtype=torch.bfloat16, device="cuda", generator=generator)
    reference = SiluAndMul().forward_cuda(x)
    expected_q, expected_scale = mxfp8_quantize_swizzled(reference)
    q, scale, activation = silu_mul_mxfp8_cuda(x, return_bf16=True)
    assert torch.equal(activation.view(torch.uint8), reference.view(torch.uint8))
    assert torch.equal(q.view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(scale, expected_scale)


@pytest.mark.parametrize(
    "rows,hidden,width,split", [(137, 128, 384, 256), (256, 512, 384, 256), (11992, 5376, 21504, 14336)]
)
def test_weight_slice_reuses_one_quantized_activation(rows, hidden, width, split):
    generator = torch.Generator(device="cuda").manual_seed(1101)
    x = torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda", generator=generator)
    weight = torch.randn(width, hidden, dtype=x.dtype, device=x.device, generator=generator)
    wq, ws = mxfp8_quantize_swizzled(weight)
    cut = _scale_numel(split, hidden)
    first, q, scale = mxfp8_quantize_project(x, wq[:split], ws[:cut])
    second = mxfp8_scaled_mm(q, wq[split:], scale, ws[cut:])
    full = mxfp8_scaled_mm(q, wq, scale, ws)
    assert torch.equal(first.view(torch.uint8), full[:, :split].contiguous().view(torch.uint8))
    assert torch.equal(second.view(torch.uint8), full[:, split:].contiguous().view(torch.uint8))
