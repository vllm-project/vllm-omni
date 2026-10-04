# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from vllm_omni.diffusion.layers.gated_residual import gated_residual

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("shape", "gate_shape"),
    [
        ((2, 7, 16), (16,)),
        ((2, 7, 16), (2, 1, 16)),
        ((2, 3, 7, 16), (2, 1, 1, 16)),
        ((2, 7, 16), (2, 7, 16)),
        ((2, 3, 7, 16), (1, 3, 1, 16)),
        ((2, 7, 16), (2, 7, 1)),
        ((2, 7, 16), ()),
    ],
)
def test_gated_residual_cpu_matches_eager(shape, gate_shape):
    torch.manual_seed(11)
    residual = torch.randn(shape)
    branch = torch.randn(shape)
    gate = torch.randn(gate_shape)

    expected = residual + branch * gate
    actual = gated_residual(residual, branch, gate)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.cpu
def test_gated_residual_noncontiguous_fallback():
    residual = torch.randn(2, 16, 5).transpose(1, 2)
    branch = torch.randn(2, 16, 5).transpose(1, 2)
    gate = torch.randn(2, 1, 16)

    actual = gated_residual(residual, branch, gate)

    torch.testing.assert_close(actual, residual + branch * gate, rtol=0, atol=0)


@pytest.mark.cpu
def test_gated_residual_rejects_invalid_shapes():
    residual = torch.randn(2, 7, 16)
    branch = torch.randn(2, 7, 8)
    gate = torch.randn(2, 1, 16)

    with pytest.raises(ValueError, match="same shape"):
        gated_residual(residual, branch, gate)
    with pytest.raises(ValueError, match="not broadcastable"):
        gated_residual(residual, residual, torch.randn(3, 16))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gate_shape", [(4096,), (2, 1, 4096), (2, 257, 4096)])
def test_gated_residual_cuda_matches_eager(dtype, gate_shape):
    torch.manual_seed(17)
    shape = (2, 257, 4096)
    residual = torch.randn(shape, device="cuda", dtype=dtype)
    branch = torch.randn(shape, device="cuda", dtype=dtype)
    gate = torch.randn(gate_shape, device="cuda", dtype=dtype)

    expected = residual + branch * gate
    actual = gated_residual(residual, branch, gate)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_gated_residual_supports_torch_compile():
    compiled = torch.compile(gated_residual, fullgraph=True)
    residual = torch.randn(2, 17, 128, device="cuda", dtype=torch.bfloat16)
    branch = torch.randn_like(residual)
    gate = torch.randn(2, 1, 128, device="cuda", dtype=torch.bfloat16)

    actual = compiled(residual, branch, gate)

    torch.testing.assert_close(actual, residual + branch * gate, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_gated_residual_supports_strided_modulation_gate():
    residual = torch.randn(2, 17, 128, device="cuda", dtype=torch.bfloat16)
    branch = torch.randn_like(residual)
    gate_storage = torch.randn(2, 17, 3, 128, device="cuda", dtype=torch.bfloat16)
    gate = gate_storage[:, :, 1, :]
    assert not gate.is_contiguous()

    actual = gated_residual(residual, branch, gate)

    torch.testing.assert_close(actual, residual + branch * gate, rtol=0, atol=0)
