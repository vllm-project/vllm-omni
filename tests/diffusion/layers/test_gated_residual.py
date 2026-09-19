# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

from vllm_omni.diffusion.layers.ops import gated_residual

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

_CUDA_ONLY = pytest.mark.skipif(
    not (HAS_TRITON and current_platform.is_cuda() and torch.cuda.is_available()),
    reason="CUDA and Triton required",
)


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
@_CUDA_ONLY
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gate_shape", [(4096,), (2, 1, 4096), (2, 257, 4096)])
def test_gated_residual_cuda_matches_eager(dtype, gate_shape, mocker):
    custom_op = mocker.spy(torch.ops.vllm_omni, "gated_residual")
    torch.manual_seed(17)
    shape = (2, 257, 4096)
    residual = torch.randn(shape, device="cuda", dtype=dtype)
    branch = torch.randn(shape, device="cuda", dtype=dtype)
    gate = torch.randn(gate_shape, device="cuda", dtype=dtype)

    expected = residual + branch * gate
    actual = gated_residual(residual, branch, gate)

    custom_op.assert_called_once()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.cuda
@_CUDA_ONLY
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gated_residual_supports_torch_compile(dtype):
    compiled = torch.compile(gated_residual, fullgraph=True)
    residual = torch.randn(2, 17, 128, device="cuda", dtype=dtype)
    branch = torch.randn_like(residual)
    gate = torch.randn(2, 1, 128, device="cuda", dtype=dtype)

    actual = compiled(residual, branch, gate)

    torch.testing.assert_close(actual, residual + branch * gate, rtol=0, atol=0)


@pytest.mark.cuda
@_CUDA_ONLY
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tokens", [1, 17])
@pytest.mark.parametrize("view", ["chunk", "unbind"])
def test_gated_residual_supports_strided_modulation_gate(dtype, tokens, view):
    residual = torch.randn(2, 17, 128, device="cuda", dtype=dtype)
    branch = torch.randn_like(residual)
    if view == "chunk":
        gate_storage = torch.randn(2, tokens, 3 * 128, device="cuda", dtype=dtype)
        gate = gate_storage.chunk(3, dim=-1)[1]
    else:
        gate_storage = torch.randn(2, tokens, 3, 128, device="cuda", dtype=dtype)
        gate = gate_storage.unbind(dim=-2)[1]
    assert not gate.is_contiguous()

    actual = gated_residual(residual, branch, gate)

    torch.testing.assert_close(actual, residual + branch * gate, rtol=0, atol=0)


@pytest.mark.cuda
@_CUDA_ONLY
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("compiled", [False, True])
def test_gated_residual_preserves_product_rounding(dtype: torch.dtype, compiled: bool):
    # The low-precision product rounds to 1 + 2 * eps. An FMA instead
    # retains eps**2 after cancellation with the residual.
    eps = torch.finfo(dtype).eps
    residual = torch.full((2, 3, 129), -(1 + 2 * eps), device="cuda", dtype=dtype)
    branch = torch.full_like(residual, 1 + eps)
    gate = torch.full((129,), 1 + eps, device="cuda", dtype=dtype)
    expected = residual + branch * gate
    single_round = (residual.float() + branch.float() * gate.float()).to(dtype)
    assert not torch.equal(expected, single_round)

    function = torch.compile(gated_residual, fullgraph=True) if compiled else gated_residual
    actual = function(residual, branch, gate)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "device",
    [pytest.param("cpu", marks=pytest.mark.cpu), pytest.param("cuda", marks=[pytest.mark.cuda, _CUDA_ONLY])],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("gate_shape", [(16,), (2, 1, 16), (2, 3, 16)])
@pytest.mark.parametrize(
    "requires_grad",
    [(True, False, False), (False, True, False), (False, False, True), (True, True, True)],
    ids=["residual", "branch", "gate", "all"],
)
def test_gated_residual_backward_matches_eager(
    device: str, dtype: torch.dtype, gate_shape: tuple[int, ...], requires_grad: tuple[bool, bool, bool]
):
    torch.manual_seed(23)
    residual = torch.randn(2, 3, 16, device=device, dtype=dtype, requires_grad=requires_grad[0])
    branch = torch.randn(2, 3, 16, device=device, dtype=dtype, requires_grad=requires_grad[1])
    gate = torch.randn(gate_shape, device=device, dtype=dtype, requires_grad=requires_grad[2])
    inputs = tuple(tensor for tensor in (residual, branch, gate) if tensor.requires_grad)
    grad_output = torch.randn_like(residual)

    with torch.enable_grad():
        expected = residual + branch * gate
        actual = gated_residual(residual, branch, gate)
        expected_grads = torch.autograd.grad(expected, inputs, grad_outputs=grad_output)
        actual_grads = torch.autograd.grad(actual, inputs, grad_outputs=grad_output)

    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.cuda
@_CUDA_ONLY
@pytest.mark.parametrize("grad_context", [torch.no_grad, torch.inference_mode])
def test_gated_residual_no_grad_keeps_fused_path(grad_context, mocker):
    residual = torch.randn(2, 7, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    branch = torch.randn_like(residual, requires_grad=True)
    gate = torch.randn(2, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    custom_op = mocker.spy(torch.ops.vllm_omni, "gated_residual")

    with grad_context():
        actual = gated_residual(residual, branch, gate)
        expected = residual + branch * gate

    custom_op.assert_called_once()
    assert not actual.requires_grad
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.cuda
@_CUDA_ONLY
@pytest.mark.parametrize(
    "case", ["float32", "mixed_dtype", "activation_stride", "gate_stride", "broadcast", "scalar_gate", "hidden_size"]
)
def test_gated_residual_cuda_fallback(case: str, mocker):
    residual = torch.randn(2, 3, 7, 128, device="cuda", dtype=torch.bfloat16)
    branch = torch.randn_like(residual)
    gate = torch.randn(2, 1, 1, 128, device="cuda", dtype=torch.bfloat16)
    if case == "float32":
        residual, branch, gate = residual.float(), branch.float(), gate.float()
    elif case == "mixed_dtype":
        residual, branch, gate = residual.float(), branch.half(), gate.float()
    elif case == "activation_stride":
        residual, branch = residual.transpose(1, 2), branch.transpose(1, 2)
    elif case == "gate_stride":
        gate = torch.randn(2, 1, 1, 256, device="cuda", dtype=gate.dtype)[..., ::2]
    elif case == "broadcast":
        gate = torch.randn(1, 3, 1, 128, device="cuda", dtype=gate.dtype)
    elif case == "scalar_gate":
        gate = torch.tensor(0.5, device="cuda", dtype=gate.dtype)
    elif case == "hidden_size":
        residual = torch.randn(2, 3, 16385, device="cuda", dtype=residual.dtype)
        branch = torch.randn_like(residual)
        gate = torch.randn(16385, device="cuda", dtype=gate.dtype)
    custom_op = mocker.spy(torch.ops.vllm_omni, "gated_residual")

    actual = gated_residual(residual, branch, gate)

    custom_op.assert_not_called()
    torch.testing.assert_close(actual, residual + branch * gate, rtol=0, atol=0)


@pytest.mark.cuda
@_CUDA_ONLY
def test_gated_residual_custom_op_schema_and_fake():
    residual = torch.randn(2, 7, 128, device="cuda", dtype=torch.bfloat16)
    branch = torch.randn_like(residual)
    gate = torch.randn(2, 1, 128, device="cuda", dtype=torch.bfloat16)
    # The internal custom op receives the per-batch layout and row stride.
    torch.library.opcheck(
        torch.ops.vllm_omni.gated_residual.default,
        (residual, branch, gate, 1, 7, 128),
        test_utils=("test_schema", "test_faketensor"),
    )
