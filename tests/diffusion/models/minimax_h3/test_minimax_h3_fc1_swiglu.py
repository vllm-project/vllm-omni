# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The fused fc1 + SwiGLU kernel matches the fc1 GEMM followed by SiluAndMul, eager and under torch.compile."""

import pytest
import torch
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_blocks import MiniMaxH3MLP
from vllm_omni.diffusion.models.minimax_h3.ops.fc1_swiglu import fc1_swiglu, fc1_swiglu_supported

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    *hardware_marks(res={"cuda": "B200"}, num_cards=1),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
        reason="fused fc1 SwiGLU is enabled on SM100-family GPUs only",
    ),
]


def _reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    import vllm._custom_ops  # noqa: F401 - registers torch.ops._C.silu_and_mul

    hidden = torch.nn.functional.linear(x, weight)
    out = torch.empty(*x.shape[:-1], weight.shape[0] // 2, device=x.device, dtype=x.dtype)
    torch.ops._C.silu_and_mul(out, hidden)
    return out


@pytest.mark.parametrize(("tokens", "hidden", "ffn"), [(1, 512, 128), (777, 512, 384), (4096, 5376, 1792)])
def test_fc1_swiglu_matches_reference(tokens: int, hidden: int, ffn: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2 * ffn, hidden, device="cuda", dtype=torch.bfloat16) / hidden**0.5
    assert fc1_swiglu_supported(x, weight)
    torch.testing.assert_close(fc1_swiglu(x, weight), _reference(x, weight))


def test_fc1_swiglu_keeps_leading_dims() -> None:
    x = torch.randn(2, 33, 512, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16) / 512**0.5
    torch.testing.assert_close(fc1_swiglu(x, weight), _reference(x, weight))


def test_fc1_swiglu_rejects_unsupported_inputs() -> None:
    x = torch.randn(8, 512, device="cuda", dtype=torch.bfloat16)
    assert not fc1_swiglu_supported(x.float(), torch.randn(256, 512, device="cuda"))
    assert not fc1_swiglu_supported(x, torch.randn(200, 512, device="cuda", dtype=torch.bfloat16))
    assert not fc1_swiglu_supported(x.cpu(), torch.randn(256, 512, dtype=torch.bfloat16))


class _Linear(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.quant_method = UnquantizedLinearMethod()
        self.bias = None
        self.weight = torch.nn.Parameter(weight, requires_grad=False)


class _PassThrough(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return x, None


@pytest.mark.parametrize("shape", [(777, 512), (2, 33, 512)])
def test_compiled_mlp_reaches_the_fused_op(shape: tuple[int, ...]) -> None:
    torch.manual_seed(0)
    mlp = MiniMaxH3MLP.__new__(MiniMaxH3MLP)
    torch.nn.Module.__init__(mlp)
    mlp.fc1 = _Linear(torch.randn(768, 512, device="cuda", dtype=torch.bfloat16) / 512**0.5)
    mlp.fc2 = _PassThrough()
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

    targets = []

    def recording_backend(gm: torch.fx.GraphModule, example_inputs):
        targets.extend(str(node.target) for node in gm.graph.nodes if node.op == "call_function")
        from torch._inductor.compile_fx import compile_fx

        return compile_fx(gm, example_inputs)

    torch._dynamo.reset()
    compiled = torch.compile(mlp.forward, backend=recording_backend, fullgraph=True)
    out = compiled(x)
    assert any(t.startswith("vllm_omni.minimax_h3_fc1_swiglu") for t in targets), targets
    assert torch.equal(out, fc1_swiglu(x, mlp.fc1.weight))
    assert torch.equal(compiled(x), out)  # replay of the compiled graph


def test_fc1_swiglu_restores_the_callers_triton_allocator() -> None:
    import triton
    from triton.runtime import _allocation

    def sentinel(size: int, align: int, stream):
        raise AssertionError("fc1_swiglu must use its own allocator for the launch")

    previous = _allocation._allocator.get()
    triton.set_allocator(sentinel)
    try:
        x = torch.randn(8, 512, device="cuda", dtype=torch.bfloat16)
        fc1_swiglu(x, torch.randn(256, 512, device="cuda", dtype=torch.bfloat16))
        assert _allocation._allocator.get() is sentinel
    finally:
        triton.set_allocator(previous)
