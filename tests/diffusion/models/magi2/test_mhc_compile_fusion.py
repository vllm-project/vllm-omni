# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for Inductor-fusable elementwise mHC stream contractions under torch.compile."""

from __future__ import annotations

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.models.magi2.layers.native.mhc import MHCHandler
from vllm_omni.platforms import current_omni_platform

pytestmark = [
    pytest.mark.diffusion,
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
    pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="requires CUDA"),
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("tokens", [1, 17, 1024])
def test_compiled_mhc_apply_pre_and_hyper_connect_parity(dtype: torch.dtype, tokens: int):
    hidden = 256
    handler = MHCHandler(num_streams=4, hidden_size=hidden)
    g = torch.Generator(device="cuda").manual_seed(tokens)
    streams = torch.randn((tokens, 4, hidden), device="cuda", dtype=dtype, generator=g)
    branch_out = torch.randn((tokens, hidden), device="cuda", dtype=dtype, generator=g)
    pre_logits = torch.randn((tokens, 4), device="cuda", dtype=torch.float32, generator=g)
    alpha = torch.tensor([0.01], device="cuda", dtype=torch.float32)
    bias = torch.randn((4,), device="cuda", dtype=torch.float32, generator=g) * 0.1
    post_coeff = torch.rand((tokens, 4), device="cuda", dtype=dtype, generator=g) * 2.0
    res_mat = torch.softmax(torch.randn((tokens, 4, 4), device="cuda", dtype=torch.float32, generator=g), dim=-1).to(
        dtype
    )

    eager_pre = handler.apply_pre(streams, (alpha, bias, pre_logits), out_dtype=dtype)
    eager_mix = handler.hyper_connect(streams, branch_out, post_coeff, res_mat)

    def compiled_fn(s, b_out, pl, a, bi, pc, rm):
        h_in = handler.apply_pre(s, (a, bi, pl), out_dtype=dtype)
        s_out = handler.hyper_connect(s, b_out, pc, rm)
        return h_in, s_out

    compiled = torch.compile(compiled_fn, fullgraph=True)
    comp_pre, comp_mix = compiled(streams, branch_out, pre_logits, alpha, bias, post_coeff, res_mat)

    tol = 1e-5 if dtype == torch.float32 else 2e-2
    assert torch.allclose(comp_pre, eager_pre, atol=tol, rtol=tol)
    assert torch.allclose(comp_mix, eager_mix, atol=tol, rtol=tol)


def test_compiled_mhc_avoids_bmm_lowering():
    """Verify compiled apply_pre and hyper_connect stay elementwise instead of lowering to aten.bmm."""
    handler = MHCHandler(num_streams=4, hidden_size=128)
    streams = torch.randn((64, 4, 128), device="cuda", dtype=torch.bfloat16)
    branch_out = torch.randn((64, 128), device="cuda", dtype=torch.bfloat16)
    pre_logits = torch.randn((64, 4), device="cuda", dtype=torch.float32)
    alpha = torch.tensor([0.01], device="cuda", dtype=torch.float32)
    bias = torch.zeros((4,), device="cuda", dtype=torch.float32)
    post_coeff = torch.ones((64, 4), device="cuda", dtype=torch.bfloat16)
    res_mat = torch.eye(4, device="cuda", dtype=torch.bfloat16).unsqueeze(0).expand(64, 4, 4).contiguous()

    captured_ops: list[str] = []

    def capture_backend(gm, example_inputs):
        for node in gm.graph.nodes:
            if node.op == "call_function":
                captured_ops.append(str(node.target))
        return gm.forward

    def fn(s, b_out, pl, a, bi, pc, rm):
        return handler.apply_pre(s, (a, bi, pl), out_dtype=s.dtype), handler.hyper_connect(s, b_out, pc, rm)

    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=True, backend=capture_backend)
    compiled(streams, branch_out, pre_logits, alpha, bias, post_coeff, res_mat)
    assert not any("bmm" in op or "einsum" in op for op in captured_ops), f"Unexpected bmm/einsum ops: {captured_ops}"
