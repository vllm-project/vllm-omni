# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


@pytest.mark.parametrize("rows, cols", [(1, 3072), (300, 3072), (257, 9216)])
def test_per_token_fp8_quant_matches_vllm(rows: int, cols: int) -> None:
    from vllm import _custom_ops as ops

    from vllm_omni.diffusion.layers.fp8_per_token_quant import per_token_fp8_quant

    torch.manual_seed(0)
    x = (torch.randn(rows, cols, device="cuda") * torch.rand(rows, 1, device="cuda") * 8).bfloat16()
    x[0].zero_()
    ref_q, ref_scale = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
    q, scale = per_token_fp8_quant(x)

    torch.testing.assert_close(scale, ref_scale, rtol=1e-6, atol=0)
    # Division vs multiplication by the reciprocal can round a code by one step.
    codes = q.view(torch.uint8).int() - ref_q.view(torch.uint8).int()
    assert codes.abs().max().item() <= 1
    assert (codes != 0).float().mean().item() < 2e-3


def test_per_token_fp8_quant_accepts_row_strided_input() -> None:
    from vllm_omni.diffusion.layers.fp8_per_token_quant import per_token_fp8_quant

    base = torch.randn(64, 2 * 3072, device="cuda").bfloat16()
    view = base[:, :3072]
    q, scale = per_token_fp8_quant(view)
    q_ref, scale_ref = per_token_fp8_quant(view.contiguous())
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.equal(scale, scale_ref)
