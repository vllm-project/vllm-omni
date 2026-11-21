# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The fused fc1 + SwiGLU kernel is bit-identical to the reference fc1 GEMM followed by SiluAndMul."""

import pytest
import torch

from tests.helpers.mark import hardware_marks
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
def test_fc1_swiglu_matches_reference_bit_for_bit(tokens: int, hidden: int, ffn: int) -> None:
    torch.manual_seed(0)
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2 * ffn, hidden, device="cuda", dtype=torch.bfloat16) / hidden**0.5
    assert fc1_swiglu_supported(x, weight)
    assert torch.equal(fc1_swiglu(x, weight), _reference(x, weight))


def test_fc1_swiglu_keeps_leading_dims() -> None:
    x = torch.randn(2, 33, 512, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16) / 512**0.5
    assert torch.equal(fc1_swiglu(x, weight), _reference(x, weight))


def test_fc1_swiglu_rejects_unsupported_inputs() -> None:
    x = torch.randn(8, 512, device="cuda", dtype=torch.bfloat16)
    assert not fc1_swiglu_supported(x.float(), torch.randn(256, 512, device="cuda"))
    assert not fc1_swiglu_supported(x, torch.randn(200, 512, device="cuda", dtype=torch.bfloat16))
    assert not fc1_swiglu_supported(x.cpu(), torch.randn(256, 512, dtype=torch.bfloat16))
