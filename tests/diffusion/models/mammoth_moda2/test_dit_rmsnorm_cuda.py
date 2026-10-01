# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MammothModa2 DiT norms on the fused shared RMSNorm with real Preview shapes."""

import pytest
import torch
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.layers.norm import RMSNorm

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    pytest.mark.cuda,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
]

# MammothModa2-Preview at 1024x1024: 77 text + 4096 image tokens, hidden 2520, 21 heads of 120.
SEQ = 77 + 4096
DIM, HEADS = 2520, 21
EPS = 1e-5

_SHAPES = [(2, SEQ, DIM), (2, SEQ, HEADS, DIM // HEADS)]
_SHAPE_IDS = ["hidden", "qk"]


def _norm_pair(hidden: int) -> tuple[Qwen2RMSNorm, RMSNorm]:
    """A Qwen2RMSNorm and a shared RMSNorm on CUDA holding the same bf16 weights."""
    qwen2 = Qwen2RMSNorm(hidden, eps=EPS).to("cuda", torch.bfloat16)
    with torch.no_grad():
        qwen2.weight.copy_(torch.rand(hidden) + 0.5)
    shared = RMSNorm(hidden, eps=EPS, dtype=torch.bfloat16).cuda()
    shared.load_state_dict(qwen2.state_dict())
    return qwen2, shared


def _float64_reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """RMSNorm evaluated in float64, independent of either implementation."""
    x64 = x.to(torch.float64)
    variance = x64.pow(2).mean(-1, keepdim=True)
    return weight.to(torch.float64) * x64 * torch.rsqrt(variance + EPS)


@pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
def test_fused_rmsnorm_is_within_one_bf16_step_of_qwen2_rmsnorm(shape):
    torch.manual_seed(0)
    qwen2, shared = _norm_pair(shape[-1])
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        got, want = shared(x), qwen2(x)
        torch.testing.assert_close(got, shared._forward_fused(x), atol=0, rtol=0)
    # bf16 keeps 7 fraction bits, so one rounding step at value v is at most |v| / 128.
    assert ((got.double() - want.double()).abs() <= want.double().abs() / 128).all()


@pytest.mark.parametrize("shape", _SHAPES, ids=_SHAPE_IDS)
def test_fused_rmsnorm_is_closer_to_float64_than_qwen2_rmsnorm(shape):
    """Qwen2RMSNorm rounds to bf16 before the weight multiply and so rounds twice;
    the fused kernel rounds once."""
    torch.manual_seed(0)
    qwen2, shared = _norm_pair(shape[-1])
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

    reference = _float64_reference(x, qwen2.weight)
    scale = reference.abs().mean()

    def relative_error(out: torch.Tensor) -> float:
        return ((out.to(torch.float64) - reference).abs().mean() / scale).item()

    with torch.no_grad():
        fused, cast_chain = relative_error(shared._forward_fused(x)), relative_error(qwen2(x))
    assert fused < cast_chain, f"fused {fused:.3e} not better than Qwen2RMSNorm {cast_chain:.3e}"
