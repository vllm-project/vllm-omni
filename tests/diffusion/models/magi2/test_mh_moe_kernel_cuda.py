# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA correctness tests for the MAGI-2 multi-head MoE fused expert kernel.

Covers triton_mh_moe_forward at the released MAGI-2 Preview dimensions
(heads=12, experts=256, top_k=6, d_head=256, d_expert=1280): numerical parity
against the small-shape PyTorch oracle, and bitwise reproducibility of the
deterministic scatter path across launches.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from vllm_omni.diffusion.models.magi2.mh_moe import (
    compute_topk_probs_and_indices,
    global_sort_routes,
    torch_mh_moe_forward,
    triton_mh_moe_forward,
)

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

_NUM_HEADS = 12
_NUM_EXPERTS = 256
_TOP_K = 6
_D_HEAD = 256
_D_EXPERT = 1280
_DTYPE = torch.bfloat16


@dataclass(frozen=True)
class _MhMoeProblem:
    x: torch.Tensor
    gather_ids: torch.Tensor
    probs: torch.Tensor
    offsets: torch.Tensor
    w_gate: torch.Tensor
    w_up: torch.Tensor
    w_down: torch.Tensor


def _make_problem(seq: int, device: torch.device, seed: int = 1234) -> _MhMoeProblem:
    """Build production-dimension routed inputs for the fused kernel."""
    torch.manual_seed(seed)
    x = torch.randn(seq, _NUM_HEADS, _D_HEAD, device=device, dtype=_DTYPE)
    w_gate = torch.randn(_NUM_HEADS * _NUM_EXPERTS, _D_HEAD, _D_EXPERT, device=device, dtype=_DTYPE) * 0.02
    w_up = torch.randn(_NUM_HEADS * _NUM_EXPERTS, _D_HEAD, _D_EXPERT, device=device, dtype=_DTYPE) * 0.02
    w_down = torch.randn(_NUM_HEADS * _NUM_EXPERTS, _D_EXPERT, _D_HEAD, device=device, dtype=_DTYPE) * 0.02
    logits = torch.randn(_NUM_HEADS, seq, _NUM_EXPERTS, device=device) * 0.5
    probs, indices = compute_topk_probs_and_indices(logits, _TOP_K)
    probs = probs * 4.9  # released routing scale
    gather_ids, sorted_probs, offsets = global_sort_routes(probs, indices, _NUM_EXPERTS)
    return _MhMoeProblem(x, gather_ids, sorted_probs, offsets, w_gate, w_up, w_down)


def test_mh_moe_production_dim_parity() -> None:
    device = torch.device("cuda:0")
    problem = _make_problem(1024, device)
    reference = torch_mh_moe_forward(
        problem.x,
        problem.gather_ids,
        problem.probs,
        problem.offsets,
        problem.w_gate,
        problem.w_up,
        problem.w_down,
    )
    for deterministic in (True, False):
        out = triton_mh_moe_forward(
            problem.x,
            problem.gather_ids,
            problem.probs,
            problem.offsets,
            problem.w_gate,
            problem.w_up,
            problem.w_down,
            deterministic=deterministic,
        )
        torch.testing.assert_close(out, reference, atol=5e-2, rtol=5e-2)


def test_mh_moe_deterministic_path_is_bitwise_reproducible() -> None:
    device = torch.device("cuda:0")
    problem = _make_problem(1024, device)
    outputs = [
        triton_mh_moe_forward(
            problem.x,
            problem.gather_ids,
            problem.probs,
            problem.offsets,
            problem.w_gate,
            problem.w_up,
            problem.w_down,
            deterministic=True,
        )
        for _ in range(3)
    ]
    torch.accelerator.synchronize()
    for later in outputs[1:]:
        torch.testing.assert_close(outputs[0], later, rtol=0.0, atol=0.0)
