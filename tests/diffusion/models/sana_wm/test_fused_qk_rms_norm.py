# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Equivalence tests for SANA-WM's TensorParallelRMSNorm q/k fusion.

`fused_qk_rms_norm` packs the q and k per-token sum-of-squares into one tensor
and issues a single tensor-parallel all-reduce instead of two. These tests lock
in that it is numerically identical to applying `norm_q`/`norm_k` separately,
and that it really issues exactly one (fused) all-reduce.

CPU-only: the collective is mocked, so a single process simulates one rank
(all-reduce is identity). This exercises the cat/split logic a real multi-rank
run depends on.
"""

from unittest.mock import patch

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.layers.norm import RMSNorm
from vllm_omni.diffusion.models.sana_wm.sana_wm_transformer import (
    TensorParallelRMSNorm,
    fused_qk_rms_norm,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_MODULE = "vllm_omni.diffusion.models.sana_wm.sana_wm_transformer"


def _make_norms(dim: int, tp_size: int) -> tuple[TensorParallelRMSNorm, TensorParallelRMSNorm]:
    """Two norms with distinct random weights so a swapped q/k slice is caught."""
    torch.manual_seed(0)
    norm_q = TensorParallelRMSNorm(dim, tp_size=tp_size)
    norm_k = TensorParallelRMSNorm(dim, tp_size=tp_size)
    norm_q.weight.data.normal_()
    norm_k.weight.data.normal_()
    return norm_q, norm_k


def _identity_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    """Single-rank all-reduce: the reduction over one rank is the identity."""
    return tensor


def test_fused_matches_separate() -> None:
    dim = 64
    norm_q, norm_k = _make_norms(dim, tp_size=2)
    q = torch.randn(2, 5, dim)
    k = torch.randn(2, 5, dim)

    with patch(f"{_MODULE}.tensor_model_parallel_all_reduce", side_effect=_identity_all_reduce):
        ref_q, ref_k = norm_q(q), norm_k(k)
        fused_q, fused_k = fused_qk_rms_norm(norm_q, norm_k, q, k)

    torch.testing.assert_close(fused_q, ref_q, rtol=0, atol=0)
    torch.testing.assert_close(fused_k, ref_k, rtol=0, atol=0)


def test_fused_issues_one_all_reduce() -> None:
    dim = 32
    norm_q, norm_k = _make_norms(dim, tp_size=4)
    q = torch.randn(1, 3, dim)
    k = torch.randn(1, 3, dim)

    with patch(f"{_MODULE}.tensor_model_parallel_all_reduce", side_effect=_identity_all_reduce) as collective:
        fused_qk_rms_norm(norm_q, norm_k, q, k)
        assert collective.call_count == 1
        # Both sums travel in one tensor: last dim is 2, not 1.
        assert collective.call_args.args[0].shape[-1] == 2

        collective.reset_mock()
        norm_q(q), norm_k(k)
        assert collective.call_count == 2


@pytest.mark.parametrize(
    "norms",
    [
        (RMSNorm(16), RMSNorm(16)),
        (nn.Identity(), nn.Identity()),
    ],
    ids=["plain_rms_norm", "identity"],
)
def test_falls_back_for_non_tp_norms(norms: tuple[nn.Module, nn.Module]) -> None:
    """qk_norm=False and TP=1 both hand in norms the fused path cannot use."""
    norm_q, norm_k = norms
    q = torch.randn(2, 4, 16)
    k = torch.randn(2, 4, 16)

    with patch(f"{_MODULE}.tensor_model_parallel_all_reduce", side_effect=_identity_all_reduce) as collective:
        fused_q, fused_k = fused_qk_rms_norm(norm_q, norm_k, q, k)
        assert collective.call_count == 0

    torch.testing.assert_close(fused_q, norm_q(q), rtol=0, atol=0)
    torch.testing.assert_close(fused_k, norm_k(k), rtol=0, atol=0)


def test_tp1_skips_the_collective() -> None:
    norm_q, norm_k = _make_norms(16, tp_size=1)
    q = torch.randn(2, 4, 16)
    k = torch.randn(2, 4, 16)

    with patch(f"{_MODULE}.tensor_model_parallel_all_reduce", side_effect=_identity_all_reduce) as collective:
        fused_q, fused_k = fused_qk_rms_norm(norm_q, norm_k, q, k)
        assert collective.call_count == 0

    torch.testing.assert_close(fused_q, norm_q(q), rtol=0, atol=0)
    torch.testing.assert_close(fused_k, norm_k(k), rtol=0, atol=0)
