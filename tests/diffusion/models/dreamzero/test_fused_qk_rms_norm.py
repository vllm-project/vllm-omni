# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Equivalence tests for DreamZero's fused q/k qk-norm TP all-reduce.

`fused_qk_rms_norm` packs the q and k per-token sum-of-squares into one tensor
and issues a single tensor-parallel all-reduce instead of two. These tests lock
in that it is numerically identical to applying `norm_q`/`norm_k` separately,
and that it really issues exactly one (fused) all-reduce.

CPU-only: the TP world size and the collective are mocked, so a single process
simulates one rank (all-reduce is identity). This exercises the cat/split and
count-scaling logic that a real multi-rank run depends on.
"""

from unittest.mock import patch

import pytest
import torch

from vllm_omni.diffusion.models.dreamzero.causal_wan_model import (
    DistributedRMSNorm,
    fused_qk_rms_norm,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_MODULE = "vllm_omni.diffusion.models.dreamzero.causal_wan_model"


def _make_norms(dim: int) -> tuple[DistributedRMSNorm, DistributedRMSNorm]:
    """Two norms with distinct random weights so a swapped q/k slice is caught."""
    torch.manual_seed(0)
    norm_q = DistributedRMSNorm(dim)
    norm_k = DistributedRMSNorm(dim)
    norm_q.weight.data.normal_()
    norm_k.weight.data.normal_()
    return norm_q, norm_k


def _identity_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    """Single-rank all-reduce: the reduction over one rank is the identity."""
    return tensor


def test_fused_matches_separate_tp1() -> None:
    dim = 64
    norm_q, norm_k = _make_norms(dim)
    q = torch.randn(2, 5, dim)
    k = torch.randn(2, 5, dim)

    with patch(f"{_MODULE}.get_tensor_model_parallel_world_size", return_value=1):
        ref_q, ref_k = norm_q(q), norm_k(k)
        fused_q, fused_k = fused_qk_rms_norm(norm_q, norm_k, q, k)

    torch.testing.assert_close(fused_q, ref_q, rtol=0, atol=0)
    torch.testing.assert_close(fused_k, ref_k, rtol=0, atol=0)


def test_fused_matches_separate_tp_gt1() -> None:
    dim = 64
    norm_q, norm_k = _make_norms(dim)
    q = torch.randn(2, 5, dim)
    k = torch.randn(2, 5, dim)

    # tp_size=2 exercises the reduce branch and the `global_count *= tp_size`
    # path in both the per-norm forward and the fused helper.
    with (
        patch(f"{_MODULE}.get_tensor_model_parallel_world_size", return_value=2),
        patch(f"{_MODULE}.tensor_model_parallel_all_reduce", side_effect=_identity_all_reduce),
    ):
        ref_q, ref_k = norm_q(q), norm_k(k)
        fused_q, fused_k = fused_qk_rms_norm(norm_q, norm_k, q, k)

    torch.testing.assert_close(fused_q, ref_q, rtol=0, atol=0)
    torch.testing.assert_close(fused_k, ref_k, rtol=0, atol=0)


def test_fused_issues_single_all_reduce_over_packed_pair() -> None:
    dim = 64
    norm_q, norm_k = _make_norms(dim)
    q = torch.randn(2, 5, dim)
    k = torch.randn(2, 5, dim)

    with (
        patch(f"{_MODULE}.get_tensor_model_parallel_world_size", return_value=2),
        patch(
            f"{_MODULE}.tensor_model_parallel_all_reduce",
            side_effect=_identity_all_reduce,
        ) as mock_ar,
    ):
        fused_qk_rms_norm(norm_q, norm_k, q, k)

    # Exactly one collective, over a tensor whose last dim packs the q+k pair.
    assert mock_ar.call_count == 1
    packed = mock_ar.call_args[0][0]
    assert packed.shape[-1] == 2


def test_fused_identity_fallback_when_qk_norm_disabled() -> None:
    # qk_norm=False makes the norms nn.Identity; the helper must fall back to
    # plain passthrough without touching the collective.
    norm_q, norm_k = torch.nn.Identity(), torch.nn.Identity()
    q = torch.randn(2, 5, 64)
    k = torch.randn(2, 5, 64)

    with patch(f"{_MODULE}.tensor_model_parallel_all_reduce") as mock_ar:
        fused_q, fused_k = fused_qk_rms_norm(norm_q, norm_k, q, k)

    torch.testing.assert_close(fused_q, q, rtol=0, atol=0)
    torch.testing.assert_close(fused_k, k, rtol=0, atol=0)
    mock_ar.assert_not_called()
