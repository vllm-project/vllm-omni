# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Q/K RMSNorm must share one collective without changing the numerics."""

import pytest
import torch

import vllm_omni.diffusion.models.helios.helios_transformer as helios_transformer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _reference(x, weight, eps):
    f = x.float()
    count = x.shape[-1]
    rms = torch.sqrt((f**2).sum(dim=-1, keepdim=True) / count + eps)
    return ((f / rms) * weight.float()).to(x.dtype)


def test_pair_matches_two_separate_norms_on_one_rank(monkeypatch):
    monkeypatch.setattr(helios_transformer, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(helios_transformer, "tensor_model_parallel_all_reduce", lambda tensor: tensor)

    query = torch.randn(1, 5, 16)
    key = torch.randn(1, 5, 16)
    weight_q = torch.randn(16)
    weight_k = torch.randn(16)

    out_q, out_k = helios_transformer.distributed_rms_norm_pair(query, weight_q, key, weight_k, 1e-6)
    assert torch.equal(out_q, _reference(query, weight_q, 1e-6))
    assert torch.equal(out_k, _reference(key, weight_k, 1e-6))


def test_pair_issues_a_single_collective(monkeypatch):
    shapes = []
    monkeypatch.setattr(helios_transformer, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(
        helios_transformer,
        "tensor_model_parallel_all_reduce",
        lambda tensor: shapes.append(tuple(tensor.shape)) or tensor,
    )
    helios_transformer.distributed_rms_norm_pair(
        torch.randn(1, 5, 16), torch.ones(16), torch.randn(1, 5, 16), torch.ones(16), 1e-6
    )
    assert shapes == [(1, 5, 2)]


def test_pair_reproduces_the_global_rms_across_two_ranks(monkeypatch):
    query = torch.randn(1, 3, 8)
    key = torch.randn(1, 3, 8)
    weight = torch.randn(8)
    expected_q = _reference(query, weight, 1e-6)
    expected_k = _reference(key, weight, 1e-6)

    def local_sums(x):
        return (x.float() ** 2).sum(dim=-1, keepdim=True)

    monkeypatch.setattr(helios_transformer, "get_tensor_model_parallel_world_size", lambda: 2)
    for lo in (0, 4):
        hi = lo + 4
        other = lo + 4 if lo == 0 else 0
        other_sums = torch.cat(
            (local_sums(query[..., other : other + 4]), local_sums(key[..., other : other + 4])), dim=-1
        )

        def fake_all_reduce(tensor, _other=other_sums):
            return tensor + _other

        monkeypatch.setattr(helios_transformer, "tensor_model_parallel_all_reduce", fake_all_reduce)
        out_q, out_k = helios_transformer.distributed_rms_norm_pair(
            query[..., lo:hi], weight[lo:hi], key[..., lo:hi], weight[lo:hi], 1e-6
        )
        assert torch.allclose(out_q, expected_q[..., lo:hi], atol=1e-6)
        assert torch.allclose(out_k, expected_k[..., lo:hi], atol=1e-6)
