# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Equivalence test for DreamZero self-attn QKV fusion.

CausalWanSelfAttention now projects q/k/v with a single QKVParallelLinear and
splits the result, instead of three separate ColumnParallelLinear. The DreamZero
checkpoint still ships separate q/k/v weights, so load_weights routes each into
the packed `qkv` param with its shard id. This test proves the chain
"load separate q/k/v via shard id -> single GEMM -> split" reproduces the three
independent projections, given identical weights.

CPU-only: a single-rank (world_size=1) tensor-parallel group is initialized and
the GEMM dispatcher is forced to the default (CPU-compatible) path, following
tests/diffusion/layers/test_adalayernorm.py.
"""

import os

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _init_distributed():
    """Minimal world_size=1 TP group required to build QKVParallelLinear."""
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29507")
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="env://",
    )
    initialize_model_parallel()
    yield
    cleanup_dist_env_and_memory()


@pytest.fixture(autouse=True)
def _force_default_gemm(monkeypatch):
    """Force CPU-compatible GEMM dispatch for CPU test tensors."""
    from vllm.model_executor.layers.utils import default_unquantized_gemm

    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.dispatch_unquantized_gemm",
        lambda: default_unquantized_gemm,
    )


def test_fused_qkv_load_and_split_matches_separate_projections():
    from vllm.model_executor.layers.linear import QKVParallelLinear

    torch.manual_seed(0)
    dim, head_dim, num_heads = 16, 4, 4  # inner_dim == dim, no GQA

    qkv = QKVParallelLinear(
        hidden_size=dim,
        head_size=head_dim,
        total_num_heads=num_heads,
        bias=True,
    )

    # Distinct random weight + bias per projection, loaded the exact way
    # DreamZero.load_weights does: param.weight_loader(param, tensor, shard_id).
    refs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for shard_id in ("q", "k", "v"):
        w = torch.randn(dim, dim)
        b = torch.randn(dim)
        refs[shard_id] = (w, b)
        qkv.weight.weight_loader(qkv.weight, w, shard_id)
        qkv.bias.weight_loader(qkv.bias, b, shard_id)

    x = torch.randn(2, 3, dim)
    fused, _ = qkv(x)
    q, k, v = fused.split([dim, dim, dim], dim=-1)

    for got, shard_id in zip((q, k, v), ("q", "k", "v"), strict=True):
        w, b = refs[shard_id]
        ref = torch.nn.functional.linear(x, w, b)
        torch.testing.assert_close(got, ref)


def test_fused_qkv_shard_ids_are_not_transposed():
    # Guard against q/k/v being routed to the wrong slice: give each projection a
    # constant, distinct weight so a swapped shard id would surface immediately.
    from vllm.model_executor.layers.linear import QKVParallelLinear

    dim, head_dim, num_heads = 16, 4, 4
    qkv = QKVParallelLinear(
        hidden_size=dim,
        head_size=head_dim,
        total_num_heads=num_heads,
        bias=True,
    )

    scale = {"q": 1.0, "k": 2.0, "v": 3.0}
    for shard_id, s in scale.items():
        qkv.weight.weight_loader(qkv.weight, torch.full((dim, dim), s), shard_id)
        qkv.bias.weight_loader(qkv.bias, torch.zeros(dim), shard_id)

    x = torch.ones(1, dim)
    fused, _ = qkv(x)
    q, k, v = fused.split([dim, dim, dim], dim=-1)

    # F.linear(ones, full(s)) == s * dim per element.
    torch.testing.assert_close(q, torch.full((1, dim), scale["q"] * dim))
    torch.testing.assert_close(k, torch.full((1, dim), scale["k"] * dim))
    torch.testing.assert_close(v, torch.full((1, dim), scale["v"] * dim))
