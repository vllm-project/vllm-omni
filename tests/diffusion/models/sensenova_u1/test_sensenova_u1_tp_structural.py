# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the parallel layers used by SenseNova-U1."""

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers import linear, vocab_parallel_embedding
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def tp(monkeypatch):
    def configure(size: int, rank: int = 0):
        monkeypatch.setattr(
            "vllm.distributed.parallel_state.get_tp_group",
            lambda: SimpleNamespace(rank_in_group=rank, world_size=size),
        )
        monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: size)
        monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: rank)
        monkeypatch.setattr(vocab_parallel_embedding, "get_tensor_model_parallel_world_size", lambda: size)
        monkeypatch.setattr(vocab_parallel_embedding, "get_tensor_model_parallel_rank", lambda: rank)

    return configure


@pytest.mark.parametrize(
    "size,expected_shape,kv_replicas",
    [
        (1, (192, 128), 1),
        (2, (96, 128), 1),
        (4, (48, 128), 1),
        (8, (24, 128), 1),
        (16, (16, 128), 2),
        (32, (12, 128), 4),
    ],
)
def test_qkv_partition_uses_the_real_parallel_layer(tp, size, expected_shape, kv_replicas):
    tp(size)
    qkv = QKVParallelLinear(128, 4, 32, 8, bias=False)

    assert qkv.weight.shape == expected_shape
    assert qkv.num_heads == 32 // size
    assert qkv.num_kv_heads == max(8 // size, 1)
    assert qkv.num_kv_head_replicas == kv_replicas


@pytest.mark.parametrize("size", [1, 2, 4, 8, 16])
def test_qkv_weight_loader_keeps_q_k_v_in_distinct_slices(tp, size):
    tp(size)
    qkv = QKVParallelLinear(128, 4, 32, 8, bias=False)
    for shard_id, rows, value in (("q", 128, 1), ("k", 32, 2), ("v", 32, 3)):
        qkv.weight.weight_loader(qkv.weight, torch.full((rows, 128), value), shard_id)

    q_rows = 128 // size
    kv_rows = max(32 // size, 4)
    for shard, value in zip(qkv.weight.split((q_rows, kv_rows, kv_rows)), (1, 2, 3), strict=True):
        torch.testing.assert_close(shard, torch.full_like(shard, value))


def test_qkv_weight_loader_selects_rank_one_rows(tp):
    tp(2, rank=1)
    qkv = QKVParallelLinear(128, 4, 32, 8, bias=False)
    for shard_id, rows in (("q", 128), ("k", 32), ("v", 32)):
        source = torch.arange(rows, dtype=torch.float32).unsqueeze(1).expand(rows, 128)
        qkv.weight.weight_loader(qkv.weight, source, shard_id)

    q, k, v = qkv.weight.split((64, 16, 16))
    torch.testing.assert_close(q[:, 0], torch.arange(64, 128, dtype=torch.float32))
    for shard in (k, v):
        torch.testing.assert_close(shard[:, 0], torch.arange(16, 32, dtype=torch.float32))


@pytest.mark.parametrize("size", [1, 2, 4, 8])
def test_mlp_projection_and_vocab_shapes_come_from_vllm(tp, size):
    tp(size)
    gate_up = MergedColumnParallelLinear(128, [64, 64], bias=False)
    down = RowParallelLinear(64, 128, bias=False, input_is_parallel=True)
    embedding = VocabParallelEmbedding(70, 16)
    head = ParallelLMHead(70, 16)

    assert gate_up.weight.shape == (128 // size, 128)
    assert down.weight.shape == (128, 64 // size)
    assert embedding.weight.shape == (128 // size, 16)
    assert head.weight.shape == (128 // size, 16)


@pytest.mark.parametrize("size", [3, 64])
def test_qkv_rejects_degrees_that_cannot_partition_query_heads(tp, size):
    tp(size)
    with pytest.raises(AssertionError):
        QKVParallelLinear(128, 4, 32, 8, bias=False)
