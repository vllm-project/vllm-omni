# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from datetime import timedelta
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm_omni.diffusion.distributed.head_parallel import (
    HeadParallelLayout,
    scatter_heads_gather_tokens,
    scatter_tokens_gather_heads,
)
from vllm_omni.diffusion.models.magi2.parallel import Magi2ParallelGroup, ep_dispatch, ep_undispatch

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


@pytest.mark.cpu
@pytest.mark.parametrize(
    "counts,rank",
    [((), 0), ([1, 2], 0), ((1, -1), 0), ((1, 2.5), 0), ((True, 1), 0), ((1, 2), -1), ((1, 2), 2), ((1,), True)],
)
def test_invalid_layout(counts, rank):
    with pytest.raises(ValueError):
        HeadParallelLayout(counts, rank)


@pytest.mark.cpu
@pytest.mark.parametrize("tokens", [0, 3])
def test_single_rank_identity(tokens):
    layout = HeadParallelLayout((tokens,), 0)
    tensor = torch.empty(tokens, 4, 8)
    assert scatter_heads_gather_tokens(tensor, None, layout) is tensor
    assert scatter_tokens_gather_heads(tensor, None, layout) is tensor
    assert layout.local_tokens == layout.total_tokens == tokens


@pytest.mark.cpu
def test_multi_rank_requires_owned_group():
    with pytest.raises(ValueError, match="caller-owned process group"):
        scatter_heads_gather_tokens(torch.empty(1, 2, 4), None, HeadParallelLayout((1, 2), 0))


def _mock_group(monkeypatch, size=2, rank=0):
    group = object()
    monkeypatch.setattr(dist, "get_world_size", lambda passed: size if passed is group else -1)
    monkeypatch.setattr(dist, "get_rank", lambda passed: rank if passed is group else -1)
    return group


@pytest.mark.cpu
@pytest.mark.parametrize("size,rank", [(3, 0), (2, 1)])
def test_group_layout_mismatch_fails_before_collective(monkeypatch, size, rank):
    group = _mock_group(monkeypatch, size, rank)
    collective = Mock()
    monkeypatch.setattr(dist, "all_to_all_single", collective)
    with pytest.raises(ValueError, match="process group size/rank"):
        scatter_heads_gather_tokens(torch.empty(1, 4, 8), group, HeadParallelLayout((1, 2), 0))
    collective.assert_not_called()


@pytest.mark.cpu
@pytest.mark.parametrize(
    "reverse,shape",
    [
        (False, (1, 4)),
        (False, (2, 4, 8)),
        (False, (1, 3, 8)),
        (False, (1, 0, 8)),
        (False, (1, 4, 0)),
        (True, (3, 2)),
        (True, (2, 2, 8)),
        (True, (3, 0, 8)),
        (True, (3, 2, 0)),
    ],
)
def test_invalid_shape_fails_before_collective(monkeypatch, reverse, shape):
    group = _mock_group(monkeypatch)
    collective = Mock()
    monkeypatch.setattr(dist, "all_to_all_single", collective)
    exchange = scatter_tokens_gather_heads if reverse else scatter_heads_gather_tokens
    with pytest.raises(ValueError):
        exchange(torch.empty(shape), group, HeadParallelLayout((1, 2), 0))
    collective.assert_not_called()


@pytest.mark.cpu
def test_explicit_layout_uses_token_counts_not_route_counts(monkeypatch):
    group = _mock_group(monkeypatch)
    collective = Mock()
    monkeypatch.setattr(dist, "all_to_all_single", collective)
    monkeypatch.setattr(dist, "all_gather", Mock(side_effect=AssertionError("no metadata collective expected")))
    tensor = torch.empty(2, 6, 4)
    layout = HeadParallelLayout((2, 5), 0)
    dispatched = scatter_heads_gather_tokens(tensor, group, layout)
    assert dispatched.shape == (7, 3, 4)
    assert collective.call_args.kwargs == {
        "output_split_sizes": [24, 60],
        "input_split_sizes": [24, 24],
        "group": group,
    }
    restored = scatter_tokens_gather_heads(dispatched, group, layout)
    assert restored.shape == tensor.shape
    assert collective.call_args.kwargs == {
        "output_split_sizes": [24, 24],
        "input_split_sizes": [24, 60],
        "group": group,
    }


def _rank_tensor(rank: int, tokens: int, heads: int, dim: int) -> torch.Tensor:
    # Noncontiguous packed view; values encode source rank and element position.
    return (
        torch.arange(tokens * heads * dim * 2, dtype=torch.float32).reshape(tokens, heads, dim * 2)[..., ::2]
        + rank * 10000
    )


def _check_group(group: dist.ProcessGroup, ranks: tuple[int, ...], counts: tuple[int, ...], device: str) -> None:
    rank = dist.get_rank(group)
    layout = HeadParallelLayout(counts, rank)
    heads, dim = 12, 4
    inputs = [_rank_tensor(source, count, heads, dim) for source, count in zip(ranks, counts)]
    # Preserve a noncontiguous layout after copying to the accelerator.
    local = inputs[rank].to(device).repeat_interleave(2, dim=-1)[..., ::2]
    local_heads = heads // len(ranks)
    dispatched = scatter_heads_gather_tokens(local, group, layout)
    expected = torch.cat(inputs)[:, rank * local_heads : (rank + 1) * local_heads]
    torch.testing.assert_close(dispatched.cpu(), expected, rtol=0, atol=0)

    # A head-owner-specific transformation catches wrong reverse permutations.
    transformed = dispatched * 2 + rank
    restored = scatter_tokens_gather_heads(transformed, group, layout)
    offsets = torch.arange(len(ranks)).repeat_interleave(local_heads).view(1, heads, 1)
    torch.testing.assert_close(restored.cpu(), inputs[rank] * 2 + offsets, rtol=0, atol=0)

    magi_group = Magi2ParallelGroup(group, len(ranks), rank)
    through_model = ep_dispatch(local, magi_group, list(counts))
    torch.testing.assert_close(through_model.cpu(), expected, rtol=0, atol=0)
    roundtrip = ep_undispatch(through_model, magi_group, list(counts))
    torch.testing.assert_close(roundtrip.cpu(), inputs[rank], rtol=0, atol=0)
    if len(set(counts)) == 1 and counts[0] > 0:
        inferred = ep_dispatch(local, magi_group)
        torch.testing.assert_close(ep_undispatch(inferred, magi_group).cpu(), inputs[rank], rtol=0, atol=0)


def _worker(rank: int, rendezvous: str, world_size: int, backend: str, device_type: str) -> None:
    torch.set_num_threads(1)
    if device_type != "cpu":
        torch.accelerator.set_device_index(rank)
    device = "cpu" if device_type == "cpu" else f"{device_type}:{rank}"
    dist.init_process_group(
        backend, init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=60)
    )
    try:
        ranks = tuple(range(world_size))
        for counts in ((2,) * world_size, tuple(range(world_size)), (0,) * world_size):
            _check_group(dist.group.WORLD, ranks, counts, device)
        if world_size == 4:
            for rank_sets in (((0, 1), (2, 3)), ((0, 2), (1, 3))):
                for members in rank_sets:
                    group = dist.new_group(ranks=list(members), backend=backend, timeout=timedelta(seconds=60))
                    if rank in members:
                        for counts in ((2, 3), (0, 3), (2, 0), (0, 0)):
                            _check_group(group, members, counts, device)
                        dist.destroy_process_group(group)
                dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.cpu
@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="requires Gloo")
def test_four_rank_exchange_and_replica_subgroups(tmp_path):
    mp.spawn(_worker, args=(f"file://{tmp_path / 'gloo-init'}", 4, "gloo", "cpu"), nprocs=4, join=True)


@pytest.mark.musa
def test_two_rank_musa_exchange(tmp_path):
    if not hasattr(torch, "musa") or not torch.musa.is_available() or torch.musa.device_count() < 2:
        pytest.skip("requires two MUSA devices")
    mp.spawn(_worker, args=(f"file://{tmp_path / 'mccl-init'}", 2, "mccl", "musa"), nprocs=2, join=True)
