# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SP subgroup construction in parallel_state.py."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.distributed.parallel_state import (
    RankGenerator,
    build_ulysses_allgather_rank_groups,
    set_seq_parallel_pg,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.parallel, pytest.mark.core_model, pytest.mark.cpu]


def _fake_new_group_factory(created_groups: list[SimpleNamespace]):
    def _fake_new_group(ranks, *args, **kwargs):
        group = SimpleNamespace(ranks=list(ranks))
        created_groups.append(group)
        return group

    return _fake_new_group


@pytest.mark.cpu
@pytest.mark.parametrize(
    "rank, expected_ulysses, expected_ring",
    [
        (0, [0, 2], [0]),
        (1, [1, 3], [1]),
        (2, [0, 2], [2]),
        (3, [1, 3], [3]),
    ],
)
def test_set_seq_parallel_pg_uses_explicit_sp_groups(rank, expected_ulysses, expected_ring, monkeypatch):
    created_groups: list[SimpleNamespace] = []
    monkeypatch.setattr(torch.distributed, "new_group", _fake_new_group_factory(created_groups))

    # tp=2, sp=2 -> SP groups are non-contiguous: [0,2] and [1,3]
    sp_group_ranks = RankGenerator(2, 2, 1, 1, 1, "tp-sp-pp-cfg-dp").get_ranks("sp")

    ulysses_pg, ring_pg, allgather_pg = set_seq_parallel_pg(
        sp_ulysses_degree=2,
        sp_ring_degree=1,
        rank=rank,
        world_size=4,
        sp_group_ranks=sp_group_ranks,
    )

    assert ulysses_pg.ranks == expected_ulysses
    assert ring_pg.ranks == expected_ring
    assert allgather_pg.ranks == [rank]


@pytest.mark.cpu
@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_set_seq_parallel_pg_builds_independent_allgather_group(rank, monkeypatch):
    created_groups: list[SimpleNamespace] = []
    monkeypatch.setattr(torch.distributed, "new_group", _fake_new_group_factory(created_groups))
    sp_group_ranks = [[0, 2], [1, 3]]

    ulysses_pg, ring_pg, allgather_pg = set_seq_parallel_pg(
        sp_ulysses_degree=1,
        sp_ring_degree=1,
        sp_allgather_degree=2,
        rank=rank,
        world_size=4,
        sp_group_ranks=sp_group_ranks,
    )

    assert ulysses_pg.ranks == [rank]
    assert ring_pg.ranks == [rank]
    assert allgather_pg.ranks == ([0, 2] if rank % 2 == 0 else [1, 3])


@pytest.mark.cpu
@pytest.mark.parametrize(
    "rank, expected_ulysses, expected_allgather, expected_ring",
    [
        (0, [0, 1], [0, 2], [0]),
        (1, [0, 1], [1, 3], [1]),
        (2, [2, 3], [0, 2], [2]),
        (3, [2, 3], [1, 3], [3]),
    ],
)
def test_set_seq_parallel_pg_builds_orthogonal_ulysses_allgather_groups(
    rank, expected_ulysses, expected_allgather, expected_ring, monkeypatch
):
    """Ulysses x AllGather-KV: contiguous Ulysses blocks + stride-U AllGather groups."""
    created_groups: list[SimpleNamespace] = []
    monkeypatch.setattr(torch.distributed, "new_group", _fake_new_group_factory(created_groups))

    ulysses_pg, ring_pg, allgather_pg = set_seq_parallel_pg(
        sp_ulysses_degree=2,
        sp_ring_degree=1,
        sp_allgather_degree=2,
        rank=rank,
        world_size=4,
        sp_group_ranks=[[0, 1, 2, 3]],
    )

    assert ulysses_pg.ranks == expected_ulysses
    assert allgather_pg.ranks == expected_allgather
    assert ring_pg.ranks == expected_ring


@pytest.mark.cpu
@pytest.mark.parametrize(("ulysses", "allgather"), [(2, 2), (4, 2), (2, 4), (4, 4), (1, 4), (4, 1)])
def test_build_ulysses_allgather_rank_groups_is_orthogonal(ulysses: int, allgather: int):
    """Every Ulysses group must intersect every AllGather group in exactly one rank.

    That orthogonality plus "Ulysses groups are contiguous" is what makes the
    composed transform reproduce the single-rank sequence order.
    """
    sp_size = ulysses * allgather
    sp_group_ranks = [list(range(sp_size))]

    ulysses_groups, allgather_groups = build_ulysses_allgather_rank_groups(sp_group_ranks, ulysses, allgather)

    assert len(ulysses_groups) == allgather
    assert len(allgather_groups) == ulysses
    assert all(len(g) == ulysses for g in ulysses_groups)
    assert all(len(g) == allgather for g in allgather_groups)

    # Orthogonal and covering: each Ulysses group meets each AllGather group in
    # exactly one rank, and the Ulysses groups tile the SP group in order.
    assert sorted(r for g in ulysses_groups for r in g) == list(range(sp_size))
    assert ulysses_groups == [list(range(i * ulysses, (i + 1) * ulysses)) for i in range(allgather)]
    for u_group in ulysses_groups:
        for a_group in allgather_groups:
            assert len(set(u_group) & set(a_group)) == 1

    # allgather_rank is the index inside the AllGather group, and it must match
    # p // U so that concatenating a region per allgather_rank rebuilds the
    # global sequence in order.
    for a_group in allgather_groups:
        for position, p in enumerate(a_group):
            assert p // ulysses == position


@pytest.mark.cpu
@pytest.mark.parametrize("rank", [0, 1, 2, 3, 4, 5, 6, 7])
def test_orthogonal_groups_span_non_contiguous_sp_group(rank, monkeypatch):
    """SP groups need not be contiguous in global rank (tp=2, sp=4 -> [0,2,4,6])."""
    created_groups: list[SimpleNamespace] = []
    monkeypatch.setattr(torch.distributed, "new_group", _fake_new_group_factory(created_groups))

    ulysses_pg, _, allgather_pg = set_seq_parallel_pg(
        sp_ulysses_degree=2,
        sp_ring_degree=1,
        sp_allgather_degree=2,
        rank=rank,
        world_size=8,
        sp_group_ranks=[[0, 2, 4, 6], [1, 3, 5, 7]],
    )

    offset = rank % 2
    local = [offset, offset + 2, offset + 4, offset + 6]
    position = local.index(rank)
    assert ulysses_pg.ranks == local[(position // 2) * 2 : (position // 2 + 1) * 2]
    assert allgather_pg.ranks == local[position % 2 :: 2]


@pytest.mark.cpu
def test_set_seq_parallel_pg_rejects_allgather_composed_with_ring(monkeypatch):
    created_groups: list[SimpleNamespace] = []
    monkeypatch.setattr(torch.distributed, "new_group", _fake_new_group_factory(created_groups))

    with pytest.raises(ValueError, match="cannot be composed with Ring"):
        set_seq_parallel_pg(
            sp_ulysses_degree=2,
            sp_ring_degree=2,
            sp_allgather_degree=2,
            rank=0,
            world_size=8,
        )


@pytest.mark.cpu
def test_set_seq_parallel_pg_validates_sp_group_ranks(monkeypatch):
    created_groups: list[SimpleNamespace] = []
    monkeypatch.setattr(torch.distributed, "new_group", _fake_new_group_factory(created_groups))

    # world_size=4, sp_size=2 -> expect 2 groups, provide 1 to trigger validation
    with pytest.raises(ValueError, match="Invalid sp_group_ranks"):
        set_seq_parallel_pg(
            sp_ulysses_degree=2,
            sp_ring_degree=1,
            rank=0,
            world_size=4,
            sp_group_ranks=[[0, 2]],
        )
