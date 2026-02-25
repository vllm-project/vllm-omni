# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SP subgroup construction in parallel_state.py."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.distributed.parallel_state import RankGenerator, set_seq_parallel_pg

pytestmark = [
    pytest.mark.diffusion,
    pytest.mark.parallel,
]


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

    ulysses_pg, ring_pg = set_seq_parallel_pg(
        sp_ulysses_degree=2,
        sp_ring_degree=1,
        rank=rank,
        world_size=4,
        sp_group_ranks=sp_group_ranks,
    )

    assert ulysses_pg.ranks == expected_ulysses
    assert ring_pg.ranks == expected_ring


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
