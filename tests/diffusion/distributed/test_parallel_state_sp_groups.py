# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for SP subgroup construction for diffusion parallel state."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import vllm.distributed.parallel_state as vllm_parallel_state

from tests.helpers.runtime import get_distributed_init_method
from vllm_omni.diffusion.distributed import parallel_state as omni_parallel_state
from vllm_omni.diffusion.distributed.parallel_state import RankGenerator, set_seq_parallel_pg

pytestmark = [pytest.mark.diffusion, pytest.mark.parallel, pytest.mark.core_model, pytest.mark.cpu]


def ensure_parallel_state_initialized():
    """Ensure torch distributed is initialized and that all world / count vars are set."""
    assert torch.distributed.is_initialized()
    assert omni_parallel_state._WORLD is not None
    assert vllm_parallel_state._WORLD is not None
    assert vllm_parallel_state._NODE_COUNT is not None


def ensure_parallel_state_not_initialized():
    """Ensure torch distributed is not initialized and that all world / count vars are unset."""
    assert not torch.distributed.is_initialized()
    assert omni_parallel_state._WORLD is None
    assert vllm_parallel_state._WORLD is None
    assert vllm_parallel_state._NODE_COUNT is None


def test_omni_manages_vllm_distributed_state(monkeypatch):
    """Verify Omni initializes and tears down vLLM's distributed state properly.

    This is a regression test for ensuring vLLM Omni also sets up vLLM's vars correctly
    to avoid potential misalignment in cases where Omni uses vLLM's native coordinator,
    e.g., MoE + diffusion.
    """
    ensure_parallel_state_not_initialized()
    monkeypatch.setattr(omni_parallel_state.current_omni_platform, "get_device_count", lambda: 1)
    monkeypatch.setattr(omni_parallel_state.current_omni_platform, "set_device", lambda _device: None)
    omni_parallel_state.init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=get_distributed_init_method(),
        backend="gloo",
    )
    try:
        ensure_parallel_state_initialized()
    finally:
        omni_parallel_state.destroy_distributed_env()
    ensure_parallel_state_not_initialized()


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
def test_set_seq_parallel_pg_rejects_allgather_with_ulysses_or_ring(monkeypatch):
    created_groups: list[SimpleNamespace] = []
    monkeypatch.setattr(torch.distributed, "new_group", _fake_new_group_factory(created_groups))

    with pytest.raises(ValueError, match="mutually exclusive"):
        set_seq_parallel_pg(
            sp_ulysses_degree=2,
            sp_ring_degree=1,
            sp_allgather_degree=2,
            rank=0,
            world_size=4,
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


@pytest.mark.parametrize("rank", [1, 3])
def test_tensor_dict_broadcast_preserves_subgroup_source(rank, monkeypatch):
    from vllm_omni.diffusion.distributed.group_coordinator import GroupCoordinator, TensorMetadata

    coordinator = object.__new__(GroupCoordinator)
    coordinator.rank = rank
    coordinator.ranks = [1, 3]
    coordinator.rank_in_group = [1, 3].index(rank)
    coordinator.world_size = 2
    coordinator.device_group = object()
    coordinator.cpu_group = object()
    metadata = [("payload", TensorMetadata("cpu", torch.float32, (2,)))]
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def broadcast_object(value, src):
        assert src == 0
        return metadata

    def broadcast(tensor, src, group, async_op):
        assert src == 1
        assert group is coordinator.cpu_group
        assert async_op
        tensor.fill_(7)
        return SimpleNamespace(wait=lambda: None)

    monkeypatch.setattr(coordinator, "broadcast_object", broadcast_object)
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    result = coordinator.broadcast_tensor_dict({"payload": torch.full((2,), 7.0)} if rank == 1 else None)
    assert torch.equal(result["payload"], torch.full((2,), 7.0))
