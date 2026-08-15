# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU-only contract tests for independently sized VAE process groups."""

import pytest

from vllm_omni.diffusion.distributed import parallel_state, vae_parallel_state
from vllm_omni.diffusion.distributed.vae_parallel_state import (
    generate_contiguous_rank_groups,
    requires_independent_vae_process_group,
    supports_independent_vae_process_group,
    validate_independent_vae_parallel_config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_independent_group_capability_is_limited_to_h3_adapters():
    assert supports_independent_vae_process_group("MiniMaxH3Pipeline")
    assert supports_independent_vae_process_group("MiniMaxH3ModularPipeline")
    assert not supports_independent_vae_process_group("WanPipeline")


def test_h3_requires_dedicated_group_only_when_smaller_than_dit_group():
    assert not requires_independent_vae_process_group("MiniMaxH3Pipeline", 4, 1)
    assert requires_independent_vae_process_group("MiniMaxH3Pipeline", 4, 2)
    assert not requires_independent_vae_process_group("MiniMaxH3Pipeline", 4, 4)
    assert not requires_independent_vae_process_group("WanPipeline", 4, 2)


def test_independent_h3_group_rejects_spatial_sharding():
    with pytest.raises(ValueError, match="tile mode only"):
        validate_independent_vae_parallel_config(4, 4, 2, "spatial_shard_height")


def test_independent_group_accepts_subgroup_within_dit_world():
    validate_independent_vae_parallel_config(4, 4, 2, "tile")


@pytest.mark.parametrize(
    ("world_size", "dit_group_size", "group_size", "message"),
    [
        (8, 9, 2, "DiT process group size"),
        (8, 4, 2, "equal diffusion world_size"),
        (8, 4, 8, "cannot exceed DiT"),
        (12, 6, 4, "evenly divide DiT"),
    ],
)
def test_independent_group_rejects_invalid_dit_composition(world_size, dit_group_size, group_size, message):
    with pytest.raises(ValueError, match=message):
        validate_independent_vae_parallel_config(world_size, dit_group_size, group_size, "tile")


@pytest.mark.parametrize(
    ("world_size", "group_size", "expected"),
    [
        (1, 1, [[0]]),
        (4, 1, [[0], [1], [2], [3]]),
        (4, 2, [[0, 1], [2, 3]]),
        (8, 4, [[0, 1, 2, 3], [4, 5, 6, 7]]),
        (8, 8, [list(range(8))]),
    ],
)
def test_generate_contiguous_rank_groups_is_deterministic(world_size, group_size, expected):
    assert generate_contiguous_rank_groups(world_size, group_size) == expected


@pytest.mark.parametrize(
    ("world_size", "group_size", "message"),
    [
        (0, 1, "world_size"),
        (4, 0, "greater than 0"),
        (4, 5, "cannot exceed"),
        (8, 3, "evenly divide"),
    ],
)
def test_generate_contiguous_rank_groups_rejects_invalid_sizes(world_size, group_size, message):
    with pytest.raises(ValueError, match=message):
        generate_contiguous_rank_groups(world_size, group_size)


def test_initialize_vae_parallel_group_uses_deterministic_creation_order(monkeypatch):
    local_rank = 2
    created_groups = []

    def fake_new_group(*, ranks, backend):
        group = (tuple(ranks), backend)
        created_groups.append(group)
        return group

    monkeypatch.setattr(vae_parallel_state, "_VAE_GROUP", None)
    monkeypatch.setattr(vae_parallel_state, "_VAE_GROUP_RANKS", None)
    monkeypatch.setattr(vae_parallel_state.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(vae_parallel_state.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(vae_parallel_state.dist, "get_rank", lambda: local_rank)
    monkeypatch.setattr(vae_parallel_state.dist, "new_group", fake_new_group)

    vae_parallel_state.initialize_vae_parallel_group(2, backend="gloo")

    assert created_groups == [((0, 1), "gloo"), ((2, 3), "gloo")]
    assert vae_parallel_state.get_vae_group() == ((2, 3), "gloo")
    assert vae_parallel_state.get_vae_group_ranks() == [2, 3]
    assert vae_parallel_state.get_vae_parallel_world_size() == 2
    assert vae_parallel_state.get_vae_parallel_rank() == 0


def test_initialize_vae_parallel_group_requires_distributed_world(monkeypatch):
    monkeypatch.setattr(vae_parallel_state, "_VAE_GROUP", None)
    monkeypatch.setattr(vae_parallel_state, "_VAE_GROUP_RANKS", None)
    monkeypatch.setattr(vae_parallel_state.dist, "is_initialized", lambda: False)

    with pytest.raises(RuntimeError, match="torch.distributed must be initialized"):
        vae_parallel_state.initialize_vae_parallel_group(2)


def test_initialize_single_rank_vae_state_does_not_create_group(monkeypatch):
    monkeypatch.setattr(vae_parallel_state, "_VAE_GROUP", None)
    monkeypatch.setattr(vae_parallel_state, "_VAE_GROUP_RANKS", None)
    monkeypatch.setattr(vae_parallel_state.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(vae_parallel_state.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(vae_parallel_state.dist, "get_rank", lambda: 3)

    def unexpected_new_group(**kwargs):
        raise AssertionError(f"single-rank VAE must not create a process group: {kwargs}")

    monkeypatch.setattr(vae_parallel_state.dist, "new_group", unexpected_new_group)

    vae_parallel_state.initialize_vae_parallel_group(1)

    assert vae_parallel_state.get_vae_group_ranks() == [3]
    assert vae_parallel_state.get_vae_parallel_world_size() == 1
    assert vae_parallel_state.get_vae_parallel_rank() == 0
    with pytest.raises(RuntimeError, match="greater than 1"):
        vae_parallel_state.get_vae_group()


def test_group_creation_failure_cleans_partial_state(monkeypatch):
    local_group = object()
    calls = 0
    destroyed = []

    def failing_new_group(*, ranks, backend):
        del backend
        nonlocal calls
        calls += 1
        if calls == 1:
            assert ranks == [0, 1]
            return local_group
        raise RuntimeError("synthetic collective creation failure")

    monkeypatch.setattr(vae_parallel_state, "_VAE_GROUP", None)
    monkeypatch.setattr(vae_parallel_state, "_VAE_GROUP_RANKS", None)
    monkeypatch.setattr(vae_parallel_state.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(vae_parallel_state.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(vae_parallel_state.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(vae_parallel_state.dist, "new_group", failing_new_group)
    monkeypatch.setattr(vae_parallel_state.dist, "destroy_process_group", destroyed.append)

    with pytest.raises(RuntimeError, match="synthetic collective creation failure"):
        vae_parallel_state.initialize_vae_parallel_group(2, backend="gloo")

    assert destroyed == [local_group]
    assert vae_parallel_state._VAE_GROUP is None
    assert vae_parallel_state._VAE_GROUP_RANKS is None


def test_duplicate_vae_group_initialization_is_rejected(monkeypatch):
    monkeypatch.setattr(vae_parallel_state, "_VAE_GROUP", object())
    monkeypatch.setattr(vae_parallel_state, "_VAE_GROUP_RANKS", [0, 1])
    monkeypatch.setattr(vae_parallel_state.dist, "is_initialized", lambda: True)

    with pytest.raises(RuntimeError, match="already initialized"):
        vae_parallel_state.initialize_vae_parallel_group(2)


def test_destroy_vae_parallel_group_clears_local_state(monkeypatch):
    group = object()
    destroyed = []
    monkeypatch.setattr(vae_parallel_state, "_VAE_GROUP", group)
    monkeypatch.setattr(vae_parallel_state, "_VAE_GROUP_RANKS", [0, 1])
    monkeypatch.setattr(vae_parallel_state.dist, "destroy_process_group", destroyed.append)

    vae_parallel_state.destroy_vae_parallel_group()

    assert destroyed == [group]
    assert vae_parallel_state._VAE_GROUP is None
    assert vae_parallel_state._VAE_GROUP_RANKS is None


def test_dit_group_override_is_scoped_and_restored(monkeypatch):
    dit_group = object()
    vae_group = object()
    nested_group = object()
    monkeypatch.setattr(parallel_state, "_DIT", dit_group)

    assert parallel_state.get_dit_group() is dit_group
    with parallel_state.override_dit_group(vae_group):
        assert parallel_state.get_dit_group() is vae_group
        with parallel_state.override_dit_group(nested_group):
            assert parallel_state.get_dit_group() is nested_group
        assert parallel_state.get_dit_group() is vae_group
    assert parallel_state.get_dit_group() is dit_group


def test_dit_group_override_is_restored_after_error(monkeypatch):
    dit_group = object()
    vae_group = object()
    monkeypatch.setattr(parallel_state, "_DIT", dit_group)

    with pytest.raises(RuntimeError, match="configuration failed"):
        with parallel_state.override_dit_group(vae_group):
            assert parallel_state.get_dit_group() is vae_group
            raise RuntimeError("configuration failed")

    assert parallel_state.get_dit_group() is dit_group
