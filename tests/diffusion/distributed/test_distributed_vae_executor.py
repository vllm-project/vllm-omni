# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import weakref
from contextlib import contextmanager
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist

from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
    DistributedVaeExecutor,
    DistributedVaeMixin,
    GridSpec,
    TileTask,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class E2EOperator:
    """tiles with (2, 3) -- (H,W)"""

    def split(self, z):
        rows_num = 2
        cols_num = 3

        h_size = z.shape[0]
        w_size = z.shape[1]

        tasks: list[TileTask] = []
        for i in range(rows_num):
            for j in range(cols_num):
                tasks.append(
                    TileTask(
                        tile_id=len(tasks),
                        grid_coord=(i, j),
                        tensor=z[
                            ((i * h_size) // rows_num) : (((i + 1) * h_size) // rows_num),
                            ((j * w_size) // cols_num) : (((j + 1) * w_size) // cols_num),
                        ],
                    )
                )

        grid_spec = GridSpec(
            split_dims=(0, 1),
            grid_shape=(2, 3),
        )
        return tasks, grid_spec

    def exec(self, task: TileTask):
        return torch.full_like(task.tensor, fill_value=task.tile_id)

    def merge(self, coord_tensor_map, grid_spec):
        tiles = []
        for r in range(grid_spec.grid_shape[0]):
            row_tiles = []
            for c in range(grid_spec.grid_shape[1]):
                coord = (r, c)
                row_tiles.append(coord_tensor_map[coord])
            tiles.append(torch.cat(row_tiles, dim=1))
        return torch.cat(tiles, dim=0)


@dataclass
class FakeWorldGroup:
    device_group: object
    cpu_group: object = None


class DummyMixin(DistributedVaeMixin):
    def __init__(self):
        self.use_tiling = True
        self.distributed_executor = DistributedVaeExecutor()
        self.distributed_executor.set_parallel_size(2)


@pytest.fixture(autouse=True)
def mock_dist(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(dist, "get_backend", lambda *args, **kwargs: "nccl")
    monkeypatch.setattr(dist, "get_world_size", lambda *args, **kwargs: 2)
    monkeypatch.setattr(dist, "get_rank", lambda *args, **kwargs: 0)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(dist, "gather", lambda *args, **kwargs: None)
    monkeypatch.setattr(dist, "broadcast", lambda *args, **kwargs: None)


@pytest.fixture(autouse=True)
def mock_world_group(monkeypatch: pytest.MonkeyPatch):
    group = object()
    monkeypatch.setattr(
        "vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor.get_world_group",
        lambda: FakeWorldGroup(device_group=group),
    )
    return group


@pytest.fixture(autouse=True)
def mock_dist_vae_executor(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(DistributedVaeExecutor, "broadcast_tensor", lambda self, x: x)


# ============================
# Unitest
# ============================


def test_uses_dedicated_world_device_group(mock_world_group):
    executor = DistributedVaeExecutor()
    assert executor.group is mock_world_group


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("backend", ["nccl", "gloo", "hccl"])
def test_gather_receive_allocation_policy(monkeypatch, rank: int, backend: str):
    executor = DistributedVaeExecutor()
    executor.rank = rank
    monkeypatch.setattr(dist, "get_backend", lambda group: backend)
    tensor = torch.arange(4, dtype=torch.float32)
    allocations = []
    empty_like = torch.empty_like
    guard_active = False
    sync_errors = executor._sync_errors

    @contextmanager
    def allocation_guard(stage):
        nonlocal guard_active
        with sync_errors(stage):
            guard_active = True
            yield
            guard_active = False

    def allocate(source):
        assert guard_active, "receive allocation must participate in error agreement"
        allocations.append(source)
        return empty_like(source)

    def gather(source, *, gather_list, dst, group):
        assert backend != "hccl" and not guard_active
        assert source is tensor and dst == 0 and group is executor.group
        if rank == 0:
            assert len(gather_list) == executor.world_size
            for peer, result in enumerate(gather_list):
                result.copy_(source + peer)
        else:
            assert gather_list is None

    def all_gather(gather_list, source, *, group):
        assert backend == "hccl" and not guard_active
        assert source is tensor and group is executor.group
        assert len(gather_list) == executor.world_size
        for peer, result in enumerate(gather_list):
            result.copy_(source + peer)

    monkeypatch.setattr(executor, "_sync_errors", allocation_guard)
    monkeypatch.setattr(torch, "empty_like", allocate)
    monkeypatch.setattr(dist, "gather", gather)
    monkeypatch.setattr(dist, "all_gather", all_gather)
    result = executor.gather_tensors(tensor)
    assert len(allocations) == (executor.world_size if rank == 0 or backend == "hccl" else 0)
    if rank == 0:
        for peer, output in enumerate(result):
            torch.testing.assert_close(output, tensor + peer)
            assert output.data_ptr() != tensor.data_ptr()
    else:
        assert result is None


@pytest.mark.parametrize("rank", [0, 1])
def test_hccl_receive_oom_is_reported_before_collective(monkeypatch, rank: int):
    executor = DistributedVaeExecutor()
    executor.rank = rank
    monkeypatch.setattr(dist, "get_backend", lambda group: "hccl")

    def fail_allocation(source):
        raise torch.OutOfMemoryError("injected receive allocation failure")

    def unexpected_collective(*args, **kwargs):
        pytest.fail("data collective must not run after receive allocation failure")

    monkeypatch.setattr(torch, "empty_like", fail_allocation)
    monkeypatch.setattr(dist, "gather", unexpected_collective)
    monkeypatch.setattr(dist, "all_gather", unexpected_collective)
    with pytest.raises(torch.OutOfMemoryError, match="Distributed VAE gather allocation failed"):
        executor.gather_tensors(torch.zeros(4))


def test_balance_tasks():
    executor = DistributedVaeExecutor()
    workloads = [2, 5, 13, 8, 2, 4]  # each is 17
    tasks = [TileTask(0, (i,), torch.tensor([i]), workload=load) for i, load in enumerate(workloads)]
    assigned = executor._balance_tasks(tasks, num_rank=2)
    assert len(assigned) == 2
    total_work = [sum(t.workload for t in group) for group in assigned]
    assert total_work[0] == total_work[1]


def test_compute_global_padding_shape():
    executor = DistributedVaeExecutor()
    executor.rank = 0

    local_results = [(0, torch.zeros((2, 3))), (1, torch.zeros((4, 2)))]
    shape = executor._compute_global_padding_shape(local_results, 2, "cpu")

    assert shape == [2, 4, 3]


def test_pack_and_unpack():
    executor = DistributedVaeExecutor()
    executor.world_size = 1

    grid_spec = GridSpec(split_dims=(0, 1), grid_shape=(2, 2))

    # ======================
    # pack
    # ======================
    local_results = [(0, torch.tensor([[1, 2], [3, 4]]))]

    global_shape = [3, 3, 3]  # (tiles, H, W)

    tile_tensor, meta_tensor = executor._pack_local_tiles(
        local_results, global_shape, grid_spec, device="cpu", dtype=torch.int64
    )

    # check pack
    assert tile_tensor.shape == torch.Size(global_shape)
    assert meta_tensor.shape == (global_shape[0], len(grid_spec.split_dims) + 1)
    assert meta_tensor[0, 0] == 0
    assert meta_tensor[0, 1] == 2
    assert meta_tensor[0, 2] == 2

    # ======================
    # unpack
    # ======================
    meta_gather = [meta_tensor]
    tile_gather = [tile_tensor]

    tid_coord_map = {0: (0, 0)}

    coord_tensor_map = executor._unpack_tiles(meta_gather, tile_gather, grid_spec, tid_coord_map)

    # check unpack
    assert torch.equal(coord_tensor_map[(0, 0)], torch.tensor([[1, 2], [3, 4]]))


def test_is_distributed_enabled():
    mixin = DummyMixin()
    assert mixin.is_distributed_enabled() is True

    mixin.use_tiling = False
    assert mixin.is_distributed_enabled() is False


@pytest.mark.parametrize("boundary", ["gather", "merge"])
@pytest.mark.parametrize("return_view", [False, True])
def test_execute_releases_copied_buffers(monkeypatch, boundary: str, return_view: bool):
    """Copied tile/send buffers must not overlap the next allocation phase."""
    executor = DistributedVaeExecutor()
    executor.world_size = 1
    decoded_refs = []
    packed_ref = None

    class Operator(E2EOperator):
        def exec(self, task):
            output = super().exec(task)
            decoded_refs.append(weakref.ref(output))
            return output

        def merge(self, coord_tensor_map, grid_spec):
            if boundary == "merge":
                assert packed_ref is not None and packed_ref() is None
            if return_view:
                # The returned view must retain gathered storage independently.
                return coord_tensor_map[(1, 2)]
            return super().merge(coord_tensor_map, grid_spec)

    def gather(tensor):
        nonlocal packed_ref
        if tensor.dtype == torch.float32:
            if boundary == "gather":
                assert decoded_refs and all(ref() is None for ref in decoded_refs)
            packed_ref = weakref.ref(tensor)
        # Match the collective's separate receive storage, not the aliasing
        # fixture used by tests that do not exercise tensor lifetimes.
        return [tensor.clone()]

    monkeypatch.setattr(executor, "gather_tensors", gather)
    output = executor.execute(torch.zeros(4, 6), Operator(), broadcast_result=False)
    expected = (
        torch.full((2, 2), 5.0)
        if return_view
        else torch.arange(6, dtype=torch.float32).reshape(2, 3).repeat_interleave(2, 0).repeat_interleave(2, 1)
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
