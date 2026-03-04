from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist

from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
    DistributedVaeExecutor,
    DistributedVaeMixin,
    GridSpec,
    TileTask,
)


class E2EOperator:
    """tiles with (2, 3) -- (H,W)"""

    def split(self, z):
        rows_num = 2
        cols_num = 3

        h_size = z.shape[0]
        w_size = z.shape[1]

        tasks = []
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


class DummyMixin(DistributedVaeMixin):
    def __init__(self):
        self.use_tiling = True
        self.distributed_decoder = MagicMock()
        self.distributed_decoder.parallel_size = 2
        self.distributed_decoder.group = None


@pytest.fixture(autouse=True)
def mock_dist():
    with (
        patch.object(dist, "get_world_size", return_value=2),
        patch.object(dist, "get_rank", return_value=0),
        patch.object(dist, "is_initialized", return_value=True),
        patch.object(dist, "gather", return_value=None),
        patch.object(dist, "broadcast", return_value=None),
    ):
        yield


@pytest.fixture(autouse=True)
def mock_dit_group():
    with patch(
        "vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor.get_dit_group",
        new=MagicMock(return_value=None),
    ):
        yield


@pytest.fixture(autouse=True)
def mock_dist_vae_executor():
    with (
        patch.object(DistributedVaeExecutor, "gather_tensors", side_effect=lambda x: [x]),
        patch.object(DistributedVaeExecutor, "broadcast_tensor", side_effect=lambda x: x),
    ):
        yield


# ============================
# Unitest
# ============================


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
