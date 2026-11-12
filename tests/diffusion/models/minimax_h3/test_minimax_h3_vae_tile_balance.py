# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for MiniMax H3 cross-clip encode tile balancing."""

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]
_PATH = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/minimax_h3/vae_tile_balance.py"
_SPEC = importlib.util.spec_from_file_location("h3_vae_tile_balance_test", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
balance = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(balance)

_REMOTE_GATHER = None


@dataclass(frozen=True)
class Device:
    type: str


class Model:
    def __init__(self):
        self.training = False
        self.encoder = SimpleNamespace()
        self.encoder_tiling = True
        self.encoder_parallel = False
        self.stack_tiling = False
        self.isolated_first_frame = False
        self.isolated_last_frame = False
        self.isolated_key_frame = False
        self.parallel_tiling = False
        self.clip_length = 4
        self.token_drop = 0
        self.vae_ratio = 1
        self.seen = []

    def split_tiles(self, length, is_decoder=False):
        del is_decoder
        if length == 4:
            return [0, 2], [2, 2], [0]
        return [0], [length], []

    def encode(self, tile):
        self.seen.append(tile.clone())
        return tile

    def blend(self, previous, current, overlap, dim):
        del previous, overlap, dim
        return current

    def encode_temporal(self, x):
        return x + 1

    def _all_gather_tiled_results(self, tasks, num_tiles):
        return all_gather_var_shape(torch.stack(tasks), group=num_tiles)


class OverlapModel(Model):
    def __init__(self):
        super().__init__()
        self.token_drop = 1

    def split_tiles(self, length, is_decoder=False):
        del is_decoder
        if length == 5:
            return [0, 2], [3, 3], [1]
        return [0], [length], []

    def encode(self, tile):
        marker = tile[..., :1, :1]
        return tile + marker

    def blend(self, previous, current, overlap, dim):
        result = current.clone()
        if dim == -2:
            result[..., :overlap, :] = (previous[..., -overlap:, :] + current[..., :overlap, :]) / 2
        else:
            result[..., :, :overlap] = (previous[..., :, -overlap:] + current[..., :, :overlap]) / 2
        return result


def all_gather_var_shape(stacked, group=None):
    del group
    if _REMOTE_GATHER is None:
        return [stacked]
    return [stacked, _REMOTE_GATHER]


def test_config_is_disabled_by_default_and_validates_positive_integer():
    assert balance.resolve_video_tile_balance_max_tasks(None) is None
    assert balance.resolve_video_tile_balance_max_tasks({}) is None
    assert balance.resolve_video_tile_balance_max_tasks({balance._CONFIG_KEY: 96}) == 96
    for value in (0, -1, 1.5, True, "96"):
        with pytest.raises(ValueError, match="positive integer"):
            balance.resolve_video_tile_balance_max_tasks({balance._CONFIG_KEY: value})


def test_windows_rotate_remainders_and_cover_each_task_once():
    windows = balance._windows(chunks=3, tiles=10, max_tasks=10)
    assert [window["task_start"] % 8 for window in windows] == [0, 2, 4]
    assert [
        [len(balance._task_ids(window["task_start"], window["task_end"], rank, 8)) for rank in range(8)]
        for window in windows
    ] == [
        [2, 2, 1, 1, 1, 1, 1, 1],
        [1, 1, 2, 2, 1, 1, 1, 1],
        [1, 1, 1, 1, 2, 2, 1, 1],
    ]
    task_ids = []
    for window in windows:
        for rank in range(8):
            task_ids.extend(balance._task_ids(window["task_start"], window["task_end"], rank, 8))
    assert sorted(task_ids) == list(range(30))


def test_tail_padding_is_clip_local_and_temporal_order_is_preserved():
    model = Model()
    x = torch.arange(1 * 1 * 5 * 4 * 4, dtype=torch.float32).reshape(1, 1, 5, 4, 4)
    plan = balance._plan(model, x, world_size=1, max_tasks=8)
    result = balance._run(model, x, plan, rank=0, group=None, all_gather_var_shape=None)

    assert plan["pad_frames"] == 3
    assert len(model.seen) == 8
    assert all(tile.shape[2] == 4 for tile in model.seen)
    for tile in model.seen[4:]:
        assert torch.equal(tile[:, :, 0], tile[:, :, 1])
        assert torch.equal(tile[:, :, 1], tile[:, :, 2])
        assert torch.equal(tile[:, :, 2], tile[:, :, 3])
    assert torch.equal(result[:, :, :4], x[:, :, :4])
    assert torch.equal(result[:, :, 4:], x[:, :, 4:].expand(-1, -1, 4, -1, -1))


def test_nonzero_overlap_blend_matches_per_clip_reference():
    model = OverlapModel()
    x = torch.arange(1 * 1 * 5 * 5 * 5, dtype=torch.float32).reshape(1, 1, 5, 5, 5)
    plan = balance._plan(model, x, world_size=1, max_tasks=8)

    def reference_clip(clip):
        raw = []
        for row in range(2):
            raw_row = []
            for column in range(2):
                tile = clip[..., row * 2 : row * 2 + 3, column * 2 : column * 2 + 3]
                raw_row.append(model.encode(tile))
            raw.append(raw_row)
        output_rows = []
        for row in range(2):
            output_row = []
            for column in range(2):
                tile = raw[row][column]
                if row:
                    tile = model.blend(raw[row - 1][column], tile, 1, -2)
                if column:
                    tile = model.blend(raw[row][column - 1], tile, 1, -1)
                output_row.append(tile[..., :-1, :] if row == 0 else tile)
            output_rows.append(
                torch.cat([tile[..., :, :-1] if index == 0 else tile for index, tile in enumerate(output_row)], -1)
            )
        return torch.cat(output_rows, -2)

    clips = [balance._clip(model, x, index, plan) for index in range(2)]
    expected = torch.cat([reference_clip(clip) for clip in clips], dim=2)[:, :, :-1]
    actual = balance._run(model, x, plan, rank=0, group=None, all_gather_var_shape=None)

    assert torch.equal(actual, expected)


def test_reassemble_restores_global_order_and_rejects_bad_counts():
    gathered = [
        torch.tensor([[0], [4], [8]]),
        torch.tensor([[1], [5]]),
        torch.tensor([[2], [6]]),
        torch.tensor([[3], [7]]),
    ]
    result = balance._reassemble(gathered, task_start=0, task_end=9, world_size=4)
    assert [int(item) for item in result] == list(range(9))
    with pytest.raises(RuntimeError, match="unexpected task count"):
        balance._reassemble([gathered[0][:-1], *gathered[1:]], task_start=0, task_end=9, world_size=4)
    with pytest.raises(RuntimeError, match="3 ranks, expected 4"):
        balance._reassemble(gathered[:-1], task_start=0, task_end=9, world_size=4)


def test_install_is_npu_only_idempotent_and_context_scoped():
    model = Model()
    adapter = SimpleNamespace(
        _device_target=Device("cpu"),
        decode_only=False,
        model=model,
        _native_parallel_state=lambda: {"sp_size": 2, "sp_rank": 0, "sp_process_group": object()},
    )
    assert balance.install_video_tile_balance(adapter, 8) is False

    adapter._device_target = Device("npu")
    assert balance.install_video_tile_balance(adapter, 8) is True
    assert balance.install_video_tile_balance(adapter, 8) is True
    original = torch.zeros(1, 1, 8, 4, 4)
    assert torch.equal(model.encode_temporal(original), original + 1)

    model.parallel_tiling = True
    unsupported = original.to(torch.float16)
    with balance.video_encode_tile_balance_context():
        assert torch.equal(model.encode_temporal(unsupported), unsupported + 1)


def test_installed_wrapper_runs_two_clip_parallel_candidate():
    global _REMOTE_GATHER

    model = Model()
    model.parallel_tiling = True
    adapter = SimpleNamespace(
        _device_target=Device("npu"),
        decode_only=False,
        model=model,
        _native_parallel_state=lambda: {"sp_size": 2, "sp_rank": 0, "sp_process_group": object()},
    )
    assert balance.install_video_tile_balance(adapter, 8) is True
    x = torch.arange(1 * 1 * 5 * 4 * 4, dtype=torch.float32).reshape(1, 1, 5, 4, 4)
    plan = balance._plan(model, x, world_size=2, max_tasks=8)
    remote = []
    for task_index in balance._task_ids(0, 8, rank=1, world_size=2):
        clip_index, tile_index = divmod(task_index, plan["tiles"])
        row, column = divmod(tile_index, plan["tile_grid"][1])
        clip = balance._clip(model, x, clip_index, plan)
        remote.append(
            clip[
                ...,
                plan["y_idx"][row] : plan["y_idx"][row] + plan["y_len"][row],
                plan["x_idx"][column] : plan["x_idx"][column] + plan["x_len"][column],
            ]
        )
    _REMOTE_GATHER = torch.stack(remote)
    try:
        with balance.video_encode_tile_balance_context():
            result = model.encode_temporal(x)
    finally:
        _REMOTE_GATHER = None

    assert plan["windows"] == [
        {
            "clip_start": 0,
            "clip_end": 2,
            "task_start": 0,
            "task_end": 8,
        }
    ]
    assert torch.equal(result[:, :, :4], x[:, :, :4])
    assert torch.equal(result[:, :, 4:], x[:, :, 4:].expand(-1, -1, 4, -1, -1))


def test_context_restores_after_error():
    assert balance._ACTIVE.get() is False
    with pytest.raises(RuntimeError, match="stop"):
        with balance.video_encode_tile_balance_context():
            assert balance._ACTIVE.get() is True
            raise RuntimeError("stop")
    assert balance._ACTIVE.get() is False
