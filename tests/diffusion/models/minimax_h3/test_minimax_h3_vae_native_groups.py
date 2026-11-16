# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Preserve native GEMM group shapes, coordinates, padding and restoration."""

import os
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from vllm_omni.diffusion.models.minimax_h3 import vae_reshape as module

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _PositionSensitiveProjection(nn.Linear):
    def __init__(self):
        super().__init__(1, 1)
        self.calls = []

    def forward(self, value):
        self.calls.append(value.shape[0])
        # Both shape and each member's position affect output; preserving only
        # a nominal batch size without original group order fails this check.
        position = torch.arange(value.shape[0], dtype=value.dtype).reshape(-1, 1, 1)
        return value + value.shape[0] * 0.25 + position


class _NativeModel(nn.Module):
    def __init__(self, rank=0, size=1):
        super().__init__()
        self.tokens_chunk_size = 2
        self.token_overlap = self.token_drop = 0
        self.vae_ratio = 1
        self.parallel_tiling = size > 1
        self.stack_tiling = True
        self.use_3d_conv = self.use_vit_decoder = self.decoder_tiling = True
        self.isolated_first_frame = self.isolated_last_frame = False
        self.decoder = nn.Module()
        self.decoder.spatial_parallel = False
        block = nn.Module()
        block.ff = nn.Module()
        block.ff.w1 = _PositionSensitiveProjection()
        self.decoder.transformer_blocks = nn.ModuleList([block])
        self.rank, self.size = rank, size
        self.eval()

    def split_tiles(self, length, latent_space):
        assert latent_space
        count = 3 if length == 3 else 1
        return list(range(count)), [1] * count, [0] * count

    def _local_tile_indices(self, count, rank, size):
        return list(range(rank, count, size))

    def decode(self, value):
        features = value.sum(dim=(1, 2, 3, 4)).reshape(-1, 1, 1)
        return self.decoder.transformer_blocks[0].ff.w1(features)

    def _run_tile_tasks(self, tiles, indices, forward, stacked, cls_agg=None):
        if stacked and indices:
            batch = tiles[0].shape[0]
            output = forward(torch.cat([tiles[i] for i in indices], dim=0))
            return list(output.unflatten(0, (len(indices), batch)).unbind(dim=0))
        return [forward(tiles[i]) for i in indices]

    def tiled_decode(self, clip):
        tiles = [clip[..., row : row + 1, :] for row in range(3)]
        size = self.size if self.parallel_tiling else 1
        rank = self.rank if self.parallel_tiling else 0
        indices = self._local_tile_indices(3, rank, size)
        results = self._run_tile_tasks(tiles, indices, self.decode, True)
        if size > 1:
            gathered = [None] * size
            dist.all_gather_object(gathered, results)
            results = [gathered[item % size][item // size] for item in range(3)]
        return torch.cat(results, dim=1).squeeze(-1)

    def _skip_decode_output(self):
        return False


class _NativeAdapter:
    def __init__(self, rank=0, size=1):
        self.model = _NativeModel(rank, size)
        self.rank, self.size = rank, size

    def _native_parallel_state(self):
        return {
            "sp_size": self.size,
            "sp_rank": self.rank,
            "sp_process_group": dist.group.WORLD if self.size > 1 else None,
        }

    def _denormalize_latent(self, latent):
        return latent

    def decode_latent(self, latent):
        return torch.cat(
            [self.model.tiled_decode(latent[:, :, i : i + 2]) for i in range(0, latent.shape[2], 2)], dim=1
        )

    def decode_with_chunks(self, latent, *, on_chunk):
        on_chunk(self.decode_latent(latent))


class _PlanModel:
    tokens_chunk_size = 5
    token_overlap = 2
    token_drop = 3
    vae_ratio = 1
    parallel_tiling = True
    stack_tiling = True

    def __init__(self, tiles):
        self.tiles = tiles

    def split_tiles(self, length, decoder):
        return (list(range(self.tiles)), [1] * self.tiles, []) if length > 1 else ([0], [1], [])

    def _local_tile_indices(self, count, rank, size):
        return list(range(rank, count, size))

    def tiled_decode(self, clip):
        return clip


@pytest.mark.parametrize("tiles", [1, 7, 8, 9, 15, 16, 28, 32])
@pytest.mark.parametrize("chunks", [1, 2, 3, 5, 17, 21])
@pytest.mark.parametrize("limit", [1, 2, 3, 4, 8])
def test_plan_preserves_every_native_group_and_member_order(tiles, chunks, limit, monkeypatch):
    monkeypatch.setenv("H3_VIDEO_DECODER_TILE_BATCH_SIZE", str(limit))
    monkeypatch.setenv("H3_VIDEO_DECODER_TILE_BATCHING", "1")
    adapter = types.SimpleNamespace(
        model=_PlanModel(tiles), _native_parallel_state=lambda: {"sp_size": 8, "sp_rank": 0}
    )
    pool = module._BalancedNativeStackTilePool(adapter, torch.zeros(1, 1, 5 * chunks + 2, 2, 1), all_outputs=True)
    pool._initialize_native_plan()
    if (tiles + 7) // 8 > limit:
        assert pool.native_fallback
        return
    actual = [group for wave in pool.wave_plans for rank in wave for group in rank]
    expected = [
        tuple(chunk * tiles + tile for tile in range(rank, tiles, 8))
        for chunk in range(chunks)
        for rank in range(8)
        if rank < tiles
    ]
    assert sorted(actual) == sorted(expected)
    assert sorted(item for group in actual for item in group) == list(range(chunks * tiles))
    assert all(sum(map(len, rank)) <= limit for wave in pool.wave_plans for rank in wave)
    for batch in (1, 2, 3, 4):
        assert all(
            sum(len(group) * batch for group in rank) == sum(map(len, rank)) * batch
            for wave in pool.wave_plans
            for rank in wave
        )


@pytest.mark.parametrize("batch", [1, 2, 3, 4])
@pytest.mark.parametrize("sizes", [(1, 1), (2, 2), (2, 1, 1), (3, 1), (4,)])
def test_projection_outputs_preserve_reference_values_and_restore(batch, sizes):
    torch.manual_seed(91)
    linear = nn.Linear(5, 7)
    model = types.SimpleNamespace(
        decoder=types.SimpleNamespace(transformer_blocks=[types.SimpleNamespace(ff=types.SimpleNamespace(w1=linear))])
    )
    groups = tuple(size * batch for size in sizes)
    value = torch.randn(sum(groups), 3, 5)
    with torch.inference_mode():
        expected = torch.cat([linear(part) for part in value.split(groups, dim=0)])
        with module._reference_ffn_projection_outputs(model, lambda: groups):
            actual = linear(value)
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
            assert model._h3_projection_out_activity["fast_calls"] == 1
            assert model._h3_projection_out_activity["fallback_calls"] == 0
    assert "forward" not in linear.__dict__
    assert "_h3_projection_out_activity" not in model.__dict__


@pytest.mark.parametrize("batch", [1, 2, 3, 4])
def test_attention_output_projection_also_keeps_native_groups(batch):
    torch.manual_seed(191)
    w1, to_out = nn.Linear(5, 7), nn.Linear(5, 7)
    model = types.SimpleNamespace(
        decoder=types.SimpleNamespace(
            transformer_blocks=[
                types.SimpleNamespace(ff=types.SimpleNamespace(w1=w1), attn=types.SimpleNamespace(to_out=to_out))
            ]
        )
    )
    groups = (2 * batch, batch, batch)
    value = torch.randn(sum(groups), 3, 5)
    with torch.inference_mode():
        references = [torch.cat([linear(part) for part in value.split(groups, dim=0)]) for linear in (w1, to_out)]
        with module._reference_ffn_projection_outputs(model, lambda: groups) as count:
            assert count == 2
            for linear, expected in zip((w1, to_out), references):
                assert torch.equal(linear(value).view(torch.uint8), expected.view(torch.uint8))
            assert model._h3_projection_out_activity["fast_calls"] == 2
    assert "forward" not in w1.__dict__ and "forward" not in to_out.__dict__


@pytest.mark.parametrize("tiles,chunks", [(15, 1), (28, 3), (32, 3)])
def test_native_schedule_without_a_load_or_wave_reduction_is_retained(tiles, chunks, monkeypatch):
    monkeypatch.setenv("H3_VIDEO_DECODER_TILE_BATCH_SIZE", "4")
    monkeypatch.setenv("H3_VIDEO_DECODER_TILE_BATCHING", "1")
    adapter = types.SimpleNamespace(
        model=_PlanModel(tiles), _native_parallel_state=lambda: {"sp_size": 8, "sp_rank": 0}
    )
    pool = module._BalancedNativeStackTilePool(adapter, torch.zeros(1, 1, 5 * chunks + 2, 2, 1), all_outputs=True)
    pool._initialize_native_plan()
    assert pool.native_fallback and not pool.stats["active"]
    assert pool.stats["status"] == "native-group-no-gain"


def test_temporal_padding_plan_keeps_all_input_bits_and_restores_methods():
    adapter = _NativeAdapter()
    adapter.model.tokens_chunk_size = 5
    adapter.model.token_drop = 3
    adapter.model.token_overlap = 2
    value = torch.arange(54, dtype=torch.float32).reshape(1, 1, 18, 3, 1)
    with module._cross_chunk_tiling(adapter, value, all_outputs=True):
        pool = getattr(adapter.model.tiled_decode, "__self__")
        assert pool.latent.shape[2] == 22 and pool.chunks == 4
        assert torch.equal(pool.latent[:, :, :18].contiguous().view(torch.uint8), value.view(torch.uint8))
        assert torch.equal(pool.latent[:, :, 18:], value[:, :, -1:].repeat(1, 1, 4, 1, 1))
        assert pool.stats["native_pad_tokens"] == 4
    assert "tiled_decode" not in adapter.model.__dict__
    assert "forward" not in adapter.model.decoder.transformer_blocks[0].ff.w1.__dict__


@pytest.mark.parametrize("batch", [1, 2, 3, 4])
def test_final_projection_retains_original_calls_and_restores_forward(batch):
    class ShapeSensitiveLinear(nn.Linear):
        def __init__(self):
            super().__init__(1, 1)
            self.calls = []

        def forward(self, value):
            self.calls.append(value.shape[0])
            return value + value.shape[0] * 0.125

    linear = ShapeSensitiveLinear()
    model = types.SimpleNamespace(decoder=types.SimpleNamespace(transformer_blocks=[], proj_out=linear))
    groups = (2 * batch, batch)
    value = torch.arange(sum(groups), dtype=torch.float32).view(-1, 1, 1)
    expected = torch.cat([linear(part) for part in value.split(groups, dim=0)])
    linear.calls.clear()
    with torch.inference_mode(), module._reference_ffn_projection_outputs(model, lambda: groups):
        actual = linear(value)
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        assert linear.calls == list(groups)
        assert model._h3_projection_out_activity["native_output_calls"] == 1
        assert model._h3_projection_out_activity["fast_calls"] == 0
    assert "forward" not in linear.__dict__


@pytest.mark.parametrize("batch", [1, 2, 3, 4])
def test_unstacked_mode_preserves_qkv_and_ffn_contraction(batch):
    class BatchSensitive(nn.Linear):
        def __init__(self):
            super().__init__(1, 1)
            self.calls = []

        def forward(self, value):
            self.calls.append(value.shape[0])
            return value + value.shape[0] * 0.125

    linears = [BatchSensitive() for _ in range(5)]
    model = types.SimpleNamespace(
        stack_tiling=False,
        decoder=types.SimpleNamespace(
            transformer_blocks=[
                types.SimpleNamespace(
                    ff=types.SimpleNamespace(w1=linears[0], w2=linears[1]),
                    attn=types.SimpleNamespace(to_qkv=linears[2], to_out=linears[3]),
                )
            ],
            proj_out=linears[4],
        ),
    )
    value = torch.arange(4 * batch, dtype=torch.float32).view(-1, 1, 1)
    expected = value + batch * 0.125
    with module._reference_ffn_projection_batches(model, batch) as count:
        assert count == 5
        for linear in linears:
            assert torch.equal(linear(value).view(torch.uint8), expected.view(torch.uint8))
            assert linear.calls == [batch] * 4
    assert all("forward" not in linear.__dict__ for linear in linears)


def worker(rank, rendezvous):
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=rendezvous)
    try:
        for batch in (1, 3):
            adapter = _NativeAdapter(rank, 2)
            value = torch.arange(batch * 30, dtype=torch.float32).reshape(batch, 1, 10, 3, 1)
            reference = adapter.decode_latent(value)
            os.environ.update(
                H3_VIDEO_DECODER_CROSS_CHUNK="1",
                H3_VIDEO_DECODER_TILE_BATCHING="1",
                H3_VIDEO_DECODER_TILE_BATCH_SIZE="4",
            )
            module.install_cross_chunk_tiling(adapter)
            actual = adapter.decode_latent(value)
            assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))
            stats = adapter.model._h3_vae_reshape_last_stats
            assert stats["active"] and stats["schedule"] == "balanced-native-groups"
            assert stats["total_items"] == 15 and sum(stats["wave_items"]) == 15
            assert max(stats["local_batch_sizes"]) <= 4
            assert all(
                sum(groups) == size * batch
                for groups, size in zip(stats["reference_group_batches"], stats["local_batch_sizes"])
            )
            assert "tiled_decode" not in adapter.model.__dict__
            assert "_run_tile_tasks" not in adapter.model.__dict__
            assert "_h3_projection_out_activity" not in adapter.model.__dict__
    finally:
        dist.destroy_process_group()


def test_balanced_members_positions_and_sample_batches(tmp_path):
    mp.spawn(worker, args=(f"file://{tmp_path / 'balanced-store'}",), nprocs=2, join=True)


def test_packer_preserves_every_group_and_its_member_order():
    groups = [
        (0, 8),
        (1, 9),
        (2, 10),
        (3, 11),
        (4, 12),
        (5, 13),
        (6, 14),
        (7,),
        (15, 23),
        (16, 24),
        (17, 25),
        (18, 26),
        (19, 27),
        (20, 28),
        (21, 29),
        (22,),
        (30, 38),
    ]
    packed = module._pack_native_groups(groups, 8, 4)
    assert packed is not None
    assert sorted(group for rank in packed for group in rank) == sorted(groups)
    assert all(sum(map(len, rank)) == 4 for rank in packed)
