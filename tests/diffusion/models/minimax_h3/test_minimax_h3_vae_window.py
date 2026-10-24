# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cross-clip scheduling, native blend order, and non-power-of-two groups."""

from math import prod

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3.vae_temporal import _iter_decoded_clips, install_temporal_stream_patches
from vllm_omni.diffusion.models.minimax_h3.vae_window import (
    decode_temporal_window,
    temporal_window_enabled,
    validate_temporal_window,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _TileModel:
    training = False
    decoder_parallel = False
    decoder_tiling = True
    parallel_tiling = True
    vae_ratio = 1

    def __init__(self, stack_tiling):
        self.stack_tiling = stack_tiling
        self.calls = []
        self.gathers = 0

    @staticmethod
    def split_tiles(length, decode):
        starts = list(range(0, length - 2, 2)) or [0]
        lengths = [min(4, length - start) for start in starts]
        return starts, lengths, [2] * (len(starts) - 1)

    @staticmethod
    def decode(z):
        return torch.sin(z * 0.5)

    def _adaptive_decode(self, z):
        return self.decode(z)

    @staticmethod
    def blend(a, b, overlap, dim):
        if not overlap:
            return b
        head = b.narrow(dim, 0, overlap)
        tail = a.narrow(dim, a.shape[dim] - overlap, overlap)
        return torch.cat(((head + tail) * 0.5, b.narrow(dim, overlap, b.shape[dim] - overlap)), dim=dim)

    def _run_tile_tasks(self, tiles, indices, forward, stack):
        self.calls.append((len(tiles), indices))
        self.tiles = tiles
        if stack:
            batch = tiles[0].shape[0]
            out = forward(torch.cat([tiles[i] for i in indices]))
            return list(out.unflatten(0, (len(indices), batch)).unbind(0))
        return [forward(tiles[i]) for i in indices]

    def _all_gather_tiled_results(self, tasks, num_tiles):
        # Simulate the other ranks' independent forwards. Check the local
        # payload before emulating the checkpoint's round-robin gather.
        assert tasks
        indices = self.calls[-1][1]
        for index, task in zip(indices, tasks):
            assert torch.equal(task, self.decode(self.tiles[index]))
        self.gathers += 1
        return [self.decode(tile) for tile in self.tiles]


def _native_decode(model, clip):
    """Independent reference: per-clip row-major tiles and native stitching."""
    ys, heights, yo = model.split_tiles(clip.shape[-2], True)
    xs, widths, xo = model.split_tiles(clip.shape[-1], True)
    rows = [[model.decode(clip[..., y : y + h, x : x + w]) for x, w in zip(xs, widths)] for y, h in zip(ys, heights)]
    stitched = []
    for i, row in enumerate(rows):
        parts = []
        for j, original in enumerate(row):
            tile = original
            if i:
                tile = model.blend(rows[i - 1][j], tile, yo[i - 1], -2)
            if j:
                tile = model.blend(row[j - 1], tile, xo[j - 1], -1)
            if i + 1 < len(rows):
                tile = tile[..., : -yo[i], :]
            if j + 1 < len(row):
                tile = tile[..., :, : -xo[j]]
            parts.append(tile)
        stitched.append(torch.cat(parts, -1))
    return torch.cat(stitched, -2)


@pytest.mark.parametrize("world_size", [2, 3, 4, 6, 7, 8, 9, 25])
@pytest.mark.parametrize("stack", [False, True])
@pytest.mark.parametrize("shapes", [((2, 3, 3, 6, 8), (2, 3, 3, 6, 8)), ((1, 3, 4, 7, 9), (1, 3, 2, 7, 9))])
def test_tiles_restore_native_order_for_all_ranks(monkeypatch, world_size, stack, shapes):
    group = object()
    barriers = []
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.minimax_h3.vae_window.dist.barrier", lambda **kw: barriers.append(kw)
    )
    clips = [torch.arange(prod(shape)).reshape(shape).float() / 100 for shape in shapes]
    reference = [_native_decode(_TileModel(stack), clip) for clip in clips]
    for rank in range(world_size):
        model = _TileModel(stack)
        outputs = decode_temporal_window(model, clips, rank=rank, world_size=world_size, process_group=group)
        assert len(outputs) == len(reference)
        for output, expected in zip(outputs, reference):
            assert torch.equal(output, expected)
        for count, indices in model.calls:
            expected = list(range(rank, count, world_size)) if count >= world_size else list(range(count))
            assert indices == expected
        assert model.gathers == sum(count >= world_size for count, _ in model.calls)
    assert all(b == {"group": group} for b in barriers)


@pytest.mark.parametrize("attribute,value", [("parallel_tiling", False), ("decoder_tiling", False)])
def test_disabled_tiling_keeps_adaptive_decode(attribute, value):
    model = _TileModel(False)
    setattr(model, attribute, value)
    clips = [torch.randn(1, 3, 2, 4, 4)]
    out = decode_temporal_window(model, clips, rank=0, world_size=4, process_group=None)
    assert torch.equal(out[0], model._adaptive_decode(clips[0]))
    assert not model.calls


def test_single_rank_uses_original_decode():
    model = _TileModel(True)
    clip = torch.randn(1, 3, 2, 4, 4)
    out = decode_temporal_window(model, [clip], rank=0, world_size=1, process_group=None)
    assert torch.equal(out[0], model._adaptive_decode(clip))
    assert not model.calls


@pytest.mark.parametrize("attribute", ["training", "decoder_parallel"])
def test_rejects_stateful_or_training_decoder(attribute):
    model = _TileModel(False)
    setattr(model, attribute, True)
    with pytest.raises(ValueError, match="require inference"):
        decode_temporal_window(model, [], rank=0, world_size=3, process_group=None)


@pytest.mark.parametrize("rank,size", [(0, 0), (-1, 3), (3, 3)])
def test_invalid_group_metadata(rank, size):
    with pytest.raises(ValueError, match="Invalid VAE tile rank"):
        decode_temporal_window(_TileModel(False), [], rank=rank, world_size=size, process_group=None)


@pytest.mark.parametrize("world_size", [1, 2, 3, 4, 6, 7, 8])
@pytest.mark.parametrize("batch", [1, 2])
def test_balanced_window_policy_follows_actual_group(world_size, batch):
    model = _TileModel(False)
    # Six equal spatial tiles. Only groups 2, 3 and 6 divide them evenly.
    z = torch.empty(batch, 3, 9, 6, 8)
    for rank in range(world_size):
        enabled = temporal_window_enabled(model, z, rank=rank, world_size=world_size)
        assert enabled == (world_size in (4, 7, 8))


def test_window_policy_rechecks_shape_and_tiling_state():
    model = _TileModel(False)
    balanced = torch.empty(1, 3, 9, 6, 8)
    # Edge tiles have different lengths; divisibility alone is insufficient.
    edge = torch.empty(1, 3, 9, 7, 9)
    assert not temporal_window_enabled(model, balanced, rank=0, world_size=3)
    assert temporal_window_enabled(model, edge, rank=0, world_size=3)
    model.stack_tiling = True
    assert temporal_window_enabled(model, balanced, rank=0, world_size=3)
    model.parallel_tiling = False
    assert not temporal_window_enabled(model, edge, rank=0, world_size=3)
    model.parallel_tiling = True
    model.decoder_tiling = False
    assert not temporal_window_enabled(model, edge, rank=0, world_size=3)


@pytest.mark.parametrize("attribute", ["training", "decoder_parallel"])
def test_window_policy_validates_before_native_fallback(attribute):
    model = _TileModel(False)
    model.parallel_tiling = False
    setattr(model, attribute, True)
    with pytest.raises(ValueError, match="require inference"):
        temporal_window_enabled(model, torch.empty(1, 3, 3, 4, 4), rank=0, world_size=1)


@pytest.mark.parametrize("rank,size", [(0, 0), (-1, 3), (3, 3)])
def test_window_policy_rejects_invalid_group_metadata(rank, size):
    with pytest.raises(ValueError, match="Invalid VAE tile rank"):
        temporal_window_enabled(_TileModel(False), torch.empty(1, 3, 3, 4, 4), rank=rank, world_size=size)


def test_window_policy_bypasses_before_second_clip_is_decoded():
    class Decoder:
        tokens_chunk_size = 2
        token_overlap = 1

        def __init__(self):
            self.calls = []

        def _adaptive_decode(self, z):
            self.calls.append(z.clone())
            return z.clone()

    model = Decoder()
    decisions = []
    windows = []

    def policy(z):
        decisions.append(tuple(z.shape))
        return z.shape[-1] > 1

    def window(clips):
        windows.append(len(clips))
        return [model._adaptive_decode(clip) for clip in clips]

    for width, expected_calls in [(1, 1), (2, 2), (1, 1)]:
        model.calls.clear()
        windows.clear()
        z = torch.arange(7 * width).reshape(1, 1, 7, 1, width)
        head, tail = z[:, :, :1], z[:, :, -1:]
        iterator = _iter_decoded_clips(model, z, head, tail, 3, window, window_enabled=policy)
        first = next(iterator)
        # Native mode must yield before decoding/buffering a second clip.
        assert len(model.calls) == expected_calls
        actual = [first, *iterator]
        reference = list(_iter_decoded_clips(Decoder(), z, head, tail, 3, None))
        assert windows == ([2, 1] if width == 2 else [])
        assert len(actual) == len(reference)
        for (i, output), (j, expected) in zip(actual, reference):
            assert i == j and torch.equal(output, expected)
    assert [shape[-1] for shape in decisions] == [1, 2, 1]


def test_default_iterator_does_not_evaluate_window_policy():
    class Decoder:
        tokens_chunk_size = 2
        token_overlap = 1

        def _adaptive_decode(self, z):
            return z

    def policy(z):
        pytest.fail("Default per-clip decode must not evaluate the opt-in policy")

    z = torch.ones(1, 1, 7, 1, 1)
    assert len(list(_iter_decoded_clips(Decoder(), z, None, None, 3, None, window_enabled=policy))) == 3


@pytest.mark.parametrize("num_chunks", [1, 2, 3, 7])
@pytest.mark.parametrize("head_tail", [False, True])
def test_window_iterator_preserves_head_tail_and_odd_clip(num_chunks, head_tail):
    class Decoder:
        tokens_chunk_size = 2
        token_overlap = 1

        def _adaptive_decode(self, z):
            return z.clone()

    model = Decoder()
    z = torch.arange(num_chunks * 2 + 1).reshape(1, 1, -1, 1, 1)
    head = torch.full_like(z[:, :, :1], -1) if head_tail else None
    tail = torch.full_like(z[:, :, :1], -2) if head_tail else None
    reference = list(_iter_decoded_clips(model, z, head, tail, num_chunks, None))
    windows = []

    def decode(clips):
        windows.append(len(clips))
        return [clip.clone() for clip in clips]

    actual = list(_iter_decoded_clips(model, z, head, tail, num_chunks, decode))
    assert windows == [2] * (num_chunks // 2) + ([1] if num_chunks % 2 else [])
    for (i, output), (j, expected) in zip(actual, reference):
        assert i == j
        assert torch.equal(output, expected)


def test_explicit_opt_in_rejects_missing_checkpoint_contract(monkeypatch):
    with pytest.raises(ValueError, match="tile methods"):
        validate_temporal_window(object())
    monkeypatch.setenv("VLLM_OMNI_VAE_LEGACY_TEMPORAL", "1")
    with pytest.raises(ValueError, match="legacy temporal"):
        install_temporal_stream_patches(object(), window_decode=lambda clips: clips)
    monkeypatch.delenv("VLLM_OMNI_VAE_LEGACY_TEMPORAL")
    monkeypatch.setenv("MINIMAX_H3_VAE_DECODER_STREAM_TEMPORAL_CAT", "0")
    with pytest.raises(ValueError, match="streaming temporal concatenation"):
        install_temporal_stream_patches(object(), window_decode=lambda clips: clips)
    monkeypatch.delenv("MINIMAX_H3_VAE_DECODER_STREAM_TEMPORAL_CAT")
    with pytest.raises(ValueError, match="streaming decode contract"):
        install_temporal_stream_patches(object(), window_decode=lambda clips: clips)


@pytest.mark.parametrize("value", [0, -1, 3, True, 1.0, "2", None])
def test_window_config_rejects_invalid_values(value):
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    with pytest.raises(ValueError, match="vae_temporal_window_size"):
        OmniDiffusionConfig(vae_temporal_window_size=value)


def test_window_config_survives_stage_and_diffusion_kwargs():
    from vllm_omni.config.omni_config import _DiffusionConfigProjection, extract_diffusion_stage_config_kwargs
    from vllm_omni.config.stage_config import StageDeployConfig
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.engine.arg_utils import OrchestratorArgs

    args = OrchestratorArgs(vae_temporal_window_size=2)
    stage = StageDeployConfig(stage_id=0, vae_temporal_window_size=args.vae_temporal_window_size)
    kwargs = extract_diffusion_stage_config_kwargs(
        {"vae_temporal_window_size": stage.vae_temporal_window_size}, stage_id=0
    )
    projection = _DiffusionConfigProjection.from_kwargs(**kwargs)
    assert projection.vae_temporal_window_size == 2
    config = OmniDiffusionConfig.from_kwargs(**kwargs)
    assert config.vae_temporal_window_size == 2
    assert OmniDiffusionConfig().vae_temporal_window_size == 1


def test_window_config_reaches_default_diffusion_stage():
    from vllm_omni.config.config_factory import StageConfigFactory
    from vllm_omni.config.stage_config import build_stage_runtime_overrides

    overrides = build_stage_runtime_overrides(0, {"vae_temporal_window_size": 2})
    stages = StageConfigFactory.create_default_diffusion(
        {"model": "unused", "model_class_name": "UnknownPipeline", **overrides}
    )
    assert stages[0]["engine_args"]["vae_temporal_window_size"] == 2


@pytest.mark.parametrize("value", [0, 3, True, 1.0, "2"])
def test_structured_window_config_rejects_invalid_values(value):
    from vllm_omni.config.omni_config import _DiffusionConfigProjection

    with pytest.raises(ValueError):
        _DiffusionConfigProjection(vae_temporal_window_size=value)
