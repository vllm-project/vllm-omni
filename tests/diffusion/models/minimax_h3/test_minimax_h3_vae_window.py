# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cross-clip scheduling, native blend order, and non-power-of-two groups."""

from math import prod

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3.vae_temporal import _iter_decoded_clips, install_temporal_stream_patches
from vllm_omni.diffusion.models.minimax_h3.vae_window import (
    decode_temporal_window,
    temporal_scheduling_unavailable_reason,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _TileModel:
    training = False
    decoder_parallel = False
    decoder_tiling = True
    parallel_tiling = True
    vae_ratio = 1
    vae_ratio_t = 1
    tokens_chunk_size = 2
    token_overlap = 1
    use_vit_decoder = False

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
        assert all(indices for _, indices in model.calls)
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

    def policy(model, z, head, tail, count):
        decisions.append(tuple(z.shape))
        return 2 if z.shape[-1] > 1 else 1

    def window(clips, **kwargs):
        windows.append(len(clips))
        return [model._adaptive_decode(clip) for clip in clips]

    for width, expected_calls in [(1, 1), (2, 2), (1, 1)]:
        model.calls.clear()
        windows.clear()
        z = torch.arange(7 * width).reshape(1, 1, 7, 1, width)
        head, tail = z[:, :, :1], z[:, :, -1:]
        iterator = _iter_decoded_clips(model, z, head, tail, 3, window, window_plan=policy)
        first = next(iterator)
        # Native mode must yield before decoding/buffering a second clip.
        assert len(model.calls) == expected_calls
        actual = [first, *iterator]
        reference = list(_iter_decoded_clips(Decoder(), z, head, tail, 3, None))
        assert windows == ([2, 1] if width == 2 else [1, 1, 1])
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

    def policy(*args):
        pytest.fail("Default per-clip decode must not evaluate the opt-in policy")

    z = torch.ones(1, 1, 7, 1, 1)
    assert len(list(_iter_decoded_clips(Decoder(), z, None, None, 3, None, window_plan=policy))) == 3


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

    def decode(clips, **kwargs):
        windows.append(len(clips))
        return [clip.clone() for clip in clips]

    actual = list(_iter_decoded_clips(model, z, head, tail, num_chunks, decode, window_plan=lambda *args: 2))
    assert windows == [2] * (num_chunks // 2) + ([1] if num_chunks % 2 else [])
    for (i, output), (j, expected) in zip(actual, reference):
        assert i == j
        assert torch.equal(output, expected)


def test_explicit_opt_in_rejects_missing_checkpoint_contract(monkeypatch):
    assert "unsupported_checkpoint" in temporal_scheduling_unavailable_reason(object())
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


@pytest.mark.parametrize("mean", [None, 1, [0, 0]])
def test_unsupported_checkpoint_normalization_falls_back(monkeypatch, mean):
    from types import SimpleNamespace

    model = _TileModel(False)
    model.use_vit_decoder = True
    monkeypatch.setattr(model, "_decode_temporal_streaming", lambda *args: None, raising=False)
    monkeypatch.setattr(
        model, "processor", SimpleNamespace(transform_rev=SimpleNamespace(mean=mean, std=[1, 1, 1])), raising=False
    )
    assert temporal_scheduling_unavailable_reason(model) == "unsupported_checkpoint_stream_contract"


@pytest.mark.parametrize("value", [0, -1, 3, 1.0, "true", None])
def test_window_config_rejects_invalid_values(value):
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    with pytest.raises(ValueError, match="enable_vae_temporal_scheduling"):
        OmniDiffusionConfig(enable_vae_temporal_scheduling=value)


def test_window_config_survives_stage_and_diffusion_kwargs():
    from vllm_omni.config.omni_config import _DiffusionConfigProjection, extract_diffusion_stage_config_kwargs
    from vllm_omni.config.stage_config import StageDeployConfig
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.engine.arg_utils import OrchestratorArgs

    args = OrchestratorArgs(enable_vae_temporal_scheduling=True)
    stage = StageDeployConfig(stage_id=0, enable_vae_temporal_scheduling=args.enable_vae_temporal_scheduling)
    kwargs = extract_diffusion_stage_config_kwargs(
        {"enable_vae_temporal_scheduling": stage.enable_vae_temporal_scheduling}, stage_id=0
    )
    projection = _DiffusionConfigProjection.from_kwargs(**kwargs)
    assert projection.enable_vae_temporal_scheduling is True
    config = OmniDiffusionConfig.from_kwargs(**kwargs)
    assert config.enable_vae_temporal_scheduling is True
    assert OmniDiffusionConfig().enable_vae_temporal_scheduling is False


def test_window_config_reaches_default_diffusion_stage():
    from vllm_omni.config.config_factory import StageConfigFactory
    from vllm_omni.config.stage_config import build_stage_runtime_overrides

    overrides = build_stage_runtime_overrides(0, {"enable_vae_temporal_scheduling": True})
    stages = StageConfigFactory.create_default_diffusion(
        {"model": "unused", "model_class_name": "UnknownPipeline", **overrides}
    )
    assert stages[0]["engine_args"]["enable_vae_temporal_scheduling"] is True


@pytest.mark.parametrize("value", [0, 3, 1.0, "true"])
def test_structured_window_config_rejects_invalid_values(value):
    from vllm_omni.config.omni_config import _DiffusionConfigProjection

    with pytest.raises(ValueError):
        _DiffusionConfigProjection(enable_vae_temporal_scheduling=value)


def test_public_cli_default_opt_in_and_removed_numeric_option():
    import argparse

    from vllm_omni.entrypoints.cli.serve import OmniServeCommand

    parser = argparse.ArgumentParser()
    OmniServeCommand().subparser_init(parser.add_subparsers())
    assert parser.parse_args(["serve", "unused"]).enable_vae_temporal_scheduling is False
    assert (
        parser.parse_args(["serve", "unused", "--enable-vae-temporal-scheduling"]).enable_vae_temporal_scheduling
        is True
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "unused", "--vae-temporal-window-size", "2"])


@pytest.mark.parametrize("budget", [0, 2**40])
def test_planner_uses_group_minimum_budget_and_real_tile_shapes(monkeypatch, budget):
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.minimax_h3.vae_window import select_temporal_window

    model = _TileModel(False)
    model.tokens_chunk_size, model.token_overlap, model.vae_ratio_t = 2, 1, 1
    model.use_vit_decoder = True
    monkeypatch.setattr(
        torch,
        "get_device_module",
        lambda device: SimpleNamespace(mem_get_info=lambda d: (2**40, 2**41)),
    )
    calls = []
    group = object()
    monkeypatch.setattr("vllm_omni.diffusion.models.minimax_h3.vae_window.dist.get_backend", lambda g: "gloo")

    def minimum(value, **kwargs):
        assert kwargs["group"] is group
        if value.dtype == torch.int64:
            calls.append(int(value.item()))
            value.fill_(min(budget, int(value.item())))

    monkeypatch.setattr("vllm_omni.diffusion.models.minimax_h3.vae_window.dist.all_reduce", minimum)

    monkeypatch.setattr(model, "decode", lambda *a: pytest.fail("planning must never decode a trial tile"))
    monkeypatch.setattr(model, "_adaptive_decode", lambda *a: pytest.fail("planning must never decode a prefix"))
    windows = [
        select_temporal_window(
            model, torch.zeros(2, 3, 17, 6, 8), None, None, 8, rank=rank, world_size=8, process_group=group
        )
        for rank in range(8)
    ]
    assert len(set(windows)) == 1
    assert len(calls) == 8
    assert (windows[0] > 1) is (budget > 0)


@pytest.mark.parametrize(
    "attribute,value", [("stack_tiling", True), ("parallel_tiling", False), ("use_vit_decoder", False)]
)
def test_unsupported_request_never_queries_memory_or_collectives(monkeypatch, attribute, value):
    from vllm_omni.diffusion.models.minimax_h3.vae_window import select_temporal_window

    model = _TileModel(False)
    model.use_vit_decoder = True
    setattr(model, attribute, value)
    monkeypatch.setattr(torch, "get_device_module", lambda *args: pytest.fail("fallback must not query device memory"))
    assert (
        select_temporal_window(
            model, torch.zeros(1, 3, 17, 6, 8), None, None, 8, rank=0, world_size=8, process_group=None
        )
        == 1
    )


@pytest.mark.parametrize("window", [1, 2, 4])
def test_all_clips_decode_once_without_calibration_prefix(window):
    model = _TileModel(False)
    z = torch.arange(15).reshape(1, 1, 15, 1, 1).float()
    head, tail = -z[:, :, :1] - 1, -z[:, :, :1] - 2
    reference = list(_iter_decoded_clips(model, z, head, tail, 7, None))
    calls: list[torch.Tensor] = []

    def plan(*args):
        assert not calls
        return window

    def decode(clips, **kwargs):
        calls.extend(clips)
        return [model._adaptive_decode(clip) for clip in clips]

    actual = list(_iter_decoded_clips(model, z, head, tail, 7, decode, window_plan=plan))
    assert len(calls) == 7
    assert all(i == j and torch.equal(a, b) for (i, a), (j, b) in zip(reference, actual))
