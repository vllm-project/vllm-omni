# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the one-piece-per-rank Wan VAE encode split (no device, no model)."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.distributed.autoencoders import wan_encode_split as ws

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

RATIO = 8
TEMPORAL = 4


@pytest.mark.parametrize(
    ("num_pieces", "latent_hw", "expected"),
    [
        (8, (90, 160), (2, 4)),  # 720p: 45x40 pieces
        (4, (90, 160), (2, 2)),
        (2, (90, 160), (1, 2)),
        (1, (90, 160), (1, 1)),
        (2, (100, 100), (1, 2)),  # tie between 1x2 and 2x1 goes to fewer rows
        (4, (1, 8), (1, 4)),  # a grid with more rows than latent rows is skipped
    ],
)
def test_rank_grid_picks_the_squarest_pieces(num_pieces, latent_hw, expected):
    assert ws.rank_grid(num_pieces, *latent_hw) == expected


@pytest.mark.parametrize(("extent", "parts"), [(90, 4), (160, 4), (45, 2), (7, 7), (10, 3)])
def test_even_bounds_cover_contiguously_with_sizes_within_one(extent, parts):
    bounds = ws.even_bounds(extent, parts)
    assert len(bounds) == parts and bounds[0][0] == 0 and bounds[-1][1] == extent
    assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:]))
    sizes = [hi - lo for lo, hi in bounds]
    assert max(sizes) - min(sizes) <= 1


def test_env_switches_default_and_validate(monkeypatch):
    monkeypatch.delenv(ws.ENCODE_SPLIT_ENV, raising=False)
    monkeypatch.delenv(ws.ENCODE_MARGIN_ENV, raising=False)
    assert ws.encode_split_mode() == "tile"
    assert ws.encode_margin() == ws.ENCODE_MARGIN_DEFAULT
    monkeypatch.setenv(ws.ENCODE_SPLIT_ENV, " Rank ")
    assert ws.encode_split_mode() == "rank"
    monkeypatch.setenv(ws.ENCODE_SPLIT_ENV, "blend")
    with pytest.raises(ValueError, match=ws.ENCODE_SPLIT_ENV):
        ws.encode_split_mode()
    monkeypatch.setenv(ws.ENCODE_MARGIN_ENV, "-1")
    with pytest.raises(ValueError, match=ws.ENCODE_MARGIN_ENV):
        ws.encode_margin()


def _fake_vae(num_pieces: int) -> SimpleNamespace:
    return SimpleNamespace(
        spatial_compression_ratio=RATIO,
        config=SimpleNamespace(patch_size=None, scale_factor_temporal=TEMPORAL),
        distributed_executor=SimpleNamespace(parallel_size=num_pieces, world_size=num_pieces),
        dtype=torch.float32,
    )


def _fake_encode(time_chunks: list[torch.Tensor]) -> torch.Tensor:
    """A stand-in encoder that is local per RATIO x RATIO block, so cropping it is exact:
    one latent frame per temporal chunk, the chunk's first frame average-pooled."""
    frames = [F.avg_pool2d(chunk[:, :, 0], RATIO).unsqueeze(2) for chunk in time_chunks]
    return torch.cat(frames, dim=2)


@pytest.mark.parametrize("num_pieces", [2, 4, 8])
@pytest.mark.parametrize("margin", [0, 1, 2])
def test_rank_split_then_merge_reassembles_the_whole_frame(monkeypatch, num_pieces, margin):
    monkeypatch.setenv(ws.ENCODE_MARGIN_ENV, str(margin))
    torch.manual_seed(0)
    num_frames, height, width = 1 + 2 * TEMPORAL, 6 * RATIO, 10 * RATIO
    x = torch.randn(1, 3, num_frames, height, width)

    tasks, grid_spec = ws.rank_split(_fake_vae(num_pieces), x)

    rows, cols = grid_spec.grid_shape
    assert rows * cols == num_pieces == len(tasks)
    assert sorted(t.grid_coord for t in tasks) == [(i, j) for i in range(rows) for j in range(cols)]
    assert all(len(t.tensor) == 1 + (num_frames - 1) // TEMPORAL for t in tasks)

    whole = _fake_encode([x[:, :, :1]] + [x[:, :, 1 + TEMPORAL * k : 1 + TEMPORAL * (k + 1)] for k in range(2)])
    merged = ws.rank_merge({t.grid_coord: _fake_encode(t.tensor) for t in tasks}, grid_spec)
    torch.testing.assert_close(merged, whole, rtol=0, atol=0)
