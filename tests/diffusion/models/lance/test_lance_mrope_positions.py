# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU parity tests for Lance's tensor-built 3-D mRoPE positions.

The reference is the per-token Python loop the vectorized builder replaced:
``(P, P, P)`` for ``start_of_image``, ``(P + 1 + int(ti * t_scale), P + 1 + hi,
P + 1 + wi)`` per latent token and ``(end_p, end_p, end_p)`` for
``end_of_image``, concatenated over the batch of requests.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.lance import lance_transformer
from vllm_omni.diffusion.models.lance.lance_transformer import LanceBagel, _mrope_position_chunk

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_LATENT_DOWNSAMPLE = 16
_DOWNSAMPLE_T = 4
_NEW_TOKEN_IDS = {"start_of_image": 11, "end_of_image": 12}


def _latent_grid(video_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    T, H, W = video_shape
    return (T - 1) // _DOWNSAMPLE_T + 1, H // _LATENT_DOWNSAMPLE, W // _LATENT_DOWNSAMPLE


def _reference_chunk(t: int, h: int, w: int, start_position_id: int, t_scale: float) -> torch.Tensor:
    """The pre-vectorization per-token loop for one request."""
    tt, hh, ww = torch.meshgrid(torch.arange(t), torch.arange(h), torch.arange(w), indexing="ij")
    per_axis_pos = [(start_position_id,) * 3]
    for ti, hi, wi in zip(tt.flatten().tolist(), hh.flatten().tolist(), ww.flatten().tolist()):
        per_axis_pos.append(
            (start_position_id + 1 + int(ti * t_scale), start_position_id + 1 + hi, start_position_id + 1 + wi)
        )
    end_p = start_position_id + max(int((t - 1) * t_scale), h - 1, w - 1) + 1 + 1
    per_axis_pos.append((end_p,) * 3)
    return torch.stack([torch.tensor([p[axis] for p in per_axis_pos], dtype=torch.long) for axis in range(3)])


def _reference_positions(video_shapes, curr_rope, t_scale: float) -> torch.Tensor:
    chunks = [
        _reference_chunk(*_latent_grid(shape), position_id, t_scale)
        for shape, position_id in zip(video_shapes, curr_rope)
    ]
    return torch.cat(chunks, dim=1) if chunks else torch.empty((3, 0), dtype=torch.long)


def _bagel_stub() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(vae_config=SimpleNamespace(downsample_temporal=_DOWNSAMPLE_T)),
        latent_downsample=_LATENT_DOWNSAMPLE,
        max_latent_size=64,
        latent_channel=4,
        latent_patch_size=1,
    )


_GRIDS = [(1, 1, 1), (1, 3, 5), (4, 3, 5), (12, 35, 47)]
_SCALES = [1, 2, 2.0, 0.5, 1.5, 2 / 3, 0.3, 0.1]


@pytest.mark.parametrize("t_scale", _SCALES)
@pytest.mark.parametrize("grid", _GRIDS)
@pytest.mark.parametrize("start_position_id", [0, 111])
def test_mrope_position_chunk_matches_python_loop(grid, start_position_id, t_scale):
    t, h, w = grid
    tt, hh, ww = torch.meshgrid(torch.arange(t), torch.arange(h), torch.arange(w), indexing="ij")
    end_p = start_position_id + max(int((t - 1) * t_scale), h - 1, w - 1) + 1 + 1

    chunk = _mrope_position_chunk(tt, hh, ww, start_position_id, end_p, t_scale)

    assert chunk.dtype == torch.long
    assert chunk.shape == (3, t * h * w + 2)
    torch.testing.assert_close(chunk, _reference_chunk(t, h, w, start_position_id, t_scale), rtol=0, atol=0)


@pytest.mark.parametrize("seconds_per_grid", [1.0, 0.75, 0.5, 1 / 3, 0.15])
def test_fractional_temporal_scale_truncates_like_python_float(monkeypatch, seconds_per_grid):
    """``int(ti * t_scale)`` in Python float64 is what the tensor path must reproduce."""
    monkeypatch.setattr(lance_transformer, "LANCE_SECONDS_PER_GRID", seconds_per_grid)
    t_scale = lance_transformer.LANCE_TOKENS_PER_SECOND * seconds_per_grid
    video_shapes = [(45, 480, 832), (9, 96, 160)]
    curr_rope = [111, 7]

    out = LanceBagel.prepare_video_latent(_bagel_stub(), [5, 3], curr_rope, video_shapes, _NEW_TOKEN_IDS)
    cfg = LanceBagel.prepare_video_latent_cfg(_bagel_stub(), [5, 3], curr_rope, video_shapes)

    expected = _reference_positions(video_shapes, curr_rope, t_scale)
    torch.testing.assert_close(out["packed_position_ids"], expected, rtol=0, atol=0)
    torch.testing.assert_close(cfg["cfg_packed_position_ids"], expected, rtol=0, atol=0)


def test_batched_requests_concatenate_per_request_positions():
    video_shapes = [(1, 32, 48), (17, 96, 160), (45, 480, 832)]
    curr_kvlens = [4, 9, 2]
    curr_rope = [3, 40, 111]
    t_scale = lance_transformer.LANCE_TOKENS_PER_SECOND * lance_transformer.LANCE_SECONDS_PER_GRID

    torch.manual_seed(0)
    out = LanceBagel.prepare_video_latent(_bagel_stub(), curr_kvlens, curr_rope, video_shapes, _NEW_TOKEN_IDS)
    cfg = LanceBagel.prepare_video_latent_cfg(_bagel_stub(), curr_kvlens, curr_rope, video_shapes)

    expected = _reference_positions(video_shapes, curr_rope, t_scale)
    num_tokens = sum(t * h * w + 2 for t, h, w in map(_latent_grid, video_shapes))
    assert expected.shape == (3, num_tokens)
    for positions in (out["packed_position_ids"], cfg["cfg_packed_position_ids"]):
        assert positions.dtype == torch.long
        assert positions.is_contiguous()
        torch.testing.assert_close(positions, expected, rtol=0, atol=0)
    # The other packed metadata follows the same token layout.
    assert out["packed_indexes"].shape[0] == num_tokens
    assert cfg["cfg_packed_query_indexes"].shape[0] == num_tokens
    assert out["packed_seqlens"].tolist() == [t * h * w + 2 for t, h, w in map(_latent_grid, video_shapes)]


def test_prepare_video_latent_cfg_without_requests_is_empty():
    cfg = LanceBagel.prepare_video_latent_cfg(_bagel_stub(), [], [], [])
    assert cfg["cfg_packed_position_ids"].shape == (3, 0)
    assert cfg["cfg_packed_position_ids"].dtype == torch.long
    assert cfg["cfg_packed_query_indexes"].numel() == 0
