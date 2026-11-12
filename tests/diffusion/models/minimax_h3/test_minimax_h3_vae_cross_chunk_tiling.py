# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU scheduling tests for the MiniMax-H3 cross-chunk tile decoder."""

import types

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.models.minimax_h3 import vae_reshape as reshape_module

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tokens_chunk_size = 2
        self.token_overlap = 0
        self.token_drop = 0
        self.vae_ratio = 1
        self.parallel_tiling = False
        self.h3_component_parallel = False
        self._h3_component_group_supported = True
        self.use_3d_conv = True
        self.use_vit_decoder = True
        self.decoder_tiling = True
        self.decoder = types.SimpleNamespace(spatial_parallel=False)
        self.isolated_first_frame = False
        self.isolated_last_frame = False

    def split_tiles(self, length, latent_space):
        assert latent_space is True
        starts = [0, 2]
        lengths = [2, 2]
        overlaps = [0, 0]
        return starts, lengths, overlaps

    def decode(self, z):
        return z.sum().reshape(1)

    def _run_tile_tasks(self, tiles, tile_indices, forward_fn, stack_tiling, cls_agg=None):
        return [forward_fn(tiles[index]) for index in tile_indices]

    def tiled_decode(self, clip):
        ordered = self._run_tile_tasks([], [], self.decode, True)
        if not ordered:
            return torch.zeros(1)
        return torch.stack([value.reshape(()) for value in ordered])

    def _skip_decode_output(self):
        return False


class _FakeAdapter:
    def __init__(self) -> None:
        self.model = _FakeModel()
        self.model.eval()
        self.config_dict = {"latent_channels": 1, "latents_mean": [0.0], "latents_std": [1.0]}

    def _native_parallel_state(self):
        return {"sp_size": 1, "sp_rank": 0, "sp_process_group": None}

    def _denormalize_latent(self, latent):
        return latent

    def _native_decode(self, latent):
        denormalized = self._denormalize_latent(latent)
        step = self.model.tokens_chunk_size
        chunks = (denormalized.shape[2] + self.model.token_drop) // step - int(self.model.token_drop > 0)
        outputs = []
        for chunk in range(chunks):
            clip = denormalized[:, :, chunk * step : (chunk + 1) * step]
            outputs.append(self.model.tiled_decode(clip))
        return torch.cat(outputs)

    def decode_latent(self, latent):
        return self._native_decode(latent)

    def decode_with_chunks(self, latent, *, on_chunk):
        result = self._native_decode(latent)
        on_chunk(result)


def _expected_tile_sums(latent: torch.Tensor) -> list[float]:
    step = 2
    rows = [0, 2]
    cols = [0, 2]
    expected = []
    for chunk in range(latent.shape[2] // step):
        for row in rows:
            for col in cols:
                expected.append(
                    float(latent[:, :, chunk * step : (chunk + 1) * step, row : row + 2, col : col + 2].sum())
                )
    return expected


def test_install_is_a_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("H3_VIDEO_DECODER_CROSS_CHUNK", raising=False)
    adapter = _FakeAdapter()
    original = adapter.decode_latent
    reshape_module.install_cross_chunk_tiling(adapter)
    assert adapter.decode_latent is original
    assert getattr(adapter, "_h3_vae_reshape_installed", False) is False


def test_install_wraps_entrypoints_and_completes_the_native_plan(monkeypatch):
    monkeypatch.setenv("H3_VIDEO_DECODER_CROSS_CHUNK", "1")
    monkeypatch.setenv("H3_VIDEO_DECODER_TILE_BATCHING", "1")
    monkeypatch.setenv("H3_VIDEO_DECODER_TILE_BATCH_SIZE", "4")
    adapter = _FakeAdapter()
    original_latent = adapter.decode_latent
    original_chunks = adapter.decode_with_chunks
    reshape_module.install_cross_chunk_tiling(adapter)
    assert adapter._h3_vae_reshape_installed is True
    assert adapter.decode_latent is not original_latent
    assert adapter.decode_with_chunks is not original_chunks

    latent = torch.arange(1, 65, dtype=torch.float32).reshape(1, 1, 4, 4, 4)
    output = adapter.decode_latent(latent)
    torch.testing.assert_close(output, torch.tensor(_expected_tile_sums(latent)))

    stats = adapter.model._h3_vae_reshape_last_stats
    assert stats["active"] is True
    assert stats["status"] == "complete"
    assert stats["chunks"] == 2
    assert stats["tiles"] == 4
    assert stats["total_items"] == 8
    assert stats["wave_items"] == [4, 4]
    assert stats["local_batch_sizes"] == [4, 4]
    assert stats["idle_item_slots"] == 0
    assert adapter.model.parallel_tiling is False
    assert "_run_tile_tasks" not in adapter.model.__dict__


def test_decode_with_chunks_uses_the_same_scheduler(monkeypatch):
    monkeypatch.setenv("H3_VIDEO_DECODER_CROSS_CHUNK", "1")
    monkeypatch.setenv("H3_VIDEO_DECODER_TILE_BATCHING", "1")
    monkeypatch.setenv("H3_VIDEO_DECODER_TILE_BATCH_SIZE", "4")
    adapter = _FakeAdapter()
    reshape_module.install_cross_chunk_tiling(adapter)
    latent = torch.arange(1, 65, dtype=torch.float32).reshape(1, 1, 4, 4, 4)
    received = []
    adapter.decode_with_chunks(latent, on_chunk=received.append)
    assert len(received) == 1
    torch.testing.assert_close(received[0], torch.tensor(_expected_tile_sums(latent)))
    assert adapter.model._h3_vae_reshape_last_stats["status"] == "complete"


def test_unsupported_geometry_keeps_the_native_path(monkeypatch):
    monkeypatch.setenv("H3_VIDEO_DECODER_CROSS_CHUNK", "1")
    adapter = _FakeAdapter()
    adapter.model.isolated_first_frame = True
    reshape_module.install_cross_chunk_tiling(adapter)
    latent = torch.arange(1, 65, dtype=torch.float32).reshape(1, 1, 4, 4, 4)
    adapter.decode_latent(latent)
    stats = adapter.model._h3_vae_reshape_last_stats
    assert stats["active"] is False
    assert "unsupported" in stats["reason"]


def test_wrong_clip_length_is_rejected(monkeypatch):
    monkeypatch.setenv("H3_VIDEO_DECODER_TILE_BATCHING", "1")
    monkeypatch.setenv("H3_VIDEO_DECODER_TILE_BATCH_SIZE", "4")
    adapter = _FakeAdapter()
    pool = reshape_module._TilePool(adapter, torch.zeros(1, 1, 4, 4, 4), all_outputs=False)
    with pytest.raises(RuntimeError, match="native temporal plan"):
        pool.decode_chunk(torch.zeros(1, 1, 1, 4, 4))
