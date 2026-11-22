# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU regressions for the pipeline's geometry, refresh, and latent output boundaries."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.cosmos3_nano_sim_auto import pipeline_cosmos3_nano_sim_auto as pipeline
from vllm_omni.diffusion.models.cosmos3_nano_sim_auto.geometry import SensorGrid, build_chunk_schedule
from vllm_omni.diffusion.models.cosmos3_nano_sim_auto.layout import build_chunk_items
from vllm_omni.diffusion.models.cosmos3_nano_sim_auto.state_cosmos3_nano_sim_auto import (
    Cosmos3NanoSimAutoSessionState,
    DenseReplayRing,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_pipeline_rgb_grid_uses_exported_compression() -> None:
    host = SimpleNamespace(manifest=SimpleNamespace(vae_spatial_compression_factor=16, latent_patch_size=2))
    grid = pipeline.Cosmos3NanoSimAutoPipeline._rgb_grid(host, 480, 832)
    assert grid.tokens_per_frame == 390


def test_dense_refresh_reads_previous_chunk_until_all_layers_finish() -> None:
    ring = DenseReplayRing(
        num_layers=2,
        cache_chunks=2,
        frame_unit=8,
        num_kv_heads=1,
        head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    state = Cosmos3NanoSimAutoSessionState(session_id="test", dense_ring=ring)
    host = SimpleNamespace(
        manifest=SimpleNamespace(cache_chunks=2, sink_chunks=1),
        device=torch.device("cpu"),
        _runtime_plan=lambda *args: None,
        _item_latents=lambda *args: [],
    )
    schedule = build_chunk_schedule(rgb_latents=12, lidar_sweeps=None, frames_per_chunk=3, rgb_seconds_per_frame=4 / 30)
    for chunk in schedule:
        items = build_chunk_items(
            chunk,
            num_views=1,
            rgb_grid=SensorGrid(1, 1, 1, 1),
            rgb_seconds_per_frame=4 / 30,
        )
        before = [(key.clone(), value.clone()) for key, value in ring.layer_kv()]

        def refresh(**kwargs):
            assert kwargs["write_slots"] is None and kwargs["kv_pools"] is None
            assert kwargs["collect_kv"]
            for actual, expected in zip(kwargs["history_kv"], before, strict=True):
                torch.testing.assert_close(actual[0], expected[0])
                torch.testing.assert_close(actual[1], expected[1])
            # Higher-layer refresh depends on the old history. Early eviction changes this value.
            count = 2 * len(chunk.rgb_frames)
            staged = [
                (
                    torch.full((1, count, 1, 2), chunk.step + layer + 1.0),
                    torch.full((1, count, 1, 2), float(before[layer][1].sum()) + 1),
                )
                for layer in range(2)
            ]
            return SimpleNamespace(current_kv=staged)

        host.transformer = refresh
        pipeline.Cosmos3NanoSimAutoPipeline._commit_chunk(host, state, chunk, items, [], ring.layer_kv(), None)
        slot = 0 if chunk.step == 0 else 1
        base = ring.slot_base(slot)
        count = 2 * len(chunk.rgb_frames)
        for layer in range(2):
            assert torch.all(ring.keys[layer][base : base + count] == chunk.step + layer + 1)
            assert torch.all(ring.values[layer][base : base + count] == before[layer][1].sum() + 1)
            assert not ring.keys[layer][base + count : base + ring.frame_unit].any()
            if slot == 1:
                torch.testing.assert_close(ring.keys[layer][:8], before[layer][0][:8])
        assert state.ring_steps[slot] == chunk.step

    before = [(key.clone(), value.clone()) for key, value in ring.layer_kv()]

    def fail(**kwargs):
        raise RuntimeError("refresh failed")

    host.transformer = fail
    with pytest.raises(RuntimeError, match="refresh failed"):
        pipeline.Cosmos3NanoSimAutoPipeline._commit_chunk(host, state, chunk, items, [], ring.layer_kv(), None)
    for actual, expected in zip(ring.layer_kv(), before, strict=True):
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])


@pytest.mark.parametrize("joint", [False, True])
def test_latents_are_opt_in_and_survive_postprocessing(monkeypatch, joint: bool) -> None:
    rgb = torch.arange(16, dtype=torch.float32).reshape(1, 2, 2, 2, 2)
    lidar = rgb + 100 if joint else None
    state = Cosmos3NanoSimAutoSessionState(
        session_id="test",
        cameras=("front", "rear"),
        joint=joint,
        rgb_target_latents=rgb,
        lidar_target_latents=lidar,
    )
    host = SimpleNamespace(
        manifest=SimpleNamespace(frames_per_chunk=3, checkpoint_id="test"),
        _decode_latents=lambda value: torch.zeros(1, 3, value.shape[2], 2, 2),
    )
    args = (host, state, {} if joint else None, [], "480", "16,9")
    payload, metadata = pipeline.Cosmos3NanoSimAutoPipeline._build_payload(*args)
    assert "latents" not in payload
    payload, metadata = pipeline.Cosmos3NanoSimAutoPipeline._build_payload(*args, return_latents=True)
    # Generic multiview processing returns video/LiDAR only; the Sim-Auto wrapper retains latents.
    monkeypatch.setattr(
        pipeline,
        "get_cosmos3_multiview_post_process_func",
        lambda config: lambda output, **kwargs: {"payload": {"video": output["payload"]["video"]}},
    )
    process = pipeline.get_cosmos3_nano_sim_auto_post_process_func(None)
    latents = process({"payload": payload, "metadata": metadata})["payload"]["latents"]
    torch.testing.assert_close(latents["vision_latent"], rgb, rtol=0, atol=0)
    assert latents["vision_latent"].device.type == "cpu"
    if joint:
        torch.testing.assert_close(latents["lidar_latent"], lidar, rtol=0, atol=0)
    else:
        assert set(latents) == {"vision_latent"}
