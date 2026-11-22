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
    state = Cosmos3NanoSimAutoSessionState(session_id="test", dense_ring=ring, rig_view_ids=torch.tensor([5]))
    host = SimpleNamespace(
        manifest=SimpleNamespace(cache_chunks=2, sink_chunks=1),
        device=torch.device("cpu"),
        _runtime_plan=lambda *args, **kwargs: None,
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
            # Committed K/V must carry the same rig identity as the denoising passes.
            assert kwargs["rig_view_ids"] is state.rig_view_ids
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


# -- Control K/V reuse ----------------------------------------------------------


def _joint_chunk():
    schedule = build_chunk_schedule(
        rgb_latents=7,
        lidar_sweeps=9,
        frames_per_chunk=3,
        rgb_seconds_per_frame=4 / 30,
        lidar_seconds_per_frame=0.1,
    )
    chunk = schedule[1]
    items = build_chunk_items(
        chunk,
        num_views=1,
        rgb_grid=SensorGrid(1, 1, 1, 1),
        rgb_seconds_per_frame=4 / 30,
        lidar_grid=SensorGrid(1, 2, 1, 1),
        lidar_seconds_per_frame=0.1,
    )
    controls = pipeline.Cosmos3NanoSimAutoPipeline._token_rows(items, controls=True, device="cpu")
    targets = pipeline.Cosmos3NanoSimAutoPipeline._token_rows(items, controls=False, device="cpu")
    return schedule, chunk, items, controls, targets


def _cache(controls: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    values = 100.0 + torch.arange(controls.numel(), dtype=torch.float32)
    return [(values.view(-1, 1, 1).expand(-1, 1, 2).clone(), -values.view(-1, 1, 1).expand(-1, 1, 2).clone())]


@pytest.mark.parametrize("reuse", [True, False])
def test_chunk_collects_control_kv_once_and_reuses_it_through_the_commit(monkeypatch, reuse: bool) -> None:
    schedule = build_chunk_schedule(rgb_latents=7, lidar_sweeps=None, frames_per_chunk=3, rgb_seconds_per_frame=4 / 30)
    chunk = schedule[1]
    items = build_chunk_items(chunk, num_views=1, rgb_grid=SensorGrid(1, 1, 1, 1), rgb_seconds_per_frame=4 / 30)
    frames = schedule[-1].rgb_prefix_end
    state = Cosmos3NanoSimAutoSessionState(
        session_id="test",
        cameras=("front",),
        schedule=schedule,
        rgb_control_latents=torch.zeros(1, 2, frames, 1, 1),
        rgb_target_latents=torch.zeros(1, 2, frames, 1, 1),
    )
    cache = [(torch.ones(3, 1, 2), torch.ones(3, 1, 2))]
    calls, commits = [], []

    def transformer(**kwargs):
        calls.append(kwargs)
        target = kwargs["latents"][1]
        return SimpleNamespace(
            targets=[torch.zeros_like(target)], control_kv=cache if kwargs["collect_control_kv"] else None
        )

    def sample(velocity, noise, **kwargs):
        for step in range(4):
            velocity(noise, torch.full((1, 1), 900.0 - 200.0 * step))
        return noise

    monkeypatch.setattr(pipeline, "fixed_step_sample", sample)
    host = SimpleNamespace(
        _reuse_control_kv=reuse,
        device=torch.device("cpu"),
        manifest=SimpleNamespace(t_list=(0.9, 0.7, 0.5, 0.3), num_train_timesteps=1000, sample_type="sde"),
        transformer=transformer,
        _items_for_chunk=lambda state, chunk: items,
        _und_kv=lambda state, bound: [],
        _history_kv=lambda state, bound: None,
        _item_latents=lambda state, items: [torch.zeros(1, 2, len(item.frames), 1, 1) for item in items],
        _runtime_plan=lambda state, chunk, metadata, kind, bound, reuse_control_kv=False: (kind, reuse_control_kv),
        _camera_major_write=pipeline.Cosmos3NanoSimAutoPipeline._camera_major_write,
        _camera_major_slice=pipeline.Cosmos3NanoSimAutoPipeline._camera_major_slice,
        _commit_chunk=lambda *args, control_kv=None: commits.append(control_kv),
    )
    pipeline.Cosmos3NanoSimAutoPipeline._run_chunk(host, state, chunk, None)

    first, *later = calls
    assert len(later) == 3
    assert first["control_kv"] is None and first["collect_control_kv"] is reuse
    assert first["runtime"] == ("noisy", False)
    for call in later:
        assert call["collect_control_kv"] is False
        assert call["control_kv"] is (cache if reuse else None)
        assert call["runtime"] == ("noisy", reuse)
    assert len(commits) == 1 and commits[0] is (cache if reuse else None)
    assert state.next_chunk == chunk.step + 1


def test_dense_commit_publishes_cached_controls_between_refreshed_targets() -> None:
    _, chunk, items, controls, targets = _joint_chunk()
    num_tokens = controls.numel() + targets.numel()
    ring = DenseReplayRing(
        num_layers=1,
        cache_chunks=2,
        frame_unit=num_tokens + 2,
        num_kv_heads=1,
        head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    state = Cosmos3NanoSimAutoSessionState(session_id="test", dense_ring=ring)
    cache = _cache(controls)
    computed = 1.0 + torch.arange(targets.numel(), dtype=torch.float32).view(1, -1, 1, 1).expand(1, -1, 1, 2)

    def refresh(**kwargs):
        assert kwargs["control_kv"] is cache and kwargs["collect_kv"]
        return SimpleNamespace(current_kv=[(computed.clone(), -computed.clone())])

    host = SimpleNamespace(
        manifest=SimpleNamespace(cache_chunks=2, sink_chunks=1),
        device=torch.device("cpu"),
        transformer=refresh,
        _runtime_plan=lambda *args, **kwargs: None,
        _item_latents=lambda *args: [],
        _chunk_kv=pipeline.Cosmos3NanoSimAutoPipeline._chunk_kv,
    )
    pipeline.Cosmos3NanoSimAutoPipeline._commit_chunk(
        host, state, chunk, items, [], ring.layer_kv(), None, control_kv=cache
    )
    published = ring.keys[0][ring.slot_base(1) : ring.slot_base(1) + num_tokens]
    torch.testing.assert_close(published[controls], cache[0][0], rtol=0, atol=0)
    torch.testing.assert_close(published[targets], computed[0], rtol=0, atol=0)
    torch.testing.assert_close(ring.values[0][ring.slot_base(1) + targets], -computed[0], rtol=0, atol=0)


def test_paged_commit_writes_cached_controls_before_the_targets_only_refresh() -> None:
    _, chunk, items, controls, targets = _joint_chunk()
    unit = controls.numel() + targets.numel() + 2
    key_cache, value_cache = torch.ones(4, unit, 1, 2), torch.ones(4, unit, 1, 2)
    pools = [(key_cache.flatten(0, 1), value_cache.flatten(0, 1))]
    committed = []
    adapter = SimpleNamespace(num_computed_tokens=unit)  # the chunk lands in table block 1 -> physical block 3
    bound = SimpleNamespace(
        adapter=lambda branch: adapter,
        kv_cache=SimpleNamespace(
            block_size=unit,
            allocate_token_slots=lambda adapter, count: [0, 3],
            commit_chunk=lambda adapter: committed.append(adapter),
        ),
    )
    state = Cosmos3NanoSimAutoSessionState(session_id="test", frame_unit=unit)
    cache = _cache(controls)
    base = 3 * unit

    def refresh(**kwargs):
        assert kwargs["control_kv"] is cache and not kwargs["collect_kv"]
        # The targets-only forward writes its own tokens; the controls are already in their pages.
        assert torch.equal(kwargs["write_slots"], base + targets)
        torch.testing.assert_close(pools[0][0][base + controls], cache[0][0], rtol=0, atol=0)
        torch.testing.assert_close(pools[0][1][base + controls], cache[0][1], rtol=0, atol=0)
        return SimpleNamespace(current_kv=[])

    host = SimpleNamespace(
        manifest=SimpleNamespace(cache_chunks=2, sink_chunks=1),
        device=torch.device("cpu"),
        transformer=refresh,
        _MAIN_BRANCH="main",
        _runtime_plan=lambda *args, **kwargs: None,
        _item_latents=lambda *args: [],
        _flat_kv_pools=lambda kv_cache: pools,
        _token_rows=pipeline.Cosmos3NanoSimAutoPipeline._token_rows,
    )
    pipeline.Cosmos3NanoSimAutoPipeline._commit_chunk(host, state, chunk, items, [], None, bound, control_kv=cache)
    assert committed == [adapter]
    assert not pools[0][0][base + unit - 2 : base + unit].any()  # padding slots are zero-filled
    assert state.ring_steps[1] == chunk.step
