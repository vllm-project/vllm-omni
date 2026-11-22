# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU regressions for the pipeline's geometry, plan cache, refresh, latent output, and session lifecycle boundaries."""

import functools
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.cosmos3_nano_sim_auto import pipeline_cosmos3_nano_sim_auto as pipeline
from vllm_omni.diffusion.models.cosmos3_nano_sim_auto.causal_plan import STREAM_HISTORY
from vllm_omni.diffusion.models.cosmos3_nano_sim_auto.geometry import SensorGrid, build_chunk_schedule, lidar_grid
from vllm_omni.diffusion.models.cosmos3_nano_sim_auto.layout import (
    CaptionDocument,
    build_chunk_items,
    chunk_token_metadata,
    committed_metadata,
)
from vllm_omni.diffusion.models.cosmos3_nano_sim_auto.state_cosmos3_nano_sim_auto import (
    Cosmos3NanoSimAutoSessionState,
    DenseReplayRing,
    ring_slot_for_step,
)
from vllm_omni.diffusion.output_formatter import format_diffusion_outputs, normalize_diffusion_postprocess_output
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_pipeline_rgb_grid_uses_exported_compression() -> None:
    host = SimpleNamespace(manifest=SimpleNamespace(vae_spatial_compression_factor=16, latent_patch_size=2))
    grid = pipeline.Cosmos3NanoSimAutoPipeline._rgb_grid(host, 480, 832)
    assert grid.tokens_per_frame == 390


class _PipelineHost(SimpleNamespace):
    """Test host whose attributes it does not set are the pipeline's methods bound to it."""

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return functools.partial(getattr(pipeline.Cosmos3NanoSimAutoPipeline, name), self)


# -- Frame unit -------------------------------------------------------------------


class _EncodeReachedError(Exception):
    """Raised by the fake encoder: session preparation got past its frame-unit checks."""


def _frame_unit_host(*, lidar_fps: float = 10.0) -> _PipelineHost:
    def encode(*args, **kwargs):
        raise _EncodeReachedError

    return _PipelineHost(
        manifest=SimpleNamespace(
            # A joint seven-camera 480p artifact.
            cameras=tuple(f"camera_{index}" for index in range(7)),
            joint=True,
            default_resolution="480",
            default_fps=30.0,
            frames_per_chunk=3,
            temporal_compression_factor=4,
            vae_spatial_compression_factor=16,
            latent_patch_size=2,
            lidar_fps=lidar_fps,
            lidar_seconds_per_frame=1 / lidar_fps,
            cache_chunks=2,
            sink_chunks=1,
            text_cache_max_len=16384,
        ),
        # V1.2 LiDAR with the current 1x1 diffusion patch: 8 x 113 = 904 tokens per sweep.
        lidar_grid=lidar_grid(patch_size=(1, 1)),
        transformer=SimpleNamespace(num_hidden_layers=2, num_kv_heads_local=1, head_dim=8),
        _MAIN_BRANCH="main",
        _TEXT_CACHE="text",
        _SESSION_CAPACITY=1,
        _encode_camera_major=encode,
        _ar_diffusion_kv_state=None,
        _bound_session_id=None,
    )


def _bound_pool(unit: int, *, block_size: int | None = None, **spec) -> SimpleNamespace:
    spec = {"chunk_size": unit, "window_chunks": 1, "sink_chunks": 1, "reset_at_boundary": False, **spec}
    return SimpleNamespace(
        session_id="s",
        kv_cache=SimpleNamespace(spec=SimpleNamespace(**spec), block_size=unit if block_size is None else block_size),
    )


@pytest.mark.parametrize(("lidar_fps", "pool_unit"), [(10.0, 24032), (11.0, 25840)])
def test_pool_is_sized_once_for_the_artifact_worst_case(lidar_fps: float, pool_unit: int) -> None:
    host = _frame_unit_host(lidar_fps=lidar_fps)
    spec = host.ar_diffusion_kv_cache_spec()
    # Every camera, joint, and 640x640: the 480p aspect ratio with the most tokens.
    assert spec.tokens_per_frame == pool_unit
    assert pool_unit == host._frame_unit(num_views=7, joint=True, height=640, width=640, fps=30.0)
    assert pool_unit > host._frame_unit(num_views=7, joint=True, height=480, width=832, fps=30.0)
    assert spec.max_scratch_frames_per_branch == 0


@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("resolution", ["480", "720"])
def test_pool_admits_every_aspect_ratio_of_the_default_resolution(resolution: str, joint: bool) -> None:
    host = _frame_unit_host()
    host.manifest.default_resolution = resolution
    host.manifest.joint = joint
    pool_unit = host.ar_diffusion_kv_cache_spec().tokens_per_frame
    # At 720 every aspect ratio has the same pixel area, but 16:9 has the most tokens.
    units = [
        host._frame_unit(num_views=7, joint=joint, height=height, width=width, fps=30.0)
        for width, height in pipeline.VIDEO_RES_SIZE_INFO[resolution].values()
    ]
    assert pool_unit == max(units)


@pytest.mark.parametrize(
    ("num_views", "joint", "num_frames", "height", "width"),
    [
        (7, True, 297, 640, 640),  # the worst case itself
        (7, True, 9, 480, 832),  # another aspect ratio, two latents and two sweeps
        (7, True, 13, 480, 832),  # three latents but three of four sweeps
        (7, False, 297, 480, 832),  # RGB only on a joint artifact
        (3, False, 5, 480, 832),  # fewer cameras, one latent per camera
        (1, True, 37, 544, 736),
    ],
)
def test_smaller_requests_run_on_the_worst_case_pool_with_its_frame_unit(
    num_views: int, joint: bool, num_frames: int, height: int, width: int
) -> None:
    host = _frame_unit_host()
    pool_unit = host.ar_diffusion_kv_cache_spec().tokens_per_frame
    units: list[int] = []

    def session_frame_unit(*args, **kwargs):
        units.append(pipeline.Cosmos3NanoSimAutoPipeline._session_frame_unit(host, *args, **kwargs))
        return units[-1]

    host._session_frame_unit = session_frame_unit
    views = [{"camera_key": camera} for camera in host.manifest.cameras[:num_views]]
    own_unit = host._frame_unit(num_views=num_views, joint=joint, height=height, width=width, fps=30.0)
    assert own_unit <= pool_unit
    # The paged path uses the pool's unit; the dense path keeps the request geometry's own.
    for bound, expected in ((_bound_pool(pool_unit), pool_unit), (None, own_unit)):
        with pytest.raises(_EncodeReachedError):
            host._prepare_session(
                Cosmos3NanoSimAutoSessionState(session_id="s"),
                views=views,
                cameras=[view["camera_key"] for view in views],
                joint=joint,
                lidar_request=None,
                lidar_condition=False,
                has_condition=False,
                height=height,
                width=width,
                num_frames=num_frames,
                fps=30.0,
                seed=0,
                caption_ids=[],
                sink_tokens=0,
                bound=bound,
            )
        assert units[-1] == expected


def test_a_pool_whose_pages_or_unit_do_not_fit_is_rejected() -> None:
    host = _frame_unit_host()
    schedule = build_chunk_schedule(rgb_latents=2, lidar_sweeps=None, frames_per_chunk=3, rgb_seconds_per_frame=4 / 30)
    geometry = {"num_views": 7, "joint": False, "height": 480, "width": 832, "fps": 30.0}
    assert host._session_frame_unit(schedule, **geometry, bound=None) == 16384
    with pytest.raises(RuntimeError, match="pages must hold one frame unit"):
        host._session_frame_unit(schedule, **geometry, bound=_bound_pool(16384, block_size=16))
    with pytest.raises(RuntimeError, match="exceeds the session's frame unit 4096"):
        host._session_frame_unit(schedule, **geometry, bound=_bound_pool(4096))


class _AdmittedError(Exception):
    """Raised by the fake caption formatter: the request passed the pool-size admission check."""


@pytest.mark.parametrize(
    ("resolution", "width", "height", "admitted"), [("480", 832, 480, True), ("704", 1280, 704, False)]
)
def test_a_request_larger_than_the_pool_is_rejected_at_admission(
    monkeypatch, resolution: str, width: int, height: int, admitted: bool
) -> None:
    host = _frame_unit_host()
    pool_unit = host.ar_diffusion_kv_cache_spec().tokens_per_frame
    views = [{"camera_key": camera, "prompt": "Driving."} for camera in host.manifest.cameras]

    def captions(*args):
        raise _AdmittedError

    monkeypatch.setattr(pipeline, "reject_multiview_negative_fields", lambda *args: None)
    monkeypatch.setattr(pipeline, "validate_multiview_request", lambda *args, **kwargs: ({}, views))
    monkeypatch.setattr(pipeline, "_resolve_multiview_num_frames", lambda *args: 9)
    monkeypatch.setattr(
        pipeline, "_resolve_multiview_geometry", lambda *args, **kwargs: (resolution, "16,9", width, height)
    )
    monkeypatch.setattr(pipeline, "format_rig_view_captions", captions)
    host.manifest.validate_request_cameras = lambda cameras: None
    host.manifest.t_list = (1.0, 0.5)
    host._ar_diffusion_kv_state = _bound_pool(pool_unit)
    host._states = OrderedDict()
    host._request_fps = lambda sp: 30.0
    host._get_sp_param = lambda sp, name, default: default
    host._view_value = pipeline.Cosmos3NanoSimAutoPipeline._view_value
    sp = SimpleNamespace(extra_args={"wsm": "control", "session_id": "s"}, num_frames=9, num_inference_steps=None)
    request = SimpleNamespace(prompts=["Driving."], sampling_params=sp)

    if admitted:
        with pytest.raises(_AdmittedError):
            host._forward_impl(request)
        return
    request_unit = host._frame_unit(num_views=7, joint=False, height=height, width=width, fps=30.0)
    message = (
        f"needs a KV frame unit of {request_unit} tokens but the pool's is {pool_unit}; "
        "the pool is sized at load for the artifact's default resolution '480'"
    )
    with pytest.raises(pipeline.ARDiffusionRequestRejectedError, match=message):
        host._forward_impl(request)
    assert not host._states


@pytest.mark.parametrize(
    "contradiction",
    [
        {"window_chunks": 3},
        {"sink_chunks": 0},
        {"reset_at_boundary": True},
        {"block_size": 16},  # superpage_tokens
        {"chunk_size": 16384},  # a pool sized for another geometry
    ],
)
def test_binding_a_pool_that_contradicts_the_manifest_ring_fails(contradiction) -> None:
    host = _frame_unit_host()
    pool_unit = host.ar_diffusion_kv_cache_spec().tokens_per_frame
    with host.bind_ar_diffusion_state("s", _bound_pool(pool_unit)):
        assert host._bound_session_id == "s"
    name = next(iter(contradiction))
    with pytest.raises(ValueError, match=f"contradicts the manifest ring: {name}="):
        with host.bind_ar_diffusion_state("s", _bound_pool(pool_unit, **contradiction)):
            pass
    assert host._ar_diffusion_kv_state is None


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
def test_latents_are_opt_in_and_reach_offline_output(monkeypatch, joint: bool) -> None:
    rgb = torch.arange(16, dtype=torch.bfloat16).reshape(1, 2, 2, 2, 2)
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
    # Generic multiview processing returns video/LiDAR only; the Sim-Auto wrapper retains latents.
    monkeypatch.setattr(
        pipeline,
        "get_cosmos3_multiview_post_process_func",
        lambda config: lambda output, **kwargs: {"payload": {"video": output["payload"]["video"]}},
    )
    request = OmniDiffusionRequest(
        prompt="Driving.",
        sampling_params=OmniDiffusionSamplingParams(seed=0),
        request_id="sim-auto-latents",
    )
    process = pipeline.get_cosmos3_nano_sim_auto_post_process_func(None)

    def format_rollout(return_latents: bool):
        payload, metadata = pipeline.Cosmos3NanoSimAutoPipeline._build_payload(
            host, state, {} if joint else None, [], "480", "16,9", return_latents=return_latents
        )
        output = DiffusionOutput(output={"payload": payload, "metadata": metadata})
        processed = process(output.output)
        [result] = format_diffusion_outputs(
            request=request,
            od_config=SimpleNamespace(model_class_name="Cosmos3NanoSimAutoPipeline"),
            diffusion_output=output,
            output_data=processed,
            postprocess_output=normalize_diffusion_postprocess_output(processed),
        )
        assert result.final_output_type == "video" and len(result.images) == 1
        return result.latents

    assert format_rollout(False) is None
    latents = format_rollout(True)
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
        _runtime_plan=lambda state, metadata, bound, reuse_control_kv=False: (
            "noisy" if bool(metadata.is_noisy.any()) else "clean",
            reuse_control_kv,
        ),
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


# -- Plan cache and paged ring mapping ------------------------------------------


def _plan_host(window_seconds: float = 0.4) -> _PipelineHost:
    return _PipelineHost(
        manifest=SimpleNamespace(cross_view_past_window_seconds=window_seconds),
        device=torch.device("cpu"),
        _plan_cache={},
        _maskless_kernel="torch",
        _MAIN_BRANCH="main",
    )


def _plan_state(num_views: int, horizon_seconds: float, *, joint: bool = False) -> Cosmos3NanoSimAutoSessionState:
    return Cosmos3NanoSimAutoSessionState(
        session_id="s",
        cameras=tuple(f"camera_{view}" for view in range(num_views)),
        joint=joint,
        frame_unit=64,
        caption_documents=tuple(
            CaptionDocument(view, 3, start_seconds=0.0, end_seconds=horizon_seconds) for view in range(num_views)
        ),
    )


def _items(chunk, *, num_views: int, rgb_seconds_per_frame: float, lidar_seconds_per_frame: float | None = None):
    return build_chunk_items(
        chunk,
        num_views=num_views,
        rgb_grid=SensorGrid(1, 2, 1, 1),
        rgb_seconds_per_frame=rgb_seconds_per_frame,
        lidar_grid=SensorGrid(1, 1, 1, 1) if lidar_seconds_per_frame else None,
        lidar_seconds_per_frame=lidar_seconds_per_frame,
    )


@pytest.mark.parametrize(
    ("cache_chunks", "sink_chunks", "fps", "lidar_seconds_per_frame"),
    [
        (2, 1, 30.0, None),  # the reference ring and clock
        (2, 1, 60.0, None),  # 0.2 s chunks: the 0.4 s window reaches chunk 0 from chunk 2, not from chunk 3
        (3, 2, 60.0, None),  # likewise for the second pinned sink
        (2, 1, 30.0, 0.1),  # four sweeps per chunk, aligned with the chunk edges
        (2, 1, 30.0, 1 / 11),  # 4.4 sweeps per chunk: their phase against the chunk edges cycles
    ],
)
def test_cached_plans_equal_fresh_plans_through_a_rollout(
    cache_chunks: int, sink_chunks: int, fps: float, lidar_seconds_per_frame: float | None
) -> None:
    rgb_spf = 4 / fps
    latents = 22
    sweeps = None
    if lidar_seconds_per_frame is not None:
        # Every sweep up to the end of the last chunk.
        sweeps = int((latents - 1) * rgb_spf / lidar_seconds_per_frame + 1e-6) + 1
    schedule = build_chunk_schedule(
        rgb_latents=latents,
        lidar_sweeps=sweeps,
        frames_per_chunk=3,
        rgb_seconds_per_frame=rgb_spf,
        lidar_seconds_per_frame=lidar_seconds_per_frame,
    )
    state = _plan_state(2, latents * rgb_spf, joint=sweeps is not None)
    host = _plan_host()
    plans_before_last_chunk = 0
    for chunk in schedule:
        plans_before_last_chunk = len(host._plan_cache)
        items = _items(
            chunk, num_views=2, rgb_seconds_per_frame=rgb_spf, lidar_seconds_per_frame=lidar_seconds_per_frame
        )
        for pass_kind in ("noisy", "clean"):
            metadata, _ = chunk_token_metadata(items, step=chunk.step, pass_kind=pass_kind)
            cached = host._runtime_plan(state, metadata, None).plan
            fresh = _plan_host()._runtime_plan(state, metadata, None).plan
            assert len(cached) == len(fresh), (chunk.step, pass_kind, state.ring_steps)
            for actual, expected in zip(cached, fresh):
                assert torch.equal(actual, expected), (chunk.step, pass_kind, state.ring_steps)
        slot = ring_slot_for_step(chunk.step, cache_chunks=cache_chunks, sink_chunks=sink_chunks)
        state.record_commit(slot=slot, step=chunk.step, metadata=committed_metadata(metadata))
    # Steady-state chunks still share plans: the last one builds none.
    assert len(host._plan_cache) == plans_before_last_chunk


def _paged_ring_state(ring_steps: dict[int, int]) -> tuple[Cosmos3NanoSimAutoSessionState, object]:
    """Ring slots holding the given steps (4/30 s latents, 1 view) and the metadata of the next chunk."""
    schedule = build_chunk_schedule(rgb_latents=19, lidar_sweeps=None, frames_per_chunk=3, rgb_seconds_per_frame=4 / 30)
    state = _plan_state(1, 19 * 4 / 30)
    for slot, step in ring_steps.items():
        items = _items(schedule[step], num_views=1, rgb_seconds_per_frame=4 / 30)
        metadata, _ = chunk_token_metadata(items, step=step, pass_kind="clean")
        state.record_commit(slot=slot, step=step, metadata=committed_metadata(metadata))
    following = schedule[max(ring_steps.values()) + 1]
    items = _items(following, num_views=1, rgb_seconds_per_frame=4 / 30)
    return state, chunk_token_metadata(items, step=following.step, pass_kind="noisy")[0]


def test_paged_ring_with_two_sinks_maps_every_slot_to_its_page() -> None:
    # cache_chunks=3, sink_chunks=2 after step 3: slots 0 and 1 pin steps 0 and 1, slot 2 holds step 3.
    state, metadata = _paged_ring_state({0: 0, 1: 1, 2: 3})
    unit = state.frame_unit
    adapter = object()
    # The pool lists its resident pages in commit order.
    bound = SimpleNamespace(
        adapter=lambda branch: adapter,
        kv_cache=SimpleNamespace(block_size=unit, window_block_ids=lambda a: [7, 2, 5] if a is adapter else None),
    )
    staging = _plan_host()._runtime_plan(state, metadata, bound).plan[-1]
    rows = staging[staging[:, 0] == STREAM_HISTORY]
    # One staged row per slot, in slot order, each starting at its page.
    assert rows[:, 1].tolist() == [7 * unit, 2 * unit, 5 * unit]
    assert rows[:, 2].tolist() == [state.ring_metadata[slot].num_tokens for slot in range(3)]


def test_paged_ring_rejects_slots_out_of_commit_order() -> None:
    # Two rotating slots after a wrap-around (cache_chunks=3, sink_chunks=1): slot 1 holds the newer step 3, but
    # the pool's commit order lists step 2 first, so slot 1 would read step 2's K/V.
    state, metadata = _paged_ring_state({0: 0, 1: 3, 2: 2})
    bound = SimpleNamespace(
        adapter=lambda branch: None,
        kv_cache=SimpleNamespace(block_size=state.frame_unit, window_block_ids=lambda adapter: [0, 1, 2]),
    )
    with pytest.raises(RuntimeError, match="commit order"):
        _plan_host()._runtime_plan(state, metadata, bound)


# -- Session lifecycle ----------------------------------------------------------


class _RunnerSessionKV:
    """The runner-owned session KV as the pipeline sees it: a paged ring of sink plus one chunk.

    As with ``ARDiffusionKVState``, ``reset()`` releases the pages and leaves the
    session open, while ``finish_request()``, which the runner calls after every
    successful forward, fails on a closed session.
    """

    def __init__(self) -> None:
        self.pages: list[int] = []
        self.allocated = 0
        self.resets = 0
        self.closed = False

    def commit(self) -> None:
        # The pinned sink page stays; the window keeps only the newest chunk.
        self.pages = [*self.pages[:1], self.allocated]
        self.allocated += 1

    def reset(self) -> None:
        self.pages = []
        self.resets += 1

    def close(self) -> None:
        self.pages = []
        self.closed = True

    def finish_request(self) -> None:
        if self.closed:
            raise RuntimeError("AR-Diffusion session is closed")


def _session_host() -> SimpleNamespace:
    host = SimpleNamespace(_states=OrderedDict(), _SESSION_CAPACITY=1, _plan_cache={})
    for name in ("_drop_session", "_get_or_create_state", "_begin_rollout", "_end_rollout"):
        setattr(host, name, functools.partial(getattr(pipeline.Cosmos3NanoSimAutoPipeline, name), host))
    return host


def _rollout(host, bound, *, reset: bool, close_session: bool, chunks: int = 3) -> Cosmos3NanoSimAutoSessionState:
    """One admitted request's session side effects, committing every chunk but the last."""
    state = host._begin_rollout("default", reset=reset, bound=bound)
    state.fingerprint = "rollout"
    for step in range(chunks):
        if bound is not None:
            # The pool must hold exactly this rollout's commits; ``_runtime_plan`` raises otherwise.
            assert len(bound.pages) == len(state.ring_steps)
        if step < chunks - 1:
            slot = ring_slot_for_step(step, cache_chunks=2, sink_chunks=1)
            state.record_commit(slot=slot, step=step, metadata=None)
            if bound is not None:
                bound.commit()
    host._end_rollout("default", close_session=close_session, bound=bound)
    if bound is not None:
        bound.finish_request()
    return state


def test_default_requests_release_the_bound_ring_so_the_next_rollout_starts_empty() -> None:
    # Omitted flags resolve to reset=True and close_session=True in the pipeline, but to
    # False in the runner, which then neither resets nor closes its session.
    host, bound = _session_host(), _RunnerSessionKV()
    for _ in range(3):
        _rollout(host, bound, reset=True, close_session=True)
        assert bound.pages == []
        assert "default" not in host._states


def test_kept_session_holds_its_ring_until_a_reset_rollout_releases_it() -> None:
    host, bound = _session_host(), _RunnerSessionKV()
    kept = _rollout(host, bound, reset=False, close_session=False)
    assert bound.resets == 0 and bound.pages == [0, 1]
    assert host._states["default"] is kept and kept.fingerprint == "rollout"

    fresh = _rollout(host, bound, reset=True, close_session=False)
    assert bound.resets == 1 and bound.pages == [2, 3]
    assert host._states["default"] is fresh and kept.fingerprint is None


@pytest.mark.parametrize("reset", [True, False])
def test_each_rollout_starts_with_an_empty_plan_cache(reset: bool) -> None:
    # Plan keys include the request's caption lengths, so plans kept across rollouts
    # would accumulate device index tensors for every distinct prompt.
    host = _session_host()
    state = _plan_state(1, 19 * 4 / 30)
    chunk = build_chunk_schedule(rgb_latents=19, lidar_sweeps=None, frames_per_chunk=3, rgb_seconds_per_frame=4 / 30)[0]
    metadata, _ = chunk_token_metadata(
        _items(chunk, num_views=1, rgb_seconds_per_frame=4 / 30), step=0, pass_kind="noisy"
    )
    planner = _plan_host()
    planner._runtime_plan(state, metadata, None)
    host._plan_cache.update(planner._plan_cache)
    assert host._plan_cache

    host._begin_rollout("default", reset=reset, bound=None)
    assert host._plan_cache == {}


def test_dense_rollouts_drop_pipeline_state_by_the_same_flags() -> None:
    host = _session_host()
    _rollout(host, None, reset=True, close_session=True)
    assert "default" not in host._states
    kept = _rollout(host, None, reset=True, close_session=False)
    assert host._states["default"] is kept and kept.ring_steps == {0: 0, 1: 1}
    fresh = _rollout(host, None, reset=True, close_session=False)
    assert host._states["default"] is fresh and kept.fingerprint is None and not kept.ring_steps


# -- Lifecycle flags at admission -----------------------------------------------


def _admit(monkeypatch, extra_args) -> SimpleNamespace:
    """Run the Sim-Auto pre-process on a bare request and check it wraps the multiview guardrail."""
    guarded = []

    def guardrail(request):
        guarded.append(request)
        return request

    monkeypatch.setattr(pipeline, "get_cosmos3_multiview_pre_process_func", lambda config: guardrail)
    request = SimpleNamespace(sampling_params=SimpleNamespace(extra_args=extra_args))
    assert pipeline.get_cosmos3_nano_sim_auto_pre_process_func(None)(request) is request
    assert guarded == [request]
    return request


@pytest.mark.parametrize("extra_args", [{"session_id": "s"}, None])
def test_admission_writes_the_rollout_defaults_for_the_runner(monkeypatch, extra_args) -> None:
    # The runner reads omitted flags as False; the pipeline treats every request as a closed rollout.
    original = None if extra_args is None else dict(extra_args)
    extra = _admit(monkeypatch, extra_args).sampling_params.extra_args
    assert extra == {**(original or {}), "reset": True, "close_session": True}
    assert extra["reset"] is True and extra["close_session"] is True
    if extra_args is not None:
        assert extra_args == original  # the caller's mapping is not modified


@pytest.mark.parametrize(("value", "expected"), [(False, False), ("false", False), ("0", False), ("true", True)])
def test_admission_writes_explicit_flags_as_the_booleans_the_pipeline_uses(monkeypatch, value, expected) -> None:
    # The runner's bool("false") would be True.
    extra = _admit(monkeypatch, {"reset": value, "close_session": value}).sampling_params.extra_args
    assert extra["reset"] is expected and extra["close_session"] is expected
    assert pipeline._lifecycle_flags(extra) == (expected, expected)


@pytest.mark.parametrize("extra_args", [{"ar_diffusion_tick": {"session_id": "s"}}, ["reset"]])
def test_admission_leaves_ticks_and_malformed_extra_args_alone(monkeypatch, extra_args) -> None:
    snapshot = list(extra_args) if isinstance(extra_args, list) else dict(extra_args)
    assert _admit(monkeypatch, extra_args).sampling_params.extra_args is extra_args
    assert extra_args == snapshot
