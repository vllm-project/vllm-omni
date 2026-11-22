# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Geometry parity and exact noncontiguous history, including partial KV pages."""

import importlib.util
import os
from dataclasses import replace

import pytest
import torch
from test_page_unaligned_frames import SMALL, bound_state, pipeline

from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.dense_attention import CosmosSimDenseAttentionCache
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.frustum_cache import FrustumKVArchive
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.frustum_retrieval import FrustumHistorySelector
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def geometry(frames=401):
    poses = torch.eye(4, dtype=torch.float64).repeat(frames, 1, 1)
    # Revisit old locations rather than monotonically moving forward.
    poses[:, 0, 3] = torch.sin(torch.arange(frames, dtype=torch.float64) / 19) * 8
    return dict(c2w=poses.tolist(), image_size=[640, 360], horizontal_fov_degrees=90.0, near=0.1, far=10.0, fps=30.0)


def test_selector_past_only_chronological_and_ties():
    g = geometry()
    g["c2w"] = torch.eye(4).repeat(401, 1, 1).tolist()
    selector = FrustumHistorySelector(g, 101, 4, 30.0)
    assert selector.select(3, 5) == [0, 1, 2]
    assert selector.select(90, 92) == list(range(8)) + list(range(26, 90))
    torch.testing.assert_close(selector.scores([0], [0, 1]), torch.ones(2, dtype=torch.float64), rtol=0, atol=0)


def test_selector_revisits_old_location():
    g = geometry()
    poses = torch.eye(4).repeat(401, 1, 1)
    poses[4:356, 0, 3] = 100
    g.update(c2w=poses.tolist(), sink_size=0, retrieved_size=1, recent_size=1)
    selected = FrustumHistorySelector(g, 101, 4, 30.0).select(89, 91)
    assert selected == [0, 88]


@pytest.mark.parametrize(
    "change",
    [
        dict(fps=15),
        dict(far=0),
        dict(sink_size=True),
        dict(recent_size=-1),
        dict(horizontal_fov_degrees=180),
        dict(c2w=[torch.eye(4).tolist()]),
        dict(intrinsics=torch.eye(3).tolist()),
    ],
)
def test_geometry_rejects_invalid_inputs(change):
    g = geometry()
    g.update(change)
    with pytest.raises(ValueError):
        FrustumHistorySelector(g, 101, 4, 30.0)


def test_reference_selector_parity():
    path = os.environ.get("FRUSTUM_REFERENCE_MODULE")
    if not path:
        pytest.skip("Set FRUSTUM_REFERENCE_MODULE to compare the pinned reference module")
    spec = importlib.util.spec_from_file_location("frustum_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    g = geometry()
    actual = FrustumHistorySelector(g, 101, 4, 30.0)
    reference = module.FrustumHistorySelector.from_rgb_geometry(g, 101, 4, 30.0)
    for start in range(1, 101, 2):
        end = min(start + 2, 101)
        assert actual.select(start, end) == reference.select(start, end)
    torch.testing.assert_close(
        actual.scores([89, 90], list(range(8, 57))),
        reference.overlap_scores([89, 90], list(range(8, 57))),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("paged", [False, True])
def test_archive_exact_tokens_and_clean_prefix(paged):
    pipe = pipeline()
    pipe.manifest = replace(pipe.manifest, window_frames=32, sink_frames=0)

    # CosmosSimDenseAttentionCache keeps a weak reference to the owner.
    class Owner:
        pass

    owner = Owner()
    tokens = SMALL.tokens_per_frame(4)
    text = [(torch.zeros(1, 2, 1, 8), torch.zeros(1, 2, 1, 8)) for _ in range(2)]
    storage = None
    if paged:
        cache = bound_state(pipe, SMALL).kv_cache
        storage = ARDiffusionKVState(cache, "test", {"main": cache.begin_request("test")}, num_layers=2)
    stage = CosmosSimDenseAttentionCache()
    archive = FrustumKVArchive(owner, text, 2, tokens, 32, 6, 2, stage, storage)
    try:
        for frame in range(17):
            selected = sorted(set([0, max(0, frame - 3)])) if frame else []
            archive.select(frame, selected)
            archive.prepare(frame)
            expected = (
                torch.cat([torch.arange(f * tokens, (f + 1) * tokens) for f in selected])
                if selected
                else torch.tensor([])
            )
            for key, value in stage.kv:
                torch.testing.assert_close(key[0, 2 : 2 + len(expected), 0, 0], expected.float())
                torch.testing.assert_close(value[0, 2 : 2 + len(expected), 0, 0], -expected.float())
            offset = 2 + stage.history_length
            values = torch.arange(frame * tokens, (frame + 1) * tokens).float().view(1, tokens, 1, 1)
            for key, value in stage.kv:
                key[:, offset : offset + tokens].copy_(values)
                value[:, offset : offset + tokens].copy_(-values)
            archive.commit(1)
        # Framewise clean refresh adds only the already committed chunk prefix.
        archive.select(17, [0, 3, 11])
        archive.prepare(17)
        offset = 2 + stage.history_length
        for key, value in stage.kv:
            key[:, offset : offset + tokens].fill_(17)
            value[:, offset : offset + tokens].fill_(-17)
        archive.commit(1)
        archive.prepare(18)
        assert archive.prepared == (0, 3, 11, 17)
        assert stage.kv[0][0][0, 2 + 3 * tokens : 2 + 4 * tokens].eq(17).all()
        with pytest.raises(ValueError):
            archive.select(18, [3, 3])
        with pytest.raises(ValueError):
            archive.select(18, [18])
    finally:
        if storage is not None:
            storage.close()


@pytest.mark.parametrize("reason", ["window", "tick", "image"])
def test_retrieval_admission_is_fail_closed(reason):
    from test_rollout import fake_pipeline, request

    pipe, state = fake_pipeline(prefix=reason != "image")
    if reason != "window":
        pipe.manifest = replace(pipe.manifest, window_frames=None, sink_frames=0)
    extra = {"frustum_geometry": geometry(61)}
    if reason == "tick":
        extra.update(chunk_only=True, num_latent_frames=4)
    with pytest.raises(ValueError, match="Frustum retrieval requires"):
        pipe.forward(request(61, output_type="latent", **extra))
    assert state.fingerprint is None
    assert pipe._states == {}


def test_session_reset_releases_retrieval_archive():
    from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.state_cosmos3_nano_sim_bimanual import (
        Cosmos3NanoSimBimanualSessionState,
    )

    state = Cosmos3NanoSimBimanualSessionState("test")
    state.frustum_archive = object()
    state.reset()
    assert state.frustum_archive is None


def test_pose_source_camera_batch():
    from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.frustum_retrieval import resolve_frustum_geometry

    poses = torch.eye(4).repeat(5, 1, 1)
    poses[:, 0, 3] = torch.arange(5)
    config = {"pose_source": "camera_batch", "image_size": [640, 360]}
    result = resolve_frustum_geometry(config, fps=30, num_frames=5, camera_poses=poses)
    assert result["c2w"] == poses.tolist()
    assert result["fps"] == 30
    assert "pose_source" not in result
    assert config == {"pose_source": "camera_batch", "image_size": [640, 360]}
    for changes in (
        {"camera_poses": None},
        {"num_frames": 9},
        {"geometry": {**config, "fps": 15}},
        {"geometry": {**config, "c2w": poses}},
    ):
        kwargs = dict(geometry=config, fps=30, num_frames=5, camera_poses=poses)
        kwargs.update(changes)
        with pytest.raises(ValueError):
            resolve_frustum_geometry(**kwargs)


def test_pose_source_action_roundtrip_across_chunk_and_domain_boundaries():
    import math

    import numpy as np

    from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.camera import (
        camera_poses_to_actions,
        parse_camera_string,
    )
    from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.frustum_retrieval import resolve_frustum_geometry

    poses = parse_camera_string("w-17,right-16,a-7")
    raw = torch.from_numpy(camera_poses_to_actions(poses, "backward_chunk_anchored_16f"))
    lo = torch.linspace(-2, -1, 59)
    hi = torch.linspace(1, 3, 59)
    rows = torch.zeros(len(raw), 59)
    rows[:, :9] = torch.asinh((raw - (hi[:9] + lo[:9]) / 2) / ((hi[:9] - lo[:9]) / 2)) / math.asinh(1)
    original = rows.clone()
    kwargs = dict(
        geometry={"pose_source": "agibot_unified_v1_chunk16"},
        fps=30,
        num_frames=len(poses),
        action=rows,
        action_space="normalized_unified_v1",
        domain_id=[15] * 20 + [2] * 20,
        normalization_stats={"global": {"q01": lo.tolist(), "q99": hi.tolist()}},
    )
    result = resolve_frustum_geometry(**kwargs)
    np.testing.assert_allclose(result["c2w"], poses, atol=2e-6)
    assert torch.equal(rows, original)
    for changes in (
        {"domain_id": 3},
        {"domain_id": [15]},
        {"action_space": "raw"},
        {"normalization_stats": None},
        {"num_frames": 10},
    ):
        with pytest.raises(ValueError):
            resolve_frustum_geometry(**{**kwargs, **changes})


def test_pose_source_explicit_and_unknown():
    from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.frustum_retrieval import resolve_frustum_geometry

    g = geometry(5)
    assert resolve_frustum_geometry(g, fps=30, num_frames=5) == g
    with pytest.raises(ValueError, match="Unsupported"):
        resolve_frustum_geometry({"pose_source": "unknown"}, fps=30, num_frames=5)
