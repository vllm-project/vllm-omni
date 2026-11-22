# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Strict artifact contract of Cosmos3-Nano-Sim-Auto."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.models.cosmos3_nano_sim_auto.config import (
    Cosmos3NanoSimAutoManifest,
    validate_sim_auto_parallel_config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

FIXTURE = Path(__file__).parent / "fixtures" / "sim_auto_artifact.json"


def artifact() -> dict:
    return json.loads(FIXTURE.read_text())


def test_fixture_parses_with_reference_values() -> None:
    manifest = Cosmos3NanoSimAutoManifest.from_artifact(artifact())
    assert manifest.joint and manifest.lidar_fps == 10.0 and manifest.lidar_seconds_per_frame == 0.1
    assert manifest.frames_per_chunk == 3 and manifest.cache_chunks == 2 and manifest.sink_chunks == 1
    assert manifest.cross_view_past_window_seconds == 0.4
    assert manifest.t_list == (1.0, 0.9375, 0.8333333333333334, 0.625)
    assert manifest.chunk_seconds == pytest.approx(0.4)
    assert manifest.lidar_latent_patch_size_hw == (2, 2)
    assert len(manifest.cameras) == 11 and manifest.text_sink_tokens == 4
    assert manifest.digest == Cosmos3NanoSimAutoManifest.from_artifact(artifact()).digest


def test_rgb_only_artifact_drops_lidar() -> None:
    payload = artifact()
    payload["lidar"] = None
    payload["lidar_latent_patch_size_hw"] = None
    manifest = Cosmos3NanoSimAutoManifest.from_artifact(payload)
    assert not manifest.joint and manifest.lidar_fps is None
    payload["lidar_latent_patch_size_hw"] = [2, 2]
    with pytest.raises(ValueError, match="requires a lidar block"):
        Cosmos3NanoSimAutoManifest.from_artifact(payload)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("replay", "deduplicate_cross_view"), False, "replay.deduplicate_cross_view"),
        (("replay", "control_visibility"), "global", "replay.control_visibility"),
        (("prompt", "policy"), "static", "prompt.policy"),
        (("ring", "sink_chunks"), 2, "at least one non-sink chunk"),
        (("fixed_step_sampler_config", "t_list"), [0.9, 0.5], "start at 1.0"),
        (("video_temporal_causal",), False, "video_temporal_causal=true"),
        (("attention_mode",), "two_way", "attention_mode"),
        (("cameras",), ["camera_front_wide_120fov", "not_a_camera"], "MADS camera keys"),
        (("prompt", "system_prompt_rgb"), "unknown", "system_prompt_rgb"),
    ],
)
def test_rejects_unsupported_values(path: tuple[str, ...], value, match: str) -> None:
    payload = artifact()
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=match):
        Cosmos3NanoSimAutoManifest.from_artifact(payload)


def test_exact_field_sets() -> None:
    payload = artifact()
    payload["extra"] = 1
    with pytest.raises(ValueError, match="unknown=\\['extra'\\]"):
        Cosmos3NanoSimAutoManifest.from_artifact(payload)
    payload = artifact()
    del payload["ring"]["sink_chunks"]
    with pytest.raises(ValueError, match="missing=\\['sink_chunks'\\]"):
        Cosmos3NanoSimAutoManifest.from_artifact(payload)


def test_from_od_config_requires_backbone_and_rejects_deploy_envelopes() -> None:
    payload = artifact()
    config = SimpleNamespace(
        tf_model_config={"backbone_type": "cosmos3_multiview", "cosmos3_nano_sim_auto": payload},
        custom_pipeline_args=None,
        model_config=None,
    )
    assert Cosmos3NanoSimAutoManifest.from_od_config(config).checkpoint_iteration == 3600
    config.tf_model_config = {"backbone_type": "cosmos3", "cosmos3_nano_sim_auto": payload}
    with pytest.raises(ValueError, match="backbone_type"):
        Cosmos3NanoSimAutoManifest.from_od_config(config)
    config.tf_model_config = {"backbone_type": "cosmos3_multiview"}
    with pytest.raises(ValueError, match="requires transformer/config.json"):
        Cosmos3NanoSimAutoManifest.from_od_config(config)
    config.tf_model_config = {"backbone_type": "cosmos3_multiview", "cosmos3_nano_sim_auto": payload}
    config.model_config = {"cosmos3_nano_sim_auto": copy.deepcopy(payload)}
    with pytest.raises(ValueError, match="artifact envelopes"):
        Cosmos3NanoSimAutoManifest.from_od_config(config)
    config.model_config = {"ring": {"cache_chunks": 8, "sink_chunks": 1}}
    with pytest.raises(ValueError, match="must not be placed at deploy root"):
        Cosmos3NanoSimAutoManifest.from_od_config(config)


def test_request_cameras_must_be_exported_rig_subset() -> None:
    manifest = Cosmos3NanoSimAutoManifest.from_artifact(artifact())
    manifest.validate_request_cameras(["camera_front_wide_120fov", "camera_rear_tele_30fov"])
    with pytest.raises(ValueError, match="not part of the exported rig"):
        manifest.validate_request_cameras(["camera_front_wide_120fov", "camera_unknown"])
    with pytest.raises(ValueError, match="unique"):
        manifest.validate_request_cameras(["camera_front_wide_120fov", "camera_front_wide_120fov"])


def test_parallel_config_rejects_sp_and_cfg_parallel() -> None:
    ok = SimpleNamespace(parallel_config=SimpleNamespace(sequence_parallel_size=1, cfg_parallel_size=1))
    validate_sim_auto_parallel_config(ok)
    bad = SimpleNamespace(parallel_config=SimpleNamespace(sequence_parallel_size=2))
    with pytest.raises(ValueError, match="sequence_parallel_size>1"):
        validate_sim_auto_parallel_config(bad)
    bad = SimpleNamespace(parallel_config=SimpleNamespace(cfg_parallel_size=2))
    with pytest.raises(ValueError, match="cfg_parallel_size>1"):
        validate_sim_auto_parallel_config(bad)
