# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Artifact contract and parallel configuration validation for Cosmos3-Nano-Sim-Auto."""

import copy
from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.models.cosmos3_nano_sim_auto.config import (
    Cosmos3NanoSimAutoManifest,
    validate_sim_auto_parallel_config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

# The full MADS rig in physical-ID order, as the imaginaire4 exporter writes it.
_CAMERAS = [
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
    "camera_rear_left_70fov",
    "camera_cross_left_120fov",
    "camera_front_tele_30fov",
    "camera_front_fisheye_200fov",
    "camera_left_fisheye_200fov",
    "camera_right_fisheye_200fov",
    "camera_rear_fisheye_200fov",
]
_RIG = {"num_embeddings": 12, "camera_ids": {camera: row for row, camera in enumerate(_CAMERAS)}, "lidar_id": 11}


def _artifact(**overrides) -> dict:
    """Schema-v1 RGB artifact as ``build_sim_auto_manifest`` emits it for the Stage 3 selector."""
    artifact = {
        "schema_version": 1,
        "checkpoint_id": "sim_auto_joint_stage3_sfdmd_mads_all@iter_000003500",
        "checkpoint_iteration": 3500,
        "checkpoint_hash": "0" * 64,
        "attention_mode": "multiview",
        "video_temporal_causal": True,
        "cameras": list(_CAMERAS),
        "rig_view_embedding": copy.deepcopy(_RIG),
        "rgb": {
            "latent_channels": 48,
            "latent_patch_size": 2,
            "vae_spatial_compression_factor": 16,
            "temporal_compression_factor": 4,
        },
        "lidar": None,
        "lidar_latent_patch_size_hw": None,
        "chunk": {"frames_per_chunk": 3},
        "ring": {"cache_chunks": 2, "sink_chunks": 1},
        "replay": {
            "control_visibility": "causal",
            "controls_read_strict_past_clean_rgb": True,
            "clean_pass_causality": "chunk",
            "attention_scope": "decomposed",
            "cross_view_past_window_seconds": 0.4,
            "deduplicate_cross_view": True,
            "lidar_attends_captions": True,
        },
        "prompt": {
            "policy": "segmented",
            "text_sink_tokens": 4,
            "text_position_mode": "fixed",
            "context_before": 0,
            "context_after": 0,
            "media_text_extent": None,
            "media_rope_scale": 1.0,
            "sink_age_cap_seconds": None,
            "use_first_chunk_only": True,
            "system_prompt_rgb": "av_multiview_transfer",
            "system_prompt_joint": "av_joint_camera_lidar_transfer",
            "fractional_duration": False,
            "emphasize_control_in_prompt": False,
            "text_cache_max_len": 16384,
        },
        "mrope": {
            "temporal_modality_margin": 15000,
            "reset_spatial_ids": True,
            "base_fps": 30.0,
            "enable_fps_modulation": True,
        },
        "fixed_step_sampler_config": {
            "t_list": [1.0, 0.9375, 5 / 6, 0.625],
            "sample_type": "sde",
            "num_train_timesteps": 1000,
        },
        "inference_defaults": {"resolution": "480", "fps": 30.0, "guidance": 1.0, "control_guidance": 1.0},
    }
    artifact.update(overrides)
    return artifact


def _parallel(heads: tuple[int, int] = (32, 8), **fields) -> SimpleNamespace:
    transformer = {"num_attention_heads": heads[0], "num_key_value_heads": heads[1]}
    return SimpleNamespace(parallel_config=SimpleNamespace(**fields), tf_model_config=transformer)


@pytest.mark.parametrize(
    "fields",
    [
        {"sequence_parallel_size": 1, "cfg_parallel_size": 1},
        {"tensor_parallel_size": 8},
        {"ulysses_degree": 8, "sequence_parallel_size": 8},
        {"tensor_parallel_size": 2, "ulysses_degree": 4, "sequence_parallel_size": 4, "ulysses_mode": "strict"},
    ],
)
def test_parallel_config_accepts_tp_and_strict_ulysses(fields: dict) -> None:
    validate_sim_auto_parallel_config(_parallel(**fields))


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"cfg_parallel_size": 2}, "cfg_parallel_size>1"),
        ({"ulysses_degree": 2, "ring_degree": 2, "sequence_parallel_size": 4}, "ring_degree>1"),
        ({"ulysses_degree": 2, "allgather_degree": 2, "sequence_parallel_size": 4}, "allgather_degree>1"),
        ({"sequence_parallel_size": 2}, "must equal ulysses_degree"),
        ({"ulysses_degree": 2, "sequence_parallel_size": 2, "ulysses_mode": "advanced_uaa"}, "ulysses_mode='strict'"),
        ({"ulysses_degree": 2, "sequence_parallel_size": 2, "ulysses_a2a_permute": True}, "ulysses_a2a_permute"),
        ({"tensor_parallel_size": 4, "ulysses_degree": 4, "sequence_parallel_size": 4}, r"KV heads \(8\)"),
    ],
)
def test_parallel_config_rejects_unsupported_topologies(fields: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_sim_auto_parallel_config(_parallel(**fields))


def test_parallel_config_checks_query_heads_too() -> None:
    with pytest.raises(ValueError, match=r"query heads \(12\)"):
        validate_sim_auto_parallel_config(_parallel((12, 8), ulysses_degree=8, sequence_parallel_size=8))


def test_rig_table_maps_request_cameras_to_physical_rows() -> None:
    manifest = Cosmos3NanoSimAutoManifest.from_artifact(_artifact())
    assert manifest.rig_view_embedding == _RIG
    # Request order, not rig order: the canonical 7-view set happens to be rows 0..6.
    assert manifest.rig_camera_ids(["camera_cross_left_120fov", "camera_front_wide_120fov"]) == [5, 0]
    assert manifest.rig_camera_ids(_CAMERAS[:7]) == list(range(7))


@pytest.mark.parametrize("present", [False, True])
def test_artifacts_without_a_rig_table_still_load(present: bool) -> None:
    artifact = _artifact(rig_view_embedding=None)
    if not present:
        del artifact["rig_view_embedding"]  # Early schema-v1 exports omit the key.
    manifest = Cosmos3NanoSimAutoManifest.from_artifact(artifact)
    assert manifest.rig_view_embedding is None
    assert manifest.rig_camera_ids(_CAMERAS[:2]) is None


@pytest.mark.parametrize(
    ("rig", "match"),
    [
        ({**_RIG, "lidar_id": 10}, "lidar_id must be the final row"),
        ({**_RIG, "camera_ids": {**_RIG["camera_ids"], "camera_front_wide_120fov": 11}}, r"camera_ids\["),
        ({**_RIG, "camera_ids": dict(list(_RIG["camera_ids"].items())[:7])}, "name exactly the exported cameras"),
        ({**_RIG, "extra": 1}, "Unknown Cosmos3 multiview rig_view_embedding fields"),
    ],
)
def test_invalid_rig_tables_are_rejected(rig: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Cosmos3NanoSimAutoManifest.from_artifact(_artifact(rig_view_embedding=rig))


def test_unknown_root_fields_are_still_rejected() -> None:
    with pytest.raises(ValueError, match=r"unknown=\['view_embedding'\]"):
        Cosmos3NanoSimAutoManifest.from_artifact(_artifact(view_embedding=None))
