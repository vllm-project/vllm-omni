# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_omni.diffusion.models.cosmos_dreams.action_contract import (
    CosmosDreamsActionSchema,
    canonical_sha256,
    float32_value,
)
from vllm_omni.diffusion.models.cosmos_dreams.action_packing import (
    build_interleaved_mrope_position_ids,
    interleave_action_vision_tokens,
    split_interleaved_action_vision_tokens,
    zero_null_action_values,
)
from vllm_omni.diffusion.models.cosmos_dreams.config import CosmosDreamsManifest
from vllm_omni.diffusion.models.cosmos_dreams.normalizer import (
    ActionAffineNormalizer,
    QuantileRotAffineNormalizer,
)
from vllm_omni.diffusion.models.cosmos_dreams.sampler import CosmosDreamsDistilledSampler
from vllm_omni.diffusion.models.cosmos_dreams.state_cosmos_dreams import (
    CosmosDreamsSessionFingerprint,
    CosmosDreamsSessionState,
    append_dense_kv_history,
)
from vllm_omni.diffusion.models.cosmos_dreams.utils import (
    estimate_kv_memory_bytes,
    iter_ar_chunk_ranges,
    iter_clean_commit_frames,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _action_layout() -> dict[str, Any]:
    return {
        "id": "agibot_backward_framewise_rot6d_v1",
        "pose_convention": "backward_framewise",
        "delta_equation": "T_i^-1 @ T_{i+1}",
        "rotation_representation": "rot6d_columns",
        "fields": [
            {"name": "head_translation", "offset": 0, "size": 3, "unit": "meter"},
            {
                "name": "head_rotation",
                "offset": 3,
                "size": 6,
                "unit": "dimensionless",
                "representation": "rot6d_columns",
            },
            {"name": "right_translation", "offset": 9, "size": 3, "unit": "meter"},
            {
                "name": "right_rotation",
                "offset": 12,
                "size": 6,
                "unit": "dimensionless",
                "representation": "rot6d_columns",
            },
            {
                "name": "right_gripper",
                "offset": 18,
                "size": 1,
                "unit": "open_fraction",
                "closed_value": 0.0,
                "open_value": 1.0,
            },
            {"name": "left_translation", "offset": 19, "size": 3, "unit": "meter"},
            {
                "name": "left_rotation",
                "offset": 22,
                "size": 6,
                "unit": "dimensionless",
                "representation": "rot6d_columns",
            },
            {
                "name": "left_gripper",
                "offset": 28,
                "size": 1,
                "unit": "open_fraction",
                "closed_value": 0.0,
                "open_value": 1.0,
            },
        ],
    }


def _camera_action_layout() -> dict[str, Any]:
    return {
        "id": "camera_pose_backward_framewise_rot6d_v1",
        "pose_convention": "backward_framewise",
        "delta_equation": "T_i^-1 @ T_{i+1}",
        "rotation_representation": "rot6d_columns",
        "fields": [
            {"name": "camera_translation", "offset": 0, "size": 3, "unit": "meter"},
            {
                "name": "camera_rotation",
                "offset": 3,
                "size": 6,
                "unit": "dimensionless",
                "representation": "rot6d_columns",
            },
        ],
    }


def _action_schema_payload() -> dict[str, Any]:
    offset = [float32_value((index - 14) / 100.0) for index in range(29)]
    training_config_excerpt = {
        "datasets": [
            {
                "dataset_class": "AgiBotWorldBetaDataset",
                "embodiment": "agibotworld",
                "method": "quantile_rot",
                "apply_forward_clamp": False,
                "pose_convention": "backward_framewise",
                "rotation_format": "rot6d",
                "stats_filename": "agibot_backward_framewise_rot6d.json",
            }
        ],
        "experiment": "interact_8b_tfdcm_chunk4_agibot",
    }
    normalizer: dict[str, Any] = {
        "schema_version": 1,
        "method": "quantile_rot",
        "transform": {
            "type": "affine",
            "offset": offset,
            "scale": [1.0] * 29,
            "forward_clamp": False,
        },
        "derivation": {
            "statistics_block": "global_raw",
            "low_key": "q01",
            "high_key": "q99",
            "range_floor": float32_value(1e-8),
        },
        "source": {
            "path": ("projects/cosmos3/cosmos3/datasets/action/normalizers/agibot_backward_framewise_rot6d.json"),
            "artifact_path": f"cosmos_dreams_action_sources/{'a' * 64}.json",
            "sha256": "a" * 64,
            "repository_revision": "d" * 40,
        },
        "training_config": {
            "experiment": "interact_8b_tfdcm_chunk4_agibot",
            "resolved_sha256": canonical_sha256(training_config_excerpt),
            "repository_revision": "d" * 40,
        },
    }
    normalizer["transform_sha256"] = canonical_sha256(
        {
            "schema_version": normalizer["schema_version"],
            "method": normalizer["method"],
            "transform": normalizer["transform"],
            "derivation": normalizer["derivation"],
        }
    )
    schema: dict[str, Any] = {
        "schema_version": 2,
        "action_tokens_per_frame": 4,
        "raw_action_dim": 29,
        "model_action_dim": 64,
        "num_embodiment_domains": 32,
        "default_embodiment": "agibotworld",
        "embodiment_to_domain": {"agibotworld": 15},
        "layout": _action_layout(),
        "padding": {"stage": "after_normalization", "value": 0.0},
        "training_config_excerpt": training_config_excerpt,
        "normalizers": {"agibotworld": normalizer},
    }
    schema["contract_sha256"] = canonical_sha256(
        {
            "schema_version": 2,
            "action_tokens_per_frame": 4,
            "raw_action_dim": 29,
            "model_action_dim": 64,
            "num_embodiment_domains": 32,
            "default_embodiment": "agibotworld",
            "embodiment_to_domain": {"agibotworld": 15},
            "layout": schema["layout"],
            "padding": schema["padding"],
            "normalizer_sha256_by_embodiment": {"agibotworld": normalizer["transform_sha256"]},
        }
    )
    return schema


def _camera_action_schema_payload() -> dict[str, Any]:
    translation_scale = 10.0
    rotation_scale = 1.0
    training_config_excerpt = {
        "datasets": [
            {
                "dataset_class": "CameraDatasetSharded",
                "embodiment": "camera_pose",
                "method": "pose_scale",
                "apply_forward_clamp": False,
                "mode": "forward_dynamics",
                "pose_convention": "backward_framewise",
                "rotation_format": "rot6d",
                "translation_scale": translation_scale,
                "rotation_scale": rotation_scale,
            }
        ],
        "experiment": "causal_8b_sf_dmd_max_4step_cam_chunk4_480p_961f",
    }
    normalizer: dict[str, Any] = {
        "schema_version": 1,
        "method": "pose_scale",
        "transform": {
            "type": "affine",
            "offset": [0.0] * 9,
            "scale": [float32_value(1.0 / translation_scale)] * 3 + [float32_value(1.0 / rotation_scale)] * 6,
            "forward_clamp": False,
        },
        "derivation": {
            "translation_scale": translation_scale,
            "rotation_scale": rotation_scale,
        },
        "training_config": {
            "experiment": training_config_excerpt["experiment"],
            "resolved_sha256": canonical_sha256(training_config_excerpt),
            "repository_revision": "d" * 40,
        },
    }
    normalizer["transform_sha256"] = canonical_sha256(
        {
            "schema_version": normalizer["schema_version"],
            "method": normalizer["method"],
            "transform": normalizer["transform"],
            "derivation": normalizer["derivation"],
        }
    )
    schema: dict[str, Any] = {
        "schema_version": 2,
        "action_tokens_per_frame": 4,
        "raw_action_dim": 9,
        "model_action_dim": 64,
        "num_embodiment_domains": 32,
        "default_embodiment": "camera_pose",
        "embodiment_to_domain": {"camera_pose": 2},
        "layout": _camera_action_layout(),
        "padding": {"stage": "after_normalization", "value": 0.0},
        "training_config_excerpt": training_config_excerpt,
        "normalizers": {"camera_pose": normalizer},
    }
    schema["contract_sha256"] = canonical_sha256(
        {
            "schema_version": schema["schema_version"],
            "action_tokens_per_frame": schema["action_tokens_per_frame"],
            "raw_action_dim": schema["raw_action_dim"],
            "model_action_dim": schema["model_action_dim"],
            "num_embodiment_domains": schema["num_embodiment_domains"],
            "default_embodiment": schema["default_embodiment"],
            "embodiment_to_domain": schema["embodiment_to_domain"],
            "layout": schema["layout"],
            "padding": schema["padding"],
            "normalizer_sha256_by_embodiment": {"camera_pose": normalizer["transform_sha256"]},
        }
    )
    return schema


def _artifact() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "checkpoint_hash": "a" * 64,
        "checkpoint_id": "checkpoint",
        "checkpoint_iteration": 1600,
        "chunk_size": 4,
        "window_frames": 64,
        "sink_frames": 0,
        "text_cache_max_len": 512,
        "deploy_resolution": [720, 1280],
        "attention_mode": "three_way",
        "video_temporal_causal": True,
        "latent_patch_size": 2,
        "vae_spatial_compression_factor": 16,
        "temporal_compression_factor": 4,
        "fixed_step_sampler_config": {
            "sample_type": "sde",
            "t_list": [1.0, 15 / 16, 5 / 6, 5 / 8],
            "num_train_timesteps": 1000,
        },
        "action_schema": _action_schema_payload(),
        "temporal_modality_margin": 15000,
        "unified_3d_mrope_reset_spatial_ids": True,
        "base_fps": 24.0,
        "enable_fps_modulation": True,
    }


def _action_manifest(**overrides: Any) -> CosmosDreamsManifest:
    return CosmosDreamsManifest(
        conditioning=CosmosDreamsActionSchema.model_validate(_action_schema_payload()),
        **overrides,
    )


def _fingerprint(**overrides) -> CosmosDreamsSessionFingerprint:
    values = {
        "prompt_hash": "prompt",
        "real_text_kv_lengths": (("main", 12),),
        "height": 720,
        "width": 1280,
        "fps": 15.0,
        "conditioning": (
            ("domain_id", 15),
            ("embodiment", "agibotworld"),
            ("action_contract_sha256", "contract"),
        ),
        "checkpoint_id": "checkpoint",
        "manifest_id": "manifest",
        "sampler_id": "sampler",
    }
    values.update(overrides)
    return CosmosDreamsSessionFingerprint(**values)


def test_manifest_reads_exported_nested_causal_fields() -> None:
    artifact = _artifact()
    config = SimpleNamespace(
        model_config={
            # Alternate deployment aliases cannot override artifact geometry.
            "height": 1,
            "width": 1,
            "cosmos_dreams": {
                "chunk_size": 4,
                "window_frames": 64,
                "deploy_resolution": [720, 1280],
            },
        },
        tf_model_config=SimpleNamespace(to_dict=lambda: {"cosmos_dreams": artifact}),
        custom_pipeline_args={},
    )

    manifest = CosmosDreamsManifest.from_od_config(config)

    assert manifest.window_frames == 64
    assert manifest.patch_grid == (23, 40)
    assert manifest.vision_tokens_per_frame == 920
    assert manifest.tokens_per_frame == 924
    assert manifest.checkpoint_hash == "a" * 64
    manifest.require_exported_artifact()
    assert manifest.resolve_domain_name(" AGIBOTWORLD ") == 15
    assert manifest.resolve_embodiment(None, 15) == "agibotworld"

    config.model_config["cosmos_dreams"]["chunk_size"] = 8
    with pytest.raises(ValueError, match="contradicts the exported artifact"):
        CosmosDreamsManifest.from_od_config(config, require_explicit=True)


def test_transformer_artifact_uses_only_the_canonical_cosmos_dreams_key() -> None:
    config = SimpleNamespace(
        model_config={},
        tf_model_config={"causal_manifest": _artifact()},
        custom_pipeline_args={},
    )

    with pytest.raises(ValueError, match="requires a causal manifest embedded"):
        CosmosDreamsManifest.from_od_config(config, require_explicit=True)


def test_deploy_manifest_cannot_replace_missing_transformer_artifact() -> None:
    config = SimpleNamespace(
        model_config={
            "cosmos_dreams": {
                "checkpoint_id": "deploy-only",
                "checkpoint_iteration": 1600,
                "checkpoint_hash": "a" * 64,
                "action_schema": _action_schema_payload(),
            }
        },
        tf_model_config={},
        custom_pipeline_args={},
    )

    with pytest.raises(ValueError, match="deployment defaults are not a validated artifact"):
        CosmosDreamsManifest.from_od_config(config, require_explicit=True)


def test_manifest_rejects_deployment_defaults_and_placeholder_hash() -> None:
    with pytest.raises(ValueError, match="checkpoint_id.*checkpoint_iteration.*checkpoint_hash"):
        CosmosDreamsManifest().require_exported_artifact()

    with pytest.raises(ValueError, match="all-zero template"):
        CosmosDreamsManifest(checkpoint_hash="0" * 64)

    CosmosDreamsManifest(
        checkpoint_id="checkpoint",
        checkpoint_iteration=1600,
        checkpoint_hash="a" * 64,
    ).require_exported_artifact()

    manifest = CosmosDreamsManifest(
        checkpoint_id="checkpoint",
        checkpoint_iteration=1600,
        checkpoint_hash="a" * 64,
        action_schema=CosmosDreamsActionSchema.model_validate(_action_schema_payload()),
    )
    with pytest.raises(ValueError, match="Unknown Cosmos-Dreams domain_name"):
        manifest.resolve_domain_name("unknown")


def test_manifest_construction_is_conditioning_neutral_but_action_properties_are_not() -> None:
    manifest = CosmosDreamsManifest()

    assert manifest.vision_tokens_per_frame == 920
    with pytest.raises(ValueError, match="action_schema is unavailable"):
        _ = manifest.action_tokens_per_frame
    with pytest.raises(ValueError, match="action_schema is unavailable"):
        _ = manifest.tokens_per_frame


def test_v2_parser_maps_action_schema_into_generic_conditioning_slot() -> None:
    artifact = _artifact()
    config = SimpleNamespace(
        model_config={},
        tf_model_config={"cosmos_dreams": artifact},
        custom_pipeline_args={},
    )

    manifest = CosmosDreamsManifest.from_od_config(config, require_explicit=True)

    assert manifest.conditioning is manifest.action_schema
    assert manifest.conditioning.digest == manifest.action_schema.digest


def test_manifest_dispatches_schema_version_before_field_validation() -> None:
    artifact = _artifact()
    artifact["schema_version"] = 4
    config = SimpleNamespace(
        model_config={},
        tf_model_config={"cosmos_dreams": artifact},
        custom_pipeline_args={},
    )

    with pytest.raises(ValueError, match="Unsupported Cosmos-Dreams manifest schema_version=4"):
        CosmosDreamsManifest.from_od_config(config, require_explicit=True)


def test_action_payload_owns_temporal_cadence_validation() -> None:
    schema = CosmosDreamsActionSchema.model_validate(_action_schema_payload())
    manifest = CosmosDreamsManifest(temporal_compression_factor=8, conditioning=schema)

    assert manifest.temporal_compression_factor == 8
    with pytest.raises(ValueError, match="action_tokens_per_frame must equal"):
        schema.validate_temporal_compression_factor(manifest.temporal_compression_factor)


def test_action_contract_rejects_legacy_unsupported_and_tampered_payloads() -> None:
    payload = _action_schema_payload()
    payload["schema_version"] = 1
    with pytest.raises(ValueError, match="schema_version"):
        CosmosDreamsActionSchema.model_validate(payload)

    payload = _action_schema_payload()
    payload["normalizers"]["agibotworld"]["method"] = "meanstd"
    with pytest.raises(ValueError, match="quantile_rot"):
        CosmosDreamsActionSchema.model_validate(payload)

    payload = _action_schema_payload()
    payload["normalizers"]["agibotworld"]["transform"]["forward_clamp"] = True
    with pytest.raises(ValueError, match="False"):
        CosmosDreamsActionSchema.model_validate(payload)

    payload = _action_schema_payload()
    transform = payload["normalizers"]["agibotworld"]["transform"]
    transform["offset"][0] = float32_value(transform["offset"][0] + 1.0)
    with pytest.raises(ValueError, match="transform_sha256"):
        CosmosDreamsActionSchema.model_validate(payload)

    payload = _action_schema_payload()
    payload["normalizers"]["agibotworld"]["transform"]["mask"] = [True] * 29
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        CosmosDreamsActionSchema.model_validate(payload)

    payload = _action_schema_payload()
    payload["normalizers"]["agibotworld"]["transform"]["offset"][0] = "0.0"
    with pytest.raises(ValueError, match="valid number"):
        CosmosDreamsActionSchema.model_validate(payload)

    payload = _action_schema_payload()
    payload["normalizers"]["agibotworld"]["source"]["artifact_path"] = "../../outside.json"
    with pytest.raises(ValueError, match="content-addressed"):
        CosmosDreamsActionSchema.model_validate(payload)


def test_action_contract_canonical_hash_has_cross_repository_golden_value() -> None:
    assert (
        canonical_sha256(
            {
                "z": -0.0,
                "offset": [0.1, -0.2],
                "range_floor": 1e-8,
                "nested": {"b": 2, "a": 1},
            }
        )
        == "4dbce8b9f13f18b829cefb38c2ea5e3efb0bcbb529d9120b9d78a3c731be2ab7"
    )


def test_action_contract_resolves_embodiment_and_rejects_domain_mismatch() -> None:
    schema = CosmosDreamsActionSchema.model_validate(_action_schema_payload())

    assert schema.resolve_embodiment(None, 15) == "agibotworld"
    assert schema.resolve_embodiment(" AGIBOTWORLD ", 15) == "agibotworld"
    with pytest.raises(ValueError, match="embodiment/domain mismatch"):
        schema.resolve_embodiment(None, 2)
    with pytest.raises(ValueError, match="Unknown Cosmos-Dreams embodiment"):
        schema.resolve_embodiment("unknown", 15)


def test_camera_action_contract_accepts_pose_scale_checkpoint_format() -> None:
    schema = CosmosDreamsActionSchema.model_validate(_camera_action_schema_payload())

    assert schema.raw_action_dim == 9
    assert schema.layout.id == "camera_pose_backward_framewise_rot6d_v1"
    assert schema.resolve_embodiment(None, 2) == "camera_pose"
    assert schema.normalizers["camera_pose"].method == "pose_scale"
    assert schema.normalizers["camera_pose"].transform.scale == pytest.approx((0.1,) * 3 + (1.0,) * 6)


def test_camera_action_contract_rejects_tampered_scale_semantics() -> None:
    payload = _camera_action_schema_payload()
    payload["normalizers"]["camera_pose"]["derivation"]["translation_scale"] = 5.0
    with pytest.raises(ValueError, match="pose_scale transform does not match"):
        CosmosDreamsActionSchema.model_validate(payload)

    payload = _camera_action_schema_payload()
    payload["normalizers"]["camera_pose"]["source"] = {}
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        CosmosDreamsActionSchema.model_validate(payload)


def test_deployment_cannot_override_artifact_action_schema() -> None:
    artifact = _artifact()
    config = SimpleNamespace(
        model_config={"cosmos_dreams": {"action_schema": artifact["action_schema"]}},
        tf_model_config={"cosmos_dreams": artifact},
        custom_pipeline_args={},
    )
    with pytest.raises(ValueError, match="may only come from transformer/config.json"):
        CosmosDreamsManifest.from_od_config(config, require_explicit=True)


def test_runtime_rejects_flattened_legacy_artifact_fields() -> None:
    artifact = _artifact()
    artifact["normalizer_id"] = "legacy"
    config = SimpleNamespace(
        model_config={},
        tf_model_config={"cosmos_dreams": artifact},
        custom_pipeline_args={},
    )
    with pytest.raises(ValueError, match="unknown fields.*normalizer_id"):
        CosmosDreamsManifest.from_od_config(config, require_explicit=True)

    config = SimpleNamespace(
        model_config={"action_normalizer": {"mean": [0.0], "std": [1.0]}},
        tf_model_config={"cosmos_dreams": _artifact()},
        custom_pipeline_args={},
    )
    with pytest.raises(ValueError, match="legacy normalizer fields"):
        CosmosDreamsManifest.from_od_config(config, require_explicit=True)


def test_chunk_partition_is_singleton_then_four_frame_chunks() -> None:
    assert list(iter_ar_chunk_ranges(0, 11, 4)) == [(0, 1), (1, 5), (5, 9), (9, 11)]
    assert list(iter_ar_chunk_ranges(5, 10, 4)) == [(5, 9), (9, 10)]


def test_clean_commit_order_is_per_frame_and_skips_only_global_terminal_frame() -> None:
    assert list(iter_clean_commit_frames(1, 5, target_frame=5, terminal_request=False)) == [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
    ]
    assert list(iter_clean_commit_frames(1, 5, target_frame=5, terminal_request=True)) == [(0, 1), (1, 2), (2, 3)]
    assert list(iter_clean_commit_frames(5, 7, target_frame=7, terminal_request=True)) == [(0, 5)]


def test_action_and_vision_tokens_round_trip_in_per_frame_order() -> None:
    action = torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]])
    vision = torch.tensor([[[[10.0], [11.0], [12.0]], [[20.0], [21.0], [22.0]]]])

    packed = interleave_action_vision_tokens(action, vision)
    unpacked_action, unpacked_vision = split_interleaved_action_vision_tokens(
        packed,
        num_frames=2,
        action_tokens_per_frame=2,
        vision_tokens_per_frame=3,
    )

    assert packed.flatten().tolist() == [1, 2, 10, 11, 12, 3, 4, 20, 21, 22]
    torch.testing.assert_close(unpacked_action, action)
    torch.testing.assert_close(unpacked_vision, vision)


def test_interleaved_mrope_uses_margin_fps_and_action_substeps() -> None:
    ids = build_interleaved_mrope_position_ids(
        frame_start=2,
        num_frames=2,
        grid_h=1,
        grid_w=2,
        text_temporal_offset=10,
        temporal_modality_margin=100,
        fps=12.0,
        base_fps=24.0,
        action_tokens_per_frame=4,
        null_action_frames=(3,),
    )

    # Frame stride is 2. Frame 2 vision time is 114; real action substeps
    # cover (112, 114]. A later all-null frame retains those architectural
    # action IDs even though its action V is zeroed.
    torch.testing.assert_close(ids[0, :4], torch.tensor([112.5, 113.0, 113.5, 114.0]))
    torch.testing.assert_close(ids[0, 4:6], torch.tensor([114.0, 114.0]))
    torch.testing.assert_close(ids[0, 6:10], torch.tensor([114.5, 115.0, 115.5, 116.0]))
    torch.testing.assert_close(ids[0, 10:12], torch.tensor([116.0, 116.0]))

    singleton_null = build_interleaved_mrope_position_ids(
        frame_start=3,
        num_frames=1,
        grid_h=1,
        grid_w=2,
        text_temporal_offset=10,
        temporal_modality_margin=100,
        fps=12.0,
        base_fps=24.0,
        action_tokens_per_frame=4,
        null_action_frames=(3,),
    )
    torch.testing.assert_close(singleton_null[0, :4], torch.full((4,), 116.0))


def test_null_action_zeroes_values_but_not_other_tokens() -> None:
    value = torch.arange(2 * 6, dtype=torch.float32).view(1, 12, 1, 1)
    result = zero_null_action_values(
        value,
        num_frames=2,
        tokens_per_frame=6,
        action_tokens_per_frame=2,
        null_frame_indexes=(1,),
    )

    torch.testing.assert_close(result[:, :6], value[:, :6])
    torch.testing.assert_close(result[:, 6:8], torch.zeros_like(result[:, 6:8]))
    torch.testing.assert_close(result[:, 8:], value[:, 8:])
    assert torch.count_nonzero(value[:, 6:8]) > 0


def test_text_padding_must_be_sliced_before_joint_softmax() -> None:
    query = torch.tensor([[[[1.0]]]])
    real_key = torch.tensor([[[[2.0]]]])
    real_value = torch.tensor([[[[3.0]]]])
    padded_key = torch.cat([real_key, torch.zeros(1, 1, 1, 1)], dim=2)
    padded_value = torch.cat([real_value, torch.zeros(1, 1, 1, 1)], dim=2)

    sliced = torch.nn.functional.scaled_dot_product_attention(
        query,
        padded_key[:, :, :1],
        padded_value[:, :, :1],
    )
    unsliced = torch.nn.functional.scaled_dot_product_attention(query, padded_key, padded_value)

    torch.testing.assert_close(sliced, torch.tensor([[[[3.0]]]]))
    assert not torch.allclose(sliced, unsliced)


def test_kv_estimator_counts_managed_scratch_and_text_pools() -> None:
    manifest = _action_manifest(window_frames=2, text_cache_max_len=3, height=32, width=32)
    estimate = estimate_kv_memory_bytes(
        manifest,
        num_layers=2,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.float16,
    )

    assert estimate.managed_blocks == 5  # window + in-flight + two allocator sentinels
    assert estimate.scratch_blocks == 1
    assert estimate.total_bytes == (
        estimate.self_attention_bytes + estimate.scratch_bytes + estimate.cross_attention_bytes
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prompt_hash", "other-prompt"),
        ("real_text_kv_lengths", (("main", 13),)),
        ("height", 704),
        ("width", 1216),
        ("fps", 24.0),
        ("conditioning", (("domain_id", 2),)),
        ("checkpoint_id", "other-checkpoint"),
        ("manifest_id", "other-manifest"),
        ("sampler_id", "other-sampler"),
    ],
)
def test_session_fingerprint_rejects_every_conditioning_mutation(field: str, value) -> None:
    state = CosmosDreamsSessionState("session")
    fingerprint = _fingerprint()
    state.initialize(fingerprint)
    state.validate_request(fingerprint, frame_idx=0)

    with pytest.raises(ValueError, match=rf"{field}.*session reset required"):
        state.validate_request(_fingerprint(**{field: value}), frame_idx=0)


def test_session_fingerprint_rejects_order_terminal_reuse_and_resets() -> None:
    state = CosmosDreamsSessionState("session")
    fingerprint = _fingerprint()
    state.initialize(fingerprint)

    with pytest.raises(ValueError, match="out of order"):
        state.validate_request(fingerprint, frame_idx=4)

    state.terminal = True
    with pytest.raises(ValueError, match="completed a full rollout"):
        state.validate_request(fingerprint, frame_idx=0)

    state.reset()
    assert state.fingerprint is None
    assert state.next_frame_idx == 0
    assert not state.terminal


def test_action_normalizer_affine_inverse_outliers_and_dimension() -> None:
    contract = CosmosDreamsActionSchema.model_validate(_action_schema_payload())
    normalizer = QuantileRotAffineNormalizer.from_contract(contract.normalizers["agibotworld"])
    offset = torch.tensor(normalizer.offset)
    raw = torch.stack([offset, offset + 1.0, offset + 100.0])

    normalized = normalizer.normalize(raw)

    torch.testing.assert_close(normalized[0], torch.zeros(29))
    assert torch.all(normalized[2] > 99.0)
    torch.testing.assert_close(normalizer.denormalize(normalized), raw)
    with pytest.raises(ValueError, match="raw action dimension"):
        normalizer.normalize(torch.zeros(1, 28))


def test_pipeline_normalizes_float32_before_zero_padding_to_model_width() -> None:
    from vllm_omni.diffusion.models.cosmos_dreams.pipeline_cosmos_dreams import (
        CosmosDreamsPipeline,
    )

    action_schema = CosmosDreamsActionSchema.model_validate(_action_schema_payload())
    normalizer = QuantileRotAffineNormalizer.from_contract(action_schema.normalizers["agibotworld"])
    raw = torch.tensor([normalizer.offset], dtype=torch.float64)
    stub = SimpleNamespace(
        action_normalizers={"agibotworld": normalizer},
        manifest=CosmosDreamsManifest(action_schema=action_schema),
        device=torch.device("cpu"),
        dtype=torch.float16,
        _get_sp_param=lambda _sp, key, default: raw if key == "action" else default,
    )

    result = CosmosDreamsPipeline._prepare_raw_action(
        stub,
        SimpleNamespace(),
        embodiment="agibotworld",
    )

    assert result is not None
    assert result.shape == (1, 64)
    assert result.dtype == torch.float16
    torch.testing.assert_close(result, torch.zeros_like(result))


def test_pipeline_normalizes_camera_pose_and_uses_artifact_default_domain() -> None:
    from vllm_omni.diffusion.models.cosmos_dreams.pipeline_cosmos_dreams import (
        CosmosDreamsPipeline,
    )

    action_schema = CosmosDreamsActionSchema.model_validate(_camera_action_schema_payload())
    manifest = CosmosDreamsManifest(action_schema=action_schema)
    stub = SimpleNamespace(manifest=manifest)
    od_config = SimpleNamespace(custom_pipeline_args={}, model_config={}, tf_model_config={})

    CosmosDreamsPipeline._init_conditioning(stub, od_config)

    assert stub.default_domain_id == 2
    assert stub.default_embodiment == "camera_pose"
    raw = torch.tensor([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    normalized = ActionAffineNormalizer.from_contract(action_schema.normalizers["camera_pose"]).normalize(raw)
    torch.testing.assert_close(
        normalized,
        torch.tensor([[10.0, 20.0, 30.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]),
    )


def test_pipeline_disables_incompatible_generic_warmup() -> None:
    from vllm_omni.diffusion.models.cosmos_dreams.pipeline_cosmos_dreams import (
        CosmosDreamsPipeline,
    )

    assert CosmosDreamsPipeline.dummy_run_num_frames == 0


def test_ar_cache_spec_uses_frame_pages_and_does_not_cap_at_resident_window() -> None:
    from vllm_omni.diffusion.models.cosmos_dreams.pipeline_cosmos_dreams import (
        CosmosDreamsPipeline,
    )

    manifest = _action_manifest(
        height=32,
        width=32,
        window_frames=2,
        sink_frames=1,
        text_cache_max_len=3,
    )
    stub = SimpleNamespace(
        transformer=SimpleNamespace(num_hidden_layers=2, num_kv_heads_local=1, head_dim=4),
        manifest=manifest,
        _MAIN_BRANCH="main",
        _SESSION_CAPACITY=1,
        _validate_ar_diffusion_deploy_overrides=lambda: None,
    )

    spec = CosmosDreamsPipeline.ar_diffusion_kv_cache_spec(stub)

    assert spec.num_kv_heads == 1  # already TP-local
    assert spec.tokens_per_frame == manifest.tokens_per_frame
    assert spec.frames_per_block == 1
    assert spec.window_frames == 2
    assert spec.sink_frames == 1
    assert spec.max_scratch_tokens_per_branch == 0
    assert spec.max_model_len == 1 << 20
    assert spec.max_model_len > (spec.sink_frames + spec.window_frames + 1) * spec.tokens_per_frame


@pytest.mark.parametrize(
    "override",
    [
        {"window_chunks": 3},
        {"sink_chunks": 2},
        {"reset_at_boundary": True},
    ],
)
def test_ar_cache_spec_rejects_semantic_window_overrides(override: dict[str, Any]) -> None:
    from vllm_omni.diffusion.models.cosmos_dreams.pipeline_cosmos_dreams import (
        CosmosDreamsPipeline,
    )

    stub = SimpleNamespace(
        manifest=CosmosDreamsManifest(window_frames=2, sink_frames=1),
        od_config=SimpleNamespace(ar_diffusion_kv_config=override),
    )

    with pytest.raises(ValueError, match="fixed by the model manifest"):
        CosmosDreamsPipeline._validate_ar_diffusion_deploy_overrides(stub)


def test_bound_ar_cache_geometry_must_match_the_manifest() -> None:
    from vllm_omni.diffusion.models.cosmos_dreams.pipeline_cosmos_dreams import (
        CosmosDreamsPipeline,
    )

    manifest = _action_manifest(
        height=32,
        width=32,
        window_frames=2,
        sink_frames=1,
        text_cache_max_len=3,
    )
    transformer = SimpleNamespace(num_hidden_layers=2, num_kv_heads_local=1, head_dim=4)
    cache = SimpleNamespace(
        num_layers=2,
        num_kv_heads=1,
        head_size=4,
        block_size=manifest.tokens_per_frame,
        spec=SimpleNamespace(window_chunks=2, sink_chunks=1, reset_at_boundary=False),
        cross_attention_lengths={"text": 3},
    )
    stub = SimpleNamespace(manifest=manifest, transformer=transformer)

    CosmosDreamsPipeline._validate_bound_kv_geometry(stub, SimpleNamespace(kv_cache=cache))

    cache.spec.window_chunks = 3
    with pytest.raises(RuntimeError, match="window_frames=expected 2, got 3"):
        CosmosDreamsPipeline._validate_bound_kv_geometry(stub, SimpleNamespace(kv_cache=cache))


def test_falsey_sampling_values_are_not_replaced_by_defaults() -> None:
    from vllm_omni.diffusion.models.cosmos_dreams.pipeline_cosmos_dreams import (
        _admission_float,
        _admission_int,
        _first_not_none,
    )
    from vllm_omni.experimental.ar_diffusion.capability import (
        ARDiffusionRequestRejectedError,
    )

    assert _first_not_none(None, 0.0, 15.0) == 0.0
    assert _first_not_none(None, False, True) is False
    with pytest.raises(ARDiffusionRequestRejectedError, match="FPS must be numeric"):
        _admission_float("invalid", "FPS")
    with pytest.raises(ARDiffusionRequestRejectedError, match="frame_idx must be an integer"):
        _admission_int(None, "frame_idx")


def test_offline_session_requires_explicit_lifecycle() -> None:
    from vllm_omni.diffusion.models.cosmos_dreams.pipeline_cosmos_dreams import (
        CosmosDreamsPipeline,
    )
    from vllm_omni.experimental.ar_diffusion.capability import (
        ARDiffusionRequestRejectedError,
    )

    validate = CosmosDreamsPipeline._validate_session_mode
    with pytest.raises(ARDiffusionRequestRejectedError, match="requires reset=True.*or close_session=True"):
        validate(
            tick=False,
            state_was_new=True,
            session_id="default",
            reset=False,
            close_session=False,
        )
    with pytest.raises(ARDiffusionRequestRejectedError, match="explicit-session.*requires reset=True"):
        validate(
            tick=False,
            state_was_new=True,
            session_id="explicit-session",
            reset=False,
            close_session=False,
        )

    for overrides in (
        {"reset": True},
        {"close_session": True},
        {"tick": True},
        {"state_was_new": False},
    ):
        values = {
            "tick": False,
            "state_was_new": True,
            "session_id": "default",
            "reset": False,
            "close_session": False,
            **overrides,
        }
        validate(**values)


def test_distilled_sampler_uses_sigma_times_training_timesteps_and_is_seeded() -> None:
    sampler = CosmosDreamsDistilledSampler([1.0, 0.5], sample_type="sde", num_train_timesteps=1000)
    timesteps: list[float] = []

    def velocity(x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        timesteps.append(float(timestep.item()))
        return torch.zeros_like(x)

    initial = torch.ones(1, 1, 1, 1, 1)
    first = sampler.sample(velocity, initial, seed=7, frame_idx=3)
    timesteps.clear()
    second = sampler.sample(velocity, initial, seed=7, frame_idx=3)

    assert timesteps == [1000.0, 500.0]
    torch.testing.assert_close(first, second)
    assert first.dtype == torch.float32


def _action_layout_pipeline_stub():
    from vllm_omni.diffusion.models.cosmos_dreams.pipeline_cosmos_dreams import (
        CosmosDreamsPipeline,
    )

    stub = SimpleNamespace(
        manifest=_action_manifest(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    return CosmosDreamsPipeline, stub


def test_dense_kv_replaces_each_layer_immediately(monkeypatch) -> None:
    old_0_k = torch.tensor([[[[1.0]], [[2.0]]]])
    old_0_v = old_0_k + 10
    old_1_k = old_0_k + 20
    old_1_v = old_0_k + 30
    old_layer_ids = (id(old_0_k), id(old_0_v))
    history = [
        (old_0_k, old_0_v),
        (old_1_k, old_1_v),
    ]
    del old_0_k, old_0_v, old_1_k, old_1_v

    current = [
        (torch.tensor([[[[3.0]]]]), torch.tensor([[[[13.0]]]])),
        (torch.tensor([[[[23.0]]]]), torch.tensor([[[[33.0]]]])),
    ]
    real_cat = torch.cat
    cat_calls = 0

    def tracked_cat(tensors, *args, **kwargs):
        nonlocal cat_calls
        cat_calls += 1
        if cat_calls == 3:
            assert tuple(id(tensor) for tensor in history[0]) != old_layer_ids
            assert history[0][0].shape[1] == 3
        return real_cat(tensors, *args, **kwargs)

    monkeypatch.setattr(torch, "cat", tracked_cat)
    updated = append_dense_kv_history(
        history,
        current,
        tokens_per_frame=1,
        sink_frames=0,
        window_frames=8,
    )

    assert cat_calls == 4
    torch.testing.assert_close(
        updated[0][0].flatten(),
        torch.tensor([1.0, 2.0, 3.0]),
    )


def test_dense_kv_first_block_reuses_transformer_output_storage() -> None:
    key = torch.tensor([[[[1.0]]]])
    value = torch.tensor([[[[2.0]]]])

    history = append_dense_kv_history(
        None,
        [(key, value)],
        tokens_per_frame=1,
        sink_frames=0,
        window_frames=8,
    )

    assert history[0][0].data_ptr() == key.data_ptr()
    assert history[0][1].data_ptr() == value.data_ptr()


def test_dense_kv_builds_the_final_sink_and_tail_without_append_then_trim() -> None:
    history = [
        (
            torch.tensor([[[[0.0]], [[1.0]], [[2.0]]]]),
            torch.tensor([[[[10.0]], [[11.0]], [[12.0]]]]),
        )
    ]

    updated = append_dense_kv_history(
        history,
        [(torch.tensor([[[[3.0]]]]), torch.tensor([[[[13.0]]]]))],
        tokens_per_frame=1,
        sink_frames=1,
        window_frames=2,
    )

    key, value = updated[0]
    torch.testing.assert_close(key.flatten(), torch.tensor([0.0, 2.0, 3.0]))
    torch.testing.assert_close(value.flatten(), torch.tensor([10.0, 12.0, 13.0]))


def _numbered_action_rows(rows: int, dim: int = 64) -> torch.Tensor:
    return torch.arange(rows, dtype=torch.float32).unsqueeze(1).repeat(1, dim)


def test_action_layout_resolution_is_per_request_and_rejects_short_actions() -> None:
    from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionRequestRejectedError

    pipeline_cls, stub = _action_layout_pipeline_stub()
    resolve = pipeline_cls._resolve_action_layout

    assert resolve(stub, None, start_frame=0, target_frame=5) is None
    # Whole-rollout jsonl layout: rows cover frames 1..target-1 globally.
    assert resolve(stub, _numbered_action_rows(16), start_frame=0, target_frame=5) == "global"
    # Tick continuation with exactly this request's rows is local, and must
    # stay local for EVERY chunk of the request (rows < global requirement).
    assert resolve(stub, _numbered_action_rows(16), start_frame=5, target_frame=9) == "local"
    with pytest.raises(ARDiffusionRequestRejectedError, match="cannot cover"):
        resolve(stub, _numbered_action_rows(20), start_frame=5, target_frame=9)


def test_actions_for_frames_local_layout_is_stable_across_chunks() -> None:
    """Regression: a two-chunk local-layout tick must consume rows in request
    order; per-chunk global/local re-detection previously flipped the first
    chunk to global indexing and misconditioned every frame."""
    pipeline_cls, stub = _action_layout_pipeline_stub()
    slice_actions = pipeline_cls._actions_for_frames
    raw = _numbered_action_rows(32)  # request [5, 13): 8 frames x 4 rows

    first_chunk, first_null = slice_actions(
        stub, raw, layout="local", request_start_frame=5, frame_start=5, frame_end=9
    )
    second_chunk, second_null = slice_actions(
        stub, raw, layout="local", request_start_frame=5, frame_start=9, frame_end=13
    )

    assert first_null == () and second_null == ()
    torch.testing.assert_close(first_chunk[0, :, 0], torch.arange(16, dtype=torch.float32))
    torch.testing.assert_close(second_chunk[0, :, 0], torch.arange(16, 32, dtype=torch.float32))


def test_actions_for_frames_global_layout_and_frame_zero_null() -> None:
    pipeline_cls, stub = _action_layout_pipeline_stub()
    slice_actions = pipeline_cls._actions_for_frames
    raw = _numbered_action_rows(32)  # global rows for frames 1..8

    chunk, nulls = slice_actions(stub, raw, layout="global", request_start_frame=0, frame_start=5, frame_end=9)
    torch.testing.assert_close(chunk[0, :, 0], torch.arange(16, 32, dtype=torch.float32))
    assert nulls == ()

    prefix, prefix_nulls = slice_actions(stub, raw, layout="global", request_start_frame=0, frame_start=0, frame_end=1)
    assert prefix_nulls == (0,)
    torch.testing.assert_close(prefix, torch.zeros_like(prefix))
