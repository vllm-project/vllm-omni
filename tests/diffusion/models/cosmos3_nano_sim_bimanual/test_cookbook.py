# SPDX-License-Identifier: Apache-2.0
"""CPU coverage of cookbook contracts and model-visible conditioning."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from pydantic import ValidationError

from vllm_omni.diffusion.models.cosmos3.resolution import find_closest_target_size
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.action_contract import (
    Cosmos3NanoSimBimanualActionSchema,
    canonical_sha256,
)
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.action_inputs import (
    prepare_action_values,
    validate_action_values,
)
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.camera import (
    camera_poses_to_actions,
    parse_camera_string,
    resolve_camera_poses,
)
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.config import Cosmos3NanoSimBimanualManifest
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.control_contract import (
    Cosmos3NanoSimBimanualActionConditioning,
    parse_cosmos3_nano_sim_bimanual_conditioning,
)
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.cookbook import prepare_cookbook_input, resolve_asset
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.normalizer import ActionAffineNormalizer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
FIXTURES = Path(__file__).with_name("fixtures")


def schema_payload(kind: str = "cookbook") -> dict:
    return json.loads((FIXTURES / f"{kind}_action_schema.json").read_text())


def manifest(kind: str = "cookbook") -> Cosmos3NanoSimBimanualManifest:
    return Cosmos3NanoSimBimanualManifest(
        checkpoint_id="cpu-fixture",
        checkpoint_iteration=1,
        checkpoint_hash="a" * 64,
        conditioning=Cosmos3NanoSimBimanualActionConditioning.model_validate(
            {"mode": "action", **schema_payload(kind)}
        ),
    )


@pytest.mark.parametrize("kind,version", [("legacy", 3), ("cookbook", 4)])
def test_contract_versions(kind: str, version: int) -> None:
    payload = schema_payload(kind)
    parsed = Cosmos3NanoSimBimanualActionSchema.model_validate(payload)
    assert parsed.schema_version == version
    assert parsed.model_dump(mode="json", exclude_none=True) == payload
    assert parsed.embodiment_to_domain == {"agibotworld": 15, "camera_pose": 2}


@pytest.mark.parametrize("field", ["contract_sha256", "normalizer_hash", "unit", "profile", "layout", "downgrade"])
def test_contract_rejects_tampering(field: str) -> None:
    payload = schema_payload()
    camera = payload["embodiments"]["camera_pose"]
    if field == "contract_sha256":
        payload[field] = "0" * 64
    elif field == "normalizer_hash":
        camera["normalizer"]["transform_sha256"] = "0" * 64
    elif field == "unit":
        camera["normalizer"]["transform"]["unit"] = 1.0
    elif field == "profile":
        payload["inference_camera_profile"]["translation_scale"] = 10.0
    elif field == "layout":
        camera["layout"]["pose_convention"] = "backward_framewise"
    else:
        payload["schema_version"] = 3
    with pytest.raises(ValidationError):
        Cosmos3NanoSimBimanualActionSchema.model_validate(payload)


def resign(payload: dict) -> dict:
    """Recompute the contract hash so a test isolates one semantic check."""
    behavior = {
        key: payload[key]
        for key in (
            "schema_version",
            "action_tokens_per_frame",
            "model_action_dim",
            "num_embodiment_domains",
            "default_embodiment",
            "padding",
            "inference_camera_profile",
        )
        if key in payload
    }
    behavior["embodiments"] = {
        name: {
            "domain_id": contract["domain_id"],
            "raw_action_dim": contract["raw_action_dim"],
            "layout": contract["layout"],
            "normalizer_sha256": contract["normalizer"]["transform_sha256"],
        }
        for name, contract in payload["embodiments"].items()
    }
    payload["contract_sha256"] = canonical_sha256(behavior)
    return payload


def test_hand_pose_contract_loads_beside_agibot_and_camera() -> None:
    payload = schema_payload("hand_pose")
    assert resign(copy.deepcopy(payload))["contract_sha256"] == payload["contract_sha256"]
    reparsed = Cosmos3NanoSimBimanualActionSchema.model_validate(payload)
    assert reparsed.model_dump(mode="json", exclude_none=True) == payload
    parsed = manifest("hand_pose").require_action_schema()
    assert parsed.embodiment_to_domain == {"agibotworld": 15, "camera_pose": 2, "hand_pose": 3}
    assert parsed.resolve_embodiment(None, 3) == "hand_pose"
    assert parsed.raw_action_dim_for("hand_pose") == 57
    assert parsed.resolve_embodiment("camera_pose", None) == "camera_pose"


def test_hand_pose_raw_actions_normalize_and_pad() -> None:
    contract = manifest("hand_pose").require_action_schema().normalizers["hand_pose"]
    normalizer = ActionAffineNormalizer.from_contract(contract)
    raw = torch.arange(3 * 57, dtype=torch.float32).reshape(3, 57) / 100
    action = prepare_action_values(raw, width=57, model_width=64, action_space="raw", normalizer=normalizer)
    offset, scale = torch.tensor(contract.transform.offset), torch.tensor(contract.transform.scale)
    torch.testing.assert_close(action[:, :57], (raw - offset) / scale, rtol=0, atol=0)
    assert torch.count_nonzero(action[:, 57:]) == 0


@pytest.mark.parametrize(
    "embodiment,key,value,message",
    [
        ("hand_pose", "delta_equation", "T_i^-1 @ T_{i+1}", "layout disagrees"),
        ("agibotworld", "id", "hand_pose_fingertips_backward_framewise_rot6d_v1", "must be used together"),
    ],
)
def test_hand_pose_layout_semantics(embodiment: str, key: str, value: str, message: str) -> None:
    payload = schema_payload("hand_pose")
    payload["embodiments"][embodiment]["layout"][key] = value
    with pytest.raises(ValidationError, match=message):
        Cosmos3NanoSimBimanualActionSchema.model_validate(resign(payload))


def test_global_asinh_matches_quantile_formula() -> None:
    contract = manifest().require_action_schema().normalizers["camera_pose"]
    normalizer = ActionAffineNormalizer.from_contract(contract)
    raw = torch.arange(45, dtype=torch.float32).reshape(5, 9) / 3 - 7
    offset, scale = torch.tensor(contract.transform.offset), torch.tensor(contract.transform.scale)
    expected = torch.asinh((raw - offset) / scale) / torch.asinh(torch.tensor(1.0))
    torch.testing.assert_close(normalizer.normalize(raw), expected, rtol=0, atol=0)
    assert normalizer.normalize(raw).abs().max() > 1  # The transform must not clamp heavy tails.


def test_sidecar_values_bypass_normalization_and_raw_values_normalize_once() -> None:
    normalizer = ActionAffineNormalizer(offset=(1.0,) * 29, scale=(2.0,) * 29, transform_sha256="a" * 64)
    action = torch.arange(116, dtype=torch.float32).reshape(4, 29)
    model = prepare_action_values(action, width=29, model_width=64, action_space="model", normalizer=normalizer)
    raw = prepare_action_values(action, width=29, model_width=64, action_space="raw", normalizer=normalizer)
    torch.testing.assert_close(model[:, :29], action)
    torch.testing.assert_close(raw[:, :29], (action - 1) / 2)
    assert torch.count_nonzero(model[:, 29:]) == torch.count_nonzero(raw[:, 29:]) == 0


@pytest.mark.parametrize(
    "value",
    [
        [[True, 1.0]],
        [["1", 2]],
        [[float("nan"), 1]],
        [[1, float("inf")]],
        [[1j, 2]],
        [],
        [[1]],
        torch.ones(2, 2, dtype=torch.bool),
    ],
)
def test_invalid_action_values(value: object) -> None:
    with pytest.raises(ValueError):
        validate_action_values(value, width=2)


@pytest.mark.parametrize("forward,backward,axis", [("w", "s", 2), ("d", "a", 0), ("n", "u", 1)])
def test_camera_translation_directions(forward: str, backward: str, axis: int) -> None:
    poses = parse_camera_string(f"{forward}-10,{backward}-10")
    np.testing.assert_allclose(poses[10, axis, 3], 1, atol=1e-12)
    np.testing.assert_allclose(poses[-1], np.eye(4), atol=1e-12)


@pytest.mark.parametrize("pair", [("up", "down"), ("left", "right"), ("cw", "ccw")])
def test_camera_inverse_rotations(pair: tuple[str, str]) -> None:
    poses = parse_camera_string(f"{pair[0]}-30,{pair[1]}-30")
    np.testing.assert_allclose(poses[-1], np.eye(4), atol=1e-12)
    assert not np.allclose(poses[30], np.eye(4))


@pytest.mark.parametrize("command", ["pano-360", "orbit-360-5.0"])
def test_camera_full_revolution(command: str) -> None:
    poses = parse_camera_string(command)
    np.testing.assert_allclose(poses[-1], np.eye(4), atol=1e-10)
    if command.startswith("orbit"):
        np.testing.assert_allclose(poses[180, 2, 3], 10, atol=1e-10)


def test_chunk_anchors_and_column_rot6d() -> None:
    poses = resolve_camera_poses("w-40", 41)
    action = camera_poses_to_actions(poses, "backward_chunk_anchored_16f")
    np.testing.assert_allclose(action[[0, 15, 16, 31, 32], 2], [0.1, 1.6, 0.1, 1.6, 0.1], atol=1e-6)
    np.testing.assert_array_equal(action[:, 3:], np.tile([1, 0, 0, 0, 1, 0], (40, 1)))
    rotated = camera_poses_to_actions(parse_camera_string("right-1"), "backward_framewise")
    angle = np.deg2rad(0.5)
    np.testing.assert_allclose(rotated[0, 3:], [np.cos(angle), 0, -np.sin(angle), 0, 1, 0], atol=1e-7)


def test_pose_padding_truncation_and_json(tmp_path: Path) -> None:
    path = tmp_path / "poses.json"
    path.write_text(json.dumps(parse_camera_string("w-4").tolist()))
    poses = resolve_camera_poses(str(path), 17)
    assert poses.shape == (17, 4, 4)
    np.testing.assert_array_equal(poses[4], poses[-1])
    # Padding absolute poses preserves the chunk-anchor displacement in the held tail.
    actions = camera_poses_to_actions(poses, "backward_chunk_anchored_16f")
    np.testing.assert_allclose(actions[3:, 2], 0.4, atol=1e-6)
    assert resolve_camera_poses("w-61", 61).shape == (61, 4, 4)


@pytest.mark.parametrize("spec", ["", "w", "w-foo", "typo-5", "pano-0", "w--1", "w-2-3", "orbit-3-0", "w-3|s-3"])
def test_invalid_camera_commands(spec: str) -> None:
    with pytest.raises(ValueError):
        resolve_camera_poses(spec, 17)


def test_invalid_camera_pose_json(tmp_path: Path) -> None:
    path = tmp_path / "poses.json"
    for poses in ([], np.ones((2, 4, 4)).tolist(), [[[float("nan")]]]):
        path.write_text(json.dumps(poses))
        with pytest.raises(ValueError):
            resolve_camera_poses(str(path), 17)
    with pytest.raises(ValueError, match="absolute"):
        resolve_camera_poses("relative.json", 17)


def action_record(tmp_path: Path, frames: int = 61) -> dict:
    Image.new("RGB", (1600, 900), (33, 55, 77)).save(tmp_path / "image.png")
    (tmp_path / "action.json").write_text(json.dumps([[2.0] * 29] * (frames - 1)))
    return {
        "prompt": "Smooth the shorts",
        "vision_path": "image.png",
        "action_path": "action.json",
        "domain_name": "agibotworld",
        "raw_action_dim": 29,
        "view_point": "ego_view",
        "fps": 30,
        "num_frames": frames,
        "action_chunk_size": frames - 1,
        "image_size": 480,
        "seed": 7,
    }


@pytest.mark.parametrize("frames", [61, 901, 1801])
def test_cookbook_action_horizons(tmp_path: Path, frames: int) -> None:
    record = action_record(tmp_path, frames)
    result = prepare_cookbook_input(record, manifest=manifest(), base_dir=tmp_path, cache_dir=tmp_path)
    assert result.action.shape == (frames - 1, 29)
    assert result.extra_args["action_space"] == "model"
    assert (result.height, result.width, result.fps, result.seed) == (480, 832, 30, 7)
    prompt = json.loads(result.prompt)
    assert prompt["cinematography"]["framing"].startswith("This video is captured from a first-person")
    assert prompt["actions"][0]["description"] == "Smooth the shorts."
    assert prompt["duration"] == f"{int(frames / 30)}s"


@pytest.mark.parametrize(
    "source_size,expected_canvas,expected_aspect",
    [
        ((720, 1080), (512, 768), "2,3"),
        ((720, 1070), (512, 768), "2,3"),
        ((720, 960), (544, 736), "3,4"),
        ((720, 1280), (480, 832), "9,16"),
        ((1600, 900), (832, 480), "16,9"),
    ],
)
def test_cookbook_aspect_buckets_and_prompt(
    tmp_path: Path, source_size: tuple[int, int], expected_canvas: tuple[int, int], expected_aspect: str
) -> None:
    record = action_record(tmp_path, 5)
    Image.new("RGB", source_size).save(tmp_path / "image.png")
    result = prepare_cookbook_input(record, manifest=manifest(), base_dir=tmp_path, cache_dir=tmp_path)
    assert (result.width, result.height) == expected_canvas
    prompt = json.loads(result.prompt)
    assert prompt["aspect_ratio"] == expected_aspect
    assert prompt["resolution"] == {"H": expected_canvas[1], "W": expected_canvas[0]}
    assert result.metadata["effective_prompt"] == result.prompt


@pytest.mark.parametrize("tier,canvas", [(256, (192, 320)), (704, (832, 1088)), (720, (832, 1104))])
def test_two_three_bucket_is_scoped_to_480(tier: int, canvas: tuple[int, int]) -> None:
    assert find_closest_target_size(1080, 720, tier) == canvas


@pytest.mark.parametrize("kind", ["legacy", "cookbook"])
@pytest.mark.parametrize("templates", ["default", "custom", "null", "empty"])
def test_resolved_camera_recipe_and_prompt_metadata(tmp_path: Path, kind: str, templates: str) -> None:
    record = {"camera_trajectory": "w-61", "model_mode": "text2video", "num_frames": 61, "prompt": "A room."}
    if kind == "legacy":
        record.update(
            camera_pose_convention="backward_framewise",
            camera_action_normalization="scale",
            camera_translation_scale=10.0,
        )
    if templates == "custom":
        record.update(duration_template=" D={duration:.1f}@{fps:.0f}", resolution_template=" R={width}x{height}")
        expected_prompt = "A room. D=2.0@30 R=832x480"
    elif templates in ("null", "empty"):
        value = None if templates == "null" else ""
        record.update(duration_template=value, resolution_template=value)
        expected_prompt = "A room."
    else:
        expected_prompt = "A room.The video is 2.0 seconds long and is of 30 FPS.This video is of 480x832 resolution."
    artifact = manifest(kind)
    result = prepare_cookbook_input(record, manifest=artifact, base_dir=tmp_path, cache_dir=tmp_path)
    recipe = result.metadata["effective_camera_recipe"]
    assert recipe == {
        "pose_convention": "backward_framewise" if kind == "legacy" else "backward_chunk_anchored_16f",
        "action_normalization": "scale" if kind == "legacy" else "global_asinh",
        "translation_scale": 10.0 if kind == "legacy" else 1.0,
        "rotation_scale": 1.0,
        "camera_num_frames": 61,
        "normalizer_method": "pose_scale" if kind == "legacy" else "global_asinh",
        "normalizer_transform_sha256": artifact.require_action_schema().normalizers["camera_pose"].transform_sha256,
    }
    assert result.metadata["effective_prompt"] == result.prompt == expected_prompt
    assert result.metadata["prompt_templates"] == {
        "duration_template": record.get(
            "duration_template", "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
        ),
        "resolution_template": record.get("resolution_template", "This video is of {height}x{width} resolution."),
    }
    assert (result.metadata["inference_camera_profile"] is None) == (kind == "legacy")


def test_cookbook_camera_profile_and_overrides(tmp_path: Path) -> None:
    record = {"camera_trajectory": "w-61", "model_mode": "text2video", "num_frames": 901}
    result = prepare_cookbook_input(
        record,
        manifest=manifest(),
        base_dir=tmp_path,
        cache_dir=tmp_path,
        overrides={"camera_num_frames": 61, "seed": 0},
    )
    assert result.num_frames == 61 and result.seed == 0
    assert result.action.shape == (60, 9)
    assert result.extra_args["action_space"] == "raw"
    assert result.metadata["requested_num_frames"] == 901
    assert result.metadata["inference_camera_profile"]["translation_scale"] == 1.0
    with pytest.raises(ValueError, match="exported contract"):
        prepare_cookbook_input(record, manifest=manifest("legacy"), base_dir=tmp_path, cache_dir=tmp_path)
    result = prepare_cookbook_input(
        record,
        manifest=manifest("legacy"),
        base_dir=tmp_path,
        cache_dir=tmp_path,
        overrides={
            "camera_pose_convention": "backward_framewise",
            "camera_translation_scale": 10.0,
            "camera_action_normalization": "scale",
        },
    )
    assert result.action.shape == (900, 9)


@pytest.mark.parametrize(
    "update",
    [
        {"num_frames": 62},
        {"action_chunk_size": 3},
        {"fps": 0},
        {"seed": True},
        {"camera_trajectory": "w-61"},
        {"action_space": "raw"},
        {"height": 480},
        {"raw_action_dim": 20},
        {"model_mode": "policy"},
        {"vision_path": None},
    ],
)
def test_cookbook_invalid_action_records(tmp_path: Path, update: dict) -> None:
    record = {**action_record(tmp_path), **update}
    with pytest.raises(ValueError):
        prepare_cookbook_input(record, manifest=manifest(), base_dir=tmp_path, cache_dir=tmp_path)


@pytest.mark.parametrize(
    "update",
    [
        {"camera_num_frames": 0},
        {"camera_translation_scale": 10.0},
        {"domain_name": "agibotworld"},
        {"action_space": "model"},
        {"autoregressive": False},
    ],
)
def test_invalid_camera_recipe(tmp_path: Path, update: dict) -> None:
    record = {"camera_trajectory": "w-61", "model_mode": "text2video", **update}
    with pytest.raises(ValueError):
        prepare_cookbook_input(record, manifest=manifest(), base_dir=tmp_path, cache_dir=tmp_path)


def test_local_asset_resolution(tmp_path: Path) -> None:
    assert resolve_asset("image.png", base_dir=tmp_path, cache_dir=tmp_path) == tmp_path / "image.png"
    assert resolve_asset(str(tmp_path / "image.png"), base_dir=tmp_path, cache_dir=tmp_path) == tmp_path / "image.png"


def test_transfer_conditioning_still_parses() -> None:
    value = {
        "mode": "control_video",
        "hints": ["edge", "blur", "depth", "seg"],
        "transfer_control_attention_mode": "causal_control_with_rgb_history",
        "share_vision_temporal_positions": True,
        "system_prompt_id": "cosmos3_transfer_v1",
        "emphasize_control_in_prompt": True,
        "no_eviction": True,
    }
    assert parse_cosmos3_nano_sim_bimanual_conditioning(copy.deepcopy(value)).model_dump(mode="json") == value
