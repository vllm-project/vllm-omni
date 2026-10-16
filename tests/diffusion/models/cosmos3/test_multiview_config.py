# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Strict deployment metadata and camera admission for the single architecture."""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.diffusion.models.cosmos3.multiview_fixtures import multiview_contract
from vllm_omni.model_extras.cosmos3 import validate_multiview_request

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _validate_multiview_contract(contract: dict[str, Any], *, camera_patch: int = 2) -> dict[str, Any]:
    from vllm_omni.diffusion.models.cosmos3.multiview_config import (
        _validated_multiview_deployment_config,
    )

    return _validated_multiview_deployment_config(
        {"backbone_type": "cosmos3_multiview", "latent_patch_size": camera_patch, "multiview": contract}
    )


def test_multiview_contract_accepts_canonical_export() -> None:
    contract = multiview_contract()
    validated = _validate_multiview_contract(json.loads(json.dumps(contract)))
    assert validated == contract
    assert validated["lidar_latent_patch_size_hw"] == [1, 1]
    assert validated["rig_view_embedding"]["camera_ids"]["camera_front_tele_30fov"] == 6


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "backend",
        "per_view_captions",
        "variable_view_count",
        "max_views",
        "attention_scope",
        "causal_training_strategy",
        "decomposed_temporal_window_seconds",
        "control_attends_sensor",
        "lidar_attends_captions",
        "align_temporal_positions_across_views",
        "share_vision_temporal_positions",
        "lidar_patch_spatial_hw",
        "separate_view_text_tokenization",
        "future_field",
    ],
)
def test_multiview_contract_rejects_obsolete_fields(field: str) -> None:
    contract = multiview_contract()
    contract[field] = None
    with pytest.raises(ValueError, match="Unknown Cosmos3 multiview contract fields"):
        _validate_multiview_contract(contract)


@pytest.mark.parametrize(
    "field",
    [
        "cameras",
        "cross_view_past_window_seconds",
        "rig_view_embedding",
        "lidar_latent_patch_size_hw",
        "inference_defaults",
    ],
)
def test_multiview_contract_rejects_missing_fields(field: str) -> None:
    contract = multiview_contract()
    del contract[field]
    with pytest.raises(ValueError, match="requires field"):
        _validate_multiview_contract(contract)


@pytest.mark.parametrize("window", [None, True, -0.1, float("nan"), float("inf"), [-0.4, 0]])
def test_multiview_contract_rejects_invalid_window(window: Any) -> None:
    contract = multiview_contract()
    contract["cross_view_past_window_seconds"] = window
    with pytest.raises(ValueError, match="finite.*non-negative"):
        _validate_multiview_contract(contract)


@pytest.mark.parametrize("patch", [[1], [1, 0], [True, 1], 1])
def test_multiview_contract_rejects_invalid_lidar_patch(patch: Any) -> None:
    contract = multiview_contract()
    contract["lidar_latent_patch_size_hw"] = patch
    with pytest.raises(ValueError, match="two positive integers"):
        _validate_multiview_contract(contract)


@pytest.mark.parametrize("field", ["sigma_max", "negative_metadata_mode"])
def test_multiview_contract_rejects_ignored_defaults(field: str) -> None:
    contract = multiview_contract()
    contract["inference_defaults"][field] = None
    with pytest.raises(ValueError, match="Unknown inference_defaults"):
        _validate_multiview_contract(contract)


def test_multiview_contract_accepts_camera_only_rig() -> None:
    contract = multiview_contract()
    del contract["lidar"]
    del contract["lidar_latent_patch_size_hw"]
    assert _validate_multiview_contract(contract) == contract


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_embeddings", 1),
        ("num_embeddings", 12.0),
        ("lidar_id", 10),
        ("lidar_id", True),
        ("lidar_id", 11.0),
        ("camera_ids", None),
        ("camera_ids", {}),
        ("unknown", 1),
    ],
)
def test_contract_rejects_invalid_rig(field: str, value: Any) -> None:
    contract = multiview_contract()
    contract["rig_view_embedding"][field] = value
    with pytest.raises((ValueError, TypeError)):
        _validate_multiview_contract(contract)


@pytest.mark.parametrize("row", [True, -1, 11, 0.0, None, 1])
def test_contract_rejects_invalid_or_duplicate_camera_rows(row: Any) -> None:
    contract = multiview_contract()
    contract["rig_view_embedding"]["camera_ids"][contract["cameras"][0]] = row
    with pytest.raises((ValueError, TypeError)):
        _validate_multiview_contract(contract)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fps", 0),
        ("fps", True),
        ("fps", float("inf")),
        ("num_steps", 1.5),
        ("num_steps", True),
        ("shift", 0),
        ("guidance", -1),
        ("normalize_cfg", 0),
        ("emphasize_control_in_prompt", 1),
        ("guidance_interval", [1, 1]),
        ("control_guidance_interval", [0, float("nan")]),
    ],
)
def test_contract_rejects_invalid_defaults(field: str, value: Any) -> None:
    contract = multiview_contract()
    contract["inference_defaults"][field] = value
    with pytest.raises(ValueError):
        _validate_multiview_contract(contract)


@pytest.mark.parametrize("change", ["version", "unknown", "missing", "null"])
def test_contract_rejects_invalid_lidar_metadata(change: str) -> None:
    contract = multiview_contract()
    if change in {"version", "unknown"}:
        contract["lidar"][change] = "1.2"
    elif change == "missing":
        del contract["lidar"]["network_config"]
    else:
        contract["lidar"] = None
    with pytest.raises((ValueError, TypeError)):
        _validate_multiview_contract(contract)


@pytest.mark.parametrize(
    "cameras", [["camera_front_tele_30fov"], ["camera_rear_tele_30fov", "camera_front_wide_120fov"]]
)
def test_requests_select_and_reorder_camera_subsets(cameras: list[str]) -> None:
    views = [{"camera_key": camera, "prompt": "Cars drive down the road."} for camera in cameras]
    _, selected = validate_multiview_request({"multiview": {"views": views}}, multiview_contract()["cameras"])
    assert selected == views


@pytest.mark.parametrize("caption", [None, "", " ", True, 42])
def test_requests_require_per_camera_captions(caption: Any) -> None:
    view = {"camera_key": "camera_front_wide_120fov", "prompt": caption}
    with pytest.raises(ValueError, match="requires one prompt per camera"):
        validate_multiview_request({"multiview": {"views": [view]}})
