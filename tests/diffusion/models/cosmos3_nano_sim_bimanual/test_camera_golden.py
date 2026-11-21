# SPDX-License-Identifier: Apache-2.0
"""Compare camera conditioning with frozen regression vectors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.action_contract import Cosmos3NanoSimBimanualActionSchema
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.action_inputs import prepare_action_values
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.camera import camera_poses_to_actions, resolve_camera_poses
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.normalizer import ActionAffineNormalizer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
FIXTURES = Path(__file__).with_name("fixtures")
CASES = (("stone_w", "w-61"), ("stone_s", "s-61"), ("stone_up", "up-61"), ("stone_left", "left-61"))


@pytest.fixture(scope="module")
def golden():
    archive = FIXTURES / "camera_stone.npz"
    metadata = json.loads(archive.with_suffix(".json").read_text())
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == metadata["archive_sha256"]
    assert metadata["schema_version"] == 1
    assert metadata["cases"] == [{"name": name, "trajectory": spec} for name, spec in CASES]
    with np.load(archive, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    assert set(arrays) == {"poses", "raw_actions", "normalized_actions"}
    assert arrays["poses"].shape == (4, 61, 4, 4) and arrays["poses"].dtype == np.float64
    for key in ("raw_actions", "normalized_actions"):
        assert arrays[key].shape == (4, 60, 9) and arrays[key].dtype == np.float32
    contract = Cosmos3NanoSimBimanualActionSchema.model_validate_json(
        (FIXTURES / "cookbook_action_schema.json").read_text()
    ).embodiments["camera_pose"]
    assert contract.normalizer.source["sha256"] == metadata["statistics_sha256"]
    override = contract.normalizer.source["pose_convention_override"]
    assert override["source_allowed_pose_conventions"] == ["backward_anchored"]
    assert override["effective_pose_convention"] == metadata["parameters"]["pose_convention"]
    assert override["authority"] == "inference_camera_profile"
    return arrays, contract


@pytest.mark.parametrize("case_index,spec", list(enumerate(spec for _, spec in CASES)), ids=[name for name, _ in CASES])
def test_camera_matches_golden(case_index: int, spec: str, golden) -> None:
    arrays, contract = golden
    poses = resolve_camera_poses(spec, 61)
    # Allow FP64 rotation roundoff and FP32 rounding during action conversion.
    np.testing.assert_allclose(poses, arrays["poses"][case_index], rtol=1e-12, atol=1e-12)
    raw = camera_poses_to_actions(poses, contract.layout.pose_convention)
    np.testing.assert_allclose(raw, arrays["raw_actions"][case_index], rtol=1e-6, atol=1e-7)
    prepared = prepare_action_values(
        raw,
        width=9,
        model_width=64,
        action_space="raw",
        normalizer=ActionAffineNormalizer.from_contract(contract.normalizer),
    )
    torch.testing.assert_close(
        prepared[:, :9],
        torch.from_numpy(arrays["normalized_actions"][case_index]),
        rtol=1e-6,
        atol=1e-6,
    )
    assert torch.count_nonzero(prepared[:, 9:]) == 0
