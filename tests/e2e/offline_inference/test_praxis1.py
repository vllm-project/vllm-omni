# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real-checkpoint Praxis-1 tests; require local weights and one CUDA GPU.

Synthetic camera/pose observations exercise cold start, context continuation,
video diagnostics and episode reset through the Omni engine. The DROID run is
also checked against actions the reference implementation produced on an H100
for the same observations. The policy is a chaotic two-step sampler in
bfloat16, so kernel-level rounding differences move individual targets by tens
of millimetres even between two builds of the reference itself; the fixture
bound is a drift guard against gross regressions, not a parity claim. Task
success is qualified separately in simulation (see the model guide).
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.core_model, pytest.mark.advanced_model, pytest.mark.diffusion]

FIXTURES = Path(__file__).parent / "fixtures"
DRIFT_BOUNDS = {"position_m": 0.05, "rotation_deg": 10.0, "gripper": 0.35}


def _droid_observation(rows):
    image = np.zeros((rows, 256, 416, 3), dtype=np.uint8)
    image[..., 0] = np.arange(416) % 256
    image[..., 1] = np.arange(256)[:, None]
    image[..., 2] = 128
    return {
        "image.exterior": image,
        "image.wrist": image,
        "robot_state.ee_pose": np.tile(np.array([0.4, 0, 0.3, 1, 0, 0, 0], np.float32), (rows, 1)),
        "grip": np.full((rows, 1), 0.5, np.float32),
        "task": "Move the gripper forward.",
        "descriptor": "praxis.sim",
    }


def _yam_observation(rows):
    obs = {
        f"image.{v}": np.full((rows, 260, 416, 3), 40 * i, np.uint8)
        for i, v in enumerate(("exterior", "wrist_left", "wrist_right"))
    }
    for i, side in enumerate(("left", "right")):
        obs[f"robot_state.{side}.ee_pose"] = np.tile([0.4, 0.3 - 0.6 * i, 0.4, 1, 0, 0, 0], (rows, 1)).astype(
            np.float32
        )
        obs[f"grip.{side}"] = np.full((rows, 1), 0.5, np.float32)
    obs["task"] = "Pick up the pen from the center of the table, then place the pen in the bag"
    return obs


def _check_targets(actions, arms):
    assert actions.dtype == np.float32 and np.isfinite(actions).all()
    assert actions.shape[1] == 13 * arms
    for arm in range(arms):
        block = actions[:, 13 * arm : 13 * (arm + 1)]
        assert np.all((block[:, -1] >= 0) & (block[:, -1] <= 1))
        rotations = block[:, 3:12].reshape(-1, 3, 3)
        np.testing.assert_allclose(
            rotations @ rotations.transpose(0, 2, 1), np.broadcast_to(np.eye(3), rotations.shape), atol=1e-5
        )


def _drift(actions, expected):
    from scipy.spatial.transform import Rotation

    position = np.linalg.norm(actions[:, :3] - expected[:, :3], axis=-1).max()
    actual_rot = Rotation.from_matrix(actions[:, 3:12].reshape(-1, 3, 3))
    expected_rot = Rotation.from_matrix(expected[:, 3:12].reshape(-1, 3, 3))
    rotation = np.rad2deg((actual_rot.inv() * expected_rot).magnitude()).max()
    return {"position_m": position, "rotation_deg": rotation, "gripper": np.abs(actions[:, -1] - expected[:, -1]).max()}


def _engine(checkpoint, deploy, tmp_path, overrides):
    from vllm_omni.entrypoints.omni import Omni

    config = yaml.safe_load(Path(deploy).read_text())
    config["stages"][0]["model_config"].update(overrides)
    path = tmp_path / "deploy.yaml"
    path.write_text(yaml.safe_dump(config))
    return Omni(model=checkpoint, deploy_config=str(path))


def _generate(engine, observation, session_id, **extra):
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    outputs = engine.generate(
        {"prompt": observation["task"]},
        OmniDiffusionSamplingParams(
            seed=7,
            num_inference_steps=2,
            extra_args={"robot_obs": observation, "session_id": session_id, "return_diagnostics": True, **extra},
        ),
        use_tqdm=False,
    )
    assert len(outputs) == 1 and outputs[0].final_output_type == "actions"
    payload = outputs[0].multimodal_output
    return np.asarray(payload["actions"]), payload["trajectory"]


@hardware_test(res={"cuda": ["H100"]}, num_cards=1)
@pytest.mark.skipif(not os.environ.get("PRAXIS_CHECKPOINT"), reason="Set PRAXIS_CHECKPOINT to local DROID weights.")
def test_praxis1_droid_context_reset_and_reference_drift(tmp_path):
    reference = json.loads((FIXTURES / "praxis1_droid_h100_reference.json").read_text())
    deploy = Path(__file__).resolve().parents[3] / "vllm_omni/deploy/praxis1.yaml"
    engine = _engine(
        os.environ["PRAXIS_CHECKPOINT"], deploy, tmp_path, {**reference["sampler"], "max_context_chunks": 3}
    )
    try:
        for index in range(4):
            actions, trace = _generate(
                engine, _droid_observation(1 if index == 0 else 25), "droid", decode_video=index == 0
            )
            assert actions.shape == (21 if index == 0 else 24, 13)
            _check_targets(actions, arms=1)
            assert trace["seed"] == 7 + index and trace["depth"] == index
            latent = torch.as_tensor(trace["pred_video_latent"])
            assert latent.shape == (1, 192, 6, 7, 26) and torch.isfinite(latent).all()
            drift = _drift(actions, np.asarray(reference["calls"][index], dtype=np.float32))
            assert all(drift[key] < bound for key, bound in DRIFT_BOUNDS.items()), drift
            if index == 0:
                first = actions.copy()
                frames = torch.as_tensor(trace["pred_frames"])
                assert frames.shape == (21, 3, 224, 832) and frames.dtype == torch.uint8
        repeated, _ = _generate(engine, _droid_observation(1), "droid", reset=True, end_session=True)
        np.testing.assert_array_equal(repeated, first)
        # A completed episode restarts even without an explicit reset.
        restarted, _ = _generate(engine, _droid_observation(1), "droid", end_session=True)
        np.testing.assert_array_equal(restarted, first)
    finally:
        engine.close()


@hardware_test(res={"cuda": ["H100"]}, num_cards=1)
@pytest.mark.skipif(
    not os.environ.get("PRAXIS_YAM_CHECKPOINT"), reason="Set PRAXIS_YAM_CHECKPOINT to local YAM weights."
)
def test_praxis1_yam_two_arm_contract(tmp_path):
    deploy = Path(__file__).resolve().parents[3] / "vllm_omni/deploy/praxis1_yam.yaml"
    engine = _engine(os.environ["PRAXIS_YAM_CHECKPOINT"], deploy, tmp_path, {})
    try:
        for index in range(2):
            actions, trace = _generate(
                engine, _yam_observation(1 if index == 0 else 25), "yam", decode_video=index == 0
            )
            assert actions.shape == (21 if index == 0 else 24, 26)
            _check_targets(actions, arms=2)
            assert trace["depth"] == index
            if index == 0:
                first = actions.copy()
                assert tuple(torch.as_tensor(trace["pred_frames"]).shape) == (21, 3, 256, 1248)
        repeated, _ = _generate(engine, _yam_observation(1), "yam", reset=True, end_session=True)
        np.testing.assert_array_equal(repeated, first)
    finally:
        engine.close()
