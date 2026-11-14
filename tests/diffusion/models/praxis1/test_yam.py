# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for the two-arm YAM rig on the shared Praxis-1 observation state machine."""

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.praxis1 import processing as common
from vllm_omni.diffusion.models.praxis1 import processing_yam as yam

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def observation(frames=25):
    obs = {keys[0]: np.full((frames, 260, 416, 3), i, np.uint8) for i, keys in enumerate(yam.VIEW_KEYS.values())}
    for i, side in enumerate(yam.SIDES):
        obs[f"robot_state.{side}.ee_pose"] = np.tile([0.4, 0.3 - 0.6 * i, 0.4, 1, 0, 0, 0], (frames, 1))
        obs[f"grip.{side}"] = np.full(frames, 1.2 if i == 0 else -0.2, np.float32)
    return obs


def stats():
    return common.NormStats(
        np.zeros(20, np.float32), np.ones(20, np.float32), np.zeros(20, np.float32), np.ones(20, np.float32)
    )


def test_yam_canvas_and_pose_keep_both_arms():
    obs = observation(1)
    canvas = yam.canvas_clip(yam.view_stacks(obs))
    assert canvas.shape == (1, 3, 1, 256, 1248)
    for i in range(3):
        assert torch.all(canvas[..., i * 416 : (i + 1) * 416] == i)
    poses = np.concatenate([yam.pose_rows(obs, side)[-1] for side in yam.SIDES])
    proprio = common.proprio_tensors(poses[None], stats(), "cpu")
    assert proprio.shape == (1, 1, 20)
    np.testing.assert_array_equal(proprio[0, 0, [9, 19]], [np.float32(1.2), np.float32(-0.2)])


@pytest.mark.parametrize("invalid", ["aspect", "quaternion", "dtype", "missing_right"])
def test_yam_refuses_invalid_observations(invalid):
    obs = observation(1)
    if invalid == "aspect":
        obs["image.exterior"] = np.zeros((1, 234, 416, 3), np.uint8)
    elif invalid == "quaternion":
        obs["robot_state.right.ee_pose"][:, 3:] = 0
    elif invalid == "dtype":
        obs["image.wrist_right"] = obs["image.wrist_right"].astype(np.float32)
    else:
        del obs["grip.right"]
    session = yam.YamServeSession(lambda x: x[:, :1, :, :1, :1], stats(), "cpu")
    with pytest.raises((KeyError, ValueError)):
        session.prepare(obs)


def test_yam_decode_keeps_independent_arm_commands():
    session = yam.YamServeSession(lambda x: x[:, :1, :, :1, :1], stats(), "cpu")
    inputs = session.prepare(observation(1))
    actions = torch.zeros(1, 6, 80)
    for offset in (0, 10):
        actions.view(1, 6, 4, 20)[..., offset + 3] = 1
        actions.view(1, 6, 4, 20)[..., offset + 7] = 1
    wire, rows, rel = session.finish(inputs, actions)
    assert len(wire) == 21 and rows.shape == rel.shape == (21, 20)
    assert set(wire[0]) == {
        "robot_command.left",
        "robot_command.right",
        "target_grip.left",
        "target_grip.right",
        "timestamp",
    }
    np.testing.assert_allclose(wire[0]["robot_command.left"]["pose"][:3], [0.9, 0.8, 0.9])
    np.testing.assert_allclose(wire[0]["robot_command.right"]["pose"][:3], [0.9, 0.2, 0.9])


def test_yam_context_continues_with_two_arm_history():
    def encode(clip):
        return torch.zeros(1, 192, (clip.shape[2] - 1) // 4 + 1, 1, 1)

    session = yam.YamServeSession(encode, stats(), "cpu", max_context_chunks=1, exec_horizon=24)
    first = session.prepare(observation(1))
    assert first.depth == 0 and first.proprio_all.shape == (1, 1, 20)
    second = session.prepare(observation(session.stream.expected_next_rows + 1))
    assert second.depth == 1 and second.proprio_all.shape == (1, 2, 20)
