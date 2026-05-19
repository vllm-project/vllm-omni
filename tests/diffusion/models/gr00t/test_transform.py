# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for F6: robot-obs transform.

Covers embodiment-tag decoding, proprio_state packing with modality_config
offsets, action unpacking into a per-key dict, and round-trip consistency.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


DROID_MODALITY = {
    "state": {
        "eef_9d": {"start": 0, "end": 9},
        "gripper_position": {"start": 9, "end": 10},
        "joint_position": {"start": 10, "end": 17},
    },
    "action": {
        "eef_9d": {"start": 0, "end": 9},
        "gripper_position": {"start": 9, "end": 10},
        "joint_position": {"start": 10, "end": 17},
    },
}


def test_embodiment_id_decode_known_tag():
    from vllm_omni.diffusion.models.gr00t.transform import embodiment_id_for_tag

    assert embodiment_id_for_tag("oxe_droid_relative_eef_relative_joint") == 24
    assert embodiment_id_for_tag("real_g1_relative_eef_relative_joints") == 25
    assert embodiment_id_for_tag("simpler_env_google") == 0


def test_embodiment_id_decode_unknown_tag_raises():
    from vllm_omni.diffusion.models.gr00t.transform import embodiment_id_for_tag

    with pytest.raises(KeyError, match="Unknown embodiment tag"):
        embodiment_id_for_tag("not_a_real_tag")


def test_pack_proprio_state_correct_offsets_and_padding():
    from vllm_omni.diffusion.models.gr00t.transform import pack_proprio_state

    state_dict = {
        "eef_9d": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "gripper_position": [0.5],
        "joint_position": [10, 11, 12, 13, 14, 15, 16],
    }
    packed = pack_proprio_state(state_dict, DROID_MODALITY["state"], max_state_dim=132)
    assert packed.shape == (132,)
    assert packed.dtype == np.float32
    np.testing.assert_array_equal(packed[0:9], [1, 2, 3, 4, 5, 6, 7, 8, 9])
    np.testing.assert_array_equal(packed[9:10], [0.5])
    np.testing.assert_array_equal(packed[10:17], [10, 11, 12, 13, 14, 15, 16])
    # Trailing padding is zeros
    assert np.all(packed[17:] == 0.0)


def test_pack_proprio_state_skips_missing_keys():
    from vllm_omni.diffusion.models.gr00t.transform import pack_proprio_state

    state_dict = {"gripper_position": [0.3]}
    packed = pack_proprio_state(state_dict, DROID_MODALITY["state"], max_state_dim=132)
    assert packed[9] == pytest.approx(0.3)
    assert packed[0] == 0.0
    assert packed[10] == 0.0


def test_pack_proprio_state_rejects_wrong_size():
    from vllm_omni.diffusion.models.gr00t.transform import pack_proprio_state

    bad = {"eef_9d": [1, 2, 3]}
    with pytest.raises(ValueError, match="expects 9 dims but got 3"):
        pack_proprio_state(bad, DROID_MODALITY["state"], max_state_dim=132)


def test_pack_proprio_state_rejects_segment_exceeding_max_dim():
    from vllm_omni.diffusion.models.gr00t.transform import pack_proprio_state

    mod = {"big": {"start": 0, "end": 200}}
    with pytest.raises(ValueError, match="exceeds max_state_dim"):
        pack_proprio_state({"big": list(range(200))}, mod, max_state_dim=132)


def test_unpack_actions_returns_per_key_dict():
    from vllm_omni.diffusion.models.gr00t.transform import unpack_actions

    horizon = 4
    actions = np.arange(horizon * 132, dtype=np.float32).reshape(horizon, 132)
    out = unpack_actions(actions, DROID_MODALITY["action"])
    assert set(out.keys()) == {"eef_9d", "gripper_position", "joint_position"}
    assert out["eef_9d"].shape == (horizon, 9)
    assert out["gripper_position"].shape == (horizon, 1)
    assert out["joint_position"].shape == (horizon, 7)
    np.testing.assert_array_equal(out["eef_9d"], actions[:, 0:9])
    np.testing.assert_array_equal(out["gripper_position"], actions[:, 9:10])
    np.testing.assert_array_equal(out["joint_position"], actions[:, 10:17])


def test_transform_input_packs_state_and_decodes_embodiment():
    from vllm_omni.diffusion.models.gr00t.transform import get_transform

    transform = get_transform(
        "oxe_droid_relative_eef_relative_joint", DROID_MODALITY
    )
    robot_obs = {
        "embodiment": "oxe_droid_relative_eef_relative_joint",
        "state": {
            "eef_9d": [1.0] * 9,
            "gripper_position": [0.7],
            "joint_position": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        },
        "prompt": "pick up the cube",
    }
    out = transform.transform_input(robot_obs, max_state_dim=132)
    assert out["state"].shape == (1, 1, 132)
    assert out["state"].dtype == np.float32
    assert out["embodiment_id"].tolist() == [24]
    assert out["prompt"] == "pick up the cube"
    assert out["input_ids"] is None  # Pre-tokenized fields not supplied here


def test_transform_input_rejects_non_dict_state():
    from vllm_omni.diffusion.models.gr00t.transform import get_transform

    transform = get_transform("simpler_env_google", DROID_MODALITY)
    with pytest.raises(TypeError, match="must be a dict"):
        transform.transform_input(
            {"state": [1, 2, 3]}, max_state_dim=132
        )


def test_transform_action_output_round_trip():
    from vllm_omni.diffusion.models.gr00t.transform import get_transform

    transform = get_transform(
        "oxe_droid_relative_eef_relative_joint", DROID_MODALITY
    )
    horizon = 4
    actions = np.arange(horizon * 132, dtype=np.float32).reshape(horizon, 132)
    out = transform.transform_action_output(actions)
    # Slices concatenate back to the original packed prefix
    recon = np.concatenate(
        [out["eef_9d"], out["gripper_position"], out["joint_position"]], axis=-1
    )
    np.testing.assert_array_equal(recon, actions[:, :17])


def test_transform_action_output_fallback_when_no_modality():
    from vllm_omni.diffusion.models.gr00t.transform import Gr00tTransform

    transform = Gr00tTransform(
        embodiment_tag="simpler_env_google",
        modality_config={"state": {}, "action": {}},
    )
    actions = np.zeros((4, 132), dtype=np.float32)
    out = transform.transform_action_output(actions)
    assert set(out.keys()) == {"actions"}
    np.testing.assert_array_equal(out["actions"], actions)


# ---------------------------------------------------------------------------
# SE(3) helpers + decode()
# ---------------------------------------------------------------------------


def test_rot6d_matrix_roundtrip():
    from vllm_omni.diffusion.models.gr00t.transform import (
        _matrix_to_rot6d, _rot6d_to_matrix,
    )

    # Identity-ish 6D rep: cols = (1,0,0) and (0,1,0) → R = I
    rot6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float64)
    R = _rot6d_to_matrix(rot6d)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(_matrix_to_rot6d(R), rot6d, atol=1e-9)


def test_compose_eef_relative_to_absolute_position_only():
    """Pure-translation delta + identity rot must add to position only."""
    from vllm_omni.diffusion.models.gr00t.transform import (
        _compose_eef_relative_to_absolute,
    )

    ref = np.array([0.5, 0.1, 0.4, 1, 0, 0, 0, 1, 0], dtype=np.float32)
    rel = np.array([[0.01, 0.02, 0.03, 1, 0, 0, 0, 1, 0]], dtype=np.float32)
    abs_ = _compose_eef_relative_to_absolute(rel, ref)
    np.testing.assert_allclose(abs_[0, :3], [0.51, 0.12, 0.43], atol=1e-5)
    np.testing.assert_allclose(abs_[0, 3:], ref[3:], atol=1e-5)


def test_decode_relative_eef_uses_se3():
    """`eef_9d` decode should reconstruct absolute pose via SE(3) compose
    (matrix mul on rot6d), not element-wise add."""
    from vllm_omni.diffusion.models.gr00t.transform import decode

    modality = {
        "action": {
            "eef_9d": {"start": 0, "end": 9},
            "gripper_position": {"start": 9, "end": 10},
        }
    }
    # Identity rotation reference, zero rotation delta → rot6d should
    # come back equal to reference, position adds.
    raw_state = {"eef_9d": np.array([0.5, 0.0, 0.3, 1, 0, 0, 0, 1, 0], dtype=np.float32)}
    packed = np.zeros((4, 132), dtype=np.float32)
    # Slot 0:3 = +0.01 pos delta, 3:9 = same rot6d.
    packed[:, 0] = 0.01
    packed[:, 3] = 1.0
    packed[:, 7] = 1.0
    # gripper absolute value 0.5 (already absolute)
    packed[:, 9] = 0.5

    out = decode(
        packed, raw_state=raw_state, modality=modality,
        action_norm_stats={}, relative_action_norm_stats={},
    )
    assert set(out.keys()) == {"eef_9d", "gripper_position"}
    # eef_9d position component: ref + 0.01 on x
    np.testing.assert_allclose(out["eef_9d"][:, 0], 0.51, atol=1e-5)
    # Rotation columns unchanged
    np.testing.assert_allclose(out["eef_9d"][:, 3:], np.tile(raw_state["eef_9d"][3:], (4, 1)), atol=1e-5)
    # gripper is absolute → no add, no denorm (we passed empty stats)
    np.testing.assert_array_equal(out["gripper_position"][:, 0], 0.5)


def test_decode_joint_position_uses_elementwise_add():
    """Non-EEF relative keys (e.g. joint_position) compose by add."""
    from vllm_omni.diffusion.models.gr00t.transform import decode

    modality = {"action": {"joint_position": {"start": 0, "end": 7}}}
    raw_state = {"joint_position": np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float32)}
    packed = np.zeros((3, 132), dtype=np.float32)
    packed[:, :7] = 0.1  # uniform delta
    out = decode(
        packed, raw_state=raw_state, modality=modality,
        action_norm_stats={}, relative_action_norm_stats={},
    )
    expected = np.tile(raw_state["joint_position"] + 0.1, (3, 1))
    np.testing.assert_allclose(out["joint_position"], expected, atol=1e-5)


def test_decode_denormalize_with_relative_action_stats():
    """Per-horizon-step relative_action stats override static action stats."""
    from vllm_omni.diffusion.models.gr00t.transform import decode

    modality = {"action": {"joint_position": {"start": 0, "end": 1}}}
    raw_state = {"joint_position": np.array([0.0], dtype=np.float32)}
    horizon = 3
    packed = np.zeros((horizon, 132), dtype=np.float32)
    packed[:, 0] = 1.0  # max-normalized → should denorm to q99
    # Per-step stats: step k has q99=k+1
    rel_stats = {
        "joint_position": {
            "q01": [[0.0]] * horizon,
            "q99": [[1.0], [2.0], [3.0]],
        }
    }
    out = decode(
        packed, raw_state=raw_state, modality=modality,
        action_norm_stats={}, relative_action_norm_stats=rel_stats,
    )
    # denorm=q99 at each step; then +raw_state (=0)
    np.testing.assert_allclose(out["joint_position"][:, 0], [1.0, 2.0, 3.0], atol=1e-5)


def test_decode_fallback_when_no_modality():
    from vllm_omni.diffusion.models.gr00t.transform import decode

    packed = np.arange(40 * 132, dtype=np.float32).reshape(40, 132)
    out = decode(
        packed, raw_state={}, modality={},
        action_norm_stats={}, relative_action_norm_stats={},
    )
    assert set(out.keys()) == {"actions"}
    np.testing.assert_array_equal(out["actions"], packed)
