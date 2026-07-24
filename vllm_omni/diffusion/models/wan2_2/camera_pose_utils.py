# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Camera class adapted from CameraCtrl (Apache-2.0)
# https://github.com/hehao13/CameraCtrl

"""Camera action -> pose trajectory -> PRoPE camera condition for DreamX-World-5B-Cam.

Ported (PRoPE path only) from the upstream DreamX-World repo
(``utils/inference_utils.py`` + ``utils/pose_utils.py``). Turns action tokens into
a pose trajectory, aligns it to the VAE 1+4k temporal pattern, and inverts it to
the ``{"viewmats": [T_lat, 4, 4], "K": [T_lat, 3, 3]}`` camera condition consumed
by ``WanCameraTransformer3DModel``'s per-block PRoPE self-attention branch.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

# Action token -> motion type. Composable tokens (e.g. "wj") are split per char.
ACTION_DICT = {
    "w": "forward",
    "s": "backward",
    "a": "left",
    "d": "right",
    "j": "left_rot",
    "l": "right_rot",
    "i": "up_rot",
    "k": "down_rot",
}

# Per-unit-speed translation (meters) and rotation (degrees) magnitudes.
TRANSLATION_BASE_UNIT = 1.0
ROTATION_BASE_UNIT = 10.0


class Camera:
    """Copied from https://github.com/hehao13/CameraCtrl/blob/main/inference.py"""

    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


def _compute_translation_step(motion_type, current_pose, translation_value, duration):
    """Compute per-frame translation step in **world coordinates**.

    The camera forward direction in world space is ``R_w2c^T @ [0,0,1]``
    (OpenCV convention: +Z is the camera's viewing direction).

    ``current_pose['position']`` accumulates world-space displacement;
    the conversion to w2c translation ``t = -R @ pos`` is done later
    when building the extrinsic matrix.
    """
    if motion_type in ["forward", "backward"]:
        yaw_rad = np.radians(current_pose["rotation"][1])
        pitch_rad = np.radians(current_pose["rotation"][0])
        forward_vec = np.array(
            [
                -math.sin(yaw_rad) * math.cos(pitch_rad),
                math.sin(pitch_rad),
                math.cos(yaw_rad) * math.cos(pitch_rad),
            ]
        )
        direction = 1 if motion_type == "forward" else -1
        total_move = forward_vec * translation_value * direction
        return total_move / duration

    if motion_type in ["left", "right"]:
        yaw_rad = np.radians(current_pose["rotation"][1])
        right_vec = np.array([math.cos(yaw_rad), 0, math.sin(yaw_rad)])
        direction = -1 if motion_type == "left" else 1
        total_move = right_vec * translation_value * direction
        return total_move / duration

    return np.zeros(3)


def _compute_rotation_step(motion_type, rotation_value, duration):
    """Compute per-frame rotation step vector for a single rotation motion type.

    rotation layout: [pitch (X-axis, up/down look), yaw (Y-axis, left/right turn), roll (Z-axis)]
    """
    if motion_type.endswith("rot"):
        axis = motion_type.split("_")[0]
        total_rotation = np.zeros(3)
        if axis == "left":
            total_rotation[1] = rotation_value
        elif axis == "right":
            total_rotation[1] = -rotation_value
        elif axis == "up":
            total_rotation[0] = -rotation_value
        elif axis == "down":
            total_rotation[0] = rotation_value
        return total_rotation / duration

    return np.zeros(3)


def generate_composite_motion_segment(
    current_pose: dict[str, np.ndarray],
    motion_types: str | list[str],
    translation_value: float,
    rotation_value: float,
    duration: int = 30,
):
    """Generate a trajectory that combines multiple motions simultaneously.

    Unlike ``_compute_translation_step``/``_compute_rotation_step`` which take a
    single motion type, this function accepts a list of motion types and blends
    them together so that, e.g., "forward" + "right_rot" produces a forward-moving
    arc.

    Parameters:
        current_pose: dict with 'position' (np.array[3]) and 'rotation' (np.array[3])
        motion_types: list of str, each one of
            ('forward', 'backward', 'left', 'right',
             'left_rot', 'right_rot', 'up_rot', 'down_rot')
        translation_value: Translation magnitude (m)
        rotation_value: Rotation magnitude (degree)
        duration: Number of frames

    Return:
        positions:    list of np.array(x, y, z)
        rotations:    list of np.array(pitch, yaw, roll)
        current_pose: updated pose dict after the motion
    """
    if isinstance(motion_types, str):
        motion_types = [motion_types]

    positions = []
    rotations = []

    translation_step = np.zeros(3)
    rotation_step = np.zeros(3)

    for motion_type in motion_types:
        translation_step += _compute_translation_step(motion_type, current_pose, translation_value, duration)
        rotation_step += _compute_rotation_step(motion_type, rotation_value, duration)

    for i in range(1, duration + 1):
        new_pos = current_pose["position"] + translation_step * i
        new_rot = current_pose["rotation"] + rotation_step * i
        positions.append(new_pos.copy())
        rotations.append(new_rot.copy())

    current_pose["position"] = positions[-1].copy()
    current_pose["rotation"] = rotations[-1].copy()

    return positions, rotations, current_pose


def euler_to_quaternion(angles: np.ndarray | Sequence[float]):
    """Convert Euler angles (pitch, yaw, roll) to quaternion.

    Uses ZYX intrinsic rotation order (roll around Z, then pitch around X',
    then yaw around Y'') which matches the w2c + OpenCV convention used by
    the translation computation in _compute_translation_step.

    Args:
        angles: [pitch, yaw, roll] in degrees.
    """
    pitch, yaw, roll = np.radians(angles)

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cy * cp * cr + sy * sp * sr
    qx = cy * sp * cr + sy * cp * sr
    qy = sy * cp * cr - cy * sp * sr
    qz = cy * cp * sr - sy * sp * cr

    return [qw, qx, qy, qz]


def quaternion_to_rotation_matrix(q):
    qw, qx, qy, qz = q
    return np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)],
        ]
    )


def _allocate_action_durations(num_frames: int, num_actions: int) -> list[int]:
    """Split the ``num_frames - 1`` motion frames across ``num_actions`` segments.

    Frame 0 is the identity pose, so only ``num_frames - 1`` frames carry motion.
    Segment boundaries are proportionally rounded, so any remainder is spread
    across the sequence instead of always shorting the final action; durations
    differ by at most one frame. Examples: 121 frames / 2 actions -> [60, 60];
    81 / 3 -> [27, 26, 27]; 11 / 3 -> [3, 4, 3].
    """
    if num_actions <= 0:
        raise ValueError(f"num_actions must be positive, got {num_actions}")
    total = num_frames - 1
    if total < num_actions:
        raise ValueError(
            f"num_frames={num_frames} leaves {total} motion frames for {num_actions} "
            f"action(s) (frame 0 is the identity pose); every action needs at least "
            f"one motion frame, so num_frames must be >= {num_actions + 1}"
        )
    boundaries = [round(i * total / num_actions) for i in range(num_actions + 1)]
    return [boundaries[i + 1] - boundaries[i] for i in range(num_actions)]


def ActionToPoseFromID(action_ids, action_speed_list, duration=33):
    """Convert action segments into a list of upstream 19-field pose strings.

    The first row is the identity (canonical) frame; subsequent rows are one per
    generated frame. ``duration`` is either a scalar (the upstream form: the same
    frame count for every action segment) or a per-action sequence, e.g. from
    ``_allocate_action_durations``.
    """
    if isinstance(duration, (list, tuple)):
        durations = list(duration)
        if len(durations) != len(action_ids):
            raise ValueError(
                f"duration list (len {len(durations)}) and action_ids (len {len(action_ids)}) must have equal length"
            )
    else:
        durations = [duration] * len(action_ids)

    all_positions = []
    all_rotations = []
    current_pose = {
        "position": np.array([0.0, 0.0, 0.0]),  # XYZ
        "rotation": np.array([0.0, 0.0, 0.0]),  # (pitch, yaw, roll)
    }
    intrinsic = [0.8, 0.5, 0.5, 0.5]

    for idx, action_id in enumerate(action_ids):
        # Normalise action_id into individual keys: "wl" -> ["w", "l"].
        keys = list(action_id)

        motion_types = [ACTION_DICT[key] for key in keys]
        speed = action_speed_list[idx]

        positions, rotations, current_pose = generate_composite_motion_segment(
            current_pose,
            motion_types=motion_types,
            translation_value=speed * TRANSLATION_BASE_UNIT,
            rotation_value=speed * ROTATION_BASE_UNIT,
            duration=durations[idx],
        )
        all_positions.extend(positions)
        all_rotations.extend(rotations)

    pose_list = []

    row = [0] + intrinsic + [0, 0] + [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    first_row = " ".join(map(str, row))
    pose_list.append(first_row)

    for i, (pos, rot) in enumerate(zip(all_positions, all_rotations)):
        quat = euler_to_quaternion(rot)
        R = quaternion_to_rotation_matrix(quat)
        # pos is world-space camera position; w2c translation is t = -R @ pos.
        t = -R @ pos
        extrinsic = np.hstack([R, t.reshape(3, 1)])

        row = [i] + intrinsic + [0, 0] + extrinsic.flatten().tolist()
        pose_list.append(" ".join(map(str, row)))
    return pose_list


def get_relative_pose(cam_params, scale_factor=1):
    """Re-express a camera trajectory relative to frame 0 (frame 0 -> identity)."""
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    cam_to_origin = 0
    target_cam_c2w = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, -cam_to_origin],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    for pose in ret_poses:
        pose[:3, -1:] *= scale_factor
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert a 4x4 SE(3) matrix."""
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def interpolate_camera_poses(cam_params, src_indices, tgt_indices):
    """Interpolate a camera trajectory onto ``tgt_indices``.

    Rotation via scipy multi-keyframe ``Slerp``; translation via linear
    ``interp1d``. Handles a left-handed coordinate system by flipping the Z
    axis when the median rotation determinant is negative.
    """
    src_indices = np.asarray(src_indices, dtype=np.float64)
    tgt_indices = np.asarray(tgt_indices, dtype=np.float64)

    src_rot_mat = np.array([cam.w2c_mat[:3, :3] for cam in cam_params])  # [N, 3, 3]
    src_trans_vec = np.array([cam.w2c_mat[:3, 3] for cam in cam_params])  # [N, 3]

    # Detect a left-handed basis and temporarily flip Z so Slerp stays valid.
    dets = np.linalg.det(src_rot_mat)
    flip_handedness = dets.size > 0 and np.median(dets) < 0.0
    if flip_handedness:
        flip_mat = np.diag([1.0, 1.0, -1.0]).astype(src_rot_mat.dtype)
        src_rot_mat = src_rot_mat @ flip_mat

    interp_func_trans = interp1d(
        src_indices,
        src_trans_vec,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    interpolated_trans_vec = interp_func_trans(tgt_indices)

    src_quat_vec = Rotation.from_matrix(src_rot_mat)
    quats = src_quat_vec.as_quat().copy()  # [N, 4] (x, y, z, w)
    # Keep neighbouring quaternions on the same hemisphere (no sign flips).
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]
    src_quat_vec = Rotation.from_quat(quats)
    slerp_func_rot = Slerp(src_indices, src_quat_vec)
    interpolated_rot_quat = slerp_func_rot(tgt_indices)
    interpolated_rot_mat = interpolated_rot_quat.as_matrix()

    if flip_handedness:
        interpolated_rot_mat = interpolated_rot_mat @ flip_mat

    ref_cam = cam_params[0]
    result_cameras = []
    for i in range(len(tgt_indices)):
        w2c_3x4 = np.hstack([interpolated_rot_mat[i], interpolated_trans_vec[i].reshape(3, 1)])
        entry = np.zeros(19, dtype=np.float32)
        entry[1:5] = [ref_cam.fx, ref_cam.fy, ref_cam.cx, ref_cam.cy]
        entry[7:] = w2c_3x4.reshape(12)
        result_cameras.append(Camera(entry))

    return result_cameras


def GetPoseEmbedsFromPosesPrope(
    poses,
    h,
    w,
    target_length,
    flip=False,
    start_index=0,
    cam_method="prope",
    dtype=torch.float32,
    device="cpu",
):
    """Pose strings -> PRoPE camera condition ``{"viewmats", "K"}``.

    The PRoPE path is resolution-independent: ``h``/``w``/``flip``/``cam_method``
    are accepted for signature parity with upstream but are unused here.

    Frames are aligned to the VAE temporal 1+4k pattern
    (``latent_frame_count = 1 + (N-1)//4``) by interpolating the trajectory,
    then re-expressed relative to frame 0 and inverted to view matrices.
    """
    poses = [pose.split(" ") for pose in poses]

    start_idx = start_index
    sample_id = [start_idx + i for i in range(target_length)]
    poses = [poses[i] for i in sample_id]

    cam_params = [[float(x) for x in pose] for pose in poses]
    assert len(cam_params) == target_length
    cam_params = [Camera(cam_param) for cam_param in cam_params]

    # Align to VAE temporal downsampling (1+4k pattern): keep frame 0, then
    # every 4th frame. latent_frame_count = 1 + (N-1)//4.
    n_frames = len(cam_params)
    latent_frame_count = 1 + (n_frames - 1) // 4

    # scipy Slerp/interp1d require >= 2 source keyframes. A single input frame
    # (the degenerate num_frames == 1 case the vLLM video API allows) cannot be
    # interpolated: keep the lone camera (T_latent == 1). With >= 2 frames the
    # interpolation is always valid, including when latent_frame_count == 1
    # (n_frames in 2..4). Upstream never hits n_frames == 1; this guard makes the
    # function robust for the vLLM API contract (num_frames >= 1).
    if n_frames > 1:
        src_indices = np.arange(n_frames, dtype=np.float64)
        tgt_indices = np.linspace(0, n_frames - 1, latent_frame_count)
        cam_params = interpolate_camera_poses(cam_params, src_indices, tgt_indices)

    c2w_poses_aligned = get_relative_pose(cam_params)
    c2ws = torch.as_tensor(c2w_poses_aligned, dtype=dtype, device=device)

    T_latent = c2ws.shape[0]
    viewmats = _invert_SE3(c2ws)  # [T_latent, 4, 4]

    # Fixed normalised pinhole intrinsics (cx, cy re-zeroed); independent of h/w.
    default_intrinsic = [
        [969.6969696969696, 0.0, 960.0],
        [0.0, 969.6969696969696, 540.0],
        [0.0, 0.0, 1.0],
    ]
    fx_norm = default_intrinsic[0][0] / (default_intrinsic[0][2] * 2)
    fy_norm = default_intrinsic[1][1] / (default_intrinsic[1][2] * 2)

    Ks = torch.zeros((T_latent, 3, 3), device=device, dtype=dtype)
    Ks[:, 0, 0] = fx_norm
    Ks[:, 1, 1] = fy_norm
    Ks[:, 0, 2] = 0
    Ks[:, 1, 2] = 0
    Ks[:, 2, 2] = 1.0

    camera_condition = {
        "viewmats": viewmats,
        "K": Ks,
    }
    return camera_condition, poses


def validate_action_sequence(action_seq, action_speed_list) -> None:
    """Validate request action controls; always raise ``ValueError`` for bad input.

    Catches every malformed-input case at the API boundary so it surfaces as a
    clean ``ValueError`` rather than a downstream ``KeyError`` (unknown token) or
    ``TypeError`` (non-string token / non-numeric speed):

    - ``action_seq`` / ``action_speed_list`` are equal-length non-empty lists.
    - Each token is a ``str`` (e.g. ``"wj"``) or a sequence of single-char
      strings, and every character is a key in ``ACTION_DICT``.
    - Each speed is a real number (``int``/``float``, not ``bool``/``str``).
    """
    if not isinstance(action_seq, (list, tuple)) or not isinstance(action_speed_list, (list, tuple)):
        raise ValueError("action_seq and action_speed_list must be lists")
    if len(action_seq) == 0:
        raise ValueError("action_seq must be non-empty")
    if len(action_seq) != len(action_speed_list):
        raise ValueError(
            f"action_seq (len {len(action_seq)}) and action_speed_list "
            f"(len {len(action_speed_list)}) must have equal length"
        )
    valid = sorted(ACTION_DICT)
    for tok in action_seq:
        # A token is a string ("wj") or a sequence of single-char strings,
        # mirroring how ActionToPoseFromID consumes it (``keys = list(tok)``).
        if isinstance(tok, str):
            keys = list(tok)
        elif isinstance(tok, (list, tuple)):
            keys = list(tok)
        else:
            raise ValueError(f"action token must be str or sequence of str, got {type(tok).__name__}")
        if len(keys) == 0:
            raise ValueError("empty action token")
        for ch in keys:
            if not isinstance(ch, str) or ch not in ACTION_DICT:
                raise ValueError(f"unknown action token '{ch}' (in {tok!r}); valid tokens are {valid}")
    for sp in action_speed_list:
        # bool is an int subclass; reject it explicitly (a boolean speed is a bug).
        if isinstance(sp, bool) or not isinstance(sp, (int, float)):
            raise ValueError(f"action speed must be int or float, got {sp!r} ({type(sp).__name__})")


def build_camera_condition(
    action_seq,
    action_speed_list,
    height,
    width,
    num_frames,
    *,
    dtype=torch.float32,
    device="cpu",
):
    """High-level entry: action controls -> PRoPE camera condition.

    ``num_frames`` MUST already be snapped to the 1+4k VAE pattern by the
    caller (the pre-process step). ``height``/``width`` are accepted for parity
    but unused in the PRoPE path. Returns
    ``{"viewmats": [T_lat, 4, 4], "K": [T_lat, 3, 3]}`` with
    ``T_lat = 1 + (num_frames - 1) // 4``.

    Motion frames are allocated explicitly per action (frame 0 is the identity
    pose, the remaining ``num_frames - 1`` frames are split evenly across
    actions). This intentionally diverges from upstream DreamX, which uses
    ``ceil(num_frames / len(action_seq))`` per action and truncates the tail —
    under-representing the final action for most frame counts. ``num_frames == 1``
    (the engine dummy-warmup shape) yields the identity-only trajectory;
    ``1 < num_frames <= len(action_seq)`` raises.
    """
    validate_action_sequence(action_seq, action_speed_list)
    if num_frames == 1:
        # Identity pose only — no room for motion (dummy warmup uses this).
        poses = ActionToPoseFromID([], [], duration=[])
    else:
        durations = _allocate_action_durations(num_frames, len(action_seq))
        poses = ActionToPoseFromID(action_seq, action_speed_list, duration=durations)
    assert len(poses) == num_frames, f"pose generation produced {len(poses)} rows, expected {num_frames}"
    camera_condition, _ = GetPoseEmbedsFromPosesPrope(
        poses,
        height,
        width,
        len(poses),
        False,
        0,
        dtype=dtype,
        device=device,
    )
    return camera_condition
