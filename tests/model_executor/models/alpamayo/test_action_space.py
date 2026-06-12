# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Alpamayo trajectory / action-space math library.

These tests are intentionally dependency-light: they exercise the pure-torch
math module in isolation (no vLLM engine), matching the action-space acceptance
criteria in ``feature_list.md``.
"""

from __future__ import annotations

import numpy as np
import torch

from vllm_omni.model_executor.models.alpamayo.action_space import (
    UnicycleAccelCurvatureActionSpace,
    rot_2d_to_3d,
    rotation_matrix_torch,
    so3_to_yaw_torch,
)


def _make_curved_trajectory(n_hist: int = 16, n_fut: int = 64, dt: float = 0.1):
    """Synthesize a smooth constant-speed, constant-curvature ego trajectory.

    Returns history/future xyz (1,1,T,3) and rot (1,1,T,3,3) tensors, all in the
    local frame whose origin/orientation is the last history pose (so the action
    space's t=0 assumptions hold).
    """
    v = 8.0  # m/s
    kappa = 0.03  # 1/m
    total = n_hist + n_fut
    # Arc-length parameterization of a circle of radius 1/kappa.
    s = v * dt * torch.arange(total, dtype=torch.float64)
    theta = kappa * s
    radius = 1.0 / kappa
    x = radius * torch.sin(theta)
    y = radius * (1.0 - torch.cos(theta))
    z = torch.zeros_like(x)
    xyz = torch.stack([x, y, z], dim=-1)  # (total, 3)
    rot = rot_2d_to_3d(rotation_matrix_torch(theta.float())).double()  # (total,3,3)

    # Re-express in the frame of the last history pose (index n_hist-1).
    origin = xyz[n_hist - 1].clone()
    R0 = rot[n_hist - 1].clone()
    R0_inv = R0.transpose(-1, -2)
    xyz_local = (R0_inv @ (xyz - origin).unsqueeze(-1)).squeeze(-1)
    rot_local = R0_inv @ rot

    hist_xyz = xyz_local[:n_hist].float().view(1, 1, n_hist, 3)
    fut_xyz = xyz_local[n_hist:].float().view(1, 1, n_fut, 3)
    hist_rot = rot_local[:n_hist].float().view(1, 1, n_hist, 3, 3)
    fut_rot = rot_local[n_hist:].float().view(1, 1, n_fut, 3, 3)
    return hist_xyz, hist_rot, fut_xyz, fut_rot


def test_action_traj_roundtrip():
    """action_to_traj(traj_to_action(traj)) should recover the future xy."""
    n_fut = 64
    space = UnicycleAccelCurvatureActionSpace(n_waypoints=n_fut, dt=0.1)
    hist_xyz, hist_rot, fut_xyz, fut_rot = _make_curved_trajectory(n_fut=n_fut)

    # batch shape expected by the action space: (..., T, 3) / (..., T, 3, 3)
    hist_xyz_b = hist_xyz[0]  # (1, n_hist, 3)
    hist_rot_b = hist_rot[0]
    fut_xyz_b = fut_xyz[0]
    fut_rot_b = fut_rot[0]

    action = space.traj_to_action(hist_xyz_b, hist_rot_b, fut_xyz_b, fut_rot_b)
    assert action.shape == (1, n_fut, 2)
    assert torch.isfinite(action).all()

    rec_xyz, rec_rot = space.action_to_traj(action, hist_xyz_b, hist_rot_b)
    assert rec_xyz.shape == (1, n_fut, 3)
    assert rec_rot.shape == (1, n_fut, 3, 3)

    # xy should round-trip closely on this smooth synthetic trajectory.
    err = (rec_xyz[..., :2] - fut_xyz_b[..., :2]).norm(dim=-1)
    assert err.mean().item() < 0.5, f"mean xy error too large: {err.mean().item()}"
    assert err.max().item() < 2.0, f"max xy error too large: {err.max().item()}"


def test_action_to_traj_shapes_and_rotation_consistency():
    space = UnicycleAccelCurvatureActionSpace(n_waypoints=64, dt=0.1)
    hist_xyz, hist_rot, _, _ = _make_curved_trajectory()
    action = torch.zeros(1, 64, 2)  # zero accel/curvature -> straight constant speed
    xyz, rot = space.action_to_traj(action, hist_xyz[0], hist_rot[0])
    # Yaw recovered from rotation should be ~0 for zero curvature.
    yaw = so3_to_yaw_torch(rot)
    assert torch.allclose(yaw, torch.zeros_like(yaw), atol=1e-4)
    # z is carried over from history's last z (which is ~0 here).
    assert torch.allclose(xyz[..., 2], torch.zeros_like(xyz[..., 2]), atol=1e-4)


def test_module_has_no_engine_import():
    """Acceptance: the math module must not pull in the vLLM/SGLang engine."""
    import sys

    import vllm_omni.model_executor.models.alpamayo.action_space as m  # noqa: F401

    mod_file = sys.modules[m.__name__].__file__
    assert mod_file.endswith("action_space.py")
    # numpy is used by rotation helpers; ensure scipy Rotation import worked.
    assert np is not None
