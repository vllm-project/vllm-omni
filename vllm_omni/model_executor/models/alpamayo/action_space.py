# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Self-contained trajectory / action-space math for Alpamayo VLA models.

Ported from the SGLang reference implementation
(``python/sglang/srt/models/alpamayo_r1.py``).

This module is intentionally free of any vLLM / SGLang engine dependency: it
only uses ``torch``, ``numpy``, ``einops`` and ``scipy`` so it can be imported
and unit-tested in isolation.

Contents:
- rotation helpers (so3<->yaw, 2d<->3d, angle wrapping)
- least-squares smoothing solvers (construct_DTD / cholesky based)
- ``UnicycleAccelCurvatureActionSpace`` (action <-> trajectory conversion)
- action input/output projection modules (Fourier + MLP)
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Any, TypeVar

import einops
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R  # noqa: N817

logger = logging.getLogger(__name__)

TensorOrNDArray = TypeVar("TensorOrNDArray", torch.Tensor, np.ndarray)


# ==============================================================================
# Minimal RMSNorm (replaces sglang.srt.layers.layernorm.RMSNorm).
# Matches the standard RMSNorm semantics used by the reference action_in_proj.
# ==============================================================================
class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return (
            (self.weight * x.to(input_dtype))
            if self.weight.dtype == input_dtype
            else (self.weight.to(input_dtype) * x.to(input_dtype))
        )


# ==============================================================================
# rotation helpers
# ==============================================================================
def so3_to_yaw_torch(rot_mat: torch.Tensor) -> torch.Tensor:
    cos_th_cos_phi = rot_mat[..., 0, 0]
    cos_th_sin_phi = rot_mat[..., 1, 0]
    return torch.atan2(cos_th_sin_phi, cos_th_cos_phi)


def so3_to_yaw_np(rot_mat: np.ndarray) -> np.ndarray:
    cos_th_cos_phi = rot_mat[..., 0, 0]
    cos_th_sin_phi = rot_mat[..., 1, 0]
    return np.arctan2(cos_th_sin_phi, cos_th_cos_phi)


def euler_2_so3(euler_angles: np.ndarray, degrees: bool = True, seq: str = "xyz") -> np.ndarray:
    return R.from_euler(seq=seq, angles=euler_angles, degrees=degrees).as_matrix().astype(np.float32)


def angle_wrap(radians: TensorOrNDArray) -> TensorOrNDArray:
    return (radians + np.pi) % (2 * np.pi) - np.pi


def rotation_matrix(angle: float | np.ndarray) -> np.ndarray:
    batch_dims = 0
    if isinstance(angle, np.ndarray):
        batch_dims = angle.ndim
    rotmat: np.ndarray = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return rotmat.transpose(*np.arange(2, batch_dims + 2), 0, 1)


def rotation_matrix_torch(angle: torch.Tensor) -> torch.Tensor:
    rotmat: torch.Tensor = torch.stack(
        [
            torch.stack([torch.cos(angle), -torch.sin(angle)], dim=-1),
            torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1),
        ],
        dim=-2,
    )
    return rotmat


def stable_gramschmidt(
    M: torch.Tensor,  # noqa: N803
) -> torch.Tensor:
    EPS = 1e-7
    x = M[..., 0]
    y = M[..., 1]
    x = x / torch.clamp_min(torch.norm(x, dim=-1, keepdim=True), EPS)
    y = y - torch.sum(x * y, dim=-1, keepdim=True) * x
    y = y / torch.clamp_min(torch.norm(y, dim=-1, keepdim=True), EPS)
    z = torch.cross(x, y, dim=-1)
    _R = torch.stack((x, y, z), dim=-1)
    return _R


def rot_3d_to_2d(rot: torch.Tensor) -> torch.Tensor:
    xu = rot[..., :2, 0]
    yu = rot[..., :2, 1]
    EPS = 1e-6
    xu = xu / (torch.norm(xu, dim=-1, keepdim=True) + EPS)
    yu = yu - torch.sum(xu * yu, dim=-1, keepdim=True) * xu
    yu = yu / (torch.norm(yu, dim=-1, keepdim=True) + EPS)
    return torch.stack((xu, yu), dim=-1)


def rot_2d_to_3d(rot: torch.Tensor) -> torch.Tensor:
    rot = torch.cat(
        [
            torch.cat([rot, torch.zeros_like(rot[..., :1])], dim=-1),
            torch.tensor([0.0, 0.0, 1.0], device=rot.device).repeat(rot.shape[:-2] + (1, 1)),
        ],
        dim=-2,
    )
    return rot


def ratan2(s: torch.Tensor, c: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    sign = (c >= 0).float() * 2 - 1
    eps = eps * (c.abs() < eps).type(c.dtype) * sign
    return torch.arctan2(s, c + eps)


def round_2pi(x: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(x), np.cos(x))


def round_2pi_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


# ==============================================================================
# action_space_utils — least-squares smoothing solvers
# ==============================================================================
def unwrap_angle(phi: torch.Tensor) -> torch.Tensor:
    d = torch.diff(phi, dim=-1)
    d = round_2pi_torch(d)
    return torch.cat([phi[..., :1], phi[..., :1] + torch.cumsum(d, dim=-1)], dim=-1)


def first_order_D(
    N: int,  # noqa: N803
    lead_shape: tuple[int, ...],
    device=torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    D = torch.zeros(*lead_shape, N - 1, N, dtype=dtype, device=device)
    rows = torch.arange(N - 1, device=device)
    D[..., rows, rows] = -1.0
    D[..., rows, rows + 1] = 1.0
    return D


def second_order_D(
    N: int,  # noqa: N803
    lead_shape: tuple[int, ...],
    device=torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    D = torch.zeros(*lead_shape, max(N - 2, 0), N, dtype=dtype, device=device)
    rows = torch.arange(max(N - 2, 0), device=device)
    D[..., rows, rows] = -1.0
    D[..., rows, rows + 1] = 2.0
    D[..., rows, rows + 2] = -1.0
    return D


def third_order_D(
    N: int,  # noqa: N803
    lead_shape: tuple[int, ...],
    device=torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    D = torch.zeros(*lead_shape, max(N - 3, 0), N, dtype=dtype, device=device)
    rows = torch.arange(max(N - 3, 0), device=device)
    D[..., rows, rows] = -1.0
    D[..., rows, rows + 1] = 3.0
    D[..., rows, rows + 2] = -3.0
    D[..., rows, rows + 3] = 1.0
    return D


@torch.amp.autocast(device_type="cuda", enabled=False)
@torch.no_grad()
@torch._dynamo.disable()
def construct_DTD(
    N: int,  # noqa: N803
    lead: tuple[int, ...],
    device=torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
    w_smooth1: float | torch.Tensor | None = None,
    w_smooth2: float | torch.Tensor | None = None,
    w_smooth3: float | torch.Tensor | None = None,
    lam: float = 1e-3,
    dt: float = 1.0,
) -> torch.Tensor:
    DTD = torch.zeros(*lead, N, N, dtype=dtype, device=device)
    if w_smooth1 is not None:
        lam_1 = lam / dt**2
        if isinstance(w_smooth1, float):
            w_smooth1_tensor = torch.full((*lead, max(N - 1, 0)), w_smooth1, dtype=dtype, device=device)
        else:
            w_smooth1_tensor = w_smooth1
        D1 = first_order_D(N, lead, device=device, dtype=dtype)
        DTD += lam_1 * einops.einsum(D1 * w_smooth1_tensor.unsqueeze(-1), D1, "... i j, ... i k -> ... j k")
    if w_smooth2 is not None:
        lam_2 = lam / dt**4
        if isinstance(w_smooth2, float):
            w_smooth2_tensor = torch.full((*lead, max(N - 2, 0)), w_smooth2, dtype=dtype, device=device)
        else:
            w_smooth2_tensor = w_smooth2
        D2 = second_order_D(N, lead, device=device, dtype=dtype)
        DTD += lam_2 * einops.einsum(D2 * w_smooth2_tensor.unsqueeze(-1), D2, "... i j, ... i k -> ... j k")
    if w_smooth3 is not None:
        lam_3 = lam / dt**6
        if isinstance(w_smooth3, float):
            w_smooth3_tensor = torch.full((*lead, max(N - 3, 0)), w_smooth3, dtype=dtype, device=device)
        else:
            w_smooth3_tensor = w_smooth3
        D3 = third_order_D(N, lead, device=device, dtype=dtype)
        DTD += lam_3 * einops.einsum(D3 * w_smooth3_tensor.unsqueeze(-1), D3, "... i j, ... i k -> ... j k")
    return DTD


@torch.amp.autocast(device_type="cuda", enabled=False)
@torch.no_grad()
@torch._dynamo.disable()
def solve_single_constraint(
    x_init: torch.Tensor,
    x_target: torch.Tensor,
    w_data: torch.Tensor | None = None,
    w_smooth1=None,
    w_smooth2=None,
    w_smooth3=None,
    lam: float = 1e-3,
    ridge: float = 0.0,
    dt: float = 1.0,
) -> torch.Tensor:
    device, dtype = x_target.device, x_target.dtype
    *lead, N = x_target.shape
    if N <= 0:
        raise ValueError("x_mid must have a positive last-dimension length N.")
    if w_data is None:
        w_data = torch.ones_like(x_target)
    x_init = torch.as_tensor(x_init, dtype=dtype, device=device)

    A_data = torch.eye(N, dtype=dtype, device=device).expand(*lead, N, N)
    Aw_data = A_data * w_data.unsqueeze(-1)
    with torch.amp.autocast(device_type="cuda", enabled=False):
        ATA = einops.einsum(Aw_data, A_data, "... i j, ... i k -> ... j k")
        rhs = einops.einsum(Aw_data, x_target, "... i j, ... i -> ... j")

    DTD = construct_DTD(
        N + 1,
        lead,
        device=device,
        dtype=dtype,
        w_smooth1=w_smooth1,
        w_smooth2=w_smooth2,
        w_smooth3=w_smooth3,
        lam=lam,
        dt=dt,
    )
    rhs -= DTD[..., 1:, 0] * x_init.unsqueeze(-1)

    ridge_term = ridge * torch.eye(N, dtype=dtype, device=device).expand(*lead, N, N)
    lhs = ATA + DTD[..., 1:, 1:] + ridge_term

    L = torch.linalg.cholesky(lhs)
    x = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
    x = torch.cat([x_init.unsqueeze(-1), x], dim=-1)
    return x


@torch.amp.autocast(device_type="cuda", enabled=False)
@torch.no_grad()
@torch._dynamo.disable()
def solve_xs_eq_y(
    s: torch.Tensor,
    y: torch.Tensor,
    w_data: torch.Tensor | None = None,
    w_smooth1=None,
    w_smooth2=None,
    w_smooth3=None,
    lam: float = 1e-3,
    ridge: float = 0.0,
    dt: float = 1.0,
) -> torch.Tensor:
    device, dtype = y.device, y.dtype
    *lead, N = y.shape
    if w_data is None:
        w_data = torch.ones_like(y)
    if w_data.shape != y.shape:
        raise ValueError("w_data must have the same shape as y")

    A_data = torch.diag_embed(s)
    Aw_data = A_data * w_data.unsqueeze(-1)
    with torch.amp.autocast(device_type="cuda", enabled=False):
        ATA = einops.einsum(Aw_data, A_data, "... i j, ... i k -> ... j k")
        rhs = einops.einsum(Aw_data, y, "... i j, ... i -> ... j")

    DTD = construct_DTD(
        N,
        lead,
        device=device,
        dtype=dtype,
        w_smooth1=w_smooth1,
        w_smooth2=w_smooth2,
        w_smooth3=w_smooth3,
        lam=lam,
        dt=dt,
    )

    L = None
    while L is None:
        try:
            ridge_term = ridge * torch.eye(N, dtype=dtype, device=device).expand(*lead, N, N)
            lhs = ATA + DTD + ridge_term
            if rhs.dtype != lhs.dtype:
                rhs = rhs.to(lhs.dtype)
            L = torch.linalg.cholesky(lhs)
        except RuntimeError as e:
            logger.error(f"Error in cholesky decomposition: {e}", exc_info=True)
            ridge *= 10
            logger.warning(f"Resolving singularity using ridge {ridge}")
    return torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)


@torch.no_grad()
@torch.amp.autocast(device_type="cuda", enabled=False)
@torch._dynamo.disable()
def dxy_theta_to_v_without_v0(
    dxy: torch.Tensor, theta: torch.Tensor, dt: float = 1.0, v_lambda: float = 1e-4, v_ridge: float = 1e-4
) -> torch.Tensor:
    *lead, N, _ = dxy.shape
    device, dtype = dxy.device, dxy.dtype
    g = 2 / dt * dxy
    w = torch.ones_like(dxy[..., 0])

    A_data = torch.zeros(*lead, 2 * N, N + 1, dtype=dtype, device=device)
    b_data = g.flatten(start_dim=-2)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    cos_rows = 2 * torch.arange(N, device=device)
    sin_rows = 2 * torch.arange(N, device=device) + 1
    cols = torch.arange(N, device=device)
    A_data[..., cos_rows, cols] = cos_theta[..., :-1]
    A_data[..., cos_rows, cols + 1] = cos_theta[..., 1:]
    A_data[..., sin_rows, cols] = sin_theta[..., :-1]
    A_data[..., sin_rows, cols + 1] = sin_theta[..., 1:]
    Aw_data = A_data * torch.repeat_interleave(w, 2, dim=-1).unsqueeze(-1)
    with torch.amp.autocast(device_type="cuda", enabled=False):
        ATA = einops.einsum(Aw_data, A_data, "... i j, ... i k -> ... j k")
        rhs = einops.einsum(Aw_data, b_data, "... i j, ... i -> ... j")

    DTD = construct_DTD(
        N + 1, lead, device=device, dtype=dtype, w_smooth1=None, w_smooth2=None, w_smooth3=1.0, lam=v_lambda, dt=dt
    )
    ridge_term = v_ridge * torch.eye(N + 1, dtype=dtype, device=device).expand(*lead, N + 1, N + 1)
    lhs = ATA + DTD + ridge_term
    L = torch.linalg.cholesky(lhs)
    y = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
    return y


@torch.no_grad()
@torch.amp.autocast(device_type="cuda", enabled=False)
@torch._dynamo.disable()
def dxy_theta_to_v(
    dxy: torch.Tensor,
    theta: torch.Tensor,
    v0: torch.Tensor,
    dt: float = 1.0,
    v_lambda: float = 1e-4,
    v_ridge: float = 1e-4,
) -> torch.Tensor:
    *lead, N, _ = dxy.shape
    device, dtype = dxy.device, dxy.dtype
    g = 2 / dt * dxy
    w = torch.ones_like(dxy[..., 0])

    A_data = torch.zeros(*lead, 2 * N, N + 1, dtype=dtype, device=device)
    b_data = g.flatten(start_dim=-2)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    cos_rows = 2 * torch.arange(N, device=device)
    sin_rows = 2 * torch.arange(N, device=device) + 1
    cols = torch.arange(N, device=device)
    A_data[..., cos_rows, cols] = cos_theta[..., :-1]
    A_data[..., cos_rows, cols + 1] = cos_theta[..., 1:]
    A_data[..., sin_rows, cols] = sin_theta[..., :-1]
    A_data[..., sin_rows, cols + 1] = sin_theta[..., 1:]
    Aw_data = A_data * torch.repeat_interleave(w, 2, dim=-1).unsqueeze(-1)
    with torch.amp.autocast(device_type="cuda", enabled=False):
        ATA = einops.einsum(Aw_data, A_data, "... i j, ... i k -> ... j k")
        rhs = einops.einsum(Aw_data[..., :, 1:], b_data, "... i j, ... i -> ... j")
    rhs -= ATA[..., 1:, 0] * v0.unsqueeze(-1)

    DTD = construct_DTD(
        N + 1, lead, device=device, dtype=dtype, w_smooth1=None, w_smooth2=None, w_smooth3=1.0, lam=v_lambda, dt=dt
    )
    rhs -= DTD[..., 1:, 0] * v0.unsqueeze(-1)

    ridge_term = v_ridge * torch.eye(N, dtype=dtype, device=device).expand(*lead, N, N)
    lhs = ATA[..., 1:, 1:] + DTD[..., 1:, 1:] + ridge_term
    L = torch.linalg.cholesky(lhs)
    y = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
    return torch.cat([v0.unsqueeze(-1), y], dim=-1)


@torch.no_grad()
@torch.amp.autocast(device_type="cuda", enabled=False)
@torch._dynamo.disable()
def theta_smooth(
    traj_future_rot: torch.Tensor, dt: float = 1.0, theta_lambda: float = 1e-4, theta_ridge: float = 1e-4
) -> torch.Tensor:
    theta = so3_to_yaw_torch(traj_future_rot)
    theta = unwrap_angle(theta)
    theta_init = torch.zeros_like(theta[..., 0])
    return solve_single_constraint(
        x_init=theta_init,
        x_target=theta,
        w_smooth1=None,
        w_smooth2=None,
        w_smooth3=1.0,
        dt=dt,
        lam=theta_lambda,
        ridge=theta_ridge,
    )


# ==============================================================================
# action space base
# ==============================================================================
class ActionSpace(ABC, nn.Module):
    """Action space base class for trajectory generation."""

    @abstractmethod
    def traj_to_action(
        self, traj_history_xyz, traj_history_rot, traj_future_xyz, traj_future_rot, *args: Any, **kwargs: Any
    ) -> torch.Tensor: ...

    @abstractmethod
    def action_to_traj(
        self, action, traj_history_xyz, traj_history_rot, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def get_action_space_dims(self) -> tuple[int, ...]: ...

    def is_within_bounds(self, action: torch.Tensor) -> torch.Tensor:
        num_action_dims = len(self.get_action_space_dims())
        batch_shape = action.shape[:-num_action_dims] if num_action_dims > 0 else action.shape
        return torch.ones(batch_shape, dtype=torch.bool, device=action.device)


# ==============================================================================
# action input projection
# ==============================================================================
class MLPEncoder(nn.Module):
    """Basic MLP encoder."""

    def __init__(self, num_input_feats: int, num_enc_layers: int, hidden_size: int, outdim: int):
        super().__init__()
        assert 1 <= num_enc_layers, f"{num_enc_layers=} must be >= 1"
        enc_layers = [nn.Linear(num_input_feats, hidden_size), nn.SiLU()]
        for layeri in range(num_enc_layers):
            if layeri < num_enc_layers - 1:
                enc_layers.extend([RMSNorm(hidden_size, eps=1e-5), nn.Linear(hidden_size, hidden_size), nn.SiLU()])
            else:
                enc_layers.extend([RMSNorm(hidden_size, eps=1e-5), nn.Linear(hidden_size, outdim)])
        self.trunk = nn.Sequential(*enc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


class FourierEncoderV2(nn.Module):
    """Fourier feature encoder with logarithmically-spaced frequencies."""

    def __init__(self, dim: int, max_freq: float = 100.0, persistent: bool = True):
        super().__init__()
        half = dim // 2
        freqs = torch.logspace(0, math.log10(max_freq), steps=half)
        self.out_dim = dim
        self.register_buffer("freqs", freqs[None, :], persistent=persistent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        arg = x[..., None] * self.freqs * 2 * torch.pi
        return torch.cat([torch.sin(arg), torch.cos(arg)], -1) * math.sqrt(2)


class PerWaypointActionInProjV2(nn.Module):
    """Per-waypoint action input projection module."""

    def __init__(
        self,
        in_dims: list[int],
        out_dim: int,
        num_enc_layers: int = 4,
        hidden_size: int = 1024,
        max_freq: float = 100.0,
        num_fourier_feats: int = 20,
        fourier_persistent: bool = True,
    ):
        super().__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim
        sinus = []
        for _ in range(in_dims[-1]):
            sinus.append(FourierEncoderV2(dim=num_fourier_feats, max_freq=max_freq, persistent=fourier_persistent))
        self.sinus = nn.ModuleList(sinus)
        self.timestep_fourier_encoder = FourierEncoderV2(
            dim=num_fourier_feats, max_freq=max_freq, persistent=fourier_persistent
        )
        num_input_feats = sum(s.out_dim for s in self.sinus) + self.timestep_fourier_encoder.out_dim
        self.encoder = MLPEncoder(
            num_input_feats=num_input_feats, num_enc_layers=num_enc_layers, hidden_size=hidden_size, outdim=out_dim
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        action_feats = torch.cat([s(x[:, :, i]) for i, s in enumerate(self.sinus)], dim=-1)
        timestep_feats = self.timestep_fourier_encoder(timesteps[..., -1])
        timestep_feats = timestep_feats.repeat(1, T, 1)
        x = torch.cat((action_feats, timestep_feats), dim=-1)
        x = x.to(dtype=self.encoder.trunk[0].weight.dtype)
        return self.norm(self.encoder(x.flatten(0, 1)).reshape(B, T, -1))


# ==============================================================================
# UnicycleAccelCurvatureActionSpace
# ==============================================================================
class UnicycleAccelCurvatureActionSpace(ActionSpace):
    """Unicycle kinematic model with acceleration and curvature as control inputs."""

    def __init__(
        self,
        accel_mean: float = 0.0,
        accel_std: float = 1.0,
        curvature_mean: float = 0.0,
        curvature_std: float = 1.0,
        accel_bounds: tuple[float, float] = (-9.8, 9.8),
        curvature_bounds: tuple[float, float] = (-0.2, 0.2),
        dt: float = 0.1,
        n_waypoints: int = 64,
        theta_lambda: float = 1e-6,
        theta_ridge: float = 1e-8,
        v_lambda: float = 1e-6,
        v_ridge: float = 1e-4,
        a_lambda: float = 1e-4,
        a_ridge: float = 1e-4,
        kappa_lambda: float = 1e-4,
        kappa_ridge: float = 1e-4,
    ):
        super().__init__()
        self.register_buffer("accel_mean", torch.tensor(accel_mean))
        self.register_buffer("accel_std", torch.tensor(accel_std))
        self.register_buffer("curvature_mean", torch.tensor(curvature_mean))
        self.register_buffer("curvature_std", torch.tensor(curvature_std))
        self.accel_bounds = accel_bounds
        self.curvature_bounds = curvature_bounds
        self.dt = dt
        self.n_waypoints = n_waypoints
        self.theta_lambda = theta_lambda
        self.theta_ridge = theta_ridge
        self.v_lambda = v_lambda
        self.v_ridge = v_ridge
        self.a_lambda = a_lambda
        self.a_ridge = a_ridge
        self.kappa_lambda = kappa_lambda
        self.kappa_ridge = kappa_ridge

    def get_action_space_dims(self) -> tuple[int, int]:
        return (self.n_waypoints, 2)

    def is_within_bounds(self, action: torch.Tensor) -> torch.Tensor:
        accel = action[..., 0]
        kappa = action[..., 1]
        accel_mean = self.accel_mean.to(accel.device)
        accel_std = self.accel_std.to(accel.device)
        kappa_mean = self.curvature_mean.to(kappa.device)
        kappa_std = self.curvature_std.to(kappa.device)
        accel = accel * accel_std + accel_mean
        kappa = kappa * kappa_std + kappa_mean
        is_accel_within_bounds = (accel >= self.accel_bounds[0]) & (accel <= self.accel_bounds[1])
        is_kappa_within_bounds = (kappa >= self.curvature_bounds[0]) & (kappa <= self.curvature_bounds[1])
        return torch.all(is_accel_within_bounds & is_kappa_within_bounds, dim=-1)

    @torch.no_grad()
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def _v_to_a(self, v: torch.Tensor) -> torch.Tensor:
        dv = (v[..., 1:] - v[..., :-1]) / self.dt
        a = solve_xs_eq_y(
            s=torch.ones_like(dv),
            y=dv,
            dt=self.dt,
            lam=self.a_lambda,
            ridge=self.a_ridge,
            w_smooth1=None,
            w_smooth2=1.0,
            w_smooth3=None,
        )
        return a

    @torch.no_grad()
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def _theta_v_a_to_kappa(self, theta: torch.Tensor, v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        dtheta = theta[..., 1:] - theta[..., :-1]
        dt = self.dt
        s = dt * v[..., :-1] + (dt**2) / 2.0 * a
        w = torch.ones_like(dtheta)
        return solve_xs_eq_y(
            s=s,
            y=dtheta,
            w_data=w,
            w_smooth1=None,
            w_smooth2=1.0,
            w_smooth3=None,
            lam=self.kappa_lambda,
            ridge=self.kappa_ridge,
            dt=self.dt,
        )

    @torch.no_grad()
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def estimate_t0_states(
        self, traj_history_xyz: torch.Tensor, traj_history_rot: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        full_xy = traj_history_xyz[..., :2]
        dxy = full_xy[..., 1:, :] - full_xy[..., :-1, :]
        theta = so3_to_yaw_torch(traj_history_rot)
        theta = unwrap_angle(theta)
        v = dxy_theta_to_v_without_v0(dxy=dxy, theta=theta, dt=self.dt, v_lambda=self.v_lambda, v_ridge=self.v_ridge)
        v_t0 = v[..., -1]
        return {"v": v_t0}

    @torch.no_grad()
    @torch._dynamo.disable()
    @torch.amp.autocast(device_type="cuda", enabled=False)
    def traj_to_action(
        self,
        traj_history_xyz,
        traj_history_rot,
        traj_future_xyz,
        traj_future_rot,
        t0_states: dict[str, torch.Tensor] | None = None,
        output_all_states: bool = False,
    ):
        if traj_future_xyz.shape[-2] != self.n_waypoints:
            raise ValueError(
                f"future trajectory must have length {self.n_waypoints} but got {traj_future_xyz.shape[-2]}"
            )
        if t0_states is None:
            t0_states = self.estimate_t0_states(traj_history_xyz, traj_history_rot)

        full_xy = torch.cat([traj_history_xyz[..., -1:, :], traj_future_xyz], dim=-2)[..., :2]
        dxy = full_xy[..., 1:, :] - full_xy[..., :-1, :]
        theta = theta_smooth(
            traj_future_rot=traj_future_rot, dt=self.dt, theta_lambda=self.theta_lambda, theta_ridge=self.theta_ridge
        )

        v0 = t0_states["v"]
        v = dxy_theta_to_v(dxy=dxy, theta=theta, v0=v0, dt=self.dt, v_lambda=self.v_lambda, v_ridge=self.v_ridge)
        accel = self._v_to_a(v)
        kappa = self._theta_v_a_to_kappa(theta, v, accel)

        accel_mean = self.accel_mean.to(accel.device)
        accel_std = self.accel_std.to(accel.device)
        kappa_mean = self.curvature_mean.to(kappa.device)
        kappa_std = self.curvature_std.to(kappa.device)
        accel = (accel - accel_mean) / accel_std
        kappa = (kappa - kappa_mean) / kappa_std

        if not output_all_states:
            return torch.stack([accel, kappa], dim=-1)
        return torch.stack([accel, kappa], dim=-1), torch.stack([v[:, :-1], accel, theta[:, :-1]], dim=-1)

    def action_to_traj(
        self,
        action: torch.Tensor,
        traj_history_xyz: torch.Tensor,
        traj_history_rot: torch.Tensor,
        t0_states: dict[str, torch.Tensor] | None = None,
    ):
        accel, kappa = action[..., 0], action[..., 1]
        accel_mean = self.accel_mean.to(accel.device)
        accel_std = self.accel_std.to(accel.device)
        kappa_mean = self.curvature_mean.to(kappa.device)
        kappa_std = self.curvature_std.to(kappa.device)
        accel = accel * accel_std + accel_mean
        kappa = kappa * kappa_std + kappa_mean

        if t0_states is None:
            t0_states = self.estimate_t0_states(traj_history_xyz, traj_history_rot)
        v0 = t0_states["v"]
        dt = self.dt

        dt_2_term = 0.5 * (self.dt**2)
        velocity = torch.cat([v0.unsqueeze(-1), (v0.unsqueeze(-1) + torch.cumsum(accel * dt, dim=-1))], dim=-1)
        initial_yaw = torch.zeros_like(v0)
        theta = torch.cat(
            [
                initial_yaw.unsqueeze(-1),
                (
                    initial_yaw.unsqueeze(-1)
                    + torch.cumsum(kappa * velocity[..., :-1] * dt, dim=-1)
                    + torch.cumsum(kappa * accel * dt_2_term, dim=-1)
                ),
            ],
            dim=-1,
        )
        half_dt_term = 0.5 * dt
        initial_x = torch.zeros_like(v0)
        initial_y = torch.zeros_like(v0)
        x = (
            initial_x.unsqueeze(-1)
            + torch.cumsum(velocity[..., :-1] * torch.cos(theta[..., :-1]) * half_dt_term, dim=-1)
            + torch.cumsum(velocity[..., 1:] * torch.cos(theta[..., 1:]) * half_dt_term, dim=-1)
        )
        y = (
            initial_y.unsqueeze(-1)
            + torch.cumsum(velocity[..., :-1] * torch.sin(theta[..., :-1]) * half_dt_term, dim=-1)
            + torch.cumsum(velocity[..., 1:] * torch.sin(theta[..., 1:]) * half_dt_term, dim=-1)
        )
        batch_dim = traj_history_xyz.shape[:-2]
        traj_future_xyz = torch.zeros(
            *batch_dim, self.n_waypoints, 3, device=traj_history_xyz.device, dtype=traj_history_xyz.dtype
        )
        traj_future_xyz[..., 0] = x
        traj_future_xyz[..., 1] = y
        traj_future_xyz[..., 2] = traj_history_xyz[..., -1:, 2]
        traj_future_rot = rot_2d_to_3d(rotation_matrix_torch(theta[..., 1:]))
        return traj_future_xyz, traj_future_rot


# ==============================================================================
# DeltaTrajectoryTokenizer — history trajectory discretization (for prompt fusion)
# ==============================================================================
class DeltaTrajectoryTokenizer:
    """Delta trajectory tokenizer: discretizes per-step xyz deltas into bins.

    Used to encode the ego history into discrete tokens that are fused into the
    VLM prompt at the ``<|traj_history|>`` placeholders so the VLM can condition
    on past motion. Defaults mirror the SGLang reference.
    """

    def __init__(
        self,
        ego_xyz_min: tuple[float, float, float] = (-4, -4, -10),
        ego_xyz_max: tuple[float, float, float] = (4, 4, 10),
        ego_yaw_min: float = -np.pi,
        ego_yaw_max: float = np.pi,
        num_bins: int = 1000,
        predict_yaw: bool = False,
        load_weights: bool = False,
    ):
        self.ego_xyz_min = ego_xyz_min
        self.ego_xyz_max = ego_xyz_max
        self.num_bins = num_bins
        self._predict_yaw = predict_yaw
        self.ego_yaw_min = ego_yaw_min
        self.ego_yaw_max = ego_yaw_max

    @property
    def vocab_size(self) -> int:
        return self.num_bins

    def encode(
        self,
        hist_xyz: torch.Tensor,
        hist_rot: torch.Tensor,
        fut_xyz: torch.Tensor,
        fut_rot: torch.Tensor,
        hist_tstamp: torch.Tensor | None = None,
        fut_tstamp: torch.Tensor | None = None,
    ) -> torch.LongTensor:
        del hist_xyz, hist_rot, hist_tstamp, fut_tstamp
        xyz = torch.nn.functional.pad(fut_xyz, [0, 0, 1, 0, 0, 0])
        xyz = xyz[:, 1:] - xyz[:, :-1]
        ego_xyz_max = torch.tensor(self.ego_xyz_max, dtype=xyz.dtype, device=xyz.device)
        ego_xyz_min = torch.tensor(self.ego_xyz_min, dtype=xyz.dtype, device=xyz.device)
        xyz = (xyz - ego_xyz_min) / (ego_xyz_max - ego_xyz_min)
        xyz = (xyz * (self.num_bins - 1)).round().long()
        xyz = xyz.clamp(0, self.num_bins - 1)
        if not self._predict_yaw:
            return einops.rearrange(xyz, "b n m -> b (n m)")
        yaw = torch.atan2(fut_rot[..., 0, 1], fut_rot[..., 0, 0])
        yaw_padded = torch.nn.functional.pad(yaw, [1, 0, 0, 0])
        delta_yaw = yaw_padded[:, 1:] - yaw_padded[:, :-1]
        delta_yaw = torch.atan2(torch.sin(delta_yaw), torch.cos(delta_yaw))
        delta_yaw = (delta_yaw - self.ego_yaw_min) / (self.ego_yaw_max - self.ego_yaw_min)
        delta_yaw = (delta_yaw * (self.num_bins - 1)).round().long()
        delta_yaw = delta_yaw.clamp(0, self.num_bins - 1)
        xyzw = torch.cat([xyz, delta_yaw.unsqueeze(-1)], dim=-1)
        return einops.rearrange(xyzw, "b n m -> b (n m)")
