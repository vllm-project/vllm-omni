# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# Adapted from NVlabs/alpamayo2 (Apache-2.0), commit 9596749e:
#   src/alpamayo2_super/models/expert.py
#   src/alpamayo2_super/models/action_in_proj.py
#   src/alpamayo2_super/action_space/unicycle_accel_curvature.py
#   src/alpamayo2_super/action_space/utils.py
#   src/alpamayo2_super/geometry/rotation.py
"""Alpamayo 2 Super action expert: flow-matching denoiser and action space."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


def _round_2pi(x: torch.Tensor) -> torch.Tensor:
    """Normalize angles to [-pi, pi]."""
    return torch.atan2(torch.sin(x), torch.cos(x))


def _so3_to_yaw(rot_mat: torch.Tensor) -> torch.Tensor:
    """Yaw of ``[..., 3, 3]`` rotation matrices (xyz order)."""
    return torch.atan2(rot_mat[..., 1, 0], rot_mat[..., 0, 0])


def _unwrap_angle(phi: torch.Tensor) -> torch.Tensor:
    """Unwrap the last dimension so consecutive differences lie in (-pi, pi]."""
    d = _round_2pi(torch.diff(phi, dim=-1))
    return torch.cat([phi[..., :1], phi[..., :1] + torch.cumsum(d, dim=-1)], dim=-1)


def _rotation_matrix_2d(angle: torch.Tensor) -> torch.Tensor:
    """``[...]`` angles to ``[..., 2, 2]`` rotation matrices."""
    return torch.stack(
        [
            torch.stack([torch.cos(angle), -torch.sin(angle)], dim=-1),
            torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1),
        ],
        dim=-2,
    )


def _rot_2d_to_3d(rot: torch.Tensor) -> torch.Tensor:
    """Embed ``[..., 2, 2]`` rotations in the flat xy plane as ``[..., 3, 3]``."""
    return torch.cat(
        [
            torch.cat([rot, torch.zeros_like(rot[..., :1])], dim=-1),
            torch.tensor([0.0, 0.0, 1.0], device=rot.device).repeat(rot.shape[:-2] + (1, 1)),
        ],
        dim=-2,
    )


def _third_order_dtd(n: int, lead: tuple[int, ...], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """``D^T D`` for the third-order finite-difference operator."""
    d = torch.zeros(*lead, max(n - 3, 0), n, dtype=dtype, device=device)
    rows = torch.arange(max(n - 3, 0), device=device)
    d[..., rows, rows] = -1.0
    d[..., rows, rows + 1] = 3.0
    d[..., rows, rows + 2] = -3.0
    d[..., rows, rows + 3] = 1.0
    return torch.einsum("...ij,...ik->...jk", d, d)


def _velocity_from_history(
    dxy: torch.Tensor,
    theta: torch.Tensor,
    *,
    dt: float,
    v_lambda: float,
    v_ridge: float,
) -> torch.Tensor:
    """Least-squares speeds from trapezoidal integration of the history.

    Solves ``v_t u_t + v_{t+1} u_{t+1} = 2 dp_t / dt`` for ``v_0..v_N`` with a
    third-order Tikhonov smoothness term, where ``u_t = (cos, sin)(theta_t)``.
    """
    *lead, n, _ = dxy.shape
    device, dtype = dxy.device, dxy.dtype
    a = torch.zeros(*lead, 2 * n, n + 1, dtype=dtype, device=device)
    b = (2 / dt * dxy).flatten(start_dim=-2)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    cos_rows = 2 * torch.arange(n, device=device)
    sin_rows = cos_rows + 1
    cols = torch.arange(n, device=device)
    a[..., cos_rows, cols] = cos_theta[..., :-1]
    a[..., cos_rows, cols + 1] = cos_theta[..., 1:]
    a[..., sin_rows, cols] = sin_theta[..., :-1]
    a[..., sin_rows, cols + 1] = sin_theta[..., 1:]
    ata = torch.einsum("...ij,...ik->...jk", a, a)
    rhs = torch.einsum("...ij,...i->...j", a, b)
    dtd = v_lambda / dt**6 * _third_order_dtd(n + 1, tuple(lead), device=device, dtype=dtype)
    ridge = v_ridge * torch.eye(n + 1, dtype=dtype, device=device).expand(*lead, n + 1, n + 1)
    chol = torch.linalg.cholesky(ata + dtd + ridge)
    return torch.cholesky_solve(rhs.unsqueeze(-1), chol).squeeze(-1)


class UnicycleAccelCurvatureActionSpace(nn.Module):
    """Unicycle kinematics with normalized acceleration and curvature controls."""

    def __init__(
        self,
        accel_mean: float = 0.0,
        accel_std: float = 1.0,
        curvature_mean: float = 0.0,
        curvature_std: float = 1.0,
        dt: float = 0.1,
        n_waypoints: int = 64,
        v_lambda: float = 1e-6,
        v_ridge: float = 1e-4,
    ) -> None:
        super().__init__()
        self.register_buffer("accel_mean", torch.tensor(accel_mean))
        self.register_buffer("accel_std", torch.tensor(accel_std))
        self.register_buffer("curvature_mean", torch.tensor(curvature_mean))
        self.register_buffer("curvature_std", torch.tensor(curvature_std))
        self.dt = dt
        self.n_waypoints = n_waypoints
        self.v_lambda = v_lambda
        self.v_ridge = v_ridge

    def get_action_space_dims(self) -> tuple[int, int]:
        return (self.n_waypoints, 2)

    def action_to_traj(
        self,
        action: torch.Tensor,
        traj_history_xyz: torch.Tensor,
        traj_history_rot: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate ``(..., T, 2)`` controls into ``(..., T, 3)`` positions and rotations."""
        accel = action[..., 0] * self.accel_std + self.accel_mean
        kappa = action[..., 1] * self.curvature_std + self.curvature_mean

        history_xy = traj_history_xyz[..., :2]
        theta_history = _unwrap_angle(_so3_to_yaw(traj_history_rot))
        v0 = _velocity_from_history(
            history_xy[..., 1:, :] - history_xy[..., :-1, :],
            theta_history,
            dt=self.dt,
            v_lambda=self.v_lambda,
            v_ridge=self.v_ridge,
        )[..., -1]

        dt = self.dt
        velocity = torch.cat([v0.unsqueeze(-1), v0.unsqueeze(-1) + torch.cumsum(accel * dt, dim=-1)], dim=-1)
        initial_yaw = torch.zeros_like(v0)
        theta = torch.cat(
            [
                initial_yaw.unsqueeze(-1),
                initial_yaw.unsqueeze(-1)
                + torch.cumsum(kappa * velocity[..., :-1] * dt, dim=-1)
                + torch.cumsum(kappa * accel * (0.5 * dt**2), dim=-1),
            ],
            dim=-1,
        )
        half_dt = 0.5 * dt
        x = torch.cumsum(velocity[..., :-1] * torch.cos(theta[..., :-1]) * half_dt, dim=-1) + torch.cumsum(
            velocity[..., 1:] * torch.cos(theta[..., 1:]) * half_dt, dim=-1
        )
        y = torch.cumsum(velocity[..., :-1] * torch.sin(theta[..., :-1]) * half_dt, dim=-1) + torch.cumsum(
            velocity[..., 1:] * torch.sin(theta[..., 1:]) * half_dt, dim=-1
        )
        traj_future_xyz = torch.zeros(
            *traj_history_xyz.shape[:-2],
            self.n_waypoints,
            3,
            device=traj_history_xyz.device,
            dtype=traj_history_xyz.dtype,
        )
        traj_future_xyz[..., 0] = x
        traj_future_xyz[..., 1] = y
        traj_future_xyz[..., 2] = traj_history_xyz[..., -1:, 2]
        traj_future_rot = _rot_2d_to_3d(_rotation_matrix_2d(theta[..., 1:]))
        return traj_future_xyz, traj_future_rot


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return output.type_as(x) * self.weight


class _MLPEncoder(nn.Module):
    def __init__(self, num_input_feats: int, num_enc_layers: int, hidden_size: int, outdim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(num_input_feats, hidden_size), nn.SiLU()]
        for layer_idx in range(num_enc_layers):
            layers.append(_RMSNorm(hidden_size, eps=1e-5))
            if layer_idx < num_enc_layers - 1:
                layers.extend([nn.Linear(hidden_size, hidden_size), nn.SiLU()])
            else:
                layers.append(nn.Linear(hidden_size, outdim))
        self.trunk = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


class _FourierEncoder(nn.Module):
    def __init__(self, dim: int, max_freq: float) -> None:
        super().__init__()
        self.out_dim = dim
        self.register_buffer("freqs", torch.logspace(0, math.log10(max_freq), steps=dim // 2)[None, :])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        arg = x[..., None] * self.freqs * 2 * torch.pi
        return torch.cat([torch.sin(arg), torch.cos(arg)], -1) * math.sqrt(2)


class PerWaypointActionInProj(nn.Module):
    """Embed per-waypoint controls and the diffusion time as expert tokens."""

    def __init__(
        self,
        in_dims: tuple[int, int],
        out_dim: int,
        num_enc_layers: int = 4,
        hidden_size: int = 1024,
        max_freq: float = 100.0,
        num_fourier_feats: int = 20,
    ) -> None:
        super().__init__()
        self.sinus = nn.ModuleList(_FourierEncoder(num_fourier_feats, max_freq) for _ in range(in_dims[-1]))
        self.timestep_fourier_encoder = _FourierEncoder(num_fourier_feats, max_freq)
        num_input_feats = num_fourier_feats * (in_dims[-1] + 1)
        self.encoder = _MLPEncoder(num_input_feats, num_enc_layers, hidden_size, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        batch_size, n_waypoints, _ = x.shape
        x = x.float()
        action_feats = torch.cat([encoder(x[:, :, idx]) for idx, encoder in enumerate(self.sinus)], dim=-1)
        timestep_feats = self.timestep_fourier_encoder(timesteps.float()[..., -1]).repeat(1, n_waypoints, 1)
        features = torch.cat((action_feats, timestep_feats), dim=-1).to(self.encoder.trunk[0].weight.dtype)
        encoded = self.encoder(features.flatten(0, 1)).reshape(batch_size, n_waypoints, -1)
        return self.norm(encoded)


# Checkpoint config keys that only the reference's training-time conversions
# use (trajectory-to-action smoothing, Hydra targets). Any other unknown key
# raises when the modules are built.
_REFERENCE_ONLY_KEYS = {
    "action_space_cfg": frozenset(
        {
            "_target_",
            "a_lambda",
            "a_ridge",
            "accel_bounds",
            "curvature_bounds",
            "kappa_lambda",
            "kappa_ridge",
            "theta_lambda",
            "theta_ridge",
        }
    ),
    "action_in_proj_cfg": frozenset({"_target_"}),
}


def _module_kwargs(expert_config: Mapping[str, Any], name: str) -> dict[str, Any]:
    return {key: value for key, value in expert_config[name].items() if key not in _REFERENCE_ONLY_KEYS[name]}


class Alpamayo2SuperActionExpert(nn.Module):
    """Non-causal Qwen3-VL text decoder that denoises trajectory controls.

    Parameter names match the checkpoint's ``expert.*`` weights.
    """

    def __init__(self, expert_config: Mapping[str, Any]) -> None:
        super().__init__()
        llm_config = expert_config["llm_config"]
        if isinstance(llm_config, Mapping):
            llm_config = AutoConfig.for_model(**llm_config)
        self.non_causal = bool(expert_config.get("expert_non_causal_attention", True))
        # The suffix attends to the VLM prefix through an explicit 4D mask,
        # which HF SDPA honors.
        self.expert = AutoModel.from_config(llm_config, attn_implementation="sdpa")
        del self.expert.embed_tokens

        self.action_space = UnicycleAccelCurvatureActionSpace(**_module_kwargs(expert_config, "action_space_cfg"))
        action_dims = self.action_space.get_action_space_dims()
        hidden_size = self.expert.config.hidden_size
        self.action_in_proj = PerWaypointActionInProj(
            in_dims=action_dims, out_dim=hidden_size, **_module_kwargs(expert_config, "action_in_proj_cfg")
        )
        self.action_out_proj = nn.Linear(hidden_size, action_dims[-1])
