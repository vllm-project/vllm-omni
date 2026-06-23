# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from .dit import get_epss_timesteps


def apply_sway_sampling(t, sway_sampling_coef):
    if sway_sampling_coef is None:
        return t
    return t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)


def build_timesteps(steps, device, dtype, use_epss=True, sway_sampling_coef=-1.0):
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    if use_epss:
        t = get_epss_timesteps(steps, device=device, dtype=dtype)
    else:
        t = torch.linspace(0, 1, steps + 1, device=device, dtype=dtype)
    return apply_sway_sampling(t, sway_sampling_coef)


class Solver:
    def __init__(self, func, y0, sigma=0.25, temperature=1.5) -> None:
        self.func = func
        self.y0 = y0
        self.sigma = sigma
        self.temperature = temperature

    def integrate(self, t):
        solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
        solution[0] = self.y0

        j = 1
        y0 = self.y0
        for t0, t1 in zip(t[:-1], t[1:]):
            dt = t1 - t0
            f0 = self.func(t0, y0)
            dy = dt * f0
            y1 = y0 + dy

            while j < len(t) and t1 >= t[j]:
                solution[j] = self._linear_interp(t0, t1, y0, y1, t[j])
                j += 1

            noise = torch.randn_like(y0)
            shift = self.sigma * (self.temperature**0.5) * (abs(dt) ** 0.5) * noise
            y0 = y1 + shift

        return solution

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)


def integrate_cfm_steps(fn, y0, t, sde_args, sde_rnd, steps):
    for step in range(steps):
        dt = t[step + 1] - t[step]
        y0 = y0 + fn(t[step], y0) * dt
        y0 = y0 + sde_args[1] * (sde_args[2] ** 0.5) * (dt.abs() ** 0.5) * sde_rnd[step]
    return y0
