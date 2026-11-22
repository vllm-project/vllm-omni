# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-local H3 solvers reconstructed from the surviving experiment records.

RES coefficients follow ComfyUI_RH_MinMaxH3 sampler_core.py at
6abdf97ae8cdce548c751ce850adc5010b583b7c. Euler delegates to the existing
implementation to preserve its exact floating-point operation order.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from .scheduling_minimax_h3_euler_ancestral import minimax_h3_euler_eta0_step


def normalize_h3_sampler(name: str | None) -> str:
    if name is None:
        return "euler"
    if name not in ("euler", "res_multistep"):
        raise ValueError(f"Unsupported MiniMax H3 sampler: {name!r}; expected euler or res_multistep")
    return name


def res_multistep_coeffs(sigmas: Sequence[float]) -> list[tuple[float, float] | None]:
    values = [float(s) for s in sigmas]
    if len(values) < 2 or any(not math.isfinite(s) or not 0 <= s <= 1 for s in values):
        raise ValueError("H3 sigmas must contain at least two finite values in [0, 1]")
    if any(a <= b for a, b in zip(values, values[1:])):
        raise ValueError("H3 sigmas must be strictly decreasing")
    coeffs = []
    for i, (curr, nxt) in enumerate(zip(values, values[1:])):
        if i == 0 or nxt == 0:
            coeffs.append(None)
            continue
        h = math.log(curr / nxt)
        c2 = math.log(curr / values[i - 1]) / h
        # Stable evaluation near h=0, where subtracting phi1-1 loses precision.
        phi1 = math.expm1(-h) / (-h)
        phi2 = (phi1 - 1) / (-h) if abs(h) >= 1e-5 else 0.5 - h / 6 + h * h / 24 - h**3 / 120
        coeffs.append((h * (phi1 - phi2 / c2), h * phi2 / c2))
    return coeffs


class H3SampleSolver:
    """Own the denoised history for one request and one video/audio stream."""

    def __init__(self, name: str | None, sigmas: Sequence[float]):
        self.name = normalize_h3_sampler(name)
        self.sigmas = tuple(float(s) for s in sigmas)
        self.coeffs = res_multistep_coeffs(self.sigmas)
        self.old_denoised: torch.Tensor | None = None
        self.step_index = 0

    def step(self, state: torch.Tensor, denoised: torch.Tensor, step: int) -> torch.Tensor:
        if step != self.step_index or not 0 <= step < len(self.coeffs):
            raise ValueError("H3 solver steps must be consumed once, in order")
        coeff = self.coeffs[step]
        curr, nxt = self.sigmas[step : step + 2]
        if self.name == "euler" or coeff is None:
            out = minimax_h3_euler_eta0_step(state, denoised, sigma_curr=curr, sigma_next=nxt)
        else:
            old = self.old_denoised
            if old is None or state.shape != denoised.shape or old.shape != state.shape:
                raise ValueError("H3 solver state and denoised history shapes must match")
            if not torch.is_floating_point(state) or not torch.is_floating_point(denoised):
                raise ValueError("H3 solver requires floating point tensors")
            if not torch.isfinite(state).all() or not torch.isfinite(denoised).all():
                raise ValueError("H3 solver inputs must be finite")
            dtype = torch.float32 if state.dtype in (torch.float16, torch.bfloat16) else state.dtype
            ratio = state.new_tensor(nxt / curr, dtype=dtype)
            hb1, hb2 = coeff
            out = (ratio * state.to(dtype) + hb1 * denoised.to(dtype) + hb2 * old.to(dtype)).to(state.dtype)
            if not torch.isfinite(out).all():
                raise ValueError("H3 solver output must be finite")
        if self.name == "res_multistep":
            self.old_denoised = denoised.detach().clone() if step + 1 < len(self.coeffs) else None
        self.step_index += 1
        return out


def create_h3_sample_solver(name: str | None, sigmas: Sequence[float]) -> H3SampleSolver:
    return H3SampleSolver(name, sigmas)
