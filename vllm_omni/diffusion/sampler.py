# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-local deterministic samplers operating on denoised (x0) predictions.

Callers own sigma scheduling, model prediction conversion and conditioning.
Create one Sampler per request, latent stream and denoising pass. Models with
specialized samplers can continue to use their own implementations.

RES coefficients follow ComfyUI_RH_MinMaxH3 sampler_core.py at
6abdf97ae8cdce548c751ce850adc5010b583b7c. Euler preserves the original
floating-point operation order.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch


def _require_finite_tensor(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must be finite")


def _validate_sigma(value: float, name: str) -> float:
    sigma = float(value)
    if not math.isfinite(sigma):
        raise ValueError(f"{name} must be finite")
    if sigma < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return sigma


def resolve_sampler_name(name: str | None) -> str:
    if name is None:
        return "euler"
    if name not in ("euler", "res_multistep"):
        raise ValueError(f"Unsupported sampler: {name!r}; expected euler or res_multistep")
    return name


def _validate_sigmas(sigmas: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(s) for s in sigmas)
    if len(values) < 2 or any(not math.isfinite(s) or not 0 <= s <= 1 for s in values):
        raise ValueError("Sampler sigmas must contain at least two finite values in [0, 1]")
    if any(a <= b for a, b in zip(values, values[1:])):
        raise ValueError("Sampler sigmas must be strictly decreasing")
    return values


class BaseSampler(ABC):
    """Share request-local sigma scheduling and ordered step tracking."""

    def __init__(self, sigmas: Sequence[float]) -> None:
        self.sigmas = _validate_sigmas(sigmas)
        self.step_index = 0

    def step(self, state: torch.Tensor, denoised: torch.Tensor, step: int) -> torch.Tensor:
        if step != self.step_index or not 0 <= step < len(self.sigmas) - 1:
            raise ValueError("Sampler steps must be consumed once, in order")
        out = self._step(state, denoised, step)
        self.step_index += 1
        return out

    @abstractmethod
    def _step(self, state: torch.Tensor, denoised: torch.Tensor, step: int) -> torch.Tensor:
        """Apply the algorithm without advancing the shared step index."""
        ...


class EulerSampler(BaseSampler):
    """Deterministic Euler updates from denoised predictions."""

    def _step(self, state: torch.Tensor, denoised: torch.Tensor, step: int) -> torch.Tensor:
        curr, nxt = self.sigmas[step : step + 2]
        return self.step_denoised(state, denoised, sigma_curr=curr, sigma_next=nxt)

    @staticmethod
    def step_denoised(
        state: torch.Tensor,
        denoised: torch.Tensor,
        *,
        sigma_curr: float,
        sigma_next: float,
    ) -> torch.Tensor:
        if state.shape != denoised.shape:
            raise ValueError(f"state and denoised shapes must match, got {state.shape} vs {denoised.shape}")
        if not torch.is_floating_point(state):
            raise ValueError("state must be a floating point tensor")
        if not torch.is_floating_point(denoised):
            raise ValueError("denoised must be a floating point tensor")
        _require_finite_tensor(state, "state")
        _require_finite_tensor(denoised, "denoised")
        sigma_curr = _validate_sigma(sigma_curr, "sigma_curr")
        sigma_next = _validate_sigma(sigma_next, "sigma_next")
        if sigma_curr == 0.0:
            if sigma_next != 0.0:
                raise ValueError("sigma_next must be 0 when sigma_curr is 0")
            return state
        compute_dtype = torch.float32
        if state.dtype not in (torch.float16, torch.bfloat16):
            compute_dtype = state.dtype
        sigma_curr_t = state.new_tensor(sigma_curr, dtype=compute_dtype)
        sigma_next_t = state.new_tensor(sigma_next, dtype=compute_dtype)
        sigma_ratio = sigma_next_t / sigma_curr_t
        out = sigma_ratio * state.to(dtype=compute_dtype) + (1.0 - sigma_ratio) * denoised.to(dtype=compute_dtype)
        out = out.to(dtype=state.dtype)
        _require_finite_tensor(out, "euler_eta0_step output")
        return out


class ResMultistepSampler(BaseSampler):
    """RES updates owning x0 history, with Euler at the first/zero-sigma step."""

    def __init__(self, sigmas: Sequence[float]) -> None:
        super().__init__(sigmas)
        self.coeffs = self.compute_coefficients(self.sigmas)
        self.old_denoised: torch.Tensor | None = None

    def _step(self, state: torch.Tensor, denoised: torch.Tensor, step: int) -> torch.Tensor:
        coeff = self.coeffs[step]
        curr, nxt = self.sigmas[step : step + 2]
        if coeff is None:
            out = EulerSampler.step_denoised(state, denoised, sigma_curr=curr, sigma_next=nxt)
        else:
            old = self.old_denoised
            if old is None or state.shape != denoised.shape or old.shape != state.shape:
                raise ValueError("Sampler state and denoised history shapes must match")
            if not torch.is_floating_point(state) or not torch.is_floating_point(denoised):
                raise ValueError("Sampler requires floating point tensors")
            if not torch.isfinite(state).all() or not torch.isfinite(denoised).all():
                raise ValueError("Sampler inputs must be finite")
            dtype = torch.float32 if state.dtype in (torch.float16, torch.bfloat16) else state.dtype
            ratio = state.new_tensor(nxt / curr, dtype=dtype)
            hb1, hb2 = coeff
            out = (ratio * state.to(dtype) + hb1 * denoised.to(dtype) + hb2 * old.to(dtype)).to(state.dtype)
            if not torch.isfinite(out).all():
                raise ValueError("Sampler output must be finite")
        self.old_denoised = denoised.detach().clone() if step + 1 < len(self.coeffs) else None
        return out

    @staticmethod
    def compute_coefficients(sigmas: Sequence[float]) -> list[tuple[float, float] | None]:
        values = _validate_sigmas(sigmas)
        coeffs: list[tuple[float, float] | None] = []
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


class Sampler:
    """Select and delegate to a sampler over decreasing sigmas in [0, 1].

    ``step`` consumes an x0 prediction, not raw model velocity or epsilon.
    Steps must be consumed once in order; start a new instance for a new pass.
    """

    def __init__(self, name: str | None, sigmas: Sequence[float]) -> None:
        self.name = resolve_sampler_name(name)
        sampler_cls = EulerSampler if self.name == "euler" else ResMultistepSampler
        self._sampler = sampler_cls(sigmas)

    def step(self, state: torch.Tensor, denoised: torch.Tensor, step: int) -> torch.Tensor:
        return self._sampler.step(state, denoised, step)
