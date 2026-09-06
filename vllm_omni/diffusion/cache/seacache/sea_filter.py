# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import math

import torch


# SeaCache paper: "Spectral-Evolution-Aware Cache for Accelerating Diffusion Models"
# https://arxiv.org/abs/2602.18993
def apply_sea_filter(hidden_states: torch.Tensor, sigma: float, power_exp: float = 3.0) -> torch.Tensor:
    """Apply the separable SEA Wiener filter over ``(T, H, W)``."""
    original_dtype = hidden_states.dtype
    hidden_states = hidden_states.contiguous().float()
    dimensions = (0, 1, 2)
    spectrum = torch.fft.fftn(hidden_states, dim=dimensions)

    sigma = max(1e-6, min(1.0 - 1e-6, float(sigma)))
    signal_scale = 1.0 - sigma
    noise_scale = sigma
    gain: torch.Tensor | None = None
    for axis in dimensions:
        frequencies = torch.fft.fftfreq(
            hidden_states.shape[axis],
            device=hidden_states.device,
            dtype=torch.float32,
        )
        clean_power = 1.0 / (frequencies.abs().pow(power_exp) + 1e-16)
        axis_gain = signal_scale * clean_power / (signal_scale**2 * clean_power + noise_scale**2 + 1e-16)
        axis_shape = [1] * hidden_states.ndim
        axis_shape[axis] = axis_gain.shape[0]
        reshaped_gain = axis_gain.reshape(axis_shape)
        gain = reshaped_gain if gain is None else gain * reshaped_gain

    assert gain is not None
    mean_gain = gain.mean()
    if torch.isfinite(mean_gain) and mean_gain > 0:
        gain = gain / mean_gain
    return torch.fft.ifftn(spectrum * gain, dim=dimensions).real.to(original_dtype)


def relative_l1_distance(current: torch.Tensor, previous: torch.Tensor, eps: float = 1e-16) -> float:
    numerator = (current.float() - previous.float()).abs().mean()
    denominator = previous.float().abs().mean() + eps
    return float((numerator / denominator).detach().cpu())


def indicator_distance(
    current: list[torch.Tensor] | None,
    previous: list[torch.Tensor] | None,
) -> float:
    """Mean relative-L1 distance, or infinity for incompatible indicators."""
    if current is None or previous is None or not current or len(current) != len(previous):
        return float("inf")

    total = 0.0
    for current_value, previous_value in zip(current, previous, strict=True):
        if (
            current_value.shape != previous_value.shape
            or current_value.device != previous_value.device
            or current_value.dtype != previous_value.dtype
        ):
            return float("inf")
        total += relative_l1_distance(current_value, previous_value)
    distance = total / len(current)
    return distance if math.isfinite(distance) else float("inf")


def extrapolate_residual(
    history: list[tuple[int, torch.Tensor]],
    step: int,
    order: int,
) -> torch.Tensor:
    """Evaluate a Newton residual polynomial at ``step``."""
    if not history:
        raise ValueError("SeaCache residual extrapolation requires non-empty history")

    polynomial_order = max(0, min(order, len(history) - 1))
    window = history[-(polynomial_order + 1) :]
    steps = [history_step for history_step, _ in window]
    coefficients = [residual.clone() for _, residual in window]

    for level in range(1, polynomial_order + 1):
        for index in range(polynomial_order, level - 1, -1):
            denominator = steps[index] - steps[index - level]
            if denominator == 0:
                return window[-1][1].clone()
            coefficients[index] = (coefficients[index] - coefficients[index - 1]) / float(denominator)

    result = coefficients[polynomial_order]
    for index in range(polynomial_order - 1, -1, -1):
        result = result * float(step - steps[index]) + coefficients[index]
    return result
