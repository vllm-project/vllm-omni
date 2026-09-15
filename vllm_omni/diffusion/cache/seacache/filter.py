# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pure SEA (spectral-evolution-aware) filter math for SeaCache: the
timestep-dependent Wiener gain of the optimal linear denoiser under a
power-law signal spectrum.
"""

from __future__ import annotations

import torch

# Keeps the flow-matching endpoints off the degenerate a=0 / b=0 filters. FLUX's
# first sigma is exactly 1.0, so this clamp is reached on every run.
_SIGMA_CLAMP = 1e-6

# Regularizes the power-law spectrum at DC: without it Sx(0) is infinite; with
# it the DC gain is the finite 1/a.
_SPECTRUM_EPS = 1e-16

_REL_L1_EPS = 1e-16

# Power-law exponent beta of the assumed spectrum; the paper fixes it per
# modality (2 for images, 3 for video).
_POWER_EXP_IMAGE = 2.0

_NORM_MODES = ("mean", "peak", "none")


def ab_from_sigma(sigma: float) -> tuple[float, float]:
    """Flow-matching mixture coefficients: x_t = a * x_0 + b * noise."""
    clamped = max(_SIGMA_CLAMP, min(1.0 - _SIGMA_CLAMP, float(sigma)))
    return 1.0 - clamped, clamped


def sea_filter_response(
    *,
    shape: torch.Size | tuple[int, ...],
    dims: tuple[int, ...],
    a: float,
    b: float,
    power_exp: float,
    norm_mode: str,
    device: torch.device,
) -> torch.Tensor:
    """Separable Wiener gain broadcast over `shape`, one 1-D factor per filtered axis."""
    response = None
    for axis in dims:
        freq = torch.fft.fftfreq(shape[axis], device=device, dtype=torch.float32).abs()
        signal_power = 1.0 / (freq**power_exp + _SPECTRUM_EPS)
        gain = (a * signal_power) / (a * a * signal_power + b * b + _SPECTRUM_EPS)
        view = [1] * len(shape)
        view[axis] = gain.shape[0]
        gain = gain.reshape(view)
        response = gain if response is None else response * gain

    if norm_mode == "mean":
        return response / response.mean()
    if norm_mode == "peak":
        return response / response.amax()
    return response


def apply_sea_filter(
    x: torch.Tensor,
    *,
    a: float,
    b: float,
    power_exp: float = _POWER_EXP_IMAGE,
    norm_mode: str = "mean",
    dims: tuple[int, ...] = (-2, -3),
) -> torch.Tensor:
    """Filter `x` along `dims` with the SEA response, in fp32, returning `x.dtype`."""
    x32 = x.contiguous().to(torch.float32)
    response = sea_filter_response(
        shape=x32.shape,
        dims=dims,
        a=a,
        b=b,
        power_exp=power_exp,
        norm_mode=norm_mode,
        device=x32.device,
    )
    spectrum = torch.fft.fftn(x32, dim=dims)
    return torch.fft.ifftn(spectrum * response, dim=dims).real.to(x.dtype)


def rel_l1(current: torch.Tensor, previous: torch.Tensor) -> float:
    """Relative L1 distance, normalized by the previous step's magnitude.

    Reduced in fp32 rather than the reference's bf16: a bf16-rounded ratio
    drifts a few percent against a threshold of order 0.3.
    """
    numerator = (current - previous).abs().float().mean()
    denominator = previous.abs().float().mean() + _REL_L1_EPS
    return float(numerator / denominator)
