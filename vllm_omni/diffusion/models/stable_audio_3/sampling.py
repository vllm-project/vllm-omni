# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""V-prediction samplers + distribution shift for Stable Audio 3.

PORT_FROM: stable_audio_3/inference/sampling.py (522 lines)
           stable_audio_3/inference/distribution_shift.py (helper file)

Unlike SA Open 1.0 (which uses diffusers' CosineDPMSolverMultistepScheduler),
SA3 uses v-prediction with one of these custom samplers:
  - dpmpp-3m-sde   (default, 8 steps)  ← sample_flow_dpmpp
  - rk4                                 ← sample_rk4
  - pingpong                            ← sample_flow_pingpong
  - discrete_euler                      ← sample_discrete_euler

Schedules come from build_schedule(); distribution shifts (logsnr/flux/full)
warp the sigma values for resolution/duration-aware sampling.

These are pure torch — no diffusers, no vllm-omni primitives. Port verbatim.
"""

from __future__ import annotations

from typing import Callable

import torch


# ---------------------------------------------------------------------------
# Distribution shift (PORT_FROM: inference/distribution_shift.py)
# ---------------------------------------------------------------------------


class IdentityDistributionShift:
    """No shift — sigma values pass through unchanged."""

    def __call__(self, sigmas: torch.Tensor, **kwargs) -> torch.Tensor:
        return sigmas


class LogSNRShift:
    """Default sampling shift: seq_len-invariant LogSNR remapping.

    PORT_FROM: inference/distribution_shift.py LogSNRShift
    Default at config: LogSNRShift(rate=0, anchor_logsnr=-6.2, logsnr_end=2.0)
    """

    def __init__(self, rate: float = 0.0, anchor_logsnr: float = -6.2, logsnr_end: float = 2.0) -> None:
        self.rate = rate
        self.anchor_logsnr = anchor_logsnr
        self.logsnr_end = logsnr_end

    def __call__(self, sigmas: torch.Tensor, **kwargs) -> torch.Tensor:
        # PORT_FROM: inference/distribution_shift.py LogSNRShift.__call__
        raise NotImplementedError


class DistributionShift:
    """Full distribution shift (config "type": "full")."""

    def __init__(self, **kwargs) -> None:
        # PORT_FROM: inference/distribution_shift.py DistributionShift
        raise NotImplementedError

    def __call__(self, sigmas: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class FluxDistributionShift:
    """Flux-style resolution-aware shift (config "type": "flux")."""

    def __init__(self, **kwargs) -> None:
        # PORT_FROM: inference/distribution_shift.py FluxDistributionShift
        raise NotImplementedError

    def __call__(self, sigmas: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Schedule construction (PORT_FROM: sampling.py:9-65)
# ---------------------------------------------------------------------------


def build_schedule(
    n_steps: int,
    sigma_max: float = 1.0,
    sigma_min: float = 0.0,
    schedule_type: str = "linear",
    rho: float = 1.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build the sigma schedule for the sampler.

    PORT_FROM: sampling.py:9-65 build_schedule
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Samplers (PORT_FROM: sampling.py:147-355)
# Each takes (model, x, sigmas) and returns the denoised latent.
# ---------------------------------------------------------------------------


def sample_discrete_euler(
    model: Callable,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    callback: Callable | None = None,
    disable_tqdm: bool = False,
    **extra_args,
) -> torch.Tensor:
    """Discrete Euler sampler. PORT_FROM: sampling.py:147-187"""
    raise NotImplementedError


def sample_rk4(
    model: Callable,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    callback: Callable | None = None,
    disable_tqdm: bool = False,
    **extra_args,
) -> torch.Tensor:
    """4th-order Runge-Kutta. PORT_FROM: sampling.py:189-225"""
    raise NotImplementedError


def sample_flow_dpmpp(
    model: Callable,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    callback: Callable | None = None,
    disable_tqdm: bool = False,
    **extra_args,
) -> torch.Tensor:
    """DPM++ 3M-SDE (DEFAULT for SA3). PORT_FROM: sampling.py:227-306"""
    raise NotImplementedError


def sample_flow_pingpong(
    model: Callable,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    callback: Callable | None = None,
    disable_tqdm: bool = False,
    **extra_args,
) -> torch.Tensor:
    """Pingpong sampler. PORT_FROM: sampling.py:308-354"""
    raise NotImplementedError


# Registry — string → sampler function
SAMPLER_REGISTRY: dict[str, Callable] = {
    "dpmpp-3m-sde": sample_flow_dpmpp,
    "rk4": sample_rk4,
    "pingpong": sample_flow_pingpong,
    "discrete_euler": sample_discrete_euler,
}


def sample_diffusion(
    model: Callable,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    sampler_type: str = "dpmpp-3m-sde",
    **kwargs,
) -> torch.Tensor:
    """Dispatcher. PORT_FROM: sampling.py:356-..."""
    if sampler_type not in SAMPLER_REGISTRY:
        raise ValueError(f"Unknown sampler: {sampler_type}. Options: {list(SAMPLER_REGISTRY)}")
    return SAMPLER_REGISTRY[sampler_type](model, x, sigmas, **kwargs)
