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

import math
from collections.abc import Callable

import torch
from tqdm import tqdm, trange

# ---------------------------------------------------------------------------
# Distribution shift (PORT_FROM: inference/distribution_shift.py)
# ---------------------------------------------------------------------------


class IdentityDistributionShift:
    """No-op distribution shift — returns timesteps unchanged.

    PORT_FROM: stable-audio-3 inference/distribution_shift.py:6-9 (verbatim).
    """

    def shift(self, t: torch.Tensor, seq_len) -> torch.Tensor:
        return t


class FluxDistributionShift:
    """Flux/SD3/Self-Flow timestep shift: t_shifted = alpha * t / (1 + (alpha-1) * t).

    PORT_FROM: stable-audio-3 inference/distribution_shift.py:12-83 (verbatim).
    """

    def __init__(
        self,
        min_length: int = 256,
        max_length: int = 4096,
        alpha_min: float = 1.0,
        alpha_max: float = 1.0,
    ) -> None:
        self.min_length = min_length
        self.max_length = max_length
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        # Precompute for log-linear interpolation
        self.log_alpha_min = math.log(max(alpha_min, 1e-8))
        self.log_alpha_max = math.log(max(alpha_max, 1e-8))
        self.log_min_seq = math.log(min_length)
        self.log_max_seq = math.log(max_length)
        if self.log_max_seq == self.log_min_seq:
            # Prevent division by zero for constant alpha
            self.log_max_seq += 1e-8

    def get_alpha(self, seq_len: int | torch.Tensor):
        """Compute alpha via log-linear interpolation in seq_len."""
        if isinstance(seq_len, torch.Tensor):
            seq_len = seq_len.float().clamp(self.min_length, self.max_length)
            log_seq = torch.log(seq_len)
            frac = (log_seq - self.log_min_seq) / (self.log_max_seq - self.log_min_seq)
            log_alpha = self.log_alpha_min + frac * (self.log_alpha_max - self.log_alpha_min)
            return torch.exp(log_alpha)
        seq_len = max(min(seq_len, self.max_length), self.min_length)
        log_seq = math.log(seq_len)
        frac = (log_seq - self.log_min_seq) / (self.log_max_seq - self.log_min_seq)
        log_alpha = self.log_alpha_min + frac * (self.log_alpha_max - self.log_alpha_min)
        return math.exp(log_alpha)

    def shift(self, t: torch.Tensor, seq_len: int | torch.Tensor) -> torch.Tensor:
        alpha = self.get_alpha(seq_len)

        if isinstance(seq_len, torch.Tensor):
            alpha = alpha.to(t.device)
            if t.dim() == 1 and alpha.dim() == 1 and t.shape[0] != alpha.shape[0]:
                t = t.unsqueeze(0)
                alpha = alpha.unsqueeze(1)

        return alpha * t / (1 + (alpha - 1.0) * t)


class DistributionShift:
    """Full distribution shift used by SD3/Flux training-time recipes.

    PORT_FROM: stable-audio-3 inference/distribution_shift.py:86-128 (verbatim).
    """

    def __init__(
        self,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
        max_length: int = 4096,
        min_length: int = 256,
        use_sine: bool = False,
    ) -> None:
        self.base_shift = base_shift
        self.max_shift = max_shift
        self.max_length = max_length
        self.min_length = min_length
        self.use_sine = use_sine

    def shift(self, t: torch.Tensor, seq_len: int | torch.Tensor) -> torch.Tensor:
        if isinstance(seq_len, torch.Tensor):
            seq_len = seq_len.to(t.device)
            seq_len_clamped = seq_len.float().clamp(self.min_length, self.max_length)
            if t.dim() == 1 and seq_len_clamped.dim() == 1 and t.shape[0] != seq_len_clamped.shape[0]:
                # t:(steps,)->(1,steps), seq_len:(batch,)->(batch,1) → (batch,steps)
                t = t.unsqueeze(0)
                seq_len_clamped = seq_len_clamped.unsqueeze(1)
            sigma = 1.0
            mu = -(
                self.base_shift
                + (self.max_shift - self.base_shift)
                * (seq_len_clamped - self.min_length)
                / (self.max_length - self.min_length)
            )
            t_out = 1 - torch.exp(mu) / (torch.exp(mu) + (1 / (1 - t) - 1) ** sigma)
            if self.use_sine:
                t_out = torch.sin(t_out * math.pi / 2)
        else:
            seq_len = min(max(seq_len, self.min_length), self.max_length)
            sigma = 1.0
            mu = -(
                self.base_shift
                + (self.max_shift - self.base_shift) * (seq_len - self.min_length) / (self.max_length - self.min_length)
            )
            t_out = 1 - math.exp(mu) / (math.exp(mu) + (1 / (1 - t) - 1) ** sigma)
            if self.use_sine:
                t_out = torch.sin(t_out * math.pi / 2)

        return t_out


class LogSNRShift:
    """Adaptive log-SNR distribution shift (default SA3 sampling shift).

    PORT_FROM: stable-audio-3 inference/distribution_shift.py:131-198 (verbatim).
    Maps t∈[0,1] to log-SNR-spaced values while preserving order (0→0, 1→1).
    """

    def __init__(
        self,
        anchor_length: int = 2000,
        anchor_logsnr: float = -6.2,
        rate: float = 1.0,
        logsnr_end: float = 2.0,
    ) -> None:
        self.anchor_length = anchor_length
        self.anchor_logsnr = anchor_logsnr
        self.rate = rate
        self.logsnr_end = logsnr_end

    def get_logsnr_start(self, seq_len: int | torch.Tensor):
        """Compute adaptive logsnr_start: drops by `rate` per doubling of seq_len."""
        if isinstance(seq_len, torch.Tensor):
            log2_ratio = torch.log2(seq_len.float() / self.anchor_length)
            return self.anchor_logsnr - self.rate * log2_ratio
        log2_ratio = math.log2(seq_len / self.anchor_length)
        return self.anchor_logsnr - self.rate * log2_ratio

    def shift(self, t: torch.Tensor, seq_len: int | torch.Tensor) -> torch.Tensor:
        t_original = t
        logsnr_start = self.get_logsnr_start(seq_len)

        if isinstance(seq_len, torch.Tensor):
            logsnr_start = logsnr_start.to(t.device)
            if t.dim() == 1 and logsnr_start.dim() == 1 and t.shape[0] != logsnr_start.shape[0]:
                t = t.unsqueeze(0)
                logsnr_start = logsnr_start.unsqueeze(1)

        # Map t through log-SNR space (monotonically: low t → high logsnr → low t_out)
        logsnr = self.logsnr_end - t * (self.logsnr_end - logsnr_start)
        t_out = torch.sigmoid(-logsnr)

        # Preserve exact endpoints
        t_out = torch.where(t_original <= 0, torch.zeros_like(t_out), t_out)
        t_out = torch.where(t_original >= 1, torch.ones_like(t_out), t_out)

        return t_out


# ---------------------------------------------------------------------------
# Schedule construction (PORT_FROM: sampling.py:9-65)
# ---------------------------------------------------------------------------


def build_schedule(
    steps: int,
    sigma_max: float = 1.0,
    dist_shift: object | None = None,
    effective_seq_len: int | torch.Tensor | None = None,
    fallback_seq_len: int | None = None,
    include_endpoint: bool = True,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Build a timestep schedule for diffusion sampling.

    PORT_FROM: stable-audio-3 inference/sampling.py:9-65 build_schedule (verbatim).

    Returns a 1D tensor of shape (steps+1,) or (steps,), or a 2D tensor of
    shape (batch_size, N) when effective_seq_len is a tensor and dist_shift
    produces per-element schedules.

    Args:
        steps: Number of sampling steps.
        sigma_max: Starting noise level (1.0 for full generation, <1.0 for variations).
        dist_shift: Optional distribution shift (LogSNRShift / DistributionShift /
            FluxDistributionShift / None). Warps the linear schedule.
        effective_seq_len: Sequence length for dist_shift. Scalar int or
            (batch_size,) tensor for per-element schedules.
        fallback_seq_len: Fallback when effective_seq_len is None (typically x.shape[-1]).
        include_endpoint: If True, schedule includes 0 as final value (RF samplers).
            If False, excludes 0 (v-diffusion DDIM).
        device: Device for the output tensor.
    """
    n_points = steps + 1 if include_endpoint else steps

    if include_endpoint:
        t = torch.linspace(sigma_max, 0, n_points, device=device)
    else:
        t = torch.linspace(sigma_max, 0, n_points + 1, device=device)[:-1]

    if dist_shift is not None:
        seq_len = effective_seq_len if effective_seq_len is not None else fallback_seq_len
        if isinstance(seq_len, torch.Tensor):
            # Clamp per-element sequence lengths to avoid zeros causing log/NaN issues
            seq_len = torch.clamp(seq_len, min=1)
        elif seq_len is not None:
            # Clamp scalar sequence length to at least 1
            seq_len = max(int(seq_len), 1)
        t = dist_shift.shift(t, seq_len)

        # Ensure the first timestep remains aligned with sigma_max after shifting.
        # This keeps the schedule consistent with the initialization in sample_diffusion(),
        # which mixes init_data using sigma_max.
        if isinstance(t, torch.Tensor):
            sigma_max_tensor = t.new_tensor(sigma_max)
            if t.ndim == 1:
                t[0] = sigma_max_tensor
            else:
                # For batched/per-element schedules, enforce sigma_max at the first time index.
                t[..., 0] = sigma_max_tensor

    return t


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
    """Discrete Euler sampler (v-prediction).

    PORT_FROM: stable-audio-3 inference/sampling.py:147-187 (verbatim).

    Args:
        sigmas: schedule tensor. Shape (steps+1,) or (batch_size, steps+1).
    """
    t = sigmas

    per_element_schedule = t.dim() == 2

    t = t.to(x.device)
    num_steps = t.shape[-1] - 1

    for i in tqdm(range(num_steps), disable=disable_tqdm):
        if per_element_schedule:
            t_curr_tensor = t[:, i].to(x.dtype)
            t_prev = t[:, i + 1].to(x.dtype)
            dt = t_prev - t_curr_tensor
            dt_broadcast = dt.view(-1, 1, 1)
        else:
            t_curr = t[i]
            t_prev = t[i + 1]
            t_curr_tensor = t_curr * torch.ones((x.shape[0],), dtype=x.dtype, device=x.device)
            dt = t_prev - t_curr
            dt_broadcast = dt

        v = model(x, t_curr_tensor, **extra_args)

        if callback is not None:
            denoised = x - t_curr_tensor[:, None, None] * v
            callback({"x": x, "t": t_curr_tensor, "sigma": t_curr_tensor, "i": i, "denoised": denoised})

        x = x + dt_broadcast * v

    return x


def sample_rk4(
    model: Callable,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    callback: Callable | None = None,
    disable_tqdm: bool = False,
    **extra_args,
) -> torch.Tensor:
    """4th-order Runge-Kutta sampler (v-prediction).

    PORT_FROM: stable-audio-3 inference/sampling.py:189-225 (verbatim).

    Note: per-element schedules NOT supported for RK4. `sigmas` must be 1-D.
    """
    # Broadcast helper: ones(B,) so we can multiply a scalar t to a per-batch tensor
    ts = x.new_ones([x.shape[0]])

    t = sigmas.to(x.device)

    for i, (t_curr, t_prev) in enumerate(tqdm(zip(t[:-1], t[1:]), disable=disable_tqdm)):
        t_curr_tensor = t_curr * ts
        dt = t_prev - t_curr  # solving backwards in time

        k1 = model(x, t_curr_tensor, **extra_args)

        if callback is not None:
            denoised = x - t_curr * k1
            callback({"x": x, "t": t_curr, "sigma": t_curr, "i": i, "denoised": denoised})

        k2 = model(x + dt / 2 * k1, (t_curr + dt / 2) * ts, **extra_args)
        k3 = model(x + dt / 2 * k2, (t_curr + dt / 2) * ts, **extra_args)

        # Clamp t_prev to avoid evaluating model at exactly t=0
        # (models aren't trained at t=0 and may return NaN)
        t_prev_eval = t_prev.clamp(min=1e-5)
        k4 = model(x + dt * k3, t_prev_eval * ts, **extra_args)

        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return x


def sample_flow_dpmpp(
    model: Callable,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    callback: Callable | None = None,
    disable_tqdm: bool = False,
    **extra_args,
) -> torch.Tensor:
    """DPM-Solver++ 2nd-order multistep for rectified-flow / v-prediction models.

    Default SA3 sampler (registered as "dpmpp-3m-sde" in SAMPLER_REGISTRY).
    Ported verbatim from Stability-AI/stable-audio-3 inference/sampling.py:227-306.

    Args:
        sigmas: schedule tensor. Shape (steps+1,) for a global schedule, or
            (batch_size, steps+1) for per-element schedules.
    """
    t = sigmas

    # Check if we have per-element schedules (batch_size, steps+1) or global schedule (steps+1,)
    per_element_schedule = t.dim() == 2

    t = t.to(x.device)
    num_steps = t.shape[-1] - 1

    old_denoised = None

    # Clamp t to avoid numerical issues with log(0) and division by zero
    def log_snr(t):
        return ((1 - t).clamp(min=1e-10) / t.clamp(min=1e-10)).log()

    for i in trange(num_steps, disable=disable_tqdm):
        if per_element_schedule:
            # Per-element schedules: t has shape (batch_size, steps+1)
            t_curr = t[:, i]
            t_next = t[:, i + 1]
            t_prev = t[:, i - 1] if i > 0 else None
            t_curr_broadcast = t_curr.view(-1, 1, 1)
            t_next_broadcast = t_next.view(-1, 1, 1)
            t_curr_tensor = t_curr
        else:
            t_curr = t[i]
            t_next = t[i + 1]
            t_prev = t[i - 1] if i > 0 else None
            t_curr_broadcast = t_curr
            t_next_broadcast = t_next
            t_curr_tensor = t_curr.expand(x.shape[0])

        model_output = model(x, t_curr_tensor, **extra_args)
        denoised = x - t_curr_broadcast * model_output

        if callback is not None:
            callback({"x": x, "i": i, "t": t_curr, "sigma": t_curr, "denoised": denoised})

        alpha_t = 1 - t_next_broadcast

        # For rectified flow, compute the DPM++ coefficient directly without log_snr
        # to avoid numerical issues at t=0 or t=1.
        dt = t_next_broadcast - t_curr_broadcast
        dpmpp_coeff = dt / ((1 - t_next_broadcast).clamp(min=1e-10) * t_curr_broadcast.clamp(min=1e-10))

        is_first_step = old_denoised is None
        is_last_step = (t_next_broadcast == 0).all() if per_element_schedule else (t_next == 0)

        if is_first_step or is_last_step:
            # First-order update (no history available, or final step)
            x = (t_next_broadcast / t_curr_broadcast.clamp(min=1e-10)) * x - alpha_t * dpmpp_coeff * denoised
        else:
            # Second-order update with Richardson extrapolation
            if per_element_schedule:
                t_prev_broadcast = t_prev.view(-1, 1, 1)
            else:
                t_prev_broadcast = t_prev
            # r = h_last / h in log-SNR space
            h = log_snr(t_next_broadcast) - log_snr(t_curr_broadcast)
            h_last = log_snr(t_curr_broadcast) - log_snr(t_prev_broadcast)
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (t_next_broadcast / t_curr_broadcast.clamp(min=1e-10)) * x - alpha_t * dpmpp_coeff * denoised_d

        old_denoised = denoised

    return x


def sample_flow_pingpong(
    model: Callable,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    callback: Callable | None = None,
    disable_tqdm: bool = False,
    **extra_args,
) -> torch.Tensor:
    """Ping-pong sampler for distilled rectified-flow models.

    PORT_FROM: stable-audio-3 inference/sampling.py:308-351 (verbatim).
    Alternates between deterministic v-prediction step and stochastic noise
    injection — used for distilled SA3 checkpoints (suffix `-base`).

    Args:
        sigmas: schedule tensor. Shape (steps+1,) or (batch_size, steps+1).
    """
    t = sigmas

    per_element_schedule = t.dim() == 2

    t = t.to(x.device)
    num_steps = t.shape[-1] - 1

    for i in trange(num_steps, disable=disable_tqdm):
        if per_element_schedule:
            t_curr = t[:, i].to(x.dtype)
            t_next = t[:, i + 1].to(x.dtype)
            t_curr_broadcast = t_curr.view(-1, 1, 1)
            t_next_broadcast = t_next.view(-1, 1, 1)
        else:
            t_curr = t[i].to(x.dtype)
            t_next = t[i + 1].to(x.dtype)
            t_curr_broadcast = t_curr
            t_next_broadcast = t_next

        if per_element_schedule:
            t_curr_tensor = t_curr
        else:
            t_curr_tensor = t_curr * torch.ones((x.shape[0],), dtype=x.dtype, device=x.device)

        denoised = x - t_curr_broadcast * model(x, t_curr_tensor, **extra_args)

        if callback is not None:
            callback({"x": x, "i": i, "t": t_curr, "sigma": t_curr, "sigma_hat": t_curr, "denoised": denoised})

        # Mix denoised with fresh noise at level t_next
        x = (1 - t_next_broadcast) * denoised + t_next_broadcast * torch.randn_like(x)

    return x


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
