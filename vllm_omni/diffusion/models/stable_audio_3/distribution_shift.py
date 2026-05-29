# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Timestep distribution shifts for Stable Audio 3.

PORT_FROM: stable_audio_3/inference/distribution_shift.py (verbatim) +
           stable_audio_3/models/diffusion.py _create_dist_shift factory.

A distribution shift remaps the uniform t-schedule onto a sequence-length-aware
schedule. SA3 uses the inference-time ``sampling_dist_shift`` (default
``LogSNRShift``) inside ``sampling.build_schedule`` via ``dist_shift.shift(t, seq_len)``.
"""

from __future__ import annotations

import math

import torch


class IdentityDistributionShift:
    """No-op distribution shift — returns timesteps unchanged."""

    def shift(self, t: torch.Tensor, seq_len: int | torch.Tensor) -> torch.Tensor:
        return t


class FluxDistributionShift:
    """Flux/SD3/Self-Flow timestep shift: t_shifted = alpha * t / (1 + (alpha-1) * t).

    Convention: t=0 is data, t=1 is noise. alpha > 1 shifts timesteps toward noise.
    Constant alpha (alpha_min == alpha_max) or seq_len-dependent (log-linear in seq_len).
    """

    def __init__(
        self, min_length: int = 256, max_length: int = 4096, alpha_min: float = 1.0, alpha_max: float = 1.0
    ) -> None:
        self.min_length = min_length
        self.max_length = max_length
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.log_alpha_min = math.log(max(alpha_min, 1e-8))
        self.log_alpha_max = math.log(max(alpha_max, 1e-8))
        self.log_min_seq = math.log(min_length)
        self.log_max_seq = math.log(max_length)
        if self.log_max_seq == self.log_min_seq:
            self.log_max_seq += 1e-8

    def get_alpha(self, seq_len: int | torch.Tensor):
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
    """Sequence-length-aware sigmoid shift (SA3 'full' training-time shift)."""

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
    """Adaptive log-SNR distribution shift (SA3 default inference-time shift).

    Maps t in [0,1] to log-SNR-spaced values, preserving endpoints (0->0, 1->1).
    logsnr_start scales with sequence length: drops by ``rate`` per doubling of
    seq_len; logsnr_end is fixed (low-t refinement is local).
    """

    def __init__(
        self, anchor_length: int = 2000, anchor_logsnr: float = -6.2, rate: float = 1.0, logsnr_end: float = 2.0
    ) -> None:
        self.anchor_length = anchor_length
        self.anchor_logsnr = anchor_logsnr
        self.rate = rate
        self.logsnr_end = logsnr_end

    def get_logsnr_start(self, seq_len: int | torch.Tensor):
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
        logsnr = self.logsnr_end - t * (self.logsnr_end - logsnr_start)
        t_out = torch.sigmoid(-logsnr)
        t_out = torch.where(t_original <= 0, torch.zeros_like(t_out), t_out)
        t_out = torch.where(t_original >= 1, torch.ones_like(t_out), t_out)
        return t_out


def create_dist_shift(options: dict):
    """Build a distribution-shift object from config options.

    PORT_FROM: diffusion.py ConditionedDiffusionModelWrapper._create_dist_shift.
    """
    dist_shift_type = options.get("type", "full")
    kwargs = {k: v for k, v in options.items() if k != "type"}
    if dist_shift_type == "none":
        return IdentityDistributionShift()
    if dist_shift_type == "flux":
        return FluxDistributionShift(**kwargs)
    if dist_shift_type == "full":
        return DistributionShift(**kwargs)
    if dist_shift_type == "logsnr":
        return LogSNRShift(**kwargs)
    raise ValueError(
        f"Unknown distribution shift type: {dist_shift_type}. Expected 'none', 'flux', 'full', or 'logsnr'."
    )
