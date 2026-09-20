# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from __future__ import annotations

import math
from typing import Any

import torch

MINIMAX_H3_SAMPLE_SOLVERS = frozenset({"euler", "res_multistep"})


def _require_finite_tensor(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must be finite")


def _validate_unit_timestep(timestep: torch.Tensor, name: str) -> None:
    if not isinstance(timestep, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if not torch.is_floating_point(timestep):
        raise ValueError(f"{name} must be a floating point tensor")
    _require_finite_tensor(timestep, name)
    out_of_range = (timestep < 0) | (timestep > 1)
    if bool(out_of_range.any().item()):
        raise ValueError(f"{name} must be in [0, 1]")


def _validate_sigma(value: float, name: str) -> float:
    sigma = float(value)
    if not math.isfinite(sigma):
        raise ValueError(f"{name} must be finite")
    if sigma < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return sigma


def _validate_timestep_sigma_pair(
    timestep: torch.Tensor,
    sigma_curr: float,
    name: str,
) -> float:
    _validate_unit_timestep(timestep, f"{name}_timestep")
    sigma = _validate_sigma(sigma_curr, f"{name}_sigma_curr")
    expected = 1.0 - timestep.detach().to(dtype=torch.float32)
    actual = torch.full_like(expected, sigma)
    if not torch.allclose(actual, expected, rtol=1e-5, atol=1e-5):
        raise ValueError(f"{name}_sigma_curr must equal 1 - {name}_timestep")
    return sigma


def minimax_h3_rf_v_to_x0(
    xt: torch.Tensor,
    v: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    if xt.shape != v.shape:
        raise ValueError(f"xt and v shapes must match, got {xt.shape} vs {v.shape}")
    if not torch.is_floating_point(xt):
        raise ValueError("xt must be a floating point tensor")
    if not torch.is_floating_point(v):
        raise ValueError("v must be a floating point tensor")
    _require_finite_tensor(xt, "xt")
    _require_finite_tensor(v, "v")
    _validate_unit_timestep(timestep, "timestep")
    cond_t = timestep.to(device=xt.device, dtype=xt.dtype)
    while cond_t.ndim < xt.ndim:
        cond_t = cond_t.unsqueeze(-1)
    sigma_t = 1 - cond_t
    x0 = xt + sigma_t * v
    _require_finite_tensor(x0, "x0")
    return x0


def minimax_h3_euler_eta0_step(
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


def minimax_h3_normalize_sample_solver(value: object | None) -> str:
    """Return the canonical H3 solver name, preserving Euler as the default."""
    solver = "euler" if value is None else str(value).strip().lower()
    if solver not in MINIMAX_H3_SAMPLE_SOLVERS:
        supported = ", ".join(sorted(MINIMAX_H3_SAMPLE_SOLVERS))
        raise ValueError(f"unsupported MiniMax H3 sample_solver={value!r}; expected one of: {supported}")
    return solver


def minimax_h3_res_multistep_eta0_step(
    state: torch.Tensor,
    denoised: torch.Tensor,
    old_denoised: torch.Tensor | None,
    *,
    sigma_prev: float | None,
    sigma_curr: float,
    sigma_next: float,
) -> torch.Tensor:
    """Apply ComfyUI's deterministic RES two-step exponential update.

    RES multistep needs the previous model estimate, so its first step is
    Euler. ComfyUI also uses Euler for the final jump to zero. ``eta=0`` keeps
    the update deterministic and makes it suitable for H3's existing sampler.
    """
    sigma_curr = _validate_sigma(sigma_curr, "sigma_curr")
    sigma_next = _validate_sigma(sigma_next, "sigma_next")
    if old_denoised is None or sigma_prev is None or sigma_next == 0.0:
        return minimax_h3_euler_eta0_step(
            state,
            denoised,
            sigma_curr=sigma_curr,
            sigma_next=sigma_next,
        )
    if state.shape != denoised.shape or state.shape != old_denoised.shape:
        raise ValueError(
            "state, denoised, and old_denoised shapes must match, got "
            f"{state.shape}, {denoised.shape}, and {old_denoised.shape}"
        )
    if not all(torch.is_floating_point(tensor) for tensor in (state, denoised, old_denoised)):
        raise ValueError("state, denoised, and old_denoised must be floating point tensors")
    _require_finite_tensor(state, "state")
    _require_finite_tensor(denoised, "denoised")
    _require_finite_tensor(old_denoised, "old_denoised")
    sigma_prev = _validate_sigma(sigma_prev, "sigma_prev")
    if sigma_prev == 0.0 or sigma_curr == 0.0:
        raise ValueError("RES multistep requires positive sigma_prev and sigma_curr")

    # t(sigma) = -log(sigma), matching ComfyUI's sample_res_multistep.
    h = math.log(sigma_curr / sigma_next)
    c2 = math.log(sigma_curr / sigma_prev) / h
    if not math.isfinite(h) or h <= 0.0 or not math.isfinite(c2) or c2 == 0.0:
        raise ValueError(
            "RES multistep requires a strictly descending, non-repeating sigma schedule; "
            f"got sigma_prev={sigma_prev}, sigma_curr={sigma_curr}, sigma_next={sigma_next}"
        )
    phi1 = math.expm1(-h) / -h
    phi2 = (phi1 - 1.0) / -h
    b1 = phi1 - phi2 / c2
    b2 = phi2 / c2

    compute_dtype = torch.float32 if state.dtype in (torch.float16, torch.bfloat16) else state.dtype
    out = math.exp(-h) * state.to(dtype=compute_dtype) + h * (
        b1 * denoised.to(dtype=compute_dtype) + b2 * old_denoised.to(dtype=compute_dtype)
    )
    out = out.to(dtype=state.dtype)
    _require_finite_tensor(out, "res_multistep_eta0_step output")
    return out


def minimax_h3_sample_step(
    state: torch.Tensor,
    denoised: torch.Tensor,
    old_denoised: torch.Tensor | None,
    *,
    sample_solver: str,
    sigma_prev: float | None,
    sigma_curr: float,
    sigma_next: float,
) -> torch.Tensor:
    """Advance one H3 sample with the selected deterministic solver."""
    solver = minimax_h3_normalize_sample_solver(sample_solver)
    if solver == "euler":
        return minimax_h3_euler_eta0_step(
            state,
            denoised,
            sigma_curr=sigma_curr,
            sigma_next=sigma_next,
        )
    return minimax_h3_res_multistep_eta0_step(
        state,
        denoised,
        old_denoised,
        sigma_prev=sigma_prev,
        sigma_curr=sigma_curr,
        sigma_next=sigma_next,
    )


class MiniMaxH3EulerAncestralEta0SchedulerAdapter:
    def __init__(self, **config: Any) -> None:
        if config:
            raise ValueError(f"{type(self).__name__} does not accept config fields: {sorted(config)}")

    def set_shift(self, _flow_shift: float) -> None:
        """Ignore flow shift, matching the previous loader-specific path."""

    def step_denoising(
        self,
        *,
        input_visual_latent: torch.Tensor,
        input_audio_latent: torch.Tensor,
        timestep: torch.Tensor,
        noise_pred_visual: torch.Tensor,
        noise_pred_audio: torch.Tensor,
        sigma_curr: float,
        sigma_next: float,
        video_timestep: torch.Tensor | None = None,
        audio_timestep: torch.Tensor | None = None,
        video_sigma_curr: float | None = None,
        video_sigma_next: float | None = None,
        audio_sigma_curr: float | None = None,
        audio_sigma_next: float | None = None,
    ) -> dict[str, torch.Tensor]:
        visual_timestep = timestep if video_timestep is None else video_timestep
        audio_timestep = timestep if audio_timestep is None else audio_timestep
        visual_sigma_curr = sigma_curr if video_sigma_curr is None else video_sigma_curr
        visual_sigma_next = sigma_next if video_sigma_next is None else video_sigma_next
        audio_sigma_curr = sigma_curr if audio_sigma_curr is None else audio_sigma_curr
        audio_sigma_next = sigma_next if audio_sigma_next is None else audio_sigma_next
        visual_sigma_curr = _validate_timestep_sigma_pair(
            visual_timestep,
            visual_sigma_curr,
            "video",
        )
        audio_sigma_curr = _validate_timestep_sigma_pair(
            audio_timestep,
            audio_sigma_curr,
            "audio",
        )

        denoised_visual = minimax_h3_rf_v_to_x0(
            input_visual_latent,
            noise_pred_visual,
            visual_timestep,
        )
        denoised_audio = minimax_h3_rf_v_to_x0(
            input_audio_latent,
            noise_pred_audio,
            audio_timestep,
        )
        return {
            "output_visual_latent": minimax_h3_euler_eta0_step(
                input_visual_latent,
                denoised_visual,
                sigma_curr=visual_sigma_curr,
                sigma_next=visual_sigma_next,
            ),
            "output_audio_latent": minimax_h3_euler_eta0_step(
                input_audio_latent,
                denoised_audio,
                sigma_curr=audio_sigma_curr,
                sigma_next=audio_sigma_next,
            ),
        }


EntryClass = MiniMaxH3EulerAncestralEta0SchedulerAdapter

__all__ = [
    "MINIMAX_H3_SAMPLE_SOLVERS",
    "MiniMaxH3EulerAncestralEta0SchedulerAdapter",
    "minimax_h3_euler_eta0_step",
    "minimax_h3_normalize_sample_solver",
    "minimax_h3_rf_v_to_x0",
    "minimax_h3_res_multistep_eta0_step",
    "minimax_h3_sample_step",
]
