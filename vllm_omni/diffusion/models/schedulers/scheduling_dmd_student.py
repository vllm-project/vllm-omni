# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Few-step sampler for DMD-distilled LingBot-Video student checkpoints (#11260).

A DMD-distilled student generates in a fixed small number of steps and must be
sampled with the exact step geometry it was trained on: a DDIM eta-ancestral
transition at high noise, the deterministic Euler ODE below a warped-sigma
threshold. Sampling with any other geometry (e.g. UniPC multistep) queries the
student off the distribution it was distilled on.

Trained sampling geometry (kept verbatim from the reference, do not "improve"
numerically):

* sigma grid — diffusers ``UniPCMultistepScheduler`` with ``use_flow_sigmas=True``,
  ``flow_shift`` warp, terminal sigma 0, and the ``sigmas[0] -= 1e-6`` guard.
  For 8 steps / shift 3 this yields
  ``[0.99999899, 0.95459503, 0.90011996, 0.83355546, 0.75037479, 0.64346898,
  0.50099897, 0.30167764, 0.0]``.
* step kernel — while ``sigma >= high_noise_threshold``, a DDIM eta-ancestral
  transition applied to the x0/x1 split of the flow prediction; below the
  threshold, the plain Euler ODE step.
* gate — ``float(sigma) >= high_noise_threshold`` on the warped sigma. With
  the default grid this makes steps 1-7 stochastic and step 8 the
  deterministic landing.

Model output convention matches the rest of this repo (``flow_prediction``):
``v = eps - x0`` so ``x0 = z - sigma*v`` and ``x1 = z + (1-sigma)*v``.

Reference recipe (the student's training-time sampling recipe): 8 steps,
guidance_scale 1.0, shift 3.0, DDIM eta=1 above warped sigma 0.5, Euler below.
"""

from __future__ import annotations

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin, SchedulerOutput
from diffusers.utils.torch_utils import randn_tensor


def _ddim_ancestral_std(sigma: torch.Tensor, sigma_next: torch.Tensor, eta: float) -> torch.Tensor:
    """DDIM eta-ancestral per-step noise std in flow coordinates (x_t = (1-σ)x0 + σε).

    The standard DDIM variance computed in the VP-normalized frame of the linear
    flow schedule
    (α̂ = s/√(s²+n²), σ̂ = n/√(s²+n²) with s = 1-σ, n = σ), mapped back to flow
    coordinates. eta=0 → deterministic DDIM; eta=1 → full ancestral. Clamped to
    σ_next (never more fresh noise than the target level); σ_next=0 → 0.
    """
    s_t, n_t = 1.0 - sigma, sigma
    s_s, n_s = 1.0 - sigma_next, sigma_next
    denom_t = 1.0 / torch.sqrt(s_t**2 + n_t**2)
    denom_s = 1.0 / torch.sqrt(s_s**2 + n_s**2)
    ratio_sq = ((s_t * denom_t) / (s_s * denom_s + 1e-8)) ** 2
    inner = (1.0 - ratio_sq).clamp(min=0.0)
    std = eta * ((n_s * denom_s) / (n_t * denom_t + 1e-8)) * inner.sqrt() / (denom_s + 1e-8)
    return torch.minimum(std, n_s)


class DMDStudentScheduler(SchedulerMixin, ConfigMixin):
    """K-step DDIM(eta)->Euler sampler matching the DMD student's training geometry.

    Drop-in for ``FlowUniPCMultistepScheduler`` in the LingBot-Video pipeline:
    same ``set_timesteps(num_inference_steps, device, shift)`` entry, same
    internal step counter, same ``step(...)[0]`` contract.
    """

    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        flow_shift: float = 3.0,
        high_noise_threshold: float = 0.5,
        ddim_eta: float = 1.0,
        prediction_type: str = "flow_prediction",
    ) -> None:
        if prediction_type != "flow_prediction":
            raise ValueError(f"DMDStudentScheduler only supports flow_prediction, got {prediction_type!r}.")
        # Full-range endpoints for API parity with FlowUniPCMultistepScheduler
        # (the t2v pipeline reads these before set_timesteps for its refiner
        # sigma helper; the DMD path never consumes them).
        s_min = 1.0 / num_train_timesteps
        self.sigma_min = flow_shift * s_min / (1 + (flow_shift - 1) * s_min)
        self.sigma_max = 1.0 - 1e-6
        self.sigmas: torch.Tensor | None = None
        self.timesteps: torch.Tensor | None = None
        self.num_inference_steps: int | None = None
        self._step_index: int | None = None

    @property
    def step_index(self) -> int | None:
        return self._step_index

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device | None = None,
        sigmas: np.ndarray | None = None,
        shift: float | None = None,
        mu: float | None = None,
    ) -> None:
        if sigmas is not None:
            raise ValueError(
                "DMDStudentScheduler defines its own sigma grid (the student was "
                "distilled on it); custom sigmas / refiner schedules are not supported."
            )
        if mu is not None:
            raise ValueError("DMDStudentScheduler does not support dynamic shifting (mu).")
        if num_inference_steps is None:
            raise ValueError("num_inference_steps is required.")
        shift = self.config.flow_shift if shift is None else float(shift)

        # Verbatim diffusers UniPCMultistepScheduler set_timesteps, use_flow_sigmas
        # branch (linspace over train sigmas -> shift warp -> sigma0 -= 1e-6 guard
        # -> integer-truncated timesteps -> terminal sigma 0).
        grid = np.linspace(1, 1 / self.config.num_train_timesteps, num_inference_steps + 1)[:-1]
        grid = shift * grid / (1 + (shift - 1) * grid)
        eps = 1e-6
        if np.fabs(grid[0] - 1) < eps:
            grid[0] -= eps
        timesteps = (grid * self.config.num_train_timesteps).copy()
        grid = np.concatenate([grid, [0.0]]).astype(np.float32)

        self.num_inference_steps = num_inference_steps
        self.sigmas = torch.from_numpy(grid)
        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)
        self._step_index = None

    def scale_model_input(self, sample: torch.Tensor, timestep=None) -> torch.Tensor:
        return sample

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.Tensor,
        return_dict: bool = True,
        generator: torch.Generator | None = None,
        **kwargs,
    ) -> SchedulerOutput | tuple[torch.Tensor]:
        if self.sigmas is None:
            raise RuntimeError("Call set_timesteps before step.")
        if self._step_index is None:
            self._step_index = 0

        # fp32 step math, exactly like the reference inference scheduler.
        v = model_output.float()
        z = sample.float()
        sigma = self.sigmas[self._step_index].to(device=z.device)
        sigma_next = self.sigmas[self._step_index + 1].to(device=z.device)

        if float(sigma) >= self.config.high_noise_threshold:
            # DDIM eta-ancestral transition (the reference x0/x1 kernel with the
            # DDIM ancestral std): split z into x0/x1, land on the mixture whose
            # total noise at sigma_next is split between the kept x1 direction
            # and std of fresh noise.
            std = _ddim_ancestral_std(sigma, sigma_next, eta=float(self.config.ddim_eta))
            x0 = z - sigma * v
            x1 = z + v * (1 - sigma)
            prev_mean = x0 * (1 - sigma_next) + x1 * torch.sqrt(sigma_next**2 - std**2)
            noise = randn_tensor(v.shape, generator=generator, device=v.device, dtype=v.dtype)
            prev_sample = prev_mean + std * noise
        else:
            # Deterministic Euler ODE below the threshold (the reference Euler kernel).
            prev_sample = z + v * (sigma_next - sigma)

        self._step_index += 1
        if not return_dict:
            return (prev_sample,)
        return SchedulerOutput(prev_sample=prev_sample)
