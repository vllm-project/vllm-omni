# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Native K6 shifted flow-matching Euler scheduler.

Self-contained (no ``diffusers`` dependency), matching the duck-typed
scheduler interface (``set_timesteps`` / ``timesteps`` / ``sigmas`` /
``step``) that other native vLLM-Omni diffusion pipelines use for their own
per-model schedulers — see ``WanEulerScheduler`` in
``vllm_omni.diffusion.models.wan2_2.scheduling_wan_euler`` for the sibling
pattern this mirrors.

The math reproduces ``kandinsky.core.algo.flow_matching.flow_match_timesteps``
exactly: for ``t`` linearly spaced from 1.0 down to 0.0,
``sigma = scale * t / (1 + (scale - 1) * t)``. This is the same shifted
rectified-flow schedule family Wan's own scheduler uses, parameterized by
``scheduler_scale`` (K6's name for what Wan calls ``shift``).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch


@dataclass
class KandinskyFlowMatchSchedulerOutput:
    prev_sample: torch.Tensor


class KandinskyFlowMatchScheduler:
    """Stateful Euler stepper over K6's shifted flow-matching sigma schedule.

    ``timesteps`` holds the model-scale (``[0, 1000]``-ish) values fed to the
    DiT's ``time`` input; ``sigmas`` holds the same schedule unscaled
    (``num_inference_steps + 1`` entries, running 1.0 -> 0.0), needed to
    derive a step size without consuming a stateful ``step()`` call when one
    modality is stepped implicitly alongside another (see the ``vllm`` port's
    ``denoise_loop`` override).
    """

    order = 1

    def __init__(self, scheduler_scale: float = 1.0, device: torch.device | str = "cpu") -> None:
        self.scheduler_scale = float(scheduler_scale)
        self.device = device
        self.config = SimpleNamespace(scheduler_scale=self.scheduler_scale)

        self._step_index: int | None = None
        self.timesteps = torch.empty(0, dtype=torch.float32)
        self.sigmas = torch.empty(0, dtype=torch.float32)

        self.set_timesteps(num_inference_steps=1, device=self.device)

    @property
    def step_index(self) -> int | None:
        return self._step_index

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: torch.device | str | None = None,
        **kwargs,  # noqa: ARG002 - kept for scheduler API compatibility
    ) -> None:
        device = device or self.device
        t = torch.linspace(1.0, 0.0, int(num_inference_steps) + 1, device=device, dtype=torch.float32)
        scale = self.scheduler_scale
        self.sigmas = scale * t / (1 + (scale - 1) * t)
        # DiT `time` input is model-scale, i.e. sigma * 1000 (matches the core
        # algo loop's `t.unsqueeze(0).expand(bs) * 1000`).
        self.timesteps = self.sigmas[:-1] * 1000.0
        self._step_index = None

    def index_for_timestep(self, timestep: torch.Tensor) -> int:
        timestep_t = torch.as_tensor(timestep, device=self.timesteps.device, dtype=self.timesteps.dtype)
        return int(torch.argmin(torch.abs(self.timesteps - timestep_t)).item())

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        return_dict: bool = True,
        **kwargs,  # noqa: ARG002 - kept for scheduler API compatibility
    ) -> KandinskyFlowMatchSchedulerOutput | tuple[torch.Tensor]:
        if self._step_index is None:
            self._step_index = self.index_for_timestep(timestep)

        sigma = self.sigmas[self._step_index]
        sigma_next = self.sigmas[self._step_index + 1]
        dt = (sigma_next - sigma).to(device=sample.device, dtype=sample.dtype)
        prev_sample = sample + dt * model_output

        self._step_index += 1

        if not return_dict:
            return (prev_sample,)
        return KandinskyFlowMatchSchedulerOutput(prev_sample=prev_sample)

    def __len__(self) -> int:
        return int(self.timesteps.shape[0])
