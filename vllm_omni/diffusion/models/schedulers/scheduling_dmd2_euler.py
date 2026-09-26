# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch
from diffusers import FlowMatchEulerDiscreteScheduler


class DMD2EulerScheduler(FlowMatchEulerDiscreteScheduler):
    """Euler scheduler that always uses the fixed DMD2 training timestep schedule."""

    def __init__(
        self,
        *args,
        dmd2_timesteps: list[int],
        stochastic_sampling: bool = False,
        **kwargs,
    ):
        super().__init__(*args, stochastic_sampling=stochastic_sampling, **kwargs)
        self._dmd2_timesteps = dmd2_timesteps

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device | None = None,
        timesteps: list[int] | None = None,
        sigmas: list[float] | None = None,
        **kwargs,
    ) -> None:
        super().set_timesteps(timesteps=self._dmd2_timesteps, device=device)
