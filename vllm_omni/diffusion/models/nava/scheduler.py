# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm_omni.diffusion.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)


class NAVAFlowMatchScheduler(FlowUniPCMultistepScheduler):
    """NAVA scheduler wrapper preserving the local pipeline API."""

    def __init__(self, *, num_train_timesteps: int = 1000, shift: float = 5.0) -> None:
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
            use_dynamic_shifting=False,
        )

    def set_timesteps(
        self,
        num_inference_steps: int,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        # Upstream NAVA keeps scheduler state on CPU and only feeds timestep
        # tensors to the model device during denoising.
        super().set_timesteps(num_inference_steps=num_inference_steps, device=None)
        if device is None:
            return self.timesteps
        return self.timesteps.to(device=device)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        return super().step(model_output, timestep, sample, return_dict=False)[0]
