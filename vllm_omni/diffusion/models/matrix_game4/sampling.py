# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Matrix-Game's shift-5 flow-matching inference grid."""

from dataclasses import dataclass

import torch


@dataclass
class Chunk:
    index: int
    frame_start: int
    frame_end: int
    pixels: torch.Tensor
    latent: torch.Tensor
    control: torch.Tensor
    denoising_steps: int
    timings_ms: dict[str, float]


class FlowMatchScheduler:
    def __init__(self, shift: float = 5.0, sigma_min: float = 0.0, extra_one_step: bool = True):
        if sigma_min != 0.0 or not extra_one_step:
            raise ValueError("Matrix-Game requires the released shift-5 inference grid")
        self.shift = shift

    def set_timesteps(self, num_inference_steps: int = 1000, training: bool = False) -> None:
        # Preserve the reference's FP32 linspace and operation order.
        self.sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1, device="cpu", dtype=torch.float32)[:-1]
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        self.timesteps = self.sigmas * 1000

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        self.sigmas = self.sigmas.to(noise.device)
        self.timesteps = self.timesteps.to(noise.device)
        timestep_id = torch.argmin((self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        return ((1 - sigma) * original_samples + sigma * noise).type_as(noise)


def warp_steps(scheduler: FlowMatchScheduler, steps: list[int]) -> torch.Tensor:
    indices = torch.as_tensor(steps, dtype=torch.long, device="cpu")
    timesteps = torch.cat((scheduler.timesteps.cpu().float(), torch.tensor([0.0], device="cpu", dtype=torch.float32)))
    return timesteps[1000 - indices]
