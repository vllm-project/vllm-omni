# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch


@dataclass
class WanEulerSchedulerOutput:
    prev_sample: torch.FloatTensor


def _unsqueeze_to_ndim(in_tensor: torch.Tensor, target_ndim: int) -> torch.Tensor:
    if in_tensor.ndim >= target_ndim:
        return in_tensor
    return in_tensor[(...,) + (None,) * (target_ndim - in_tensor.ndim)]


def _get_timesteps(num_steps: int, max_steps: int = 1000) -> np.ndarray:
    # Keep num_steps + 1 points so Euler update can always access sigma_next.
    return np.linspace(max_steps, 0, num_steps + 1, dtype=np.float32)


def _timestep_shift(timesteps: torch.Tensor, shift: float = 1.0) -> torch.Tensor:
    return shift * timesteps / (1 + (shift - 1) * timesteps)


class WanEulerScheduler:
    order = 1

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        device: torch.device | str = "cpu",
    ) -> None:
        self.num_train_timesteps = int(num_train_timesteps)
        self._shift = float(shift)
        self.device = device
        self.config = SimpleNamespace(num_train_timesteps=self.num_train_timesteps)
        self.init_noise_sigma = 1.0

        self._step_index: int | None = None
        self._begin_index: int | None = None

        self.timesteps = torch.empty(0, dtype=torch.float32)
        self.sigmas = torch.empty(0, dtype=torch.float32)
        self.timesteps_ori = torch.empty(0, dtype=torch.float32)

        self.set_timesteps(num_inference_steps=self.num_train_timesteps, device=self.device)

    @property
    def step_index(self) -> int | None:
        return self._step_index

    @property
    def begin_index(self) -> int | None:
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._begin_index = int(begin_index)

    def index_for_timestep(self, timestep: torch.Tensor) -> int:
        indices = (self.timesteps == timestep).nonzero()
        if len(indices) > 0:
            pos = 1 if len(indices) > 1 else 0
            return int(indices[pos].item())
        # Fallback for tiny float drift
        return int(torch.argmin(torch.abs(self.timesteps - timestep)).item())

    def _init_step_index(self, timestep: float | torch.Tensor) -> None:
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep_t = timestep.to(self.timesteps.device, dtype=self.timesteps.dtype)
            else:
                timestep_t = torch.tensor(timestep, device=self.timesteps.device, dtype=self.timesteps.dtype)
            self._step_index = self.index_for_timestep(timestep_t)
        else:
            self._step_index = self._begin_index

    def set_shift(self, shift: float = 1.0) -> None:
        # Compute shifted sigma schedule on [0, 1].
        sigmas_full = self.timesteps_ori / float(self.num_train_timesteps)
        sigmas_full = _timestep_shift(sigmas_full, shift=float(shift))
        self.sigmas = sigmas_full
        # Public timesteps are the first N points; next point is consumed as sigma_next.
        self.timesteps = self.sigmas[:-1] * self.num_train_timesteps
        self._shift = float(shift)

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: torch.device | str | int | None = None,
        **kwargs,  # noqa: ARG002 - kept for scheduler API compatibility
    ) -> None:
        timesteps = _get_timesteps(
            num_steps=int(num_inference_steps),
            max_steps=self.num_train_timesteps,
        )
        self.timesteps_ori = torch.from_numpy(timesteps).to(
            dtype=torch.float32,
            device=device or self.device,
        )
        self.set_shift(self._shift)
        self._step_index = None
        self._begin_index = None

    def scale_model_input(self, sample: torch.Tensor, timestep: int | None = None) -> torch.Tensor:  # noqa: ARG002
        return sample

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: float | torch.FloatTensor,
        sample: torch.FloatTensor,
        return_dict: bool = True,
        **kwargs,  # noqa: ARG002 - kept for scheduler API compatibility
    ) -> WanEulerSchedulerOutput | tuple[torch.FloatTensor]:
        if isinstance(timestep, (int, torch.IntTensor, torch.LongTensor)):
            raise ValueError(
                "Passing integer indices as timesteps is not supported. Use one value from scheduler.timesteps instead."
            )

        if self.step_index is None:
            self._init_step_index(timestep)
        assert self._step_index is not None

        sample_fp32 = sample.to(torch.float32)
        sigma = _unsqueeze_to_ndim(self.sigmas[self._step_index], sample_fp32.ndim).to(sample_fp32.device)
        sigma_next = _unsqueeze_to_ndim(self.sigmas[self._step_index + 1], sample_fp32.ndim).to(sample_fp32.device)

        prev_sample = sample_fp32 + (sigma_next - sigma) * model_output
        prev_sample = prev_sample.to(model_output.dtype)

        self._step_index += 1

        if not return_dict:
            return (prev_sample,)
        return WanEulerSchedulerOutput(prev_sample=prev_sample)

    def __len__(self) -> int:
        return self.num_train_timesteps
