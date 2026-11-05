# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from Tencent-Hunyuan/Hy-Nomas and Diffusers (Apache-2.0).
# Copyright 2024 Stability AI, Katherine Crowson and the HuggingFace Team.

from types import SimpleNamespace

import torch


class EulerSolver:
    def __init__(self, *, shift=5.0, reverse=True, solver="euler"):
        if shift != 5.0 or not reverse or solver != "euler":
            raise ValueError("Hy-Nomas requires the released reverse Euler schedule with shift=5.")
        self._step_index = None

    def set_timesteps(self, steps, device=None):
        if type(steps) is not int or steps <= 0:
            raise ValueError("Hy-Nomas inference steps must be a positive integer.")
        sigmas = torch.linspace(1, 0, steps + 1)
        self.sigmas = (5.0 * sigmas) / (1 + 4.0 * sigmas)
        self.timesteps = (self.sigmas[:-1] * 1000).to(dtype=torch.float32, device=device)
        self._step_index = None

    def step(self, model_output, timestep, sample, *, return_dict=True):
        if not isinstance(timestep, torch.Tensor) or not timestep.is_floating_point():
            raise ValueError("Hy-Nomas Euler step requires a floating-point schedule timestep.")
        if self._step_index is None:
            matches = (self.timesteps == timestep.to(self.timesteps.device)).nonzero().flatten()
            if not matches.numel():
                raise ValueError("Hy-Nomas timestep is absent from the Euler schedule.")
            self._step_index = int(matches[1 if len(matches) > 1 else 0])
        dt = self.sigmas[self._step_index + 1] - self.sigmas[self._step_index]
        result = sample.float() + model_output.float() * dt
        self._step_index += 1
        return SimpleNamespace(prev_sample=result) if return_dict else (result,)
