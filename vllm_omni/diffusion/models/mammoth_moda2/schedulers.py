# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. on 2025-09-30.
#
# Original file was released under Apache-2.0, with the full license text
# available at https://www.apache.org/licenses/LICENSE-2.0
#
# This modified file is released under the same license.
#
# --- Upstream header preserved below ---
#
# Copyright 2024 Stability AI, Katherine Crowson and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class FlowMatchEulerDiscreteSchedulerOutput:
    prev_sample: torch.FloatTensor


class FlowMatchEulerDiscreteScheduler:
    order = 1

    def __init__(self, num_train_timesteps: int = 1000, dynamic_time_shift: bool = True):
        self.num_train_timesteps = int(num_train_timesteps)
        self.dynamic_time_shift = bool(dynamic_time_shift)

        timesteps = torch.linspace(0, 1, self.num_train_timesteps + 1, dtype=torch.float32)[:-1]
        self.timesteps = timesteps
        self._timesteps = torch.cat([timesteps, torch.ones(1, dtype=timesteps.dtype)])

        self._step_index: int | None = None
        self._begin_index: int | None = None

    @property
    def step_index(self) -> int | None:
        return self._step_index

    @property
    def begin_index(self) -> int | None:
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._begin_index = int(begin_index)

    def index_for_timestep(self, timestep: torch.Tensor, schedule_timesteps: torch.Tensor | None = None) -> int:
        schedule_timesteps = self._timesteps if schedule_timesteps is None else schedule_timesteps
        indices = (schedule_timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return int(indices[pos].item())

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device | None = None,
        timesteps: list[float] | None = None,
        num_tokens: int | None = None,
    ) -> None:
        if timesteps is None:
            if num_inference_steps is None:
                raise ValueError("`num_inference_steps` must be provided when `timesteps` is None.")
            timesteps_np = np.linspace(0, 1, num_inference_steps + 1, dtype=np.float32)[:-1]
            if self.dynamic_time_shift and num_tokens is not None:
                m = np.sqrt(float(num_tokens)) / 40.0
                timesteps_np = timesteps_np / (m - m * timesteps_np + timesteps_np)
        else:
            timesteps_np = np.asarray(timesteps, dtype=np.float32)

        timesteps_t = torch.from_numpy(timesteps_np).to(dtype=torch.float32, device=device)
        self.timesteps = timesteps_t
        self._timesteps = torch.cat([timesteps_t, torch.ones(1, device=timesteps_t.device, dtype=timesteps_t.dtype)])

        self._step_index = None
        self._begin_index = None

    def _init_step_index(self, timestep: float | torch.Tensor) -> None:
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            else:
                timestep = torch.tensor(timestep, device=self.timesteps.device, dtype=self.timesteps.dtype)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: float | torch.FloatTensor,
        sample: torch.FloatTensor,
        generator: torch.Generator | None = None,  # noqa: ARG002 - kept for API compatibility
        return_dict: bool = True,
    ) -> FlowMatchEulerDiscreteSchedulerOutput | tuple[torch.FloatTensor]:
        if isinstance(timestep, (int, torch.IntTensor, torch.LongTensor)):
            raise ValueError(
                "Integer timesteps (e.g. a loop index from enumerate) are not supported; "
                "pass a value from `scheduler.timesteps` instead.",
            )

        if self.step_index is None:
            self._init_step_index(timestep)
        assert self._step_index is not None

        sample_fp32 = sample.to(torch.float32)
        t = self._timesteps[self._step_index]
        t_next = self._timesteps[self._step_index + 1]

        prev_sample = sample_fp32 + (t_next - t) * model_output
        prev_sample = prev_sample.to(model_output.dtype)

        self._step_index += 1

        if not return_dict:
            return (prev_sample,)
        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)

    def __len__(self) -> int:
        return self.num_train_timesteps
