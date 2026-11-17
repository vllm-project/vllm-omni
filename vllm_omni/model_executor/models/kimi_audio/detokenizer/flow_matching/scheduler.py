# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from MoonshotAI/Kimi-Audio (MIT), revision
# 349251e1d8f4f98d58fda59246381faecd7392e0, kimia_infer/models/detokenizer/.
# See vllm_omni/model_executor/models/kimi_audio/NOTICE for the upstream license.

import torch


class StreamingFlowMatchingScheduler:
    def __init__(
        self,
        timesteps=1000,
        sigma_min=1e-4,
    ) -> None:
        super().__init__()

        self.sigma_min = sigma_min
        self.timesteps = timesteps
        self.t_min = 0
        self.t_max = 1 - self.sigma_min

    def set_timesteps(self, timesteps=15):
        self.timesteps = timesteps

    def sample(self, ode_wrapper, time_steps, xt, verbose=False, x0=None):
        h = (self.t_max - self.t_min) / self.timesteps
        h = h * torch.ones(xt.shape[0], dtype=xt.dtype, device=xt.device)

        if verbose:
            gt_v = x0 - xt

        for t in time_steps:
            predicted_v = ode_wrapper(t, xt)
            if verbose:
                dist = torch.mean(torch.nn.functional.l1_loss(gt_v, predicted_v))
                print(f"Time: {t}, Distance: {dist}")
            xt = xt + h * predicted_v
        return xt

    def sample_by_neuralode(self, ode_wrapper, time_steps, xt, verbose=False, x0=None):
        # Fixed-step Euler over time_steps, reproducing torchdyn's
        # NeuralODE(solver="euler") step by step: time advances as t + dt and
        # the next dt is measured from that accumulated t. The wrapper
        # quantizes t to (t * 1000).long(), so the time grid must match exactly.
        time_steps = time_steps.to(xt.device)
        t, dt = time_steps[0], time_steps[1] - time_steps[0]
        for step in range(1, len(time_steps)):
            xt = xt + dt * ode_wrapper(t, xt)
            t = t + dt
            if step < len(time_steps) - 1:
                dt = time_steps[step + 1] - t
        return xt
