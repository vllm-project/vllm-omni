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

        self.neural_ode = None

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
        try:
            from torchdyn.core import NeuralODE
        except ImportError as exc:
            raise ImportError("Kimi-Audio decoding requires vllm-omni[kimi-audio]") from exc

        if self.neural_ode is None:
            self.neural_ode = NeuralODE(
                ode_wrapper,
                solver="euler",
                sensitivity="adjoint",
                atol=self.sigma_min,
                rtol=self.sigma_min,
            )

        eval_points, traj = self.neural_ode(xt, time_steps)
        return traj[-1]
