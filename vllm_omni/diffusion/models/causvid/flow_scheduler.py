# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Per-chunk scheduler wrapper for CausVid stream-batch execution."""

from __future__ import annotations

from typing import Any

import torch


class CausVidFlowScheduler:
    def __init__(
        self,
        inner: Any,
        timesteps: torch.Tensor,
    ) -> None:
        # Reference to the (already-``set_timesteps``-d) inner scheduler.
        self._inner = inner
        self.timesteps = timesteps
        self.sigmas = inner.sigmas
        self._full_timesteps = inner.timesteps

    def step(
        self,
        noise_pred: torch.Tensor,
        t: torch.Tensor,
        latents: torch.Tensor,
        return_dict: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor]:
        x0 = self._convert_flow_pred_to_x0(noise_pred, latents, t)

        ts_eq = (self.timesteps == t).nonzero(as_tuple=False)
        chunk_step = int(ts_eq[0].item()) if ts_eq.numel() > 0 else 0

        if chunk_step + 1 < self.timesteps.shape[0]:
            next_t = self.timesteps[chunk_step + 1]
            noise = torch.randn(x0.shape, generator=generator, device=x0.device, dtype=x0.dtype)
            return (self._inner.add_noise(x0, noise, next_t.view(1)),)
        return (x0,)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        return self._inner.add_noise(original_samples, noise, timesteps)

    def _convert_flow_pred_to_x0(
        self,
        flow_pred: torch.Tensor,
        xt: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device),
            [flow_pred, xt, self.sigmas, self._full_timesteps],
        )
        timestep_id = torch.argmin((timesteps - timestep).abs())
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        return (xt - sigma_t * flow_pred).to(original_dtype)