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
    default_height = 480
    default_width = 864
    default_num_frames = 125
    default_num_inference_steps = 50
    default_guidance_scale = 5.0

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


class KandinskyPiFlowScheduler(KandinskyFlowMatchScheduler):
    """Pi-Flow sampler for Kandinsky6 Pro Distil checkpoints.

    Distilled checkpoints predict a grid of denoising outputs per modality.
    Each outer model call uses that grid to construct a DX policy, then rolls
    the policy forward with the trained Pi-Flow integration schedule.
    """

    default_num_frames = 121
    default_num_inference_steps = 10
    default_guidance_scale = 1.0

    def __init__(
        self,
        scheduler_scale: float = 5.0,
        device: torch.device | str = "cpu",
        n_grid: int = 10,
        num_policy_substeps: int = 128,
        final_step_size_scale: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        if n_grid < 2:
            raise ValueError(f"Pi-Flow requires n_grid >= 2, got {n_grid}")
        if num_policy_substeps < 1:
            raise ValueError(f"Pi-Flow requires num_policy_substeps >= 1, got {num_policy_substeps}")
        if not 0.0 < final_step_size_scale <= 1.0:
            raise ValueError("final_step_size_scale must be in (0, 1]")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.n_grid = int(n_grid)
        self.num_policy_substeps = int(num_policy_substeps)
        self.final_step_size_scale = float(final_step_size_scale)
        self.eps = float(eps)
        super().__init__(scheduler_scale=scheduler_scale, device=device)

    @staticmethod
    def _shift(t: torch.Tensor, shift: float) -> torch.Tensor:
        return shift * t / (1.0 + (shift - 1.0) * t)

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: torch.device | str | None = None,
        **kwargs,
    ) -> None:
        del kwargs
        if num_inference_steps < 1:
            raise ValueError(f"num_inference_steps must be positive, got {num_inference_steps}")
        device = device or self.device
        base_segment = 1.0 / (num_inference_steps - (1.0 - self.final_step_size_scale))
        raw_src = torch.ones(num_inference_steps, device=device, dtype=torch.float32)
        raw_dst = torch.empty_like(raw_src)
        for index in range(num_inference_steps):
            segment = base_segment * (self.final_step_size_scale if index == num_inference_steps - 1 else 1.0)
            raw_dst[index] = (raw_src[index] - segment).clamp(min=self.eps)
            if index + 1 < num_inference_steps:
                raw_src[index + 1] = raw_dst[index]
        self.raw_t_src = raw_src
        self.raw_t_dst = raw_dst
        self.sigmas = torch.cat(
            [self._shift(raw_src, self.scheduler_scale), self._shift(raw_dst[-1:], self.scheduler_scale)]
        )
        self.timesteps = self.sigmas[:-1] * 1000.0
        self._step_index = None

    def _rollout(self, prediction: torch.Tensor, sample: torch.Tensor, step_index: int) -> torch.Tensor:
        channels = sample.shape[-1]
        expected = self.n_grid * channels
        if prediction.shape[:-1] != sample.shape[:-1] or prediction.shape[-1] != expected:
            raise ValueError(
                "Pi-Flow prediction must have the checkpoint grid packed in its last dimension: "
                f"expected {(*sample.shape[:-1], expected)}, got {tuple(prediction.shape)}"
            )
        # Match Diffusers PiFlow: accumulate policy updates in FP32 even when
        # the transformer predicts BF16, then restore the caller's sample dtype.
        sample_dtype = sample.dtype
        sample = sample.float()
        prediction = prediction.float()
        # [tokens, ..., grid * channels] -> [tokens, grid, ..., channels]
        grid = prediction.reshape(*sample.shape[:-1], self.n_grid, channels).movedim(-2, 1)
        raw_src = self.raw_t_src[step_index]
        raw_dst = self.raw_t_dst[step_index]
        sigma_src = self.sigmas[step_index]
        x0_grid = sample.unsqueeze(1) - sigma_src * grid

        def policy(x_t: torch.Tensor, sigma_t: torch.Tensor) -> torch.Tensor:
            sigma = sigma_t.clamp(min=self.eps)
            raw_t = sigma_t / (self.scheduler_scale + (1.0 - self.scheduler_scale) * sigma_t)
            position = ((raw_t - raw_dst) / (raw_src - raw_dst).clamp(min=self.eps)).clamp(0.0, 1.0)
            position = position * (self.n_grid - 1)
            lower = position.floor().long().clamp(0, self.n_grid - 2)
            upper = lower + 1
            weight = position - lower
            x0 = torch.lerp(
                x0_grid.index_select(1, lower.reshape(1)).squeeze(1),
                x0_grid.index_select(1, upper.reshape(1)).squeeze(1),
                weight,
            )
            return (x_t - x0) / sigma

        delta = raw_src - raw_dst
        substeps = max(1, int(round(float(delta.item()) * self.num_policy_substeps)))
        substep_size = delta / substeps
        raw_t = raw_src
        sigma_t = sigma_src
        x_t = sample
        for _ in range(substeps):
            velocity = policy(x_t, sigma_t)
            raw_next = (raw_t - substep_size).clamp(min=0.0)
            sigma_next = self._shift(raw_next, self.scheduler_scale)
            x_t = x_t + velocity * (sigma_next - sigma_t)
            raw_t, sigma_t = raw_next, sigma_next
        return x_t.to(sample_dtype)

    def step_pair(
        self,
        prediction_video: torch.Tensor,
        prediction_audio: torch.Tensor,
        timestep: torch.Tensor,
        sample_video: torch.Tensor,
        sample_audio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one synchronized Pi-Flow step to the video and audio states."""
        if self._step_index is None:
            self._step_index = self.index_for_timestep(timestep)
        index = self._step_index
        video = self._rollout(prediction_video, sample_video, index)
        audio = self._rollout(prediction_audio, sample_audio, index)
        self._step_index += 1
        return video, audio

    def step(self, *args, **kwargs):
        raise RuntimeError("KandinskyPiFlowScheduler requires synchronized video/audio step_pair() calls")
