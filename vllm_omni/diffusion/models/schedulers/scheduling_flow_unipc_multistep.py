# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/hao-ai-lab/FastVideo
# Originally from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/schedulers/scheduling_unipc_multistep.py
# Convert unipc for flow matching
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
FlowUniPCMultistepScheduler - A training-free framework for fast sampling of flow-matching diffusion models.

This scheduler implements the UniPC (Unified Predictor-Corrector) algorithm adapted for flow matching,
providing faster convergence than simple Euler methods while maintaining quality.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput
from diffusers.utils import deprecate

from vllm_omni.diffusion.models.schedulers.base import BaseScheduler


class FlowUniPCMultistepScheduler(SchedulerMixin, ConfigMixin, BaseScheduler):
    """
    `FlowUniPCMultistepScheduler` is a training-free framework designed for the fast sampling of
    flow-matching diffusion models.

    This scheduler implements the UniPC (Unified Predictor-Corrector) algorithm adapted for flow matching,
    which can achieve the same quality as Euler methods in fewer steps (typically 20-30 steps vs 40-50).

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        solver_order (`int`, default `2`):
            The UniPC order which can be any positive integer. The effective order of accuracy is `solver_order + 1`
            due to the UniC. It is recommended to use `solver_order=2` for guided sampling, and `solver_order=3` for
            unconditional sampling.
        prediction_type (`str`, defaults to "flow_prediction"):
            Prediction type of the scheduler function; must be `flow_prediction` for this scheduler.
        shift (`float`, defaults to 1.0):
            The shift parameter for the noise schedule. For Wan2.2: use 5.0 for 720p, 12.0 for 480p.
        use_dynamic_shifting (`bool`, defaults to False):
            Whether to use dynamic shifting based on image resolution.
        thresholding (`bool`, defaults to `False`):
            Whether to use the "dynamic thresholding" method.
        dynamic_thresholding_ratio (`float`, defaults to 0.995):
            The ratio for the dynamic thresholding method.
        sample_max_value (`float`, defaults to 1.0):
            The threshold value for dynamic thresholding.
        predict_x0 (`bool`, defaults to `True`):
            Whether to use the updating algorithm on the predicted x0.
        solver_type (`str`, default `bh2`):
            Solver type for UniPC. Use `bh1` for unconditional sampling when steps < 10, `bh2` otherwise.
        lower_order_final (`bool`, default `True`):
            Whether to use lower-order solvers in the final steps. Stabilizes sampling for steps < 15.
        disable_corrector (`list`, default `[]`):
            Steps to disable the corrector to mitigate misalignment with large guidance scales.
        timestep_spacing (`str`, defaults to `"linspace"`):
            The way the timesteps should be scaled.
        final_sigmas_type (`str`, defaults to `"zero"`):
            The final `sigma` value for the noise schedule. Either `"zero"` or `"sigma_min"`.
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        solver_order: int = 2,
        prediction_type: str = "flow_prediction",
        shift: float | None = 1.0,
        use_dynamic_shifting: bool = False,
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        predict_x0: bool = True,
        solver_type: str = "bh2",
        lower_order_final: bool = True,
        disable_corrector: tuple = (),
        solver_p: SchedulerMixin | None = None,
        timestep_spacing: str = "linspace",
        steps_offset: int = 0,
        final_sigmas_type: str | None = "zero",
        **kwargs,
    ):
        if solver_type not in ["bh1", "bh2"]:
            if solver_type in ["midpoint", "heun", "logrho"]:
                self.register_to_config(solver_type="bh2")
            else:
                raise NotImplementedError(f"{solver_type} is not implemented for {self.__class__}")

        self.predict_x0 = predict_x0
        self.num_inference_steps: int | None = None

        # Initialize sigma schedule
        alphas = np.linspace(1, 1 / num_train_timesteps, num_train_timesteps)[::-1].copy()
        sigmas = 1.0 - alphas
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)

        if not use_dynamic_shifting:
            # Apply timestep shifting based on shift parameter
            assert shift is not None
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.sigmas = sigmas
        self.timesteps = sigmas * num_train_timesteps
        self.num_train_timesteps = num_train_timesteps

        # State for multistep solver
        self.model_outputs: list[torch.Tensor | None] = [None] * solver_order
        self.timestep_list: list[Any | None] = [None] * solver_order
        self.lower_order_nums = 0
        self.disable_corrector = list(disable_corrector)
        self.solver_p = solver_p
        self.last_sample: torch.Tensor | None = None
        self._step_index: int | None = None
        self._begin_index: int | None = None
        self.this_order: int = 1

        # Move sigmas to CPU to reduce GPU/CPU communication
        self.sigmas = self.sigmas.to("cpu")
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

        BaseScheduler.__init__(self)

    @property
    def step_index(self) -> int | None:
        """The index counter for current timestep. Increases by 1 after each scheduler step."""
        return self._step_index

    @property
    def begin_index(self) -> int | None:
        """The index for the first timestep. Should be set from pipeline with `set_begin_index` method."""
        return self._begin_index

    def set_shift(self, shift: float) -> None:
        """Set the shift parameter for the scheduler."""
        self.config.shift = shift

    def set_begin_index(self, begin_index: int = 0) -> None:
        """
        Sets the begin index for the scheduler. Run from pipeline before inference.

        Args:
            begin_index (`int`): The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device | None = None,
        sigmas: list[float] | None = None,
        mu: float | None = None,
        shift: float | None = None,
    ) -> None:
        """
        Sets the discrete timesteps used for the diffusion chain (run before inference).

        Args:
            num_inference_steps (`int`):
                Total number of timesteps.
            device (`str` or `torch.device`, *optional*):
                The device to move timesteps to.
            sigmas (`list[float]`, *optional*):
                Custom sigma schedule.
            mu (`float`, *optional*):
                Parameter for dynamic shifting.
            shift (`float`, *optional*):
                Override shift parameter.
        """
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError("Must pass a value for `mu` when `use_dynamic_shifting` is True")

        if sigmas is None:
            assert num_inference_steps is not None
            sigmas = np.linspace(self.sigma_max, self.sigma_min, num_inference_steps + 1).copy()[:-1]

        if self.config.use_dynamic_shifting:
            assert mu is not None
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            if shift is None:
                shift = self.config.shift
            assert isinstance(sigmas, np.ndarray)
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        if self.config.final_sigmas_type == "sigma_min":
            sigma_last = self.sigma_min
        elif self.config.final_sigmas_type == "zero":
            sigma_last = 0
        else:
            raise ValueError(f"`final_sigmas_type` must be 'zero' or 'sigma_min', got {self.config.final_sigmas_type}")

        timesteps = sigmas * self.config.num_train_timesteps
        sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)

        self.num_inference_steps = len(timesteps)

        # Reset state
        self.model_outputs = [None] * self.config.solver_order
        self.timestep_list = [None] * self.config.solver_order
        self.lower_order_nums = 0
        self.last_sample = None

        if self.solver_p:
            self.solver_p.set_timesteps(self.num_inference_steps, device=device)

        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")

    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Dynamic thresholding to prevent pixel saturation.

        From "Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding"
        https://arxiv.org/abs/2205.11487
        """
        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape

        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()

        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))
        abs_sample = sample.abs()

        s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(s, min=1, max=self.config.sample_max_value)
        s = s.unsqueeze(1)
        sample = torch.clamp(sample, -s, s) / s

        sample = sample.reshape(batch_size, channels, *remaining_dims)
        sample = sample.to(dtype)

        return sample

    def _sigma_to_t(self, sigma: torch.Tensor) -> torch.Tensor:
        """Convert sigma to timestep."""
        return sigma * self.config.num_train_timesteps

    def _sigma_to_alpha_sigma_t(self, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert sigma to alpha and sigma_t for flow matching."""
        return 1 - sigma, sigma

    def time_shift(self, mu: float, sigma: float, t: np.ndarray) -> np.ndarray:
        """Apply time shift transformation."""
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def convert_model_output(
        self,
        model_output: torch.Tensor,
        *args,
        sample: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Convert the model output to the format needed by the UniPC algorithm.

        Args:
            model_output (`torch.Tensor`): Direct output from the diffusion model.
            sample (`torch.Tensor`): Current sample in the diffusion process.

        Returns:
            `torch.Tensor`: Converted model output.
        """
        timestep = args[0] if len(args) > 0 else kwargs.pop("timestep", None)
        if sample is None:
            if len(args) > 1:
                sample = args[1]
            else:
                raise ValueError("missing `sample` as a required keyword argument")
        if timestep is not None:
            deprecate(
                "timesteps",
                "1.0.0",
                "Passing `timesteps` is deprecated and has no effect as model output conversion "
                "is now handled via an internal counter `self.step_index`",
            )

        sigma = self.sigmas[self.step_index].to(sample.device)
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)

        if self.predict_x0:
            if self.config.prediction_type == "flow_prediction":
                sigma_t = sigma.to(sample.device)
                x0_pred = sample - sigma_t * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be `flow_prediction` "
                    "for the FlowUniPCMultistepScheduler."
                )

            if self.config.thresholding:
                x0_pred = self._threshold_sample(x0_pred)

            return x0_pred
        else:
            if self.config.prediction_type == "flow_prediction":
                sigma_t = sigma.to(sample.device)
                epsilon = sample - (1 - sigma_t) * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be `flow_prediction` "
                    "for the FlowUniPCMultistepScheduler."
                )

            if self.config.thresholding:
                sigma_t = sigma.to(sample.device)
                x0_pred = sample - sigma_t * model_output
                x0_pred = self._threshold_sample(x0_pred)
                epsilon = model_output + x0_pred

            return epsilon

    def multistep_uni_p_bh_update(
        self,
        model_output: torch.Tensor,
        *args,
        sample: torch.Tensor | None = None,
        order: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        One step for the UniP (B(h) version) predictor.

        Args:
            model_output (`torch.Tensor`): Direct output from the diffusion model.
            sample (`torch.Tensor`): Current sample.
            order (`int`): The order of UniP at this timestep.

        Returns:
            `torch.Tensor`: The sample tensor at the previous timestep.
        """
        prev_timestep = args[0] if len(args) > 0 else kwargs.pop("prev_timestep", None)
        if sample is None:
            if len(args) > 1:
                sample = args[1]
            else:
                raise ValueError("missing `sample` as a required keyword argument")
        if order is None:
            if len(args) > 2:
                order = args[2]
            else:
                raise ValueError("missing `order` as a required keyword argument")
        if prev_timestep is not None:
            deprecate(
                "prev_timestep",
                "1.0.0",
                "Passing `prev_timestep` is deprecated and has no effect.",
            )

        model_output_list = self.model_outputs

        s0 = self.timestep_list[-1]
        m0 = model_output_list[-1]
        x = sample

        if self.solver_p:
            x_t = self.solver_p.step(model_output, s0, x).prev_sample
            return x_t

        device = sample.device
        sigma_t, sigma_s0 = (
            self.sigmas[self.step_index + 1].to(device),
            self.sigmas[self.step_index].to(device),
        )
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)

        h = lambda_t - lambda_s0

        rks = []
        D1s: list[Any] | None = []
        for i in range(1, order):
            si = self.step_index - i
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si].to(device))
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            assert mi is not None
            D1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=device)

        R = []
        b = []

        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.config.solver_type == "bh1":
            B_h = hh
        elif self.config.solver_type == "bh2":
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.tensor(b, device=device)

        if D1s is not None and len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)
            if order == 2:
                rhos_p = torch.tensor([0.5], dtype=x.dtype, device=device)
            else:
                assert isinstance(R, torch.Tensor)
                rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1]).to(device).to(x.dtype)
        else:
            D1s = None

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            if D1s is not None:
                pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s)
            else:
                pred_res = 0
            x_t = x_t_ - alpha_t * B_h * pred_res
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            if D1s is not None:
                pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s)
            else:
                pred_res = 0
            x_t = x_t_ - sigma_t * B_h * pred_res

        x_t = x_t.to(x.dtype)
        return x_t

    def multistep_uni_c_bh_update(
        self,
        this_model_output: torch.Tensor,
        *args,
        last_sample: torch.Tensor | None = None,
        this_sample: torch.Tensor | None = None,
        order: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        One step for the UniC (B(h) version) corrector.

        Args:
            this_model_output (`torch.Tensor`): Model outputs at `x_t`.
            last_sample (`torch.Tensor`): Sample before the last predictor `x_{t-1}`.
            this_sample (`torch.Tensor`): Sample after the last predictor `x_{t}`.
            order (`int`): The order of UniC-p. Effective accuracy is `order + 1`.

        Returns:
            `torch.Tensor`: The corrected sample tensor.
        """
        this_timestep = args[0] if len(args) > 0 else kwargs.pop("this_timestep", None)
        if last_sample is None:
            if len(args) > 1:
                last_sample = args[1]
            else:
                raise ValueError("missing `last_sample` as a required keyword argument")
        if this_sample is None:
            if len(args) > 2:
                this_sample = args[2]
            else:
                raise ValueError("missing `this_sample` as a required keyword argument")
        if order is None:
            if len(args) > 3:
                order = args[3]
            else:
                raise ValueError("missing `order` as a required keyword argument")
        if this_timestep is not None:
            deprecate(
                "this_timestep",
                "1.0.0",
                "Passing `this_timestep` is deprecated and has no effect.",
            )

        model_output_list = self.model_outputs

        m0 = model_output_list[-1]
        x = last_sample
        x_t = this_sample
        model_t = this_model_output

        device = this_sample.device
        sigma_t, sigma_s0 = (
            self.sigmas[self.step_index].to(device),
            self.sigmas[self.step_index - 1].to(device),
        )
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)

        h = lambda_t - lambda_s0

        rks = []
        D1s: list[Any] | None = []
        for i in range(1, order):
            si = self.step_index - (i + 1)
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si].to(device))
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            assert mi is not None
            D1s.append((mi - m0) / rk)

        rks.append(1.0)
        rks = torch.tensor(rks, device=device)

        R = []
        b = []

        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.config.solver_type == "bh1":
            B_h = hh
        elif self.config.solver_type == "bh2":
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.tensor(b, device=device)

        if D1s is not None and len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)
        else:
            D1s = None

        if order == 1:
            rhos_c = torch.tensor([0.5], dtype=x.dtype, device=device)
        else:
            rhos_c = torch.linalg.solve(R, b).to(device).to(x.dtype)

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            if D1s is not None:
                corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
            else:
                corr_res = 0
            D1_t = model_t - m0
            x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            if D1s is not None:
                corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
            else:
                corr_res = 0
            D1_t = model_t - m0
            x_t = x_t_ - sigma_t * B_h * (corr_res + rhos_c[-1] * D1_t)

        x_t = x_t.to(x.dtype)
        return x_t

    def index_for_timestep(self, timestep: torch.Tensor, schedule_timesteps: torch.Tensor | None = None) -> int:
        """Get the index for a given timestep."""
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        step_index: int = indices[pos].item()

        return step_index

    def _init_step_index(self, timestep: torch.Tensor) -> None:
        """Initialize the step_index counter for the scheduler."""
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.Tensor,
        return_dict: bool = True,
        generator: torch.Generator | None = None,
    ) -> SchedulerOutput | tuple:
        """
        Predict the sample from the previous timestep by reversing the SDE using multistep UniPC.

        Args:
            model_output (`torch.Tensor`): Direct output from the diffusion model.
            timestep (`int`): Current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`): Current sample created by the diffusion process.
            return_dict (`bool`): Whether to return a SchedulerOutput or tuple.

        Returns:
            `SchedulerOutput` or `tuple`: The sample tensor at the previous timestep.
        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        use_corrector = (
            self.step_index > 0 and self.step_index - 1 not in self.disable_corrector and self.last_sample is not None
        )

        model_output_convert = self.convert_model_output(model_output, sample=sample)

        if use_corrector:
            sample = self.multistep_uni_c_bh_update(
                this_model_output=model_output_convert,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
            )

        # Update model output history
        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self.timestep_list[i] = self.timestep_list[i + 1]

        self.model_outputs[-1] = model_output_convert
        self.timestep_list[-1] = timestep

        # Determine order for this step
        if self.config.lower_order_final:
            this_order = min(self.config.solver_order, len(self.timesteps) - self.step_index)
        else:
            this_order = self.config.solver_order

        self.this_order = min(this_order, self.lower_order_nums + 1)  # warmup for multistep
        assert self.this_order > 0

        self.last_sample = sample
        prev_sample = self.multistep_uni_p_bh_update(
            model_output=model_output,
            sample=sample,
            order=self.this_order,
        )

        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1

        assert self._step_index is not None
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    def scale_model_input(self, sample: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input.

        Args:
            sample (`torch.Tensor`): The input sample.

        Returns:
            `torch.Tensor`: A scaled input sample (unchanged for this scheduler).
        """
        return sample

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        """
        Add noise to the original samples.

        Args:
            original_samples (`torch.Tensor`): Original samples.
            noise (`torch.Tensor`): Noise to add.
            timesteps (`torch.IntTensor`): Timesteps for noise addition.

        Returns:
            `torch.Tensor`: Noisy samples.
        """
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)

        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)

        if self.begin_index is None:
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        elif self.step_index is not None:
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            step_indices = [self.begin_index] * timesteps.shape[0]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        noisy_samples = alpha_t * original_samples + sigma_t * noise
        return noisy_samples

    def __len__(self) -> int:
        return self.config.num_train_timesteps
