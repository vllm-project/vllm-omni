# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 SandAI. All Rights Reserved.
# Ported from daVinci-MagiHuman inference/pipeline/video_generate.py
# Adapted for vllm-omni: single-GPU, diffusers VAE, configurable dit_subfolder.

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import whisper
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import (
    KarrasDiffusionSchedulers,
    SchedulerMixin,
    SchedulerOutput,
)
from diffusers.utils import deprecate, load_image
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from einops import rearrange
from PIL import Image
from safetensors.torch import load_file
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from transformers import AutoTokenizer
from transformers.models.t5gemma import T5GemmaEncoderModel
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
    DistributedAutoencoderKLWan,
)
from vllm_omni.diffusion.model_loader.diffusers_loader import (
    DiffusersPipelineLoader,
)
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.models.t5_encoder.t5_gemma_encoder import T5GemmaEncoderModelTP
from vllm_omni.diffusion.models.utils import _load_json
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import (
    DiffusionPipelineProfilerMixin,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest

from .magi_human_dit import (
    DiTModel,
    FFAHandler,
    MagiHumanDiTConfig,
    Modality,
    VarlenHandler,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Scheduler (ported from daVinci-MagiHuman inference/pipeline/scheduler_unipc.py)
# ===========================================================================
class FlowUniPCMultistepScheduler(SchedulerMixin, ConfigMixin):
    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        solver_order: int = 2,
        prediction_type: str = "flow_prediction",
        shift: float = 1.0,
        use_dynamic_shifting=False,
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        predict_x0: bool = True,
        solver_type: str = "bh2",
        lower_order_final: bool = True,
        disable_corrector: list[int] = [],
        solver_p: SchedulerMixin = None,
        timestep_spacing: str = "linspace",
        steps_offset: int = 0,
        final_sigmas_type: str | None = "zero",
    ):
        if solver_type not in ["bh1", "bh2"]:
            if solver_type in ["midpoint", "heun", "logrho"]:
                self.register_to_config(solver_type="bh2")
            else:
                raise NotImplementedError(f"{solver_type} is not implemented for {self.__class__}")

        self.predict_x0 = predict_x0
        self.num_inference_steps = None
        alphas = np.linspace(1, 1 / num_train_timesteps, num_train_timesteps)[::-1].copy()
        sigmas = 1.0 - alphas
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)

        if not use_dynamic_shifting:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.sigmas = sigmas
        self.timesteps = sigmas * num_train_timesteps

        self.model_outputs = [None] * solver_order
        self.timestep_list = [None] * solver_order
        self.lower_order_nums = 0
        self.disable_corrector = disable_corrector
        self.solver_p = solver_p
        self.last_sample = None
        self._step_index: int | None = None
        self._begin_index: int | None = None

        self.sigmas = self.sigmas.to("cpu")
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = begin_index

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        device: str | torch.device = None,
        sigmas: list[float] | None = None,
        mu: float | None | None = None,
        shift: float | None | None = None,
    ):
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError(" you have to pass a value for `mu` when `use_dynamic_shifting` is set to be `True`")

        if sigmas is None:
            sigmas = np.linspace(self.sigma_max, self.sigma_min, num_inference_steps + 1).copy()[:-1]

        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            if shift is None:
                shift = self.config.shift
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        if self.config.final_sigmas_type == "sigma_min":
            sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0]) ** 0.5
        elif self.config.final_sigmas_type == "zero":
            sigma_last = 0
        else:
            raise ValueError(
                f"`final_sigmas_type` must be one of 'zero', or 'sigma_min', but got {self.config.final_sigmas_type}"
            )

        timesteps = sigmas * self.config.num_train_timesteps
        sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)

        self.num_inference_steps = len(timesteps)

        self.model_outputs = [None] * self.config.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        if self.solver_p:
            self.solver_p.set_timesteps(self.num_inference_steps, device=device)

        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")

    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
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
        return sample.to(dtype)

    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    def _sigma_to_alpha_sigma_t(self, sigma):
        return 1 - sigma, sigma

    def time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def convert_model_output(
        self, model_output: torch.Tensor, *args, sample: torch.Tensor = None, **kwargs
    ) -> torch.Tensor:
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
                "Passing `timesteps` is deprecated and has no effect as model output "
                "conversion is now handled via an internal counter `self.step_index`",
            )

        sigma = self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)

        if self.predict_x0:
            if self.config.prediction_type == "flow_prediction":
                sigma_t = self.sigmas[self.step_index]
                x0_pred = sample - sigma_t * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`,"
                    " `v_prediction` or `flow_prediction` for the UniPCMultistepScheduler."
                )
            if self.config.thresholding:
                x0_pred = self._threshold_sample(x0_pred)
            return x0_pred
        else:
            if self.config.prediction_type == "flow_prediction":
                sigma_t = self.sigmas[self.step_index]
                epsilon = sample - (1 - sigma_t) * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`,"
                    " `v_prediction` or `flow_prediction` for the UniPCMultistepScheduler."
                )
            if self.config.thresholding:
                sigma_t = self.sigmas[self.step_index]
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
        prev_timestep = args[0] if len(args) > 0 else kwargs.pop("prev_timestep", None)
        if sample is None:
            if len(args) > 1:
                sample = args[1]
            else:
                raise ValueError(" missing `sample` as a required keyword argument")
        if order is None:
            if len(args) > 2:
                order = args[2]
            else:
                raise ValueError(" missing `order` as a required keyword argument")
        if prev_timestep is not None:
            deprecate("prev_timestep", "1.0.0", "Passing `prev_timestep` is deprecated and has no effect.")

        model_output_list = self.model_outputs
        s0 = self.timestep_list[-1]
        m0 = model_output_list[-1]
        x = sample

        if self.solver_p:
            return self.solver_p.step(model_output, s0, x).prev_sample

        sigma_t, sigma_s0 = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0
        device = sample.device

        rks = []
        D1s: list[Any] | None = []
        for i in range(1, order):
            si = self.step_index - i
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
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

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)
            if order == 2:
                rhos_p = torch.tensor([0.5], dtype=x.dtype, device=device)
            else:
                rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1]).to(device).to(x.dtype)
        else:
            D1s = None

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s) if D1s is not None else 0
            x_t = x_t_ - alpha_t * B_h * pred_res
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s) if D1s is not None else 0
            x_t = x_t_ - sigma_t * B_h * pred_res

        return x_t.to(x.dtype)

    def multistep_uni_c_bh_update(
        self,
        this_model_output: torch.Tensor,
        *args,
        last_sample: torch.Tensor = None,
        this_sample: torch.Tensor = None,
        order: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        this_timestep = args[0] if len(args) > 0 else kwargs.pop("this_timestep", None)
        if last_sample is None:
            if len(args) > 1:
                last_sample = args[1]
            else:
                raise ValueError(" missing`last_sample` as a required keyword argument")
        if this_sample is None:
            if len(args) > 2:
                this_sample = args[2]
            else:
                raise ValueError(" missing`this_sample` as a required keyword argument")
        if order is None:
            if len(args) > 3:
                order = args[3]
            else:
                raise ValueError(" missing`order` as a required keyword argument")
        if this_timestep is not None:
            deprecate("this_timestep", "1.0.0", "Passing `this_timestep` is deprecated and has no effect.")

        model_output_list = self.model_outputs
        m0 = model_output_list[-1]
        x = last_sample
        x_t = this_sample
        model_t = this_model_output

        sigma_t, sigma_s0 = self.sigmas[self.step_index], self.sigmas[self.step_index - 1]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0
        device = this_sample.device

        rks = []
        D1s: list[Any] | None = []
        for i in range(1, order):
            si = self.step_index - (i + 1)
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
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

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)
        else:
            D1s = None

        if order == 1:
            rhos_c = torch.tensor([0.5], dtype=x.dtype, device=device)
        else:
            rhos_c = torch.linalg.solve(R, b).to(device).to(x.dtype)

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s) if D1s is not None else 0
            D1_t = model_t - m0
            x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s) if D1s is not None else 0
            D1_t = model_t - m0
            x_t = x_t_ - sigma_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        return x_t.to(x.dtype)

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
        indices = (schedule_timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()

    def _init_step_index(self, timestep):
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
        generator=None,
    ) -> SchedulerOutput | tuple:
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

        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self.timestep_list[i] = self.timestep_list[i + 1]

        self.model_outputs[-1] = model_output_convert
        self.timestep_list[-1] = timestep

        if self.config.lower_order_final:
            this_order = min(self.config.solver_order, len(self.timesteps) - self.step_index)
        else:
            this_order = self.config.solver_order

        self.this_order = min(this_order, self.lower_order_nums + 1)
        assert self.this_order > 0

        self.last_sample = sample
        prev_sample = self.multistep_uni_p_bh_update(model_output=model_output, sample=sample, order=self.this_order)

        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1

        self._step_index += 1

        if not return_dict:
            return (prev_sample,)
        return SchedulerOutput(prev_sample=prev_sample)

    def step_ddim(
        self,
        velocity: torch.FloatTensor,
        t: int,
        curr_state: torch.FloatTensor,
        prev_state: torch.FloatTensor | None = None,
        generator: torch.Generator | None = None,
    ):
        device = curr_state.device
        curr_t = self.sigmas[t]
        prev_t = self.sigmas[t + 1]
        variance_noise = randn_tensor(curr_state.shape, generator=generator, device=device, dtype=curr_state.dtype)
        cur_clean_ = curr_state - curr_t * velocity
        return prev_t * variance_noise + (1 - prev_t) * cur_clean_

    def step_sde(
        self,
        velocity: torch.FloatTensor,
        t: int,
        curr_state: torch.FloatTensor,
        noise_theta: float = 1.0,
        prev_state: torch.FloatTensor | None = None,
        generator: torch.Generator | None = None,
    ):
        device = curr_state.device
        curr_t = self.sigmas[t]
        prev_t = self.sigmas[t + 1]
        cos = torch.cos(torch.tensor(noise_theta) * torch.pi / 2).to(device)
        sin = torch.sin(torch.tensor(noise_theta) * torch.pi / 2).to(device)
        prev_sample_mean = (1 - prev_t + prev_t * cos) * (curr_state - curr_t * velocity) + prev_t * cos * velocity
        std_dev_t = prev_t * sin
        std_dev_t = torch.ones((1, 1)).to(curr_state) * std_dev_t
        if prev_state is None:
            variance_noise = randn_tensor(curr_state.shape, generator=generator, device=device, dtype=curr_state.dtype)
            prev_state = prev_sample_mean + std_dev_t * variance_noise
        else:
            prev_state = prev_sample_mean + (prev_state - prev_sample_mean.detach())
        return prev_state

    def scale_model_input(self, sample: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return sample

    def add_noise(
        self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.IntTensor
    ) -> torch.Tensor:
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
        return alpha_t * original_samples + sigma_t * noise

    def __len__(self):
        return self.config.num_train_timesteps


# ===========================================================================
# Audio VAE (ported from daVinci-MagiHuman inference/model/sa_audio/)
# ===========================================================================
def _snake_beta(x, alpha, beta):
    return x + (1.0 / (beta + 1e-9)) * torch.pow(torch.sin(x * alpha), 2)


class _SnakeBeta(nn.Module):
    def __init__(self, in_features: int, alpha: float = 1.0, alpha_trainable: bool = True, alpha_logscale: bool = True):
        super().__init__()
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)
        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return _snake_beta(x, alpha, beta)


def _vae_sample(mean, scale):
    stdev = F.softplus(scale) + 1e-4
    var = stdev * stdev
    logvar = torch.log(var)
    latents = torch.randn_like(mean) * stdev + mean
    kl = (mean * mean + var - logvar - 1).sum(1).mean()
    return latents, kl


class _VAEBottleneck(nn.Module):
    def encode(self, x, return_info=False, **kwargs):
        info = {}
        mean, scale = x.chunk(2, dim=1)
        x, kl = _vae_sample(mean, scale)
        info["kl"] = kl
        return (x, info) if return_info else x

    def decode(self, x):
        return x


def _WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def _WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def _checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)


def _get_activation(activation: Literal["elu", "snake", "none"], antialias: bool = False, channels=None) -> nn.Module:
    if antialias:
        raise NotImplementedError("antialias activation not supported")
    if activation == "elu":
        return nn.ELU()
    if activation == "snake":
        return _SnakeBeta(channels)
    if activation == "none":
        return nn.Identity()
    raise ValueError(f"Unknown activation {activation}")


class _ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, use_snake=False, antialias_activation=False):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.layers = nn.Sequential(
            _get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
            _WNConv1d(in_channels, out_channels, kernel_size=7, dilation=dilation, padding=padding),
            _get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
            _WNConv1d(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return (_checkpoint(self.layers, x) if self.training else self.layers(x)) + x


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_snake=False, antialias_activation=False):
        super().__init__()
        self.layers = nn.Sequential(
            _ResidualUnit(in_channels, in_channels, 1, use_snake=use_snake),
            _ResidualUnit(in_channels, in_channels, 3, use_snake=use_snake),
            _ResidualUnit(in_channels, in_channels, 9, use_snake=use_snake),
            _get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels),
            _WNConv1d(in_channels, out_channels, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)),
        )

    def forward(self, x):
        return self.layers(x)


class _DecoderBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride, use_snake=False, antialias_activation=False, use_nearest_upsample=False
    ):
        super().__init__()
        if use_nearest_upsample:
            upsample_layer = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode="nearest"),
                _WNConv1d(in_channels, out_channels, kernel_size=2 * stride, stride=1, bias=False, padding="same"),
            )
        else:
            upsample_layer = _WNConvTranspose1d(
                in_channels, out_channels, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)
            )
        self.layers = nn.Sequential(
            _get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels),
            upsample_layer,
            _ResidualUnit(out_channels, out_channels, 1, use_snake=use_snake),
            _ResidualUnit(out_channels, out_channels, 3, use_snake=use_snake),
            _ResidualUnit(out_channels, out_channels, 9, use_snake=use_snake),
        )

    def forward(self, x):
        return self.layers(x)


class _OobleckEncoder(nn.Module):
    def __init__(
        self,
        in_channels=2,
        channels=128,
        latent_dim=32,
        c_mults=[1, 2, 4, 8],
        strides=[2, 4, 8, 8],
        use_snake=False,
        antialias_activation=False,
    ):
        super().__init__()
        c_mults = [1] + c_mults
        depth = len(c_mults)
        layers = [_WNConv1d(in_channels, c_mults[0] * channels, kernel_size=7, padding=3)]
        for i in range(depth - 1):
            layers.append(
                _EncoderBlock(c_mults[i] * channels, c_mults[i + 1] * channels, strides[i], use_snake=use_snake)
            )
        layers.extend(
            [
                _get_activation(
                    "snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[-1] * channels
                ),
                _WNConv1d(c_mults[-1] * channels, latent_dim, kernel_size=3, padding=1),
            ]
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class _OobleckDecoder(nn.Module):
    def __init__(
        self,
        out_channels=2,
        channels=128,
        latent_dim=32,
        c_mults=[1, 2, 4, 8],
        strides=[2, 4, 8, 8],
        use_snake=False,
        antialias_activation=False,
        use_nearest_upsample=False,
        final_tanh=True,
    ):
        super().__init__()
        c_mults = [1] + c_mults
        depth = len(c_mults)
        layers = [_WNConv1d(latent_dim, c_mults[-1] * channels, kernel_size=7, padding=3)]
        for i in range(depth - 1, 0, -1):
            layers.append(
                _DecoderBlock(
                    c_mults[i] * channels,
                    c_mults[i - 1] * channels,
                    strides[i - 1],
                    use_snake=use_snake,
                    antialias_activation=antialias_activation,
                    use_nearest_upsample=use_nearest_upsample,
                )
            )
        layers.extend(
            [
                _get_activation(
                    "snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[0] * channels
                ),
                _WNConv1d(c_mults[0] * channels, out_channels, kernel_size=7, padding=3, bias=False),
                nn.Tanh() if final_tanh else nn.Identity(),
            ]
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class _AudioAutoencoder(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        latent_dim,
        downsampling_ratio,
        sample_rate,
        io_channels=2,
        bottleneck=None,
        in_channels=None,
        out_channels=None,
        soft_clip=False,
    ):
        super().__init__()
        self.downsampling_ratio = downsampling_ratio
        self.sample_rate = sample_rate
        self.latent_dim = latent_dim
        self.io_channels = io_channels
        self.in_channels = in_channels if in_channels is not None else io_channels
        self.out_channels = out_channels if out_channels is not None else io_channels
        self.bottleneck = bottleneck
        self.encoder = encoder
        self.decoder = decoder
        self.soft_clip = soft_clip

    def encode(self, audio, skip_bottleneck=False, return_info=False, **kwargs):
        info = {}
        latents = self.encoder(audio)
        info["pre_bottleneck_latents"] = latents
        if self.bottleneck is not None and not skip_bottleneck:
            latents, bottleneck_info = self.bottleneck.encode(latents, return_info=True, **kwargs)
            info.update(bottleneck_info)
        return (latents, info) if return_info else latents

    def decode(self, latents, skip_bottleneck=False, **kwargs):
        if self.bottleneck is not None and not skip_bottleneck:
            latents = self.bottleneck.decode(latents)
        decoded = self.decoder(latents, **kwargs)
        if self.soft_clip:
            decoded = torch.tanh(decoded)
        return decoded


def _create_encoder_from_config(cfg: dict[str, Any]):
    assert cfg.get("type") == "oobleck", f"Only 'oobleck' encoder supported, got: {cfg.get('type')}"
    enc = _OobleckEncoder(**cfg["config"])
    if not cfg.get("requires_grad", True):
        for p in enc.parameters():
            p.requires_grad = False
    return enc


def _create_decoder_from_config(cfg: dict[str, Any]):
    assert cfg.get("type") == "oobleck", f"Only 'oobleck' decoder supported, got: {cfg.get('type')}"
    dec = _OobleckDecoder(**cfg["config"])
    if not cfg.get("requires_grad", True):
        for p in dec.parameters():
            p.requires_grad = False
    return dec


def _create_bottleneck_from_config(cfg: dict[str, Any]):
    assert cfg.get("type") == "vae", f"Only 'vae' bottleneck supported, got: {cfg.get('type')}"
    bn = _VAEBottleneck()
    if not cfg.get("requires_grad", True):
        for p in bn.parameters():
            p.requires_grad = False
    return bn


def _create_autoencoder_from_config(config: dict[str, Any]):
    ae_config = config["model"]
    if ae_config.get("pretransform") is not None:
        raise NotImplementedError("Nested pretransform not supported")
    encoder = _create_encoder_from_config(ae_config["encoder"])
    decoder = _create_decoder_from_config(ae_config["decoder"])
    bottleneck_cfg = ae_config.get("bottleneck")
    bottleneck = _create_bottleneck_from_config(bottleneck_cfg) if bottleneck_cfg else None
    return _AudioAutoencoder(
        encoder=encoder,
        decoder=decoder,
        latent_dim=ae_config["latent_dim"],
        downsampling_ratio=ae_config["downsampling_ratio"],
        sample_rate=config["sample_rate"],
        io_channels=ae_config["io_channels"],
        bottleneck=bottleneck,
        in_channels=ae_config.get("in_channels"),
        out_channels=ae_config.get("out_channels"),
        soft_clip=ae_config["decoder"].get("soft_clip", False),
    )


class SAAudioFeatureExtractor:
    def __init__(self, device, model_path):
        self.device = device
        self.vae_model, self.sample_rate = self._load_vae(model_path)
        self.resampler = None

    def _load_vae(self, model_path):
        if not (isinstance(model_path, str) and Path(model_path).is_dir()):
            raise ValueError("model_path must be a local directory")

        model_config_path = os.path.join(model_path, "model_config.json")
        with open(model_config_path) as f:
            full_config = json.load(f)

        vae_config = full_config["model"]["pretransform"]["config"]
        sample_rate = full_config["sample_rate"]

        autoencoder_config = {
            "model_type": "autoencoder",
            "sample_rate": sample_rate,
            "model": vae_config,
        }
        vae_model = _create_autoencoder_from_config(autoencoder_config)

        weights_path = Path(model_path) / "model.safetensors"
        if not weights_path.exists():
            raise FileNotFoundError(f"Weight file does not exist: {weights_path}")

        full_state_dict = load_file(weights_path, device=str(self.device))
        vae_state_dict = {}
        for key, value in full_state_dict.items():
            if key.startswith("pretransform.model."):
                vae_state_dict[key[len("pretransform.model.") :]] = value

        model_keys = set(vae_model.state_dict().keys())
        vae_keys = set(vae_state_dict.keys())
        missing = model_keys - vae_keys
        extra = vae_keys - model_keys
        if missing:
            logger.warning("Audio VAE missing keys (%d): %s", len(missing), list(missing)[:5])
        if extra:
            logger.warning("Audio VAE unexpected keys (%d): %s", len(extra), list(extra)[:5])

        vae_model.load_state_dict(vae_state_dict)
        vae_model.to(self.device)
        return vae_model, sample_rate

    def decode(self, latents):
        with torch.no_grad():
            return self.vae_model.decode(latents)

    def encode(self, waveform):
        with torch.no_grad():
            return self.vae_model.encode(waveform)


# ===========================================================================
# Audio utilities (ported from daVinci-MagiHuman inference/pipeline/video_process.py)
# ===========================================================================
_SAMPLE_RATE = 51200
_AUDIO_CHUNK_DURATION = 29
_OVERLAP_RATIO = 0.5


def _merge_overlapping_vae_features(audio_feats: list[torch.Tensor], overlap_ratio: float = 0.5) -> torch.Tensor | None:
    if not audio_feats:
        return None
    if len(audio_feats) == 1:
        return audio_feats[0]

    batch_size, total_frames, feature_dim = audio_feats[0].shape
    overlap_frames = int(total_frames * overlap_ratio)
    step_frames = total_frames - overlap_frames
    final_length = (len(audio_feats) - 1) * step_frames + total_frames
    output_feat = torch.zeros(
        batch_size, final_length, feature_dim, device=audio_feats[0].device, dtype=audio_feats[0].dtype
    )

    for block_idx, current_feat in enumerate(audio_feats):
        output_start = block_idx * step_frames
        if block_idx == 0:
            output_feat[:, output_start : output_start + total_frames, :] = current_feat
            continue

        non_overlap_start = output_start + overlap_frames
        non_overlap_end = output_start + total_frames
        output_feat[:, non_overlap_start:non_overlap_end, :] = current_feat[:, overlap_frames:, :]

        for frame_idx in range(overlap_frames):
            output_pos = output_start + frame_idx
            prev_weight = (overlap_frames - frame_idx) / overlap_frames
            curr_weight = frame_idx / overlap_frames
            output_feat[:, output_pos, :] = (
                prev_weight * output_feat[:, output_pos, :] + curr_weight * current_feat[:, frame_idx, :]
            )
    return output_feat


def load_audio_and_encode(audio_vae, audio_path: str, seconds: int | None = None) -> torch.Tensor:
    """Load audio from file and encode to latent space using the Stable Audio VAE."""
    audio_full = whisper.load_audio(audio_path, sr=_SAMPLE_RATE)
    if seconds is not None:
        audio_full = audio_full[: min(int(seconds * _SAMPLE_RATE), audio_full.shape[0])]
    total_samples = audio_full.shape[0]

    window_size = int(_AUDIO_CHUNK_DURATION * _SAMPLE_RATE)
    step_size = int(window_size * (1 - _OVERLAP_RATIO))
    if total_samples <= window_size:
        audio = torch.from_numpy(audio_full).cuda()
        audio = audio.unsqueeze(0).expand(2, -1)
        return audio_vae.vae_model.encode(audio)

    encoded_chunks = []
    latent_to_audio_ratio = None
    for offset_start in range(0, total_samples, step_size):
        offset_end = min(offset_start + window_size, total_samples)
        chunk = whisper.pad_or_trim(audio_full[offset_start:offset_end], length=window_size)
        chunk_tensor = torch.from_numpy(chunk).cuda().unsqueeze(0).expand(2, -1)
        encoded_chunk = audio_vae.vae_model.encode(chunk_tensor)

        if latent_to_audio_ratio is None:
            latent_to_audio_ratio = encoded_chunk.shape[-1] / window_size

        encoded_chunks.append(encoded_chunk.permute(0, 2, 1))
        if offset_end >= total_samples:
            break

    final_feat = _merge_overlapping_vae_features(encoded_chunks, overlap_ratio=_OVERLAP_RATIO).permute(0, 2, 1)
    final_target_len = math.ceil(total_samples * latent_to_audio_ratio)
    return final_feat[:, :, :final_target_len]


# ===========================================================================
# Data proxy (ported from daVinci-MagiHuman inference/pipeline/data_proxy.py)
# ===========================================================================
def _unfold_3d(x: torch.Tensor, kernel_size: tuple[int, int, int], stride: tuple[int, int, int]) -> torch.Tensor:
    """Pure-PyTorch 3D unfold matching UnfoldAnd behavior.

    After N unfold ops the shape is (batch, C, oD, oH, oW, kD, kH, kW).
    UnfoldAnd permutes kernel dims next to channel before reshape so that the
    col_dim axis is ordered as (C, kD, kH, kW) -- matching F.unfold semantics.
    Without this permute, .view() interleaves spatial and kernel positions.

    Args:
        x: (N, C, D, H, W)
        kernel_size: (kD, kH, kW)
        stride: (sD, sH, sW)
    Returns:
        (N, C*kD*kH*kW, L) where L = product of output spatial dims.
    """
    ndim = len(kernel_size)
    for d in range(ndim):
        x = x.unfold(d + 2, kernel_size[d], stride[d])
    perm = [0, 1] + list(range(ndim + 2, 2 * ndim + 2)) + list(range(2, ndim + 2))
    x = x.permute(*perm).contiguous()

    batch_size = x.shape[0]
    col_dim = 1
    for i in range(1, ndim + 2):
        col_dim *= x.shape[i]
    spatial = 1
    for i in range(ndim + 2, 2 * ndim + 2):
        spatial *= x.shape[i]
    return x.view(batch_size, col_dim, spatial)


def _calc_local_qk_range(num_video_tokens, num_audio_and_txt_tokens, num_frames, frame_receptive_field):
    token_per_frame = num_video_tokens // num_frames
    total_tokens = num_video_tokens + num_audio_and_txt_tokens

    q_range_list = []
    k_range_list = []
    for i in range(num_frames):
        q_range_list.append(torch.tensor([i * token_per_frame, (i + 1) * token_per_frame]))
        k_range_list.append(
            torch.tensor(
                [
                    (i - frame_receptive_field) * token_per_frame,
                    (i + frame_receptive_field + 1) * token_per_frame,
                ]
            )
        )
    local_q_range = torch.stack(q_range_list, dim=0)
    local_k_range = torch.stack(k_range_list, dim=0)

    local_k_range[local_k_range < 0] = 0
    local_k_range[local_k_range > num_video_tokens] = num_video_tokens

    video_q_range = torch.tensor([[0, num_video_tokens]])
    video_k_range = torch.tensor([[num_video_tokens, num_video_tokens + num_audio_and_txt_tokens]])

    at_q_ranges = torch.tensor([[num_video_tokens, total_tokens]])
    at_k_ranges = torch.tensor([[0, total_tokens]])

    q_ranges = (
        torch.cat([local_q_range, video_q_range, at_q_ranges], dim=0).to(torch.int32).to("cuda", non_blocking=True)
    )
    k_ranges = (
        torch.cat([local_k_range, video_k_range, at_k_ranges], dim=0).to(torch.int32).to("cuda", non_blocking=True)
    )
    return q_ranges, k_ranges


def _calc_local_attn_ffa_handler(num_video_tokens, num_audio_and_txt_tokens, num_frames, frame_receptive_field):
    q_ranges, k_ranges = _calc_local_qk_range(
        num_video_tokens, num_audio_and_txt_tokens, num_frames, frame_receptive_field
    )
    total = num_video_tokens + num_audio_and_txt_tokens
    return FFAHandler(
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        max_seqlen_q=total,
        max_seqlen_k=total,
        attn_type_map=torch.zeros([q_ranges.shape[0]], device="cuda", dtype=torch.int32),
        softmax_scale=None,
    )


def _get_coords(
    shape: list[int],
    ref_feat_shape: list[int],
    offset_thw: list[int] | None = None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
):
    if offset_thw is None:
        offset_thw = [0, 0, 0]
    ori_t, ori_h, ori_w = shape
    ref_t, ref_h, ref_w = ref_feat_shape

    offset_t, offset_h, offset_w = offset_thw
    time_rng = torch.arange(ori_t, device=device, dtype=dtype) + offset_t
    height_rng = torch.arange(ori_h, device=device, dtype=dtype) + offset_h
    width_rng = torch.arange(ori_w, device=device, dtype=dtype) + offset_w

    time_grid, height_grid, width_grid = torch.meshgrid(time_rng, height_rng, width_rng, indexing="ij")
    coords_flat = torch.stack([time_grid, height_grid, width_grid], dim=-1).reshape(-1, 3)

    meta = torch.tensor([ori_t, ori_h, ori_w, ref_t, ref_h, ref_w], device=device, dtype=dtype)
    meta_expanded = meta.expand(coords_flat.size(0), -1)
    return torch.cat([coords_flat, meta_expanded], dim=-1)


@dataclass
class _SingleData:
    video_x_t: torch.Tensor
    audio_x_t: torch.Tensor
    audio_feat_len: int
    txt_feat: torch.Tensor
    txt_feat_len: int
    t: int
    h: int
    w: int
    patch_size: int
    t_patch_size: int
    spatial_rope_interpolation: Literal["inter", "extra"]
    ref_audio_offset: int
    text_offset: int
    coords_style: Literal["v1", "v2"] = "v1"

    def __post_init__(self):
        self.video_token_num = self.video_x_t.shape[0]
        self.audio_x_t = self.audio_x_t[: self.audio_feat_len]
        self.txt_feat = self.txt_feat[: self.txt_feat_len]
        self.video_channel = self.video_x_t.shape[-1]
        self.audio_channel = self.audio_x_t.shape[-1]
        self.txt_channel = self.txt_feat.shape[-1]

    @property
    def device(self):
        return self.video_x_t.device

    @property
    def default_dtype(self):
        return self.video_x_t.dtype

    @property
    def total_token_num(self):
        return self.video_token_num + self.audio_feat_len + self.txt_feat_len

    @property
    def token_sequence(self):
        tensors = [self.video_x_t, self.audio_x_t, self.txt_feat]
        max_channel = max(t.shape[-1] for t in tensors)
        padded = [F.pad(t, (0, max_channel - t.shape[-1])) for t in tensors]
        return torch.cat(padded, dim=0)

    @property
    def modality_mapping(self):
        v_map = torch.full((self.video_token_num,), Modality.VIDEO, dtype=torch.int64, device=self.device)
        a_map = torch.full((self.audio_feat_len,), Modality.AUDIO, dtype=torch.int64, device=self.device)
        t_map = torch.full((self.txt_feat_len,), Modality.TEXT, dtype=torch.int64, device=self.device)
        return torch.cat([v_map, a_map, t_map], dim=0)

    def _default_coords(self, shape, ref_feat_shape, offset_thw=None):
        if offset_thw is None:
            offset_thw = [0, 0, 0]
        return _get_coords(
            shape=shape,
            ref_feat_shape=ref_feat_shape,
            offset_thw=offset_thw,
            device=self.device,
            dtype=self.default_dtype,
        )

    @property
    def coords_mapping(self):
        if self.spatial_rope_interpolation == "inter":
            video_ref_feat_shape = (self.t // self.t_patch_size, 32, 32)
        else:
            video_ref_feat_shape = (self.t // self.t_patch_size, self.h // self.patch_size, self.w // self.patch_size)

        video_coords = self._default_coords(
            shape=(self.t // self.t_patch_size, self.h // self.patch_size, self.w // self.patch_size),
            ref_feat_shape=video_ref_feat_shape,
        )

        if self.coords_style == "v1":
            audio_coords = self._default_coords(
                shape=(self.audio_feat_len, 1, 1),
                ref_feat_shape=(self.t // self.t_patch_size, 1, 1),
            )
            text_coords = self._default_coords(
                shape=(self.txt_feat_len, 1, 1),
                ref_feat_shape=(2, 1, 1),
                offset_thw=[self.text_offset, 0, 0],
            )
        elif self.coords_style == "v2":
            magic_audio_ref_t = (self.audio_feat_len - 1) // 4 + 1
            audio_coords = self._default_coords(
                shape=(self.audio_feat_len, 1, 1),
                ref_feat_shape=(magic_audio_ref_t // self.t_patch_size, 1, 1),
            )
            text_coords = self._default_coords(
                shape=(self.txt_feat_len, 1, 1),
                ref_feat_shape=(1, 1, 1),
                offset_thw=[-self.txt_feat_len, 0, 0],
            )
        else:
            raise ValueError(f"Unknown coords_style: {self.coords_style}")

        return torch.cat([video_coords, audio_coords, text_coords], dim=0)

    def depack_token_sequence(self, token_sequence):
        video_x_t = token_sequence[: self.video_token_num, : self.video_channel]
        video_x_t = rearrange(
            video_x_t,
            "(T H W) (pT pH pW C) -> C (T pT) (H pH) (W pW)",
            H=self.h // self.patch_size,
            W=self.w // self.patch_size,
            pT=self.t_patch_size,
            pH=self.patch_size,
            pW=self.patch_size,
        ).contiguous()
        audio_x_t = token_sequence[
            self.video_token_num : self.video_token_num + self.audio_feat_len, : self.audio_channel
        ]
        return video_x_t, audio_x_t


@dataclass
class _SimplePackedData:
    items: list[_SingleData]

    @property
    def token_sequence(self):
        return torch.cat([item.token_sequence for item in self.items], dim=0)

    @property
    def modality_mapping(self):
        return torch.cat([item.modality_mapping for item in self.items], dim=0)

    @property
    def coords_mapping(self):
        return torch.cat([item.coords_mapping for item in self.items], dim=0)

    @property
    def total_token_num(self):
        return sum(item.total_token_num for item in self.items)

    def __getitem__(self, index):
        return self.items[index]

    @property
    def cu_seqlen(self):
        cu = torch.cumsum(torch.tensor([item.total_token_num for item in self.items]), dim=0)
        return F.pad(cu, (1, 0))

    @property
    def max_seqlen(self):
        return torch.tensor(max(item.total_token_num for item in self.items))

    def depack_token_sequence(self, token_sequence):
        video_list, audio_list = [], []
        parts = torch.split(token_sequence, [item.total_token_num for item in self.items], dim=0)
        for item, part in zip(self.items, parts):
            v, a = item.depack_token_sequence(part)
            video_list.append(v)
            audio_list.append(a)
        return torch.stack(video_list, dim=0), torch.stack(audio_list, dim=0)


class MagiDataProxy:
    def __init__(
        self,
        patch_size: int = 2,
        t_patch_size: int = 1,
        frame_receptive_field: int = 11,
        spatial_rope_interpolation: str = "extra",
        ref_audio_offset: int = 1000,
        text_offset: int = 0,
        coords_style: str = "v2",
    ):
        self.patch_size = patch_size
        self.t_patch_size = t_patch_size
        self.frame_receptive_field = frame_receptive_field
        self.spatial_rope_interpolation = spatial_rope_interpolation
        self.ref_audio_offset = ref_audio_offset
        self.text_offset = text_offset
        self.coords_style = coords_style
        self._kernel = (t_patch_size, patch_size, patch_size)
        self._stride = (t_patch_size, patch_size, patch_size)
        self._saved_data: dict[str, Any] = {}

    def saved_for_output(self, **kwargs):
        self._saved_data.update(kwargs)

    def get_saved_data(self, key: str):
        return self._saved_data[key]

    def img2tokens(self, x_t: torch.Tensor):
        x_t_unfolded = _unfold_3d(x_t, self._kernel, self._stride)
        return rearrange(x_t_unfolded, "N col_dim num_tokens -> N num_tokens col_dim").contiguous()

    def process_input(self, transported_data: EvalInput):
        batch_size, _, t, h, w = transported_data.x_t.shape
        x_t = self.img2tokens(transported_data.x_t)
        audio_x_t = transported_data.audio_x_t.contiguous()
        text_in = transported_data.txt_feat.contiguous()

        simple_packed_data = _SimplePackedData(items=[])
        for i in range(batch_size):
            single_data = _SingleData(
                video_x_t=x_t[i],
                audio_x_t=audio_x_t[i],
                audio_feat_len=transported_data.audio_feat_len[i],
                txt_feat=text_in[i],
                txt_feat_len=transported_data.txt_feat_len[i],
                t=t,
                h=h,
                w=w,
                patch_size=self.patch_size,
                t_patch_size=self.t_patch_size,
                spatial_rope_interpolation=self.spatial_rope_interpolation,
                ref_audio_offset=self.ref_audio_offset,
                text_offset=self.text_offset,
                coords_style=self.coords_style,
            )
            simple_packed_data.items.append(single_data)

        if self.frame_receptive_field != -1:
            assert batch_size == 1, "local attention only supports batch size 1"
            local_attn_handler = _calc_local_attn_ffa_handler(
                num_video_tokens=simple_packed_data[0].video_token_num,
                num_audio_and_txt_tokens=simple_packed_data[0].audio_feat_len + simple_packed_data[0].txt_feat_len,
                num_frames=t,
                frame_receptive_field=self.frame_receptive_field,
            )
            if isinstance(local_attn_handler.max_seqlen_k, torch.Tensor):
                local_attn_handler.max_seqlen_k = local_attn_handler.max_seqlen_k.item()
            if isinstance(local_attn_handler.max_seqlen_q, torch.Tensor):
                local_attn_handler.max_seqlen_q = local_attn_handler.max_seqlen_q.item()
        else:
            local_attn_handler = None

        varlen_handler = VarlenHandler(
            cu_seqlens_q=simple_packed_data.cu_seqlen.to(torch.int32).cuda(),
            cu_seqlens_k=simple_packed_data.cu_seqlen.to(torch.int32).cuda(),
            max_seqlen_q=simple_packed_data.max_seqlen.to(torch.int32).cuda(),
            max_seqlen_k=simple_packed_data.max_seqlen.to(torch.int32).cuda(),
        )

        self.saved_for_output(simple_packed_data=simple_packed_data)

        x = simple_packed_data.token_sequence
        coords_mapping = simple_packed_data.coords_mapping
        modality_mapping = simple_packed_data.modality_mapping
        return (x, coords_mapping, modality_mapping, varlen_handler, local_attn_handler)

    def process_output(self, x: torch.Tensor):
        simple_packed_data: _SimplePackedData = self.get_saved_data("simple_packed_data")
        return simple_packed_data.depack_token_sequence(x)


# ===========================================================================
# Pipeline helpers
# ===========================================================================
@dataclass
class EvalInput:
    x_t: torch.Tensor
    audio_x_t: torch.Tensor
    audio_feat_len: torch.Tensor | list[int]
    txt_feat: torch.Tensor
    txt_feat_len: torch.Tensor | list[int]


class _T5GemmaEncoder:
    def __init__(self, model_path: str, device: str, weight_dtype: torch.dtype, subfolder: str | None = None):
        from vllm.distributed import get_tensor_model_parallel_world_size

        self.device = device
        hf_kwargs: dict[str, Any] = {}
        if subfolder is not None:
            hf_kwargs["subfolder"] = subfolder
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **hf_kwargs)

        tp_size = get_tensor_model_parallel_world_size()
        if tp_size > 1:
            from transformers.models.t5gemma.configuration_t5gemma import T5GemmaConfig

            config = T5GemmaConfig.from_pretrained(model_path, **hf_kwargs)
            # The config we need is the encoder config
            config_encoder = config.encoder
            # Propagate some outer config values
            config_encoder.vocab_size = config.vocab_size
            config_encoder.rms_norm_eps = getattr(config, "rms_norm_eps", config_encoder.rms_norm_eps)
            self.model = T5GemmaEncoderModelTP(config_encoder).to(device).to(weight_dtype)
            self.is_tp = True
        else:
            self.model = T5GemmaEncoderModel.from_pretrained(
                model_path, is_encoder_decoder=False, dtype=weight_dtype, **hf_kwargs
            ).to(device)
            self.is_tp = False

    @torch.inference_mode()
    def encode(self, prompt: str) -> torch.Tensor:
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        if self.is_tp:
            # T5GemmaEncoderModelTP just returns the hidden states tensor
            return outputs.half()
        else:
            # HF model returns BaseModelOutput
            return outputs["last_hidden_state"].half()


def _pad_or_trim(tensor: torch.Tensor, target_size: int, dim: int, pad_value: float = 0.0) -> tuple[torch.Tensor, int]:
    current_size = tensor.size(dim)
    if current_size < target_size:
        padding_amount = target_size - current_size
        padding_tuple = [0] * (2 * tensor.dim())
        padding_dim_index = tensor.dim() - 1 - dim
        padding_tuple[2 * padding_dim_index + 1] = padding_amount
        return F.pad(tensor, tuple(padding_tuple), "constant", pad_value), current_size
    slicing = [slice(None)] * tensor.dim()
    slicing[dim] = slice(0, target_size)
    return tensor[tuple(slicing)], target_size


def _get_padded_t5_gemma_embedding(
    prompt: str,
    encoder: _T5GemmaEncoder,
    target_length: int,
) -> tuple[torch.Tensor, int]:
    txt_feat = encoder.encode(prompt)
    txt_feat, original_len = _pad_or_trim(txt_feat, target_size=target_length, dim=1)
    return txt_feat.to(torch.float32), original_len


def _resizecrop(img: Image.Image, target_height: int, target_width: int) -> Image.Image:
    """Centre-crop resize keeping aspect ratio then letterbox to target."""
    pil_image = img.convert("RGB")
    original_width, original_height = pil_image.size
    scale_x = target_width / original_width
    scale_y = target_height / original_height
    scale = max(scale_x, scale_y)
    new_width = int(round(original_width * scale))
    new_height = int(round(original_height * scale))
    resized_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    return resized_image.crop((left, top, left + target_width, top + target_height))


class ZeroSNRDDPMDiscretization:
    """ZeroSNR DDPM sigma schedule, ported from daVinci-MagiHuman.
    Used to compute sigma values for SR noise injection.
    """

    def __init__(
        self,
        linear_start: float = 0.00085,
        linear_end: float = 0.0120,
        num_timesteps: int = 1000,
        shift_scale: float = 1.0,
        keep_start: bool = False,
        post_shift: bool = False,
    ):
        from functools import partial

        if keep_start and not post_shift:
            linear_start = linear_start / (shift_scale + (1 - shift_scale) * linear_start)
        self.num_timesteps = num_timesteps
        betas = torch.linspace(linear_start**0.5, linear_end**0.5, num_timesteps, dtype=torch.float64) ** 2
        alphas = 1.0 - betas.cpu().numpy()
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.to_torch = partial(torch.tensor, dtype=torch.float32)
        if not post_shift:
            self.alphas_cumprod = self.alphas_cumprod / (shift_scale + (1 - shift_scale) * self.alphas_cumprod)
        self.post_shift = post_shift
        self.shift_scale = shift_scale

    def __call__(
        self,
        n: int,
        do_append_zero: bool = True,
        device: str = "cpu",
        flip: bool = False,
        return_idx: bool = False,
    ):
        from functools import partial

        if n < self.num_timesteps:
            timesteps = np.linspace(self.num_timesteps - 1, 0, n, endpoint=False).astype(int)[::-1]
            alphas_cumprod = self.alphas_cumprod[timesteps]
        elif n == self.num_timesteps:
            alphas_cumprod = self.alphas_cumprod
        else:
            raise ValueError(f"n={n} > num_timesteps={self.num_timesteps}")

        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        alphas_cumprod = to_torch(alphas_cumprod)
        alphas_cumprod_sqrt = alphas_cumprod.sqrt()
        alphas_cumprod_sqrt_0 = alphas_cumprod_sqrt[0].clone()
        alphas_cumprod_sqrt_T = alphas_cumprod_sqrt[-1].clone()
        alphas_cumprod_sqrt -= alphas_cumprod_sqrt_T
        alphas_cumprod_sqrt *= alphas_cumprod_sqrt_0 / (alphas_cumprod_sqrt_0 - alphas_cumprod_sqrt_T)

        if self.post_shift:
            alphas_cumprod_sqrt = (
                alphas_cumprod_sqrt**2 / (self.shift_scale + (1 - self.shift_scale) * alphas_cumprod_sqrt**2)
            ) ** 0.5

        sigmas = torch.flip(alphas_cumprod_sqrt, (0,))
        sigmas = torch.cat([sigmas, sigmas.new_zeros([1])]) if do_append_zero else sigmas
        if return_idx:
            return sigmas if not flip else torch.flip(sigmas, (0,)), timesteps
        return sigmas if not flip else torch.flip(sigmas, (0,))


def _schedule_latent_step(
    *,
    video_scheduler: FlowUniPCMultistepScheduler,
    audio_scheduler: FlowUniPCMultistepScheduler,
    latent_video: torch.Tensor,
    latent_audio: torch.Tensor,
    t,
    idx: int,
    steps,
    v_cfg_video: torch.Tensor,
    v_cfg_audio: torch.Tensor,
    is_a2v: bool,
    cfg_number: int,
    using_sde_flag: bool,
    use_sr_model: bool = False,
):
    # Fast DDIM path for cfg_number==1, only used during the BR stage
    if cfg_number == 1 and not use_sr_model:
        latent_video = video_scheduler.step_ddim(v_cfg_video, idx, latent_video)
        latent_audio = audio_scheduler.step_ddim(v_cfg_audio, idx, latent_audio)
        return latent_video, latent_audio

    if using_sde_flag:
        if use_sr_model:
            # SR stage with SDE: only update video, keep audio unchanged
            latent_video = video_scheduler.step(v_cfg_video, t, latent_video, return_dict=False)[0]
            return latent_video, latent_audio
        if idx < int(len(steps) * (3 / 4)):
            noise_theta = 1.0 if (idx + 1) % 2 == 0 else 0.0
        else:
            noise_theta = 1.0 if idx % 3 == 0 else 0.0
        latent_video = video_scheduler.step_sde(v_cfg_video, idx, latent_video, noise_theta=noise_theta)
        if not is_a2v:
            latent_audio = audio_scheduler.step_sde(v_cfg_audio, idx, latent_audio, noise_theta=noise_theta)
        return latent_video, latent_audio

    latent_video = video_scheduler.step(v_cfg_video, t, latent_video, return_dict=False)[0]
    # Do not update audio latent during the SR stage
    if not is_a2v and not use_sr_model:
        latent_audio = audio_scheduler.step(v_cfg_audio, t, latent_audio, return_dict=False)[0]
    return latent_video, latent_audio


_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
    "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, walking backwards"
    ", low quality, worst quality, poor quality, noise, background noise, hiss, hum, buzz, crackle, static, "
    "compression artifacts, MP3 artifacts, digital clipping, distortion, muffled, muddy, unclear, echo, "
    "reverb, room echo, over-reverberated, hollow sound, distant, washed out, harsh, shrill, piercing, "
    "grating, tinny, thin sound, boomy, bass-heavy, flat EQ, over-compressed, abrupt cut, jarring transition, "
    "sudden silence, looping artifact, music, instrumental, sirens, alarms, crowd noise, unrelated sound "
    "effects, chaotic, disorganized, messy, cheap sound"
    ", emotionless, flat delivery, deadpan, lifeless, apathetic, robotic, mechanical, monotone, flat "
    "intonation, undynamic, boring, reading from a script, AI voice, synthetic, text-to-speech, TTS, "
    "insincere, fake emotion, exaggerated, overly dramatic, melodramatic, cheesy, cringey, hesitant, "
    "unconfident, tired, weak voice, stuttering, stammering, mumbling, slurred speech, mispronounced, "
    "bad articulation, lisp, vocal fry, creaky voice, mouth clicks, lip smacks, wet mouth sounds, heavy "
    "breathing, audible inhales, plosives, p-pops, coughing, clearing throat, sneezing, speaking too fast, "
    "rushed, speaking too slow, dragged out, unnatural pauses, awkward silence, choppy, disjointed, multiple "
    "speakers, two voices, background talking, out of tune, off-key, autotune artifacts"
)


# ===========================================================================
# Pre/post process funcs (registered in registry)
# ===========================================================================
def get_magi_human_pre_process_func(*args, **kwargs):
    def pre_process(request: OmniDiffusionRequest):
        return request

    return pre_process


def get_magi_human_post_process_func(*args, **kwargs):
    def post_process(output):
        if isinstance(output, tuple) and len(output) == 2:
            video, audio = output
            return {
                "video": video,
                "audio": audio,
                "audio_sample_rate": 44100,
                "fps": 25,
            }
        return output

    return post_process


# ===========================================================================
# HF Hub / local path helpers
# ===========================================================================


def _resolve_subdir(
    model_path: str,
    subfolder: str,
    local_files_only: bool = True,
    required_files: list[str] | None = None,
) -> str:
    """Resolve a model subfolder to a local directory path.

    For HF Hub repos, downloads all ``required_files`` (default: ``["config.json"]``)
    into the HF cache and returns the parent directory.
    """
    if local_files_only:
        return os.path.join(model_path, subfolder)
    from huggingface_hub import hf_hub_download

    files = required_files or ["config.json"]
    last_cached: str | None = None
    for fname in files:
        last_cached = hf_hub_download(repo_id=model_path, filename=f"{subfolder}/{fname}")
    return os.path.dirname(last_cached)


# ===========================================================================
# Main Pipeline
# ===========================================================================
class MagiHumanPipeline(nn.Module, ProgressBarMixin, DiffusionPipelineProfilerMixin):
    def __init__(self, od_config: OmniDiffusionConfig, **kwargs):
        super().__init__()
        model_path = od_config.model
        local_files_only = os.path.exists(model_path)
        device = f"cuda:{torch.cuda.current_device()}"
        self.device_str = device
        self.dtype = od_config.dtype or torch.bfloat16

        model_index = _load_json(model_path, "model_index.json", local_files_only)
        eval_cfg = model_index
        dp_cfg = model_index.get("data_proxy", {})

        dit_subfolder = "transformer"

        dit_json = _load_json(model_path, f"{dit_subfolder}/config.json", local_files_only)
        dit_model_config = MagiHumanDiTConfig(**dit_json)

        self.dit = DiTModel(dit_model_config)
        self.dit.eval()

        self.vae = DistributedAutoencoderKLWan.from_pretrained(model_path, subfolder="vae")
        self.vae.to(device)
        self.vae.eval()
        vae_cfg = _load_json(model_path, "vae/config.json", local_files_only)
        self.vae_latent_mean = torch.tensor(vae_cfg["latents_mean"], dtype=torch.float32)
        self.vae_latent_std = torch.tensor(vae_cfg["latents_std"], dtype=torch.float32)

        self.audio_vae = SAAudioFeatureExtractor(
            device=device,
            model_path=_resolve_subdir(
                model_path,
                "audio_vae",
                local_files_only,
                required_files=["config.json", "model_config.json", "model.safetensors"],
            ),
        )

        logger.info("Loading T5Gemma text encoder from %s (subfolder=text_encoder)", model_path)
        if local_files_only:
            txt_enc_path = os.path.join(model_path, "text_encoder")
            txt_enc_subfolder = None
        else:
            txt_enc_path = model_path
            txt_enc_subfolder = "text_encoder"
        self.text_encoder = _T5GemmaEncoder(
            model_path=txt_enc_path,
            device=device,
            weight_dtype=self.dtype,
            subfolder=txt_enc_subfolder,
        )

        self.data_proxy = MagiDataProxy(
            patch_size=dp_cfg.get("patch_size", 2),
            t_patch_size=dp_cfg.get("t_patch_size", 1),
            frame_receptive_field=dp_cfg.get("frame_receptive_field", 11),
            spatial_rope_interpolation=dp_cfg.get("spatial_rope_interpolation", "extra"),
            ref_audio_offset=dp_cfg.get("ref_audio_offset", 1000),
            text_offset=dp_cfg.get("text_offset", 0),
            coords_style=dp_cfg.get("coords_style", "v2"),
        )
        # SR DataProxy forces v1 coordinate style (consistent with the original)
        self.sr_data_proxy = MagiDataProxy(
            patch_size=dp_cfg.get("patch_size", 2),
            t_patch_size=dp_cfg.get("t_patch_size", 1),
            frame_receptive_field=dp_cfg.get("frame_receptive_field", 11),
            spatial_rope_interpolation=dp_cfg.get("spatial_rope_interpolation", "extra"),
            ref_audio_offset=dp_cfg.get("ref_audio_offset", 1000),
            text_offset=dp_cfg.get("text_offset", 0),
            coords_style="v1",
        )

        self.fps = eval_cfg.get("fps", 25)
        self.num_inference_steps_default = eval_cfg.get("num_inference_steps", 32)
        self.video_txt_guidance_scale = eval_cfg.get("video_txt_guidance_scale", 5.0)
        self.audio_txt_guidance_scale = eval_cfg.get("audio_txt_guidance_scale", 5.0)
        self.shift = eval_cfg.get("shift", 5.0)
        self.cfg_number = eval_cfg.get("cfg_number", 2)
        self.use_cfg_trick = eval_cfg.get("use_cfg_trick", True)
        self.cfg_trick_start_frame = eval_cfg.get("cfg_trick_start_frame", 13)
        self.cfg_trick_value = eval_cfg.get("cfg_trick_value", 2.0)
        self.using_sde_flag = eval_cfg.get("using_sde_flag", False)
        self.t5_gemma_target_length = eval_cfg.get("t5_gemma_target_length", 640)
        self.vae_stride = eval_cfg.get("vae_stride", [4, 16, 16])
        self.z_dim = eval_cfg.get("z_dim", 48)
        self.patch_size = eval_cfg.get("patch_size", [1, 2, 2])
        # SR-specific hyperparameters
        self.sr_num_inference_steps_default = eval_cfg.get("sr_num_inference_steps", 5)
        self.sr_cfg_number = eval_cfg.get("sr_cfg_number", 2)
        self.sr_video_txt_guidance_scale = eval_cfg.get("sr_video_txt_guidance_scale", 3.5)
        self.noise_value = eval_cfg.get("noise_value", 220)
        self.sr_audio_noise_scale = eval_cfg.get("sr_audio_noise_scale", 0.7)
        # ZeroSNR sigma schedule for SR noise injection (flip=True, high to low)
        self.zerosnr_sigmas = ZeroSNRDDPMDiscretization()(1000, do_append_zero=False, flip=True)

        self.context_null, self.original_context_null_len = _get_padded_t5_gemma_embedding(
            _NEGATIVE_PROMPT,
            self.text_encoder,
            self.t5_gemma_target_length,
        )
        self.video_processor = VideoProcessor(vae_scale_factor=16)

        # SR DiT model (loaded from the sr/ subdirectory)
        sr_dit_subfolder = "sr"
        sr_dit_json = _load_json(model_path, f"{sr_dit_subfolder}/config.json", local_files_only)
        sr_dit_model_config = MagiHumanDiTConfig(**sr_dit_json)
        self.sr_dit = DiTModel(sr_dit_model_config)
        self.sr_dit.eval()

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model_path,
                subfolder=dit_subfolder,
                revision=None,
                prefix="dit.",
                fall_back_to_pt=True,
            ),
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model_path,
                subfolder=sr_dit_subfolder,
                revision=None,
                prefix="sr_dit.",
                fall_back_to_pt=True,
            ),
        ]
        if getattr(self.text_encoder, "is_tp", False):
            self.weights_sources.append(
                DiffusersPipelineLoader.ComponentSource(
                    model_or_path=model_path,
                    subfolder="text_encoder",
                    revision=None,
                    prefix="text_encoder.",
                    fall_back_to_pt=True,
                ),
            )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Weight loading for MagiHuman DiT with TP support.
        #
        # The checkpoint stores weights with these naming patterns:
        #   - attention.linear_qkv.weight: fused [Q, K, V, G] for shared layers,
        #     or stacked per-expert [expert0_Q|K|V|G, expert1_..., expert2_...] for MoE.
        #   - attention.linear_proj.weight: single for shared, stacked per-expert for MoE.
        #   - mlp.up_gate_proj.weight / mlp.down_proj.weight: similarly stacked for MoE.
        #
        # The model now uses per-expert vLLM parallel layers for MoE blocks:
        #   attention.linear_qkv.experts.{i}.weight  (QKVParallelLinear per expert)
        #   attention.linear_gating.experts.{i}.weight  (ColumnParallelLinear per expert)
        #   attention.linear_proj.experts.{i}.weight  (RowParallelLinear per expert)
        #   mlp.up_gate_proj.experts.{i}.weight  (ColumnParallelLinear per expert)
        #   mlp.down_proj.experts.{i}.weight  (RowParallelLinear per expert)
        #
        # Shared layers keep the same naming (no .experts.).
        params_dict = dict(self.named_parameters())
        modules_dict = dict(self.named_modules())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            # ── Text Encoder weights ──
            if name.startswith("text_encoder."):
                if getattr(self.text_encoder, "is_tp", False):
                    # Strip "text_encoder." prefix for the T5Gemma TP model
                    # The T5GemmaEncoderModelTP load_weights handles the "encoder." prefix itself
                    sub_name = name[len("text_encoder.") :]
                    loaded_params.update(
                        f"text_encoder.{k}" for k in self.text_encoder.model.load_weights([(sub_name, loaded_weight)])
                    )
                else:
                    loaded_params.add(name)
                continue

            # ── Shared attention QKV + Gating split ──
            # Checkpoint: attention.linear_qkv.weight = [Q, K, V, G] fused.
            # Model: attention.linear_qkv.weight (QKVParallelLinear) + attention.linear_gating.weight.
            if "attention.linear_qkv.weight" in name:
                gating_name = name.replace("attention.linear_qkv.weight", "attention.linear_gating.weight")
                # Check if this is a shared layer (direct param exists, no .experts.)
                if name in params_dict and gating_name in params_dict:
                    qkv_param = params_dict[name]
                    gating_param = params_dict[gating_name]

                    mod_path = name[: -len(".weight")]
                    qkv_mod = modules_dict.get(mod_path)
                    if qkv_mod is not None and hasattr(qkv_mod, "total_num_heads"):
                        total_heads_q = qkv_mod.total_num_heads
                        total_heads_kv = qkv_mod.total_num_kv_heads
                        head_dim = qkv_mod.head_size
                    else:
                        head_dim = 128
                        tp_size = get_tensor_model_parallel_world_size()
                        total_heads_q = gating_param.data.shape[0] * tp_size
                        total_heads_kv = (loaded_weight.shape[0] - total_heads_q * head_dim - total_heads_q) // (
                            2 * head_dim
                        )

                    q_size = total_heads_q * head_dim
                    kv_size = total_heads_kv * head_dim

                    q_w = loaded_weight[:q_size]
                    k_w = loaded_weight[q_size : q_size + kv_size]
                    v_w = loaded_weight[q_size + kv_size : q_size + 2 * kv_size]
                    g_w = loaded_weight[q_size + 2 * kv_size :]

                    qkv_loader = getattr(qkv_param, "weight_loader", default_weight_loader)
                    qkv_loader(qkv_param, q_w, "q")
                    qkv_loader(qkv_param, k_w, "k")
                    qkv_loader(qkv_param, v_w, "v")

                    gating_loader = getattr(gating_param, "weight_loader", default_weight_loader)
                    gating_loader(gating_param, g_w)

                    loaded_params.add(name)
                    loaded_params.add(gating_name)
                    continue

                # ── MoE attention QKV + Gating split ──
                # Checkpoint: attention.linear_qkv.weight = stacked [expert0_QKVG, expert1_QKVG, ...].
                # Model: attention.linear_qkv.experts.{i}.weight (QKVParallelLinear per expert)
                #       + attention.linear_gating.experts.{i}.weight (ColumnParallelLinear per expert).
                expert0_name = name.replace("attention.linear_qkv.weight", "attention.linear_qkv.experts.0.weight")
                if expert0_name in params_dict:
                    # Determine num_experts by checking which expert indices exist.
                    moe_qkv_mod_path = name[: -len(".weight")]
                    moe_qkv_mod = modules_dict.get(moe_qkv_mod_path)
                    num_experts = moe_qkv_mod.num_experts if moe_qkv_mod is not None else 3

                    # Get head info from the first expert's QKVParallelLinear.
                    expert0_mod_path = name.replace("attention.linear_qkv.weight", "attention.linear_qkv.experts.0")
                    expert0_mod = modules_dict.get(expert0_mod_path)
                    if expert0_mod is not None and hasattr(expert0_mod, "total_num_heads"):
                        total_heads_q = expert0_mod.total_num_heads
                        total_heads_kv = expert0_mod.total_num_kv_heads
                        head_dim = expert0_mod.head_size
                    else:
                        head_dim = 128
                        # Infer from checkpoint weight shape.
                        # We'll get exact sizes from model config below.
                        total_heads_q = 40  # fallback for default config
                        total_heads_kv = 8

                    q_size = total_heads_q * head_dim
                    kv_size = total_heads_kv * head_dim
                    # Check if gating is present.
                    gating_expert0_name = name.replace(
                        "attention.linear_qkv.weight", "attention.linear_gating.experts.0.weight"
                    )
                    has_gating = gating_expert0_name in params_dict

                    # Split stacked checkpoint weight into per-expert chunks.
                    expert_weights = loaded_weight.chunk(num_experts, dim=0)

                    for i in range(num_experts):
                        expert_w = expert_weights[i]
                        # Each expert chunk: [Q, K, V, G (optional)].
                        q_w = expert_w[:q_size]
                        k_w = expert_w[q_size : q_size + kv_size]
                        v_w = expert_w[q_size + kv_size : q_size + 2 * kv_size]

                        expert_param_name = name.replace(
                            "attention.linear_qkv.weight",
                            f"attention.linear_qkv.experts.{i}.weight",
                        )
                        expert_param = params_dict[expert_param_name]
                        expert_loader = getattr(expert_param, "weight_loader", default_weight_loader)
                        expert_loader(expert_param, q_w, "q")
                        expert_loader(expert_param, k_w, "k")
                        expert_loader(expert_param, v_w, "v")
                        loaded_params.add(expert_param_name)

                        if has_gating:
                            g_w = expert_w[q_size + 2 * kv_size :]
                            gating_param_name = name.replace(
                                "attention.linear_qkv.weight",
                                f"attention.linear_gating.experts.{i}.weight",
                            )
                            gating_param = params_dict[gating_param_name]
                            gating_loader = getattr(gating_param, "weight_loader", default_weight_loader)
                            gating_loader(gating_param, g_w)
                            loaded_params.add(gating_param_name)
                    continue

            # ── MoE stacked weight splitting for proj / MLP layers ──
            # Checkpoint: x.y.weight (stacked [expert0, expert1, ...]).
            # Model: x.y.experts.{i}.weight.
            if name not in params_dict:
                # Check if this is a stacked MoE weight by looking for .experts.0.
                base, _, suffix = name.rpartition(".")
                expert0_name = f"{base}.experts.0.{suffix}" if base else None
                if expert0_name and expert0_name in params_dict:
                    # Determine num_experts.
                    moe_mod = modules_dict.get(base)
                    num_experts = getattr(moe_mod, "num_experts", 3) if moe_mod is not None else 3

                    # Split stacked weight into per-expert chunks.
                    expert_weights = loaded_weight.chunk(num_experts, dim=0)
                    for i in range(num_experts):
                        expert_param_name = f"{base}.experts.{i}.{suffix}"
                        if expert_param_name not in params_dict:
                            continue
                        expert_param = params_dict[expert_param_name]
                        expert_loader = getattr(expert_param, "weight_loader", default_weight_loader)
                        expert_loader(expert_param, expert_weights[i])
                        loaded_params.add(expert_param_name)
                    continue
                # Truly unknown weight — skip.
                continue

            # ── Standard weight loading (shared layers + non-MoE params) ──
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        if getattr(self.text_encoder, "is_tp", False):
            self.context_null, self.original_context_null_len = _get_padded_t5_gemma_embedding(
                _NEGATIVE_PROMPT,
                self.text_encoder,
                self.t5_gemma_target_length,
            )

        return loaded_params

    def _dit_forward(self, eval_input: EvalInput) -> tuple[torch.Tensor, torch.Tensor]:
        packed = self.data_proxy.process_input(eval_input)
        noise_pred = self.dit(*packed)
        return self.data_proxy.process_output(noise_pred)

    def _sr_dit_forward(self, eval_input: EvalInput) -> tuple[torch.Tensor, torch.Tensor]:
        """SR stage uses sr_data_proxy (coords_style=v1) and sr_dit model."""
        packed = self.sr_data_proxy.process_input(eval_input)
        noise_pred = self.sr_dit(*packed)
        return self.sr_data_proxy.process_output(noise_pred)

    @torch.inference_mode()
    def _evaluate_with_latent(
        self,
        context: torch.Tensor,
        original_context_len: int,
        latent_image: torch.Tensor | None,
        latent_video: torch.Tensor,
        latent_audio: torch.Tensor,
        num_inference_steps: int,
        is_a2v: bool = False,
        use_sr_model: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Select cfg_number and guidance_scale based on BR/SR stage
        cfg_number = self.sr_cfg_number if use_sr_model else self.cfg_number
        video_guidance = self.sr_video_txt_guidance_scale if use_sr_model else self.video_txt_guidance_scale
        forward_fn = self._sr_dit_forward if use_sr_model else self._dit_forward

        video_scheduler = FlowUniPCMultistepScheduler()
        audio_scheduler = FlowUniPCMultistepScheduler()
        video_scheduler.set_timesteps(num_inference_steps, device=self.device_str, shift=self.shift)
        audio_scheduler.set_timesteps(num_inference_steps, device=self.device_str, shift=self.shift)
        timesteps = video_scheduler.timesteps

        latent_length = latent_video.shape[2]
        cfg_trick_guidance = (
            torch.tensor(video_guidance, device=self.device_str).expand(1, 1, latent_length, 1, 1).clone()
        )
        if self.use_cfg_trick:
            cfg_trick_guidance[:, :, : self.cfg_trick_start_frame] = min(self.cfg_trick_value, video_guidance)

        with self.progress_bar(total=len(timesteps)) as pbar:
            for idx, t in enumerate(timesteps):
                if latent_image is not None:
                    latent_video[:, :, :1] = latent_image[:, :, :1]

                # Reduce guidance when t<=500 during BR stage (original behavior)
                cur_video_guidance = video_guidance if (use_sr_model or t > 500) else 2.0

                eval_input_cond = EvalInput(
                    x_t=latent_video,
                    audio_x_t=latent_audio,
                    audio_feat_len=[latent_audio.shape[1]],
                    txt_feat=context,
                    txt_feat_len=[original_context_len],
                )

                v_cond_video, v_cond_audio = forward_fn(eval_input_cond)

                if cfg_number == 1:
                    v_cfg_video = v_cond_video
                    v_cfg_audio = v_cond_audio
                elif cfg_number == 2:
                    eval_input_uncond = EvalInput(
                        x_t=latent_video,
                        audio_x_t=latent_audio,
                        audio_feat_len=[latent_audio.shape[1]],
                        txt_feat=self.context_null,
                        txt_feat_len=[self.original_context_null_len],
                    )
                    v_uncond_video, v_uncond_audio = forward_fn(eval_input_uncond)
                    v_cfg_video = v_uncond_video + cur_video_guidance * (v_cond_video - v_uncond_video)
                    v_cfg_audio = v_uncond_audio + self.audio_txt_guidance_scale * (v_cond_audio - v_uncond_audio)
                else:
                    raise ValueError(f"Invalid cfg_number: {cfg_number}")

                latent_video, latent_audio = _schedule_latent_step(
                    video_scheduler=video_scheduler,
                    audio_scheduler=audio_scheduler,
                    latent_video=latent_video,
                    latent_audio=latent_audio,
                    t=t,
                    idx=idx,
                    steps=timesteps,
                    v_cfg_video=v_cfg_video,
                    v_cfg_audio=v_cfg_audio,
                    is_a2v=is_a2v,
                    cfg_number=cfg_number,
                    using_sde_flag=self.using_sde_flag,
                    use_sr_model=use_sr_model,
                )

                pbar.update()

        if latent_image is not None:
            latent_video[:, :, :1] = latent_image[:, :, :1]
        return latent_video, latent_audio

    def _encode_image(self, image: Image.Image, height: int, width: int) -> torch.Tensor:
        image = load_image(image)
        image = _resizecrop(image, height, width)
        image = self.video_processor.preprocess(image, height=height, width=width)
        image = image.to(device=self.device_str, dtype=self.dtype).unsqueeze(2)
        vae_out = self.vae.encode(image)
        if hasattr(vae_out, "latent_dist"):
            return vae_out.latent_dist.mode().to(torch.float32)
        return vae_out.to(torch.float32)

    def _decode_video(self, latent: torch.Tensor) -> list[np.ndarray]:
        mean = self.vae_latent_mean.to(latent.device, dtype=latent.dtype).view(1, -1, 1, 1, 1)
        std = self.vae_latent_std.to(latent.device, dtype=latent.dtype).view(1, -1, 1, 1, 1)
        latent = latent * std + mean

        videos = self.vae.decode(latent.to(self.dtype))
        if hasattr(videos, "sample"):
            videos = videos.sample
        videos.mul_(0.5).add_(0.5).clamp_(0, 1)
        videos = [v.float().cpu().permute(1, 2, 3, 0) * 255 for v in videos]
        return [v.numpy().astype(np.uint8) for v in videos]

    def _decode_audio(self, latent_audio: torch.Tensor) -> np.ndarray:
        latent_audio = latent_audio.squeeze(0).to(self.dtype)
        audio_output = self.audio_vae.decode(latent_audio.T)
        audio_np = audio_output.squeeze(0).T.float().cpu().numpy()
        target_len = int(audio_np.shape[0] * 441 / 512)
        from scipy.signal import resample

        return resample(audio_np, target_len)

    @torch.inference_mode()
    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | None = None,
        height: int = 256,
        width: int = 448,
        num_inference_steps: int | None = None,
        seconds: int = 10,
        seed: int | None = None,
        image_path: str | None = None,
        audio_path: str | None = None,
        **kwargs,
    ) -> DiffusionOutput:
        if len(req.prompts) >= 1:
            p = req.prompts[0]
            prompt = p if isinstance(p, str) else p.get("prompt", prompt)
            if not isinstance(p, str):
                image_path = p.get("image_path", image_path)
                audio_path = p.get("audio_path", audio_path)
        if prompt is None:
            raise ValueError("prompt is required")

        height = req.sampling_params.height or height
        width = req.sampling_params.width or width
        seed = req.sampling_params.seed if req.sampling_params.seed is not None else seed
        num_steps = req.sampling_params.num_inference_steps or num_inference_steps or self.num_inference_steps_default
        sr_height: int | None = None
        sr_width: int | None = None
        sr_num_steps: int | None = None
        if hasattr(req.sampling_params, "extra_args") and req.sampling_params.extra_args:
            seconds = req.sampling_params.extra_args.get("seconds", seconds)
            audio_path = req.sampling_params.extra_args.get("audio_path", audio_path)
            image_path = req.sampling_params.extra_args.get("image_path", image_path)
            sr_height = req.sampling_params.extra_args.get("sr_height", None)
            sr_width = req.sampling_params.extra_args.get("sr_width", None)
            sr_num_steps = req.sampling_params.extra_args.get("sr_num_inference_steps", None)

        device = self.device_str

        br_latent_height = height // self.vae_stride[1] // self.patch_size[1] * self.patch_size[1]
        br_latent_width = width // self.vae_stride[2] // self.patch_size[2] * self.patch_size[2]
        br_height = br_latent_height * self.vae_stride[1]
        br_width = br_latent_width * self.vae_stride[2]

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        if audio_path is not None:
            latent_audio = load_audio_and_encode(self.audio_vae, audio_path, seconds)
            latent_audio = latent_audio.permute(0, 2, 1)
            num_frames = latent_audio.shape[1]
            is_a2v = True
        else:
            num_frames = seconds * self.fps + 1
            latent_audio = torch.randn(1, num_frames, 64, dtype=torch.float32, device=device)
            is_a2v = False

        latent_length = (num_frames - 1) // 4 + 1
        latent_video = torch.randn(
            1,
            self.z_dim,
            latent_length,
            br_latent_height,
            br_latent_width,
            dtype=torch.float32,
            device=device,
        )

        context, original_context_len = _get_padded_t5_gemma_embedding(
            prompt,
            self.text_encoder,
            self.t5_gemma_target_length,
        )

        if image_path is not None:
            br_image = self._encode_image(load_image(image_path), br_height, br_width)
        else:
            br_image = None

        # ── BR stage ─────────────────────────────────────────────────────────
        br_latent_video, br_latent_audio = self._evaluate_with_latent(
            context,
            original_context_len,
            br_image,
            latent_video.clone(),
            latent_audio.clone(),
            num_steps,
            is_a2v,
            use_sr_model=False,
        )

        # ── SR stage (optional, triggered when sr_height/sr_width are provided) ──
        if sr_height is not None and sr_width is not None:
            sr_latent_height = sr_height // self.vae_stride[1] // self.patch_size[1] * self.patch_size[1]
            sr_latent_width = sr_width // self.vae_stride[2] // self.patch_size[2] * self.patch_size[2]
            sr_height = sr_latent_height * self.vae_stride[1]
            sr_width = sr_latent_width * self.vae_stride[2]

            # Image condition (at SR resolution)
            if image_path is not None:
                sr_image = self._encode_image(load_image(image_path), sr_height, sr_width)
            else:
                sr_image = None

            # Trilinear interpolation of BR latent to SR resolution
            sr_latent_video = torch.nn.functional.interpolate(
                br_latent_video,
                size=(latent_length, sr_latent_height, sr_latent_width),
                mode="trilinear",
                align_corners=True,
            )

            # Noise injection: sigma-weighted blend (noise_value indexes the ZeroSNR sigma schedule)
            if self.noise_value != 0:
                noise = torch.randn_like(sr_latent_video)
                sigma = self.zerosnr_sigmas.to(sr_latent_video.device)[self.noise_value]
                sr_latent_video = sr_latent_video * sigma + noise * (1 - sigma**2) ** 0.5

            # Audio: blend with noise (noised version used during SR inference; final audio keeps BR result)
            sr_latent_audio = torch.randn_like(br_latent_audio) * self.sr_audio_noise_scale + br_latent_audio * (
                1 - self.sr_audio_noise_scale
            )

            torch.cuda.empty_cache()
            sr_steps = sr_num_steps or self.sr_num_inference_steps_default
            final_latent_video, _ = self._evaluate_with_latent(
                context,
                original_context_len,
                sr_image,
                sr_latent_video.clone(),
                sr_latent_audio.clone(),
                sr_steps,
                is_a2v,
                use_sr_model=True,
            )
            # SR stage does not update audio; keep the BR result
            final_latent_video = final_latent_video
            final_latent_audio = br_latent_audio
        else:
            final_latent_video = br_latent_video
            final_latent_audio = br_latent_audio

        torch.cuda.empty_cache()
        videos_np = self._decode_video(final_latent_video)
        torch.cuda.empty_cache()
        audio_np = self._decode_audio(final_latent_audio)

        return DiffusionOutput(output=(videos_np, audio_np))
