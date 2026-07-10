# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LTX-2.3 guidance math and pass metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class _LTX23GuidanceParams:
    video_cfg_scale: float
    audio_cfg_scale: float
    video_stg_scale: float
    audio_stg_scale: float
    video_modality_scale: float
    audio_modality_scale: float
    video_rescale_scale: float
    audio_rescale_scale: float
    video_stg_blocks: tuple[int, ...]
    audio_stg_blocks: tuple[int, ...]

    @property
    def do_cfg(self) -> bool:
        return self.video_cfg_scale != 1.0 or self.audio_cfg_scale != 1.0

    @property
    def do_stg(self) -> bool:
        return (self.video_stg_scale != 0.0 and bool(self.video_stg_blocks)) or (
            self.audio_stg_scale != 0.0 and bool(self.audio_stg_blocks)
        )

    @property
    def do_modality_guidance(self) -> bool:
        return self.video_modality_scale != 1.0 or self.audio_modality_scale != 1.0

    @property
    def do_rescale(self) -> bool:
        return self.video_rescale_scale != 0.0 or self.audio_rescale_scale != 0.0


def denoise_pass_count(guidance_params: _LTX23GuidanceParams) -> int:
    return 1 + int(guidance_params.do_cfg) + int(guidance_params.do_stg) + int(guidance_params.do_modality_guidance)


def x0_from_noise(sample: torch.Tensor, noise_pred: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    calc_dtype = torch.float32
    sigma = sigma.to(calc_dtype) if isinstance(sigma, torch.Tensor) else sigma
    return (sample.to(calc_dtype) - noise_pred.to(calc_dtype) * sigma).to(sample.dtype)


def noise_from_x0(sample: torch.Tensor, x0: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    calc_dtype = torch.float32
    sigma = sigma.to(calc_dtype) if isinstance(sigma, torch.Tensor) else sigma
    return ((sample.to(calc_dtype) - x0.to(calc_dtype)) / sigma).to(sample.dtype)


def euler_step_from_velocity(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigmas: torch.Tensor,
    step_index: int,
) -> torch.Tensor:
    sigma = sigmas[step_index].to(torch.float32)
    sigma_next = sigmas[step_index + 1].to(torch.float32)
    dt = sigma_next - sigma
    return (sample.to(torch.float32) + velocity.to(torch.float32) * dt).to(sample.dtype)


def rescale_guided_x0(cond: torch.Tensor, pred: torch.Tensor, rescale_scale: float) -> torch.Tensor:
    if rescale_scale == 0.0:
        return pred
    factor = cond.float().std() / pred.float().std()
    factor = rescale_scale * factor + (1 - rescale_scale)
    return pred * factor


def combine_guided_x0(
    *,
    cond: torch.Tensor,
    uncond_text: torch.Tensor | float,
    uncond_perturbed: torch.Tensor | float,
    uncond_modality: torch.Tensor | float,
    cfg_scale: float,
    stg_scale: float,
    modality_scale: float,
    rescale_scale: float,
) -> torch.Tensor:
    dtype = cond.dtype
    cond = cond.float()
    uncond_text = uncond_text.float() if isinstance(uncond_text, torch.Tensor) else uncond_text
    uncond_perturbed = uncond_perturbed.float() if isinstance(uncond_perturbed, torch.Tensor) else uncond_perturbed
    uncond_modality = uncond_modality.float() if isinstance(uncond_modality, torch.Tensor) else uncond_modality
    pred = (
        cond
        + (cfg_scale - 1) * (cond - uncond_text)
        + stg_scale * (cond - uncond_perturbed)
        + (modality_scale - 1) * (cond - uncond_modality)
    )
    pred = rescale_guided_x0(cond, pred, rescale_scale)
    return pred.to(dtype)


def repeat_batch_tensor(tensor: torch.Tensor, repeats: int) -> torch.Tensor:
    return tensor.repeat((repeats,) + (1,) * (tensor.ndim - 1))


def mask_for_pass(
    *,
    pass_count: int,
    batch_size: int,
    pass_index: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    mask = torch.ones((pass_count * batch_size, 1, 1), device=device, dtype=dtype)
    start = pass_index * batch_size
    mask[start : start + batch_size] = 0
    return mask


def build_ltx23_perturbation_kwargs(
    *,
    pass_names: list[str],
    batch_size: int,
    guidance_params: _LTX23GuidanceParams,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    perturbation_kwargs: dict[str, Any] = {}
    pass_count = len(pass_names)
    if "ptb" in pass_names:
        ptb_index = pass_names.index("ptb")
        if guidance_params.video_stg_scale != 0.0 and guidance_params.video_stg_blocks:
            perturbation_kwargs["video_self_attention_mask"] = mask_for_pass(
                pass_count=pass_count,
                batch_size=batch_size,
                pass_index=ptb_index,
                device=device,
                dtype=dtype,
            )
            perturbation_kwargs["video_self_attention_blocks"] = guidance_params.video_stg_blocks
        if guidance_params.audio_stg_scale != 0.0 and guidance_params.audio_stg_blocks:
            perturbation_kwargs["audio_self_attention_mask"] = mask_for_pass(
                pass_count=pass_count,
                batch_size=batch_size,
                pass_index=ptb_index,
                device=device,
                dtype=dtype,
            )
            perturbation_kwargs["audio_self_attention_blocks"] = guidance_params.audio_stg_blocks

    if "mod" in pass_names:
        mod_index = pass_names.index("mod")
        mod_mask = mask_for_pass(
            pass_count=pass_count,
            batch_size=batch_size,
            pass_index=mod_index,
            device=device,
            dtype=dtype,
        )
        perturbation_kwargs["a2v_cross_attention_mask"] = mod_mask
        perturbation_kwargs["v2a_cross_attention_mask"] = mod_mask

    return perturbation_kwargs
