# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Image DiT denoise loops for Flux3 sampling.

One explicit ``denoise_<mode>`` function per image generation mode, each a short
rectified-flow Euler loop over the DiT. Both modes share two micro-helpers:

- ``_cfg_step``: split the CFG-doubled prediction, blend, take one Euler step, re-double.
- ``_stream_timesteps``: broadcast a scalar timestep across a stream's tokens.

The latent accumulator stays in its sampled dtype (fp32) and is downcast to the
model's compute dtype only at the ``model(...)`` boundary.
"""

import torch
from torch import Tensor, nn


def _cfg_step(x: Tensor, pred: Tensor, t_curr: Tensor, t_prev: Tensor, cfg: float) -> Tensor:
    pred_uc, pred_c = pred.chunk(2)
    v = pred_uc + cfg * (pred_c - pred_uc)
    half = x.chunk(2)[0] + (t_prev - t_curr) * v
    return torch.cat([half, half], 0)


def _stream_timesteps(x: Tensor, t: Tensor) -> Tensor:
    return t * torch.ones(x.shape[:2], device=x.device, dtype=x.dtype)


def denoise_t2i(
    model: nn.Module,
    x_image: Tensor,
    x_image_ids: Tensor,
    ctx: Tensor,
    ctx_ids: Tensor,
    vector: Tensor,
    timesteps_ctx: Tensor,
    timesteps: Tensor,
    cfg: float,
    pag_single_layers: list[int] | None = None,
    pag_strength: float = 1.0,
) -> Tensor:
    compute_dtype = next(model.parameters()).dtype
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:], strict=True):
        x_in = x_image.to(compute_dtype)
        pred = model(
            x_image=x_in,
            x_image_ids=x_image_ids,
            x_image_timesteps=_stream_timesteps(x_in, t_curr),
            ctx=ctx,
            ctx_ids=ctx_ids,
            timesteps_ctx=timesteps_ctx,
            vector=vector,
            pag_single_layers=pag_single_layers,
            pag_strength=pag_strength,
        )["x_image"]
        x_image = _cfg_step(x_image, pred, t_curr, t_prev, cfg)
    return x_image.chunk(2)[-1]


def denoise_i2i(
    model: nn.Module,
    x_image: Tensor,
    x_image_ids: Tensor,
    x_image_cond: Tensor,
    x_image_cond_ids: Tensor,
    x_image_cond_timesteps: Tensor,
    ctx: Tensor,
    ctx_ids: Tensor,
    vector: Tensor,
    timesteps_ctx: Tensor,
    timesteps: Tensor,
    cfg: float,
    pag_single_layers: list[int] | None = None,
    pag_strength: float = 1.0,
) -> Tensor:
    compute_dtype = next(model.parameters()).dtype
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:], strict=True):
        x_in = x_image.to(compute_dtype)
        pred = model(
            x_image=x_in,
            x_image_ids=x_image_ids,
            x_image_timesteps=_stream_timesteps(x_in, t_curr),
            x_image_cond=x_image_cond,
            x_image_cond_ids=x_image_cond_ids,
            x_image_cond_timesteps=x_image_cond_timesteps,
            ctx=ctx,
            ctx_ids=ctx_ids,
            timesteps_ctx=timesteps_ctx,
            vector=vector,
            pag_single_layers=pag_single_layers,
            pag_strength=pag_strength,
        )["x_image"]
        x_image = _cfg_step(x_image, pred, t_curr, t_prev, cfg)
    return x_image.chunk(2)[-1]
