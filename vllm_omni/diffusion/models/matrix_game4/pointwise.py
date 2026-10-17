# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compile tensor-only normalization chains; keep GEMMs and Attention shared."""

from collections.abc import Callable
from functools import lru_cache
from typing import NamedTuple

import torch
import torch.nn.functional as F


def modulate(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    normalized = F.layer_norm(x.float(), (x.shape[-1],), eps=1e-6).to(x.dtype)
    return normalized * (1 + scale) + shift


def residual_norm(
    x: torch.Tensor, residual: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = x + residual * gate
    normalized = F.layer_norm(hidden.float(), (hidden.shape[-1],), weight.float(), bias.float(), 1e-6)
    return normalized.to(x.dtype), hidden


def residual_modulate(
    x: torch.Tensor, residual: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = x + residual
    return modulate(hidden, scale, shift), hidden


def gated_residual(x: torch.Tensor, residual: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return x + residual * gate


def rms(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    value = x.float()
    normalized = (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype)
    return normalized * weight


def rms_rope(x: torch.Tensor, weight: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    normalized = rms(x, weight).reshape(x.shape[0], x.shape[1], 24, 64, 2).float()
    real, imag = normalized[..., 0], normalized[..., 1]
    cosine, sine = rotation.real.float(), rotation.imag.float()
    return torch.stack((real * cosine - imag * sine, imag * cosine + real * sine), dim=-1).flatten(-2).to(x.dtype)


class MatrixOps(NamedTuple):
    modulate: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    residual_norm: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
    ]
    residual_modulate: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
    ]
    gated_residual: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    rms: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    rms_rope: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@lru_cache(maxsize=1)
def compiled_ops() -> MatrixOps:
    # Preserve materialized BF16 rounding points and disable FP contraction.
    # PyTorch owns kernel generation and its compilation cache.
    return MatrixOps(
        *(
            torch.compile(fn, fullgraph=True, options={"emulate_precision_casts": True})
            for fn in (modulate, residual_norm, residual_modulate, gated_residual, rms, rms_rope)
        )
    )
