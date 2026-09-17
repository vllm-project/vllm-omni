# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Hardware dispatch for MiniMax H3 VAE operators."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import torch
from vllm.triton_utils import HAS_TRITON

from vllm_omni.platforms import current_omni_platform

from .qk_norm_rope import try_qk_norm_rope_exact
from .scaled_residual import try_scaled_residual_exact

MINIMAX_H3_VAE_EXACT_OPS_SM120_ENV = "VLLM_OMNI_MINIMAX_H3_VAE_EXACT_OPS_SM120"

QKNormRopeOp = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
        float,
    ],
    tuple[torch.Tensor, torch.Tensor] | None,
]
ScaledResidualOp = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor | None,
]


@dataclass(frozen=True)
class H3VAEOperatorSet:
    """One validated hardware implementation of the H3 VAE operations."""

    supports: Callable[[torch.device], bool]
    qk_norm_rope: QKNormRopeOp
    scaled_residual: ScaledResidualOp


def _supports_cuda_capability(device: torch.device, expected: int) -> bool:
    if (
        not HAS_TRITON
        or device.type != "cuda"
        or torch.version.hip is not None
        or not current_omni_platform.is_cuda()
        or not current_omni_platform.is_available()
    ):
        return False
    device_index = 0 if device.index is None else int(device.index)
    capability = current_omni_platform.get_device_capability(device_index)
    return capability is not None and int(capability.to_int()) == expected


def _supports_cuda_sm90(device: torch.device) -> bool:
    return _supports_cuda_capability(device, 90)


def _supports_cuda_sm100(device: torch.device) -> bool:
    return _supports_cuda_capability(device, 100)


def _supports_cuda_sm103(device: torch.device) -> bool:
    return _supports_cuda_capability(device, 103)


def resolve_minimax_h3_vae_exact_ops_sm120(raw: str | None = None) -> bool:
    """Resolve the experimental SM120 exact-op gate without silent fallback."""

    if raw is None:
        raw = os.environ.get(MINIMAX_H3_VAE_EXACT_OPS_SM120_ENV, "0")
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "off", "disable", "disabled", ""}:
        return False
    raise ValueError(f"{MINIMAX_H3_VAE_EXACT_OPS_SM120_ENV} must be 0 or 1, got {raw!r}")


def _supports_cuda_sm120(device: torch.device) -> bool:
    # Parse the opt-in only on the exact experimental target. An invalid value
    # must fail that worker at startup, but must not affect other architectures.
    return _supports_cuda_capability(device, 120) and resolve_minimax_h3_vae_exact_ops_sm120()


# Keep hardware selection flat: adding a backend means adding one operator set,
# without changing the installer or the model execution path.
H3_VAE_OPERATOR_TABLE: tuple[H3VAEOperatorSet, ...] = (
    H3VAEOperatorSet(
        supports=_supports_cuda_sm90,
        qk_norm_rope=try_qk_norm_rope_exact,
        scaled_residual=try_scaled_residual_exact,
    ),
    H3VAEOperatorSet(
        supports=_supports_cuda_sm100,
        qk_norm_rope=try_qk_norm_rope_exact,
        scaled_residual=try_scaled_residual_exact,
    ),
    H3VAEOperatorSet(
        supports=_supports_cuda_sm103,
        qk_norm_rope=try_qk_norm_rope_exact,
        scaled_residual=try_scaled_residual_exact,
    ),
    H3VAEOperatorSet(
        supports=_supports_cuda_sm120,
        qk_norm_rope=try_qk_norm_rope_exact,
        scaled_residual=try_scaled_residual_exact,
    ),
)


def resolve_h3_vae_operators(device: torch.device) -> H3VAEOperatorSet | None:
    for operators in H3_VAE_OPERATOR_TABLE:
        if operators.supports(device):
            return operators
    return None


__all__ = [
    "MINIMAX_H3_VAE_EXACT_OPS_SM120_ENV",
    "H3VAEOperatorSet",
    "H3_VAE_OPERATOR_TABLE",
    "resolve_minimax_h3_vae_exact_ops_sm120",
    "resolve_h3_vae_operators",
]
