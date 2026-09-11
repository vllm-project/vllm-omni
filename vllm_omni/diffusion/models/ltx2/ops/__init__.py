# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Shared operator selection for LTX-2 eager kernels and their tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import torch
from vllm.triton_utils import HAS_TRITON
from vllm.utils.import_utils import has_tilelang

from vllm_omni.platforms import current_omni_platform

# Pointwise fusions are launch/memory-traffic reductions whose first execution
# for every runtime key is checked bit-for-bit; failures use the ordinary model
# math. The FNA schedule is performance-tuned for SM90 and has no silent
# fallback, so qualify the two paths independently.
_FUSION_CUDA_COMPUTE_CAPABILITIES = frozenset({90, 100, 103})
_FNA_CUDA_COMPUTE_CAPABILITIES = frozenset({90})


@dataclass(frozen=True)
class LTX2VAEOperatorSet:
    qk_norm_rope: Callable[..., tuple[torch.Tensor, torch.Tensor] | None]
    residual_norm: Callable[..., torch.Tensor | None]
    residual_add: Callable[..., torch.Tensor | None]
    swiglu: Callable[..., torch.Tensor | None]
    fna: Callable[..., torch.Tensor] | None


def _fna3d_tilelang(*args, **kwargs):
    # Selecting an operator set must not import the optional compiler backend.
    from .fna import fna3d_tilelang

    return fna3d_tilelang(*args, **kwargs)


@cache
def _operator_set(with_fna: bool) -> LTX2VAEOperatorSet:
    from .qk_rms_norm import try_qk_rms_norm_scale_rope_3d_exact
    from .residual_adaln import try_residual_add3_exact, try_residual_rms_norm_modulate_exact
    from .swiglu import try_swiglu_tiled_exact

    return LTX2VAEOperatorSet(
        qk_norm_rope=try_qk_rms_norm_scale_rope_3d_exact,
        residual_norm=try_residual_rms_norm_modulate_exact,
        residual_add=try_residual_add3_exact,
        swiglu=try_swiglu_tiled_exact,
        fna=_fna3d_tilelang if with_fna else None,
    )


@cache
def _cuda_compute_capability(device_index: int) -> int | None:
    if not HAS_TRITON or not current_omni_platform.is_cuda() or not current_omni_platform.is_available():
        return None
    capability = current_omni_platform.get_device_capability(device_id=device_index)
    return capability.to_int() if capability is not None else None


def resolve_ltx2_vae_operators(device: torch.device) -> LTX2VAEOperatorSet | None:
    """Select the production operators, or return None for unsupported devices."""
    if device.type != "cuda" or not current_omni_platform.is_cuda() or not current_omni_platform.is_available():
        return None
    device_index = device.index if device.index is not None else torch.accelerator.current_device_index()
    capability = _cuda_compute_capability(int(device_index))
    if capability not in _FUSION_CUDA_COMPUTE_CAPABILITIES:
        return None
    return _operator_set(capability in _FNA_CUDA_COMPUTE_CAPABILITIES and has_tilelang())


def _eligible_operators(tensor: torch.Tensor) -> LTX2VAEOperatorSet | None:
    if not tensor.is_cuda or torch.compiler.is_compiling() or torch.is_grad_enabled():
        return None
    return resolve_ltx2_vae_operators(tensor.device)


def is_ltx2_fusion_eligible(tensor: torch.Tensor) -> bool:
    """Allow self-verifying pointwise fusions on qualified CUDA devices."""

    return _eligible_operators(tensor) is not None


def is_ltx2_fna_eligible(tensor: torch.Tensor) -> bool:
    """Keep the strict, Hopper-tuned TileLang FNA schedule on SM90."""

    operators = _eligible_operators(tensor)
    return operators is not None and operators.fna is not None


__all__ = ["LTX2VAEOperatorSet", "resolve_ltx2_vae_operators", "is_ltx2_fna_eligible", "is_ltx2_fusion_eligible"]
