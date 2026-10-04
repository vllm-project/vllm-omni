# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-local CUDA JIT norms matching the frozen ZONOS2 reference.

FlashInfer 0.6 defaults to CuTe DSL norms. Selecting its CUDA JIT entrypoint
explicitly preserves the reference reduction/rounding without changing a
process-wide environment variable or affecting other models.
"""

from __future__ import annotations

import torch


@torch.library.custom_op("vllm_omni::zonos2_cuda_rmsnorm", mutates_args=())
def zonos2_cuda_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    from flashinfer.norm import get_norm_module

    output = torch.empty_like(x)
    get_norm_module().rmsnorm(output, x, weight, eps, False)
    return output


@zonos2_cuda_rmsnorm.register_fake
def _rmsnorm_fake(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("vllm_omni::zonos2_cuda_fused_add_rmsnorm", mutates_args=("x", "residual"))
def zonos2_cuda_fused_add_rmsnorm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
    from flashinfer.norm import get_norm_module

    get_norm_module().fused_add_rmsnorm(x, residual, weight, eps, False)


@zonos2_cuda_fused_add_rmsnorm.register_fake
def _fused_norm_fake(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
    pass
