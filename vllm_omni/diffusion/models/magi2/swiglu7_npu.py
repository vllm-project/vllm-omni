# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2026 SandAI. All Rights Reserved.

"""Opt-in Ascend clipped-SwiGLU7 for MAGI-2 Preview, selected by MAGI2_FUSED_SWIGLU7."""

from __future__ import annotations

import os

import torch

from vllm_omni.diffusion.layers.swiglu7 import SwiGLU7


def fused_swiglu7_enabled() -> bool:
    return os.environ.get("MAGI2_FUSED_SWIGLU7") == "1"


def _fused_applies(
    x: torch.Tensor,
    alpha: float,
    limit: float,
    out_dtype: torch.dtype | None,
) -> bool:
    return (
        fused_swiglu7_enabled()
        and x.device.type == "npu"
        and x.dtype == torch.bfloat16
        and out_dtype in (None, x.dtype)
        and x.ndim == 2
        and x.numel() > 0
        and x.shape[-1] % 2 == 0
        and x.is_contiguous()
        and (alpha, limit) == (1.702, 7.0)
        and not torch.compiler.is_compiling()
        and not (torch.is_grad_enabled() and x.requires_grad)
    )


class Magi2SwiGLU7(SwiGLU7):
    """``SwiGLU7`` with MAGI-2's opt-in Ascend kernel; unchanged on other backends."""

    def forward_npu(
        self,
        x: torch.Tensor,
        alpha: float = 1.702,
        limit: float = 7.0,
        out_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if not _fused_applies(x, alpha, limit, out_dtype):
            return self.forward_native(x, alpha, limit, out_dtype)

        import torch_npu

        # Gate and up arrive in adjacent columns; bias=1.0 is the native formula's (up + 1).
        return torch_npu.npu_clipped_swiglu(x, alpha=alpha, limit=limit, bias=1.0, interleaved=True)


__all__ = ["Magi2SwiGLU7", "fused_swiglu7_enabled"]
