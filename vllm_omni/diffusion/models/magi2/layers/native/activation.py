# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2026 SandAI. All Rights Reserved.

"""Native MAGI-2 activation layers."""

from __future__ import annotations

import torch


def swiglu7(
    x: torch.Tensor,
    alpha: float = 1.702,
    limit: float = 7.0,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Released GPT-OSS-style clamped SwiGLU activation."""

    out_dtype = x.dtype if out_dtype is None else out_dtype
    x = x.to(torch.float32)
    gate, linear = x[..., ::2], x[..., 1::2]
    gate = gate.clamp(max=limit)
    linear = linear.clamp(min=-limit, max=limit)
    return (gate * torch.sigmoid(alpha * gate) * (linear + 1.0)).to(out_dtype)


__all__ = ["swiglu7"]
