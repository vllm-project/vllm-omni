# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight-only FP8 support for Ideogram-4 models.

This module provides FP8 (E4M3FN) weight-only quantization support that matches
Ideogram-4's offline quantization format:
    weight:      (out_features, in_features) float8_e4m3fn
    weight_scale: (out_features,) float32  (per-row / per-output-channel)
    bias:        (out_features,) compute_dtype (optional)
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    pass

FP8_E4M3_MAX = 448.0
FP8_WEIGHT_DTYPE = torch.float8_e4m3fn
FP8_SCALE_SUFFIX = ".weight_scale"


class Ideogram4Fp8Linear(nn.Module):
    """Linear layer with weight-only FP8 (E4M3FN) + per-row scale.

    Weight and scale are registered as buffers (not parameters) so they load
    via load_state_dict and are excluded from optimizer/grad machinery.

    The dequantized matmul runs in compute_dtype (e.g. bfloat16), so this
    needs no FP8 tensor-core hardware and works on any device that can
    store float8.
    """

    weight: torch.Tensor
    weight_scale: torch.Tensor
    bias: torch.Tensor | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype

        # Weight stored as FP8 E4M3FN
        self.register_buffer(
            "weight",
            torch.empty(out_features, in_features, dtype=FP8_WEIGHT_DTYPE),
        )
        # Per-row scale (per-output-channel)
        self.register_buffer(
            "weight_scale",
            torch.empty(out_features, dtype=torch.float32),
        )
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=compute_dtype))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize: weight_fp8 * scale
        # weight: (out, in), weight_scale: (out,)
        # Move weights to the same device as input
        w = self.weight.to(device=x.device, dtype=x.dtype)
        scale = self.weight_scale.to(device=x.device, dtype=x.dtype).unsqueeze(1)
        w = w * scale
        bias = self.bias.to(device=x.device, dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"weight_dtype={FP8_WEIGHT_DTYPE}, compute_dtype={self.compute_dtype}"
        )


def is_ideogram_fp8_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    """Check if checkpoint uses Ideogram-4's FP8 format.

    Returns True if any key ends with .weight_scale or any tensor has FP8 dtype.
    """
    return any(k.endswith(FP8_SCALE_SUFFIX) for k in state_dict) or any(
        v.dtype == FP8_WEIGHT_DTYPE for v in state_dict.values()
    )


def swap_linears_to_fp8(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    compute_dtype: torch.dtype = torch.bfloat16,
    prefix: str = "",
) -> None:
    """Replace each nn.Linear that has a saved FP8 scale with Ideogram4Fp8Linear.

    Gating on the presence of <name>.weight_scale means only layers that were
    actually quantized at save time are swapped; everything else loads normally.
    """
    for name, child in list(module.named_children()):
        child_prefix = f"{prefix}{name}"

        if isinstance(child, nn.Linear):
            scale_key = f"{child_prefix}{FP8_SCALE_SUFFIX}"
            if scale_key in state_dict:
                fp8_linear = Ideogram4Fp8Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=compute_dtype,
                )
                setattr(module, name, fp8_linear)
        else:
            swap_linears_to_fp8(child, state_dict, compute_dtype, prefix=f"{child_prefix}.")


def load_ideogram_fp8_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    assign: bool = True,
    strict: bool = True,
) -> None:
    """Load Ideogram-4 FP8 checkpoint into model.

    Model must already have its FP8 Linear layers swapped in (see swap_linears_to_fp8).
    FP8 weights are kept as float8, scales stay float32, and every other floating
    tensor is cast to dtype.

    Args:
        assign: Replace the module's tensors with the prepared ones rather than
                copying into them. Use True when model was built with from_config.
        strict: If True, raise on missing keys. If False, downgrade to warning.
    """
    prepared: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if v.dtype == FP8_WEIGHT_DTYPE:
            prepared[k] = v.to(device=device)
        elif k.endswith(FP8_SCALE_SUFFIX):
            prepared[k] = v.to(device=device, dtype=torch.float32)
        elif v.is_floating_point():
            prepared[k] = v.to(device=device, dtype=dtype)
        else:
            prepared[k] = v.to(device=device)

    missing, unexpected = model.load_state_dict(prepared, strict=False, assign=assign)
    if unexpected:
        raise RuntimeError(f"unexpected keys after fp8 load: {unexpected[:10]}")
    if missing and strict:
        raise RuntimeError(f"missing keys after fp8 load: {missing[:10]}")
    elif missing:
        warnings.warn(f"missing keys after fp8 load: {missing[:10]}", stacklevel=2)

    model.to(device)


def quantize_weight_to_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2-D Linear weight to e4m3 float8 with per-row scales.

    Returns (weight_fp8, scale) where weight_fp8 has shape (out, in) in
    float8_e4m3fn and scale has shape (out,) in float32 such that
    weight ≈ weight_fp8.to(dtype) * scale[:, None].
    """
    w = weight.detach().to(torch.float32)
    amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = amax / FP8_E4M3_MAX
    q = (w / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(FP8_WEIGHT_DTYPE)
    return q, scale.squeeze(1).to(torch.float32)
