# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in FP8 GEMMs for the token-wise linears of the AuK DiT blocks.

Weights are stored as FP8 E4M3 with one scale per tensor, activations are
cast to FP8 at unit scale, and ``torch._scaled_mm`` runs the GEMM. E4M3 is a
floating-point format, so its relative precision does not depend on how a
tensor is scaled as long as the values stay inside its range: on real
requests per-token, per-tensor and fixed activation scales all give the same
velocity error. The fixed unit scale is therefore used, because inside the
regionally compiled blocks the saturating cast then fuses with the op that
produces the activation and adds no kernel of its own.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

__all__ = ["Fp8Linear", "fp8_supported", "quantize_block_linears"]

_FP8 = torch.float8_e4m3fn
_FP8_MAX = float(torch.finfo(_FP8).max)


def fp8_supported(device: torch.device) -> bool:
    """FP8 GEMMs through ``torch._scaled_mm`` are validated on NVIDIA Ada and Hopper GPUs."""
    # ROCm also uses the CUDA namespace, but this E4M3 GEMM path is NVIDIA-only.
    if device.type != "cuda" or torch.version.hip is not None or not torch.cuda.is_available():
        return False
    return (8, 9) <= torch.cuda.get_device_capability(device) < (10, 0)


class Fp8Linear(nn.Module):
    """``nn.Linear`` over FP8 weights and FP8 activations, with the output in the weight's dtype."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        weight = linear.weight.detach().float()
        scale = weight.abs().amax().clamp_min(1e-12) / _FP8_MAX
        # _scaled_mm wants the second operand column-major: W^T as a view of the row-major weight.
        self.register_buffer("weight", (weight / scale).to(_FP8).t(), persistent=False)
        self.register_buffer("weight_scale", scale.reshape(()), persistent=False)
        self.register_buffer("input_scale", torch.ones((), device=weight.device, dtype=torch.float32), persistent=False)
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach().clone(), persistent=False)
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.out_dtype = linear.weight.dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = x.reshape(-1, self.in_features)
        # E4M3 has no infinity: saturate instead of overflowing into NaN.
        quantized = rows.clamp(-_FP8_MAX, _FP8_MAX).to(_FP8)
        # _scaled_mm only fuses the bias into half-precision outputs.
        fuse_bias = self.out_dtype != torch.float32
        out = torch._scaled_mm(
            quantized,
            self.weight,
            scale_a=self.input_scale,
            scale_b=self.weight_scale,
            bias=self.bias if fuse_bias else None,
            out_dtype=self.out_dtype,
        )
        if self.bias is not None and not fuse_bias:
            out = out + self.bias
        return out.reshape(*x.shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


def _block_linears(block: nn.Module) -> list[tuple[nn.Module, str]]:
    """``(parent, attribute)`` of every token-wise linear of a DiT block.

    The adaLN projections are left alone: they act on one timestep embedding
    per request, not on the token sequence.
    """
    attn = block.attn
    targets: list[tuple[nn.Module, str]] = [(attn, "to_qkv"), (attn.to_out, "0")]
    if hasattr(attn, "to_qkv_c"):
        targets += [(attn, "to_qkv_c"), (attn, "to_out_c")]
    for name in ("ff", "ff_c", "ff_x"):
        feed_forward = getattr(block, name, None)
        if feed_forward is not None:
            targets += [(feed_forward, "linear_in"), (feed_forward, "linear_out")]
    return targets


def _matches(name: str, pattern: str, match_mode: str) -> bool:
    if match_mode == "exact":
        return name == pattern
    if match_mode == "substring":
        return pattern in name
    if match_mode == "suffix":
        return name.endswith(pattern)
    raise ValueError(f"Unsupported ignored_layers match mode {match_mode!r}")


def quantize_block_linears(
    dit: nn.Module,
    ignored_layers: Sequence[str] = (),
    match_mode: str = "exact",
    prefix: str = "dit",
) -> int:
    """Swap the token-wise linears of every DiT block for :class:`Fp8Linear`; returns the count.

    The embeddings, the adaLN modulations and the output projection keep the
    model dtype. Linears named in ``ignored_layers`` (full names under
    ``prefix``, e.g. ``dit.transformer_blocks.0.attn.to_qkv``, matched as
    vLLM's ``Fp8Config`` does) also keep it; an entry that names no module of
    the DiT is rejected rather than ignored. Call this after the weights are
    loaded and on the inference device, and before any compilation or CUDA
    graph capture.
    """
    names = {id(module): f"{prefix}.{name}" if name else prefix for name, module in dit.named_modules()}
    unmatched = [p for p in ignored_layers if not any(_matches(n, p, match_mode) for n in names.values())]
    if unmatched:
        raise ValueError(f"ignored_layers {unmatched} match no module of the AuK DiT (match mode {match_mode!r})")
    count = 0
    for block in list(dit.transformer_blocks) + list(dit.single_transformer_blocks):
        for parent, attribute in _block_linears(block):
            linear = parent._modules[attribute]
            if any(_matches(names[id(linear)], p, match_mode) for p in ignored_layers):
                continue
            # The FP8 GEMM needs both feature dimensions to be multiples of 16.
            if isinstance(linear, nn.Linear) and linear.in_features % 16 == 0 and linear.out_features % 16 == 0:
                parent._modules[attribute] = Fp8Linear(linear)
                count += 1
    return count
