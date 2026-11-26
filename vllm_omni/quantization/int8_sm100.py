# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""INT8 W8A8 GEMM fallback for SM100 (Blackwell) built on ``torch._int_mm``.

vLLM's CUTLASS INT8 kernel refuses SM100. This kernel keeps the CUTLASS weight
layout (``[K, N]`` view of a contiguous ``[N, K]`` tensor, which is the
column-major operand ``torch._int_mm`` needs), so ``process_weights_after_loading``
and the Cosmos3 MAPS strategy work unchanged. Only ``apply_weights`` differs.
"""

import torch
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    CutlassInt8ScaledMMLinearKernel,
)

logger = init_logger(__name__)

_INT_MM_MIN_M = 17  # torch._int_mm needs M > 16
_TRITON_MAX_K = 16384  # one Triton program quantizes one token row


def _triton_kernels():
    """Fused Triton quant and epilogue kernels, or None when Triton is unavailable."""
    try:
        from vllm_omni.quantization import int8_sm100_triton

        return int8_sm100_triton
    except Exception:  # pragma: no cover - Triton missing
        return None


def sm100_int8_fallback_needed() -> bool:
    """True when the current CUDA device is SM100 or newer."""
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 10


def quantize_activations_reference(
    x: torch.Tensor, input_scale: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric INT8 activation quantization in plain torch.

    Dynamic (``input_scale`` is None): per-token, ``q = round(x * (127 / amax))``.
    Static: per-tensor, ``q = round(x / input_scale)``.
    Returns ``(q int8 [M, K], scale float32 [M, 1] or [1, 1])``.
    """
    xf = x.float()
    if input_scale is None:
        amax = xf.abs().amax(dim=-1, keepdim=True).clamp(min=torch.finfo(torch.float32).tiny)
        q = torch.round(xf * (127.0 / amax)).clamp(-128, 127).to(torch.int8)
        return q, amax / 127.0
    scale = input_scale.reshape(1, 1).float()
    q = torch.round(xf / scale).clamp(-128, 127).to(torch.int8)
    return q, scale


def int8_epilogue(
    acc: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """``acc [M, N] int32`` to ``out_dtype``: ``acc * x_scale[M,1] * w_scale[1,N] + bias``."""
    out = acc.float() * x_scale.reshape(-1, 1).float() * w_scale.reshape(1, -1).float()
    if bias is not None:
        out = out + bias.float()
    return out.to(out_dtype)


class Int8Sm100ScaledMMLinearKernel(CutlassInt8ScaledMMLinearKernel):
    """``torch._int_mm`` INT8 kernel for SM100+. CUTLASS INT8 is not available there."""

    _quant_fallback_logged = False

    @classmethod
    def is_supported(cls, compute_capability: int | None = None) -> tuple[bool, str | None]:
        if not torch.cuda.is_available():
            return False, "requires CUDA."
        return True, None

    @classmethod
    def can_implement(cls, c) -> tuple[bool, str | None]:
        if not c.input_symmetric:
            return False, "SM100 INT8 fallback supports only symmetric activations."
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        w_q_name = self.layer_param_names[0]
        k, n = getattr(layer, w_q_name).shape
        if k % 8 or n % 8:
            raise ValueError(f"torch._int_mm needs K and N divisible by 8, got K={k}, N={n}")

    def fuses_pre_quant_scale(self, layer: torch.nn.Module, x: torch.Tensor) -> bool:
        """True when ``apply_weights`` multiplies ``layer.pre_quant_scale`` into ``x`` itself.

        Only the dynamic Triton quant fuses it; the static path quantizes with ``ops.scaled_int8_quant``.
        """
        i_s = self._get_layer_params(layer)[2]
        return _triton_kernels() is not None and i_s is None and x.shape[-1] <= _TRITON_MAX_K

    def _quantize(self, x: torch.Tensor, input_scale: torch.Tensor | None, pre_quant_scale=None):
        tk = _triton_kernels()
        if tk is not None and input_scale is None and x.shape[-1] <= _TRITON_MAX_K:
            return tk.dynamic_per_token_quant(x, pre_quant_scale)
        if pre_quant_scale is not None:
            x = x * pre_quant_scale.to(x.dtype)
        try:
            x_q, x_s, _ = ops.scaled_int8_quant(x, input_scale, None, symmetric=True)
            return x_q, x_s
        except Exception as exc:  # op unavailable for this arch
            if not Int8Sm100ScaledMMLinearKernel._quant_fallback_logged:
                logger.warning("ops.scaled_int8_quant failed on SM100 (%s); using torch reference.", exc)
                Int8Sm100ScaledMMLinearKernel._quant_fallback_logged = True
            return quantize_activations_reference(x, input_scale)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        pre_quant_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``pre_quant_scale`` ([K]) is multiplied into ``x`` first (SmoothQuant), fused when possible."""
        w_q, w_s, i_s, _, _ = self._get_layer_params(layer)
        x = x.contiguous()
        x_q, x_s = self._quantize(x, i_s, pre_quant_scale)
        m = x_q.shape[0]
        if m < _INT_MM_MIN_M:
            x_q = torch.nn.functional.pad(x_q, (0, 0, 0, _INT_MM_MIN_M - m))
        acc = torch._int_mm(x_q, w_q)[:m]
        tk = _triton_kernels()
        if tk is not None:
            return tk.int8_scale_epilogue(acc, x_s, w_s, bias, x.dtype)
        return int8_epilogue(acc, x_s, w_s, bias, x.dtype)
