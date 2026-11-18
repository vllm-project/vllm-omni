# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GELU fused with the MXFP8 activation quantization of the following linear layer.

A DiT feed-forward runs ``linear -> GELU -> linear``. With MXFP8 weights the second
linear quantizes its input, so the unfused path writes the GELU output in bf16 and
reads it back once more to quantize it (two passes over the widest activation of the
block). ``fused_gelu_quant`` produces the quantized input directly, as the
``QuantizedActivation`` the vLLM MXFP8 linear kernels accept, in one pass.
"""

import torch
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

try:
    from vllm.model_executor.layers.fusion.quant_activation import (
        QuantizedActivation,
        get_input_quant_key,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import kMxfp8Dynamic
except ImportError:  # older vLLM without the fusion contract
    QuantizedActivation = None  # type: ignore[assignment]
    get_input_quant_key = None  # type: ignore[assignment]
    kMxfp8Dynamic = None  # noqa: N816

_MX_GROUP = 32


def _xpu_fused_gelu_mxfp8_available() -> bool:
    if not current_omni_platform.is_xpu():
        return False
    try:
        import vllm_xpu_kernels._C  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch.ops._C, "fused_gelu_mxfp8_quant")


_FUSED_GELU_MXFP8 = _xpu_fused_gelu_mxfp8_available()
_logged_active = False


def fused_gelu_quant(
    next_linear: torch.nn.Module,
    x: torch.Tensor,
    approximate: str = "none",
) -> "QuantizedActivation | torch.Tensor":
    """GELU(x) quantized for ``next_linear`` when that layer takes MXFP8 input.

    Returns the ``QuantizedActivation`` the layer's kernel consumes, or the plain
    bf16/fp16 GELU output when the fused kernel does not apply (other platforms,
    non-MX quantization, unsupported shape), so callers can use it unconditionally.
    """
    if (
        not _FUSED_GELU_MXFP8
        or get_input_quant_key is None
        or get_input_quant_key(next_linear) != kMxfp8Dynamic
        or x.dtype not in (torch.bfloat16, torch.float16)
        or x.shape[-1] % _MX_GROUP != 0
    ):
        return F.gelu(x, approximate=approximate)
    global _logged_active
    if not _logged_active:
        _logged_active = True
        logger.info("fused GELU + MXFP8 activation quantization active for %s", type(next_linear).__name__)
    x2 = x.reshape(-1, x.shape[-1])
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    num_tokens, hidden = x2.shape
    out = torch.empty((num_tokens, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    # The SYCL-TLA MXFP8 GEMM reads block-major ([K/32, M]) activation scales
    # directly; the linear kernel records which layout its GEMM wants.
    transposed = bool(getattr(next_linear, "mxfp8_block_major_a_scale", False))
    if transposed:
        scales = torch.empty((hidden // _MX_GROUP, num_tokens), dtype=torch.float8_e8m0fnu, device=x.device).t()
    else:
        scales = torch.empty((num_tokens, hidden // _MX_GROUP), dtype=torch.float8_e8m0fnu, device=x.device)
    torch.ops._C.fused_gelu_mxfp8_quant(out, x2, scales, _MX_GROUP, approximate, transposed)
    return QuantizedActivation(
        data=out,
        scale=scales,
        orig_dtype=x.dtype,
        orig_shape=x.shape,
        quant_key=kMxfp8Dynamic,
    )
