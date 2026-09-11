# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""XPU-specific patches for upstream vLLM behavior."""

from vllm.logger import init_logger

logger = init_logger(__name__)


def _patch_w8a16_fp8_nd_activation() -> None:
    """Restore the N-D output shape for the XPU W8A16 FP8 linear kernel.

    ``XPUW8A16FP8LinearKernel.apply_weights`` hands its activation straight to
    ``torch.ops._xpu_C.fp8_gemm_w8a16``, skipping the flatten-then-reshape every
    other ScaledMM kernel does. The op's registered fake always returns 2-D, so a
    3-D activation makes inductor bake in a 2-D ``assert_size_stride`` that the
    rank-3 result trips, killing FLUX.1 ``--quantization fp8`` at startup warmup.
    Only >2-D inputs are affected, so LLM (token-flattened, 2-D) paths are not.
    """
    try:
        from vllm.model_executor.kernels.linear.scaled_mm.xpu import (
            XPUW8A16FP8LinearKernel,
        )
    except ImportError:
        logger.debug("XPUW8A16FP8LinearKernel not available; skipping FP8 shape patch.")
        return

    _original_apply_weights = XPUW8A16FP8LinearKernel.apply_weights

    def _patched_apply_weights(self, layer, x, bias=None):
        output_shape = (*x.shape[:-1], layer.weight.shape[1])
        x = x.reshape(-1, x.shape[-1])
        return _original_apply_weights(self, layer, x, bias).view(*output_shape)

    XPUW8A16FP8LinearKernel.apply_weights = _patched_apply_weights


def apply_patches() -> None:
    """Apply all XPU-specific patches."""
    _patch_w8a16_fp8_nd_activation()
