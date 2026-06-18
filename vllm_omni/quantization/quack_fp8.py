# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 scaled-MM with a fused bias epilogue, backed by quack's CuteDSL GEMM."""

from __future__ import annotations

import os

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}
_gemm_interface = None


def _is_blackwell() -> bool:
    try:
        if not torch.cuda.is_available():
            return False
        return torch.cuda.get_device_capability()[0] >= 10
    except Exception:  # noqa: BLE001
        return False


def quack_enabled() -> bool:
    override = os.environ.get("VLLM_OMNI_USE_QUACK_FP8")
    if override is not None:
        return override.lower() in _TRUTHY
    return _is_blackwell()


def _set_persistent_cache_dir() -> None:
    if os.environ.get("QUACK_CACHE_DIR"):
        return
    root = (
        os.environ.get("VLLM_CACHE_ROOT")
        or os.environ.get("XDG_CACHE_HOME")
        or os.path.join(os.path.expanduser("~"), ".cache")
    )
    os.environ["QUACK_CACHE_DIR"] = os.path.join(root, "vllm_omni", "quack")


def _load_quack():
    global _gemm_interface
    if _gemm_interface is not None:
        return _gemm_interface or None
    try:
        _set_persistent_cache_dir()

        import cutlass
        import cutlass.base_dsl
        import cutlass.base_dsl.arch as arch

        if not hasattr(cutlass.base_dsl, "Arch"):
            cutlass.base_dsl.Arch = arch.Arch

        import quack.gemm_interface as gemm_interface
        from quack.cute_dsl_utils import torch2cute_dtype_map

        torch2cute_dtype_map.setdefault(torch.float8_e4m3fn, cutlass.Float8E4M3FN)
        torch2cute_dtype_map.setdefault(torch.float8_e5m2, cutlass.Float8E5M2)

        _gemm_interface = gemm_interface
        logger.info("Quack FP8 fused-bias GEMM enabled (CuteDSL).")
        return gemm_interface
    except Exception as exc:  # noqa: BLE001
        logger.warning("Quack FP8 unavailable, using FlashInfer: %s", exc)
        _gemm_interface = False
        return None


def quack_scaled_fp8_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor | None:
    gemm = _load_quack()
    if gemm is None:
        return None
    out = torch.empty(a.shape[0], b.shape[1], device=a.device, dtype=out_dtype)
    alpha = scale_a.reshape(1).float() * scale_b.reshape(1).float()
    gemm.gemm_tuned(a, b, out, bias=bias, alpha=alpha)
    return out


def install_quack_fp8_patch() -> None:
    if not quack_enabled():
        return
    if _load_quack() is None:
        return
    try:
        from vllm.model_executor.kernels.linear.scaled_mm.flashinfer import (
            FlashInferFP8ScaledMMLinearKernel,
        )
    except ImportError:
        return

    original = FlashInferFP8ScaledMMLinearKernel.apply_scaled_mm
    if getattr(original, "_omni_quack_patched", False):
        return

    def apply_scaled_mm(self, *, A, B, out_dtype, As, Bs, bias, output_shape):  # noqa: N803
        try:
            out = quack_scaled_fp8_mm(A, B, As, Bs, out_dtype, bias)
            if out is not None:
                return out.view(*output_shape)
        except Exception as exc:  # noqa: BLE001
            logger.warning_once("Quack FP8 GEMM failed (%s); using FlashInfer.", exc)
        return original(self, A=A, B=B, out_dtype=out_dtype, As=As, Bs=Bs, bias=bias, output_shape=output_shape)

    apply_scaled_mm._omni_quack_patched = True
    FlashInferFP8ScaledMMLinearKernel.apply_scaled_mm = apply_scaled_mm
    logger.info("Patched FlashInfer FP8 ScaledMM to use quack fused-bias GEMM.")


def warmup_quack_fp8(
    shapes: list[tuple[int, int, int]],
    device: str = "cuda",
    out_dtype: torch.dtype = torch.bfloat16,
) -> None:
    if _load_quack() is None:
        return
    scale = torch.ones(1, device=device, dtype=torch.float32)
    for m, k, n in shapes:
        a = torch.zeros(m, k, device=device, dtype=torch.float8_e4m3fn)
        b = torch.zeros(k, n, device=device, dtype=torch.float8_e4m3fn)
        quack_scaled_fp8_mm(a, b, scale, scale, out_dtype)
    if torch.cuda.is_available():
        torch.accelerator.synchronize()
