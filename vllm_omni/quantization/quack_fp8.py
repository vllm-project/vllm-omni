# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 scaled-MM with a fused bias epilogue, backed by quack's CuteDSL GEMM."""

from __future__ import annotations

import os
from contextlib import contextmanager
from importlib import import_module
from multiprocessing import current_process

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}
_gemm_interface = None


def _is_quack_capable() -> bool:
    """quack's CuteDSL FP8 / block-scaled MMA is built on the 5th-gen tensor-core
    ``tcgen05`` instruction family, which is datacenter-Blackwell only
    (``sm_100a`` / ``sm_101a`` / ``sm_103a``, compute capability ``10.x``).

    Workstation/consumer Blackwell (``sm_120`` / ``sm_121``, compute capability
    ``12.x``, e.g. RTX PRO 6000 / RTX 50-series) lacks ``tcgen05``, so quack can
    never run there — CuteDSL rejects the arch and every GEMM falls back to
    FlashInfer one call at a time, which is catastrophically slow. Those GPUs
    have working native FlashInfer FP8 kernels, so default quack off for them.
    """
    try:
        if not torch.cuda.is_available():
            return False
        return torch.cuda.get_device_capability()[0] == 10
    except Exception:  # noqa: BLE001
        return False


def quack_enabled() -> bool:
    override = os.environ.get("VLLM_OMNI_USE_QUACK_FP8")
    if override is not None:
        return override.lower() in _TRUTHY
    return _is_quack_capable()


def _set_persistent_cache_dir() -> None:
    if os.environ.get("QUACK_CACHE_DIR"):
        return
    root = (
        os.environ.get("VLLM_CACHE_ROOT")
        or os.environ.get("XDG_CACHE_HOME")
        or os.path.join(os.path.expanduser("~"), ".cache")
    )
    os.environ["QUACK_CACHE_DIR"] = os.path.join(root, "vllm_omni", "quack")


def _configure_quack_compilation() -> None:
    """Keep autotuning in daemon workers without starting compiler children."""
    try:
        async_compile = import_module("quack.cache.async_compile")
    except ModuleNotFoundError as exc:
        # Older Quack releases compile synchronously and have no async pool.
        if exc.name not in {"quack.cache", "quack.cache.async_compile"}:
            raise
        return
    original_pool_scope = getattr(async_compile, "pool_scope", None)
    suppress_pool = getattr(async_compile, "suppress_pool", None)
    if not callable(original_pool_scope) or not callable(suppress_pool):
        logger.warning(
            "Quack async compilation API is unsupported: pool_scope and suppress_pool must be callable. "
            "Skipping the compilation patch; autotuning in daemon workers may fail."
        )
        return
    if getattr(original_pool_scope, "_omni_daemon_safe", False):
        return

    @contextmanager
    def daemon_safe_pool_scope():
        # spawn can import this module before installing the child's daemon flag.
        # Check at tuning time; suppress_pool keeps compilation in-process.
        if current_process().daemon:
            with suppress_pool():
                yield None
        else:
            with original_pool_scope() as pool:
                yield pool

    daemon_safe_pool_scope._omni_daemon_safe = True
    async_compile.pool_scope = daemon_safe_pool_scope


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

        _configure_quack_compilation()
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
    gemm.gemm(a, b, out=out, bias=bias, alpha=alpha, tuned=True)
    return out


_valid_scale_ptrs: set[tuple[int, int]] = set()


def _scales_valid(scale_a: torch.Tensor, scale_b: torch.Tensor) -> bool:
    """True when both per-tensor scales are finite and positive.

    vLLM initializes a per-tensor FP8 scale to ``finfo(float32).min`` and fills it at
    weight-load time, so a call made before that (the dummy profiling forward) would
    make ``alpha = scale_a * scale_b`` overflow to ``+inf`` and return an all-inf tile.
    Cache only positive results: a buffer still holding the sentinel is re-checked and
    picks up the fast path once the real scale is written.

    Called inside the dispatch custom op, so pointer checks and cache updates
    neither specialize Dynamo graphs on each layer's addresses nor break them.
    """
    key = (scale_a.data_ptr(), scale_b.data_ptr())
    if key in _valid_scale_ptrs:
        return True
    ok = bool(
        torch.isfinite(scale_a).all() and torch.isfinite(scale_b).all() and (scale_a > 0).all() and (scale_b > 0).all()
    )
    if ok:
        _valid_scale_ptrs.add(key)
    return ok


@torch.library.custom_op("vllm_omni::quack_fp8_scaled_mm", mutates_args=())
def _quack_fp8_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Keep scale validation and runtime fallback inside one opaque graph node."""
    if scale_a.numel() == 1 and scale_b.numel() == 1 and _scales_valid(scale_a, scale_b):
        try:
            out = quack_scaled_fp8_mm(a, b, scale_a, scale_b, out_dtype, bias)
            if out is not None:
                return out
        except Exception as exc:  # noqa: BLE001
            logger.warning_once("Quack FP8 GEMM failed (%s); using FlashInfer.", exc)

    # Use the same fallback as FlashInferFP8ScaledMMLinearKernel, without
    # passing its Python instance through the custom-op schema. Unpopulated
    # scales also take this path: FlashInfer tolerates the profiling sentinel.
    from vllm.utils.flashinfer import flashinfer_scaled_fp8_mm

    out = flashinfer_scaled_fp8_mm(a, b, out_dtype=out_dtype, scale_a=scale_a, scale_b=scale_b, bias=bias)
    # FlashInfer's separate bias add may promote the dtype. Both branches must
    # match the fake kernel's output metadata, including with an FP32 bias.
    return out.to(out_dtype)


@_quack_fp8_scaled_mm.register_fake
def _quack_fp8_scaled_mm_fake(a, b, scale_a, scale_b, out_dtype, bias=None):
    return torch.empty(a.shape[0], b.shape[1], device=a.device, dtype=out_dtype)


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
        return _quack_fp8_scaled_mm(A, B, As, Bs, out_dtype, bias).view(*output_shape)

    apply_scaled_mm._omni_quack_patched = True
    FlashInferFP8ScaledMMLinearKernel.apply_scaled_mm = apply_scaled_mm
    logger.info("Patched FlashInfer FP8 ScaledMM to use quack fused-bias GEMM.")


@torch.inference_mode()
def warmup_quack_fp8(
    shapes: list[tuple[int, int, int]],
    device: str = "cuda",
    out_dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Warm no-bias GEMMs with the transposed weight layout used by vLLM."""
    if _load_quack() is None:
        return
    scale = torch.ones(1, device=device, dtype=torch.float32)
    for m, k, n in shapes:
        a = torch.zeros(m, k, device=device, dtype=torch.float8_e4m3fn)
        b = torch.zeros(n, k, device=device, dtype=torch.float8_e4m3fn).t()
        quack_scaled_fp8_mm(a, b, scale, scale, out_dtype)
    if torch.cuda.is_available():
        torch.accelerator.synchronize()
