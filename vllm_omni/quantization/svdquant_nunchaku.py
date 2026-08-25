# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Nunchaku backend for SVDQuant W4A4.

Covers consumer NVIDIA GPUs (SM_75 Turing through consumer Blackwell
SM_120) via the external `nunchaku` package. Hopper SM_90 and
datacenter Blackwell SM_100/103 are intentionally excluded — the
former has no validated kernel family; the latter is planned in
FlashInfer.

Exposes the backend interface consumed by `svdquant_dispatch` and
`DiffusionSVDQuantLinearMethod`:

    supports(cap, precision) -> bool
    prepare_weights(layer, precision) -> None
    apply(layer, x, bias) -> Tensor

Plus `has_nunchaku()` / `has_nunchaku_w4a4()` / `has_nunchaku_w4a16()`
for callers that need capability detection (notably the hardware gate).

Install note: the PyPI `nunchaku` package is an unrelated Bayesian
library; SVDQuant kernels ship as GitHub release wheels from
https://github.com/nunchaku-ai/nunchaku/releases only.
"""

from __future__ import annotations

import functools
import importlib
import importlib.util
from collections.abc import Callable
from typing import Any, NoReturn

import torch
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability

logger = init_logger(__name__)


# ── Capability detection ────────────────────────────────────────────


@functools.cache
def has_nunchaku() -> bool:
    """Return True if the `nunchaku` package is importable."""
    if importlib.util.find_spec("nunchaku") is None:
        logger.debug_once("Nunchaku unavailable: package not installed")
        return False
    return True


def _get_submodule(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        return None


@functools.cache
def has_nunchaku_w4a4() -> bool:
    """True iff both the W4A4 GEMM and the fused act-quantize+LoRA op exist."""
    if not has_nunchaku():
        return False
    required = [
        ("nunchaku.ops.gemm", "svdq_gemm_w4a4_cuda"),
        ("nunchaku.ops.quantize", "svdq_quantize_w4a4_act_fuse_lora_cuda"),
    ]
    for module_name, attr_name in required:
        mod = _get_submodule(module_name)
        if mod is None or not hasattr(mod, attr_name):
            logger.debug_once("Nunchaku W4A4 unavailable: missing %s.%s", module_name, attr_name)
            return False
    return True


@functools.cache
def has_nunchaku_w4a16() -> bool:
    """True iff Nunchaku's W4A16 AWQ GEMV op exists (decode-style paths)."""
    if not has_nunchaku():
        return False
    mod = _get_submodule("nunchaku.ops.gemv")
    return mod is not None and hasattr(mod, "awq_gemv_w4a16_cuda")


# ── Lazy call wrappers ──────────────────────────────────────────────


def _missing(*_: Any, **__: Any) -> NoReturn:
    raise RuntimeError(
        "Nunchaku is not installed. SVDQuant requires the nunchaku-ai "
        "wheels from https://github.com/nunchaku-ai/nunchaku/releases "
        "(do NOT `pip install nunchaku` — that pulls an unrelated PyPI "
        "package). Source: https://github.com/nunchaku-ai/nunchaku"
    )


def _lazy_import_wrapper(module_name: str, attr_name: str, fallback_fn: Callable[..., Any] = _missing):
    @functools.cache
    def _get_impl():
        if not has_nunchaku():
            return None
        mod = _get_submodule(module_name)
        return getattr(mod, attr_name, None) if mod else None

    def wrapper(*args, **kwargs):
        impl = _get_impl()
        if impl is None:
            return fallback_fn(*args, **kwargs)
        return impl(*args, **kwargs)

    wrapper.__name__ = attr_name
    wrapper.__qualname__ = f"nunchaku::{attr_name}"
    return wrapper


_svdq_gemm_w4a4 = _lazy_import_wrapper("nunchaku.ops.gemm", "svdq_gemm_w4a4_cuda")
_svdq_quantize_w4a4_act_fuse_lora = _lazy_import_wrapper(
    "nunchaku.ops.quantize", "svdq_quantize_w4a4_act_fuse_lora_cuda"
)


# ── Backend interface ───────────────────────────────────────────────

# Compute capabilities the nunchaku PTX-MMA family targets. Hopper SM_90
# and datacenter Blackwell SM_100/103 are deliberately absent.
_SUPPORTED_CAPS: set[tuple[int, int]] = {
    (7, 5),  # Turing
    (8, 0),  # Ampere A100
    (8, 6),  # Ampere consumer (RTX 30xx)
    (8, 9),  # Ada (RTX 40xx)
    (12, 0),  # Consumer Blackwell (RTX 5090)
}


def supports(cap: DeviceCapability | None, precision: str) -> bool:
    """Return True iff this backend can serve (cap, precision)."""
    if cap is None:
        return False
    if not has_nunchaku_w4a4():
        return False
    # nvfp4 needs tcgen05's SM_100+ tensor units; in this backend that
    # means consumer Blackwell only.
    if precision == "nvfp4" and (cap.major, cap.minor) != (12, 0):
        return False
    return (cap.major, cap.minor) in _SUPPORTED_CAPS


def prepare_weights(layer: torch.nn.Module, precision: str) -> None:
    """Post-load weight prep for the nunchaku kernel.

    On-disk format is canonical row-major NVFP4 (or INT4-nibble); the
    nunchaku kernel wants a PTX-MMA fragment-permuted layout. For
    NVFP4 we repack in-place via the bit-preserving pack chain in
    `tools/svdquant_nvfp4_layout`; for INT4 the on-disk layout is
    already what the kernel expects.

    Also caches the kernel's per-tensor `alpha` from `wtscale`. Do NOT
    fold `wcscales` into `alpha`: the kernel applies them as
    `(accumulator * alpha) * wcscales` and conflating them
    double-counts the per-channel factors.
    """
    if precision == "nvfp4":
        _pack_nvfp4_to_nunchaku_fragment(layer)

    alpha: float = 1.0
    wtscale = getattr(layer, "wtscale", None)
    if wtscale is not None:
        value = float(wtscale.detach().cpu().item())
        if abs(value - 1.0) > 1e-6:
            alpha = value
    layer._svdquant_alpha = alpha


def _pack_nvfp4_to_nunchaku_fragment(layer: torch.nn.Module) -> None:
    """Repack row-major NVFP4 params in-place to nunchaku fragment layout.

    On-disk (canonical row-major):
      * qweight   : [N, K/2] int8/uint8 (FP4 nibbles, low = even-k)
      * wscales   : [K/16, N] fp8_e4m3fn
      * proj_up   : [N, R]
      * proj_down : [K, R]

    After repack (nunchaku PTX-MMA fragment):
      * qweight   : [N, K/2] int8 (permuted into MMA fragment)
      * wscales   : [K/16, N] fp8 (permuted into MMA fragment)
      * proj_up   : [N, R] (permuted into MMA fragment)
      * proj_down : [K, R] (permuted into MMA fragment)
    """
    # Lazy imports: nunchaku is a soft dep on non-consumer hardware,
    # and the layout helpers pull in torch ops we only need here.
    from nunchaku.lora.flux.nunchaku_converter import pack_lowrank_weight

    from vllm_omni.quantization.tools.svdquant_nvfp4_layout import (
        _unpack_nibbles,
        pack_nunchaku_qweight_fp4,
        pack_nunchaku_wscales_fp4,
    )

    device = layer.qweight.device

    # qweight: stored as [N, K/2] packed-nibble bytes (low = even-k).
    # `pack_nunchaku_qweight_fp4` expects [N, K] one-nibble-per-byte —
    # unpack to that form first, then pack to nunchaku fragment.
    qw_rm_packed = layer.qweight.data.view(torch.uint8)  # [N, K/2]
    qw_rm_nibs = _unpack_nibbles(qw_rm_packed)  # [N, K]
    layer.qweight.data = pack_nunchaku_qweight_fp4(qw_rm_nibs).to(device)

    # wscales: pack pair operates in fp8_e4m3fn.
    layer.wscales.data = pack_nunchaku_wscales_fp4(layer.wscales.data).to(device)

    # proj_up: row-major [N, R] → nunchaku frag [N, R]. down=False.
    layer.proj_up.data = pack_lowrank_weight(layer.proj_up.data, down=False).to(device)

    # proj_down: canonical row-major is [K, R]; nunchaku's pack expects
    # [R, K] (transpose-quirk on the down=True path). Transpose then pack;
    # output is fragment [K, R].
    pd_rk = layer.proj_down.data.transpose(0, 1).contiguous()
    layer.proj_down.data = pack_lowrank_weight(pd_rk, down=True).to(device)


def apply(
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the nunchaku W4A4 GEMM."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])

    is_fp4 = layer.precision == "nvfp4"
    out_features = layer.out_features_per_partition

    quantized_x, ascales, lora_act_out = _svdq_quantize_w4a4_act_fuse_lora(
        x_2d,
        lora_down=layer.proj_down,
        smooth=layer.smooth_factor,
        fp4=is_fp4,
        pad_size=256,
    )

    # The quantize kernel may pad the batch dim up to a multiple of
    # `pad_size`; the GEMM consumes the padded shape, then we trim back
    # below.
    out_2d = torch.empty(
        quantized_x.shape[0],
        out_features,
        dtype=layer.proj_up.dtype,
        device=x_2d.device,
    )

    _svdq_gemm_w4a4(
        act=quantized_x,
        wgt=layer.qweight,
        out=out_2d,
        ascales=ascales,
        wscales=layer.wscales,
        lora_act_in=lora_act_out,
        lora_up=layer.proj_up,
        bias=bias,
        fp4=is_fp4,
        alpha=getattr(layer, "_svdquant_alpha", 1.0),
        wcscales=layer.wcscales,
        act_unsigned=layer.act_unsigned,
    )

    actual_batch = x_2d.shape[0]
    if out_2d.shape[0] > actual_batch:
        out_2d = out_2d[:actual_batch]

    return out_2d.reshape(*orig_shape[:-1], out_features)


__all__ = [
    "has_nunchaku",
    "has_nunchaku_w4a4",
    "has_nunchaku_w4a16",
    "supports",
    "prepare_weights",
    "apply",
]
