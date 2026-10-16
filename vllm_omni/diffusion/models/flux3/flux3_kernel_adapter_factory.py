# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Select the Flux3 kernel adapter from runtime configuration and support."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_omni.diffusion.models.flux3.flux3_kernel_adapter import Flux3KernelAdapter, QKNormRopeFn

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)


class Flux3PrimsTSMode(str, Enum):
    """User-visible selection policy for the optional PrimsTS kernels."""

    OFF = "off"
    AUTO = "auto"
    REQUIRED = "required"


@dataclass(frozen=True)
class Flux3PrimsTSConfig:
    """Validated Flux3 PrimsTS configuration."""

    # PrimsTS wherever the GPU, FlashInfer build and checkpoint support it,
    # with a logged fallback to vLLM kernels otherwise.
    mode: Flux3PrimsTSMode = Flux3PrimsTSMode.AUTO

    @classmethod
    def from_additional_config(cls, additional_config: Mapping[str, Any] | None) -> Flux3PrimsTSConfig:
        raw = (additional_config or {}).get("prims_ts")
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise TypeError("additional_config['prims_ts'] must be a mapping")

        unknown = set(raw) - {"mode"}
        if unknown:
            names = ", ".join(sorted(str(name) for name in unknown))
            raise ValueError(f"Unknown additional_config['prims_ts'] options: {names}")

        mode = raw.get("mode", Flux3PrimsTSMode.AUTO.value)
        if not isinstance(mode, str):
            raise TypeError("additional_config['prims_ts']['mode'] must be a string")
        try:
            return cls(mode=Flux3PrimsTSMode(mode.lower()))
        except ValueError as exc:
            choices = ", ".join(item.value for item in Flux3PrimsTSMode)
            raise ValueError(f"additional_config['prims_ts']['mode'] must be one of {choices}; got {mode!r}") from exc


@dataclass(frozen=True)
class Flux3PrimsTSSupport:
    """Result of probing whether PrimsTS can be selected in this worker."""

    available: bool
    reason: str | None = None


_BASE_PRIMS_TS_SYMBOLS = ("PrimsTsGemmConfig",)
_SUPPORTED_PRIMS_TS_ARCHES = (100, 103, 107)
_FP8_PRIMS_TS_SYMBOLS = (
    *_BASE_PRIMS_TS_SYMBOLS,
    "fp8_linear",
    "fp8_linear_swiglu",
    "fp8_qkv_qknorm_rope",
)
_NVFP4_PRIMS_TS_SYMBOLS = (
    *_BASE_PRIMS_TS_SYMBOLS,
    "prepare_fp4_linear",
    "prepare_fp4_linear_swiglu",
    "prepare_fp4_qkv_qknorm_rope",
)


def probe_prims_ts_support(
    required_symbols: tuple[str, ...] = _BASE_PRIMS_TS_SYMBOLS,
) -> Flux3PrimsTSSupport:
    """Check hardware and optional FlashInfer APIs without importing them globally."""
    if not current_platform.is_cuda():
        return Flux3PrimsTSSupport(False, "PrimsTS requires a CUDA device")

    capability = current_platform.get_device_capability()
    if capability is None or capability.to_int() not in _SUPPORTED_PRIMS_TS_ARCHES:
        sm = capability.to_int() if capability is not None else "unknown"
        supported = ", ".join(f"SM{arch}" for arch in _SUPPORTED_PRIMS_TS_ARCHES)
        return Flux3PrimsTSSupport(False, f"PrimsTS requires one of {supported}; found SM{sm}")

    try:
        gemm = importlib.import_module("flashinfer.prims_ts.gemm")
    except (ImportError, OSError) as exc:
        return Flux3PrimsTSSupport(False, f"FlashInfer PrimsTS is unavailable: {exc}")

    missing = tuple(name for name in required_symbols if not hasattr(gemm, name))
    if missing:
        return Flux3PrimsTSSupport(False, f"FlashInfer PrimsTS is missing required symbols: {', '.join(missing)}")

    if "prepare_fp4_linear" in required_symbols:
        try:
            flashinfer = importlib.import_module("flashinfer")
        except (ImportError, OSError) as exc:
            return Flux3PrimsTSSupport(False, f"FlashInfer NVFP4 quantization is unavailable: {exc}")
        quantizer_symbols = ("nvfp4_quantize", "SfLayout")
        missing = tuple(name for name in quantizer_symbols if not hasattr(flashinfer, name))
        if missing:
            return Flux3PrimsTSSupport(
                False,
                f"FlashInfer NVFP4 quantization is missing required symbols: {', '.join(missing)}",
            )
    return Flux3PrimsTSSupport(True)


def resolve_prims_ts_support(config: Flux3PrimsTSConfig) -> Flux3PrimsTSSupport:
    """Resolve an optional backend request and enforce ``required`` mode."""
    if config.mode is Flux3PrimsTSMode.OFF:
        return Flux3PrimsTSSupport(False, "disabled by configuration")

    support = probe_prims_ts_support()
    if not support.available and config.mode is Flux3PrimsTSMode.REQUIRED:
        raise RuntimeError(f"Flux3 PrimsTS was required but is unavailable: {support.reason}")
    return support


def _quant_format(quant_config: object | None) -> str | None:
    get_name = getattr(quant_config, "get_name", None)
    quant_name = get_name() if callable(get_name) else None
    if quant_name == "modelopt_fp4" and getattr(quant_config, "quant_method", None) == "NVFP4":
        return "nvfp4"
    if quant_name == "modelopt" and getattr(quant_config, "quant_method", None) == "FP8_PER_CHANNEL_PER_TOKEN":
        return "fp8"
    return None


def _incompatibility(
    od_config: OmniDiffusionConfig | None,
    quant_config: object | None,
) -> tuple[str | None, str | None]:
    quant_format = _quant_format(quant_config)
    if quant_format is None:
        return None, "PrimsTS requires a ModelOpt FP8_PER_CHANNEL_PER_TOKEN or NVFP4 checkpoint"

    dtype = getattr(od_config, "dtype", None)
    if dtype is not None and dtype is not torch.bfloat16:
        return quant_format, f"PrimsTS requires bfloat16 model execution; found {dtype}"

    parallel_config = getattr(od_config, "parallel_config", None)
    if getattr(parallel_config, "use_hsdp", False):
        return quant_format, "PrimsTS does not currently support HSDP"
    return quant_format, None


def build_flux3_kernel_adapter(
    *,
    qk_norm_rope: QKNormRopeFn,
    od_config: OmniDiffusionConfig | None,
    quant_config: object | None,
) -> Flux3KernelAdapter:
    """Select PrimsTS unless disabled or unavailable; otherwise use the default adapter."""
    additional_config = od_config.additional_config if od_config is not None else None
    prims_ts_config = Flux3PrimsTSConfig.from_additional_config(additional_config)
    if prims_ts_config.mode is Flux3PrimsTSMode.OFF:
        return Flux3KernelAdapter(qk_norm_rope)

    quant_format, reason = _incompatibility(od_config, quant_config)
    if reason is not None:
        if prims_ts_config.mode is Flux3PrimsTSMode.REQUIRED:
            raise RuntimeError(f"Flux3 PrimsTS was required but is unavailable: {reason}")
        logger.info_once("Flux3 PrimsTS is unavailable; using vLLM kernels: %s", reason)
        return Flux3KernelAdapter(qk_norm_rope)

    required_symbols = _FP8_PRIMS_TS_SYMBOLS if quant_format == "fp8" else _NVFP4_PRIMS_TS_SYMBOLS
    prims_ts_support = probe_prims_ts_support(required_symbols)
    if not prims_ts_support.available:
        if prims_ts_config.mode is Flux3PrimsTSMode.REQUIRED:
            raise RuntimeError(f"Flux3 PrimsTS was required but is unavailable: {prims_ts_support.reason}")
        logger.info_once("Flux3 PrimsTS is unavailable; using vLLM kernels: %s", prims_ts_support.reason)
        return Flux3KernelAdapter(qk_norm_rope)

    # Keep the specialized module and its custom-op registrations optional.
    from vllm_omni.diffusion.models.flux3.flux3_prims_ts import Flux3PrimsTSKernelAdapter

    adapter = Flux3PrimsTSKernelAdapter(
        qk_norm_rope,
        prims_ts_config=prims_ts_config,
        prims_ts_support=prims_ts_support,
        quant_config=quant_config,
        quant_format=quant_format,
    )
    logger.info_once("Flux3 PrimsTS %s kernels enabled", quant_format.upper())
    return adapter


__all__ = [
    "Flux3PrimsTSConfig",
    "Flux3PrimsTSMode",
    "Flux3PrimsTSSupport",
    "build_flux3_kernel_adapter",
    "probe_prims_ts_support",
    "resolve_prims_ts_support",
]
