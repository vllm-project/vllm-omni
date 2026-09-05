# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Instance-level installer for the Wan VAE decoder fast path."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MethodType
from typing import Any

import torch
import torch.nn as nn
from diffusers.models.autoencoders import AutoencoderKLWan
from diffusers.models.autoencoders.autoencoder_kl_wan import (
    WanCausalConv3d,
    WanDecoder3d,
    WanResample,
    WanResidualBlock,
    WanResidualUpBlock,
    WanUpsample,
)
from vllm.logger import init_logger

from . import forwards
from . import triton_rms_norm as rn

logger = init_logger(__name__)

VAE_FAST_PATH_LEVELS: tuple[str, ...] = ("off", "lossless", "channels_last")

REPORT_ATTR = "_vllm_omni_wan_fastpath_report"
_UNDO_ATTR = "_vllm_omni_wan_fastpath_undo"


@dataclass(frozen=True)
class WanVaeFastPathReport:
    """What :func:`install_wan_vae_fastpath` did to one VAE instance."""

    level: str
    installed: bool
    reason: str | None = None
    patched: Mapping[str, int] = field(default_factory=dict)
    fused_silu_dtypes: tuple[str, ...] = ()
    channels_last: bool = False


def is_installed(vae: nn.Module) -> bool:
    report = getattr(vae, REPORT_ATTR, None)
    return report is not None and report.installed


def _skip(level: str, reason: str) -> WanVaeFastPathReport:
    logger.info("Wan VAE fast path (%s) not installed: %s", level, reason)
    return WanVaeFastPathReport(level=level, installed=False, reason=reason)


def _convert_conv_memory_format(modules: list[nn.Module], *, channels_last: bool) -> int:
    """Convert conv weights to (or back from) channels-last layouts.

    Only 4D/5D parameters change (``nn.Module.to(memory_format=...)`` skips the
    rest), so parameter names, dtypes and devices are untouched. Inputs are not
    converted: cuDNN selects the channels-last algorithm when either operand
    suggests it, so the layout propagates from the first convolution.
    """
    count = 0
    for module in modules:
        if isinstance(module, nn.Conv3d):
            module.to(memory_format=torch.channels_last_3d if channels_last else torch.contiguous_format)
            count += 1
        elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            module.to(memory_format=torch.channels_last if channels_last else torch.contiguous_format)
            count += 1
    return count


def install_wan_vae_fastpath(vae: nn.Module, *, level: str = "lossless") -> WanVaeFastPathReport:
    """Install the Wan decoder fast path on one loaded ``AutoencoderKLWan``. Idempotent.

    ``level``:
      * ``"off"``: do nothing.
      * ``"lossless"``: bind the bit-exact replacement forwards (Tier 1).
      * ``"channels_last"``: Tier 1 plus channels-last conv weights (not bit-exact).

    The installer refuses (and reports why) when the VAE is not a diffusers Wan
    VAE, or when spatially-sharded decode is configured or already installed:
    ``wan_spatial_shard`` rebinds the same forwards and replaces the causal
    convolutions with halo-exchanging variants.
    """
    if level not in VAE_FAST_PATH_LEVELS:
        raise ValueError(f"vae_fast_path must be one of {list(VAE_FAST_PATH_LEVELS)}, got {level!r}")
    existing = getattr(vae, REPORT_ATTR, None)
    if existing is not None:
        if existing.level != level:
            logger.warning(
                "Wan VAE fast path already installed at level %r; ignoring request for level %r",
                existing.level,
                level,
            )
        return existing
    if level == "off":
        return WanVaeFastPathReport(level=level, installed=False, reason="disabled")
    if not isinstance(vae, AutoencoderKLWan):
        return _skip(level, f"{type(vae).__name__} is not a diffusers AutoencoderKLWan")
    decoder = getattr(vae, "decoder", None)
    if type(decoder) is not WanDecoder3d:
        return _skip(level, f"decoder is {type(decoder).__name__}, not diffusers WanDecoder3d")
    if getattr(vae, "_vllm_omni_wan_spatial_shard_installed", False):
        return _skip(level, "spatial-shard decode is already installed on this VAE")
    parallel_mode = getattr(getattr(vae, "distributed_executor", None), "parallel_mode", "tile")
    if isinstance(parallel_mode, str) and parallel_mode.startswith("spatial_shard"):
        return _skip(level, f"vae_parallel_mode={parallel_mode!r} is not supported")

    device = next((p.device for p in decoder.parameters()), torch.device("cpu"))
    fused_silu_dtypes: frozenset[torch.dtype] = frozenset()
    if device.type == "cuda":
        fused_silu_dtypes = frozenset(
            dtype for dtype in (torch.bfloat16, torch.float16) if rn.silu_epilogue_is_exact(device, dtype)
        )
    cfg = forwards.FastPathConfig(fused_silu_dtypes=fused_silu_dtypes, channels_last=level == "channels_last")

    undo: list[Callable[[], None]] = []
    patched: dict[str, int] = {}

    def bind(module: nn.Module, forward: Callable[..., Any]) -> None:
        module.forward = MethodType(forward, module)
        setattr(module, forwards.CFG_ATTR, cfg)
        patched[type(module).__name__] = patched.get(type(module).__name__, 0) + 1

        def restore(module: nn.Module = module) -> None:
            module.__dict__.pop("forward", None)
            module.__dict__.pop(forwards.CFG_ATTR, None)

        undo.append(restore)

    bind(decoder, forwards.decoder_forward)
    convs: list[nn.Module] = []
    for module in decoder.modules():
        module_type = type(module)
        if module_type is WanCausalConv3d:
            bind(module, forwards.causal_conv_forward)
            convs.append(module)
        elif module_type is WanResidualBlock:
            bind(module, forwards.residual_block_forward)
        elif module_type is WanResidualUpBlock:
            bind(module, forwards.residual_up_block_forward)
        elif module_type is WanResample:
            bind(module, forwards.resample_forward)
        elif module_type is WanUpsample:
            bind(module, forwards.upsample_forward)
        elif forwards.is_diffusers_rms_norm(module):
            bind(module, forwards.rms_norm_forward)
        elif isinstance(module, (nn.Conv2d, nn.Conv3d)):
            convs.append(module)
    post_quant_conv = getattr(vae, "post_quant_conv", None)
    if type(post_quant_conv) is WanCausalConv3d:
        bind(post_quant_conv, forwards.causal_conv_forward)
        convs.append(post_quant_conv)

    if cfg.channels_last:
        converted = _convert_conv_memory_format(convs, channels_last=True)
        undo.append(lambda: _convert_conv_memory_format(convs, channels_last=False))
        logger.info("Wan VAE decoder: %d convolution weights converted to channels-last layout", converted)

    report = WanVaeFastPathReport(
        level=level,
        installed=True,
        patched=dict(patched),
        fused_silu_dtypes=tuple(str(dtype).removeprefix("torch.") for dtype in sorted(fused_silu_dtypes, key=str)),
        channels_last=cfg.channels_last,
    )
    setattr(vae, REPORT_ATTR, report)
    setattr(vae, _UNDO_ATTR, undo)
    logger.info(
        "Wan VAE fast path (%s) installed: patched=%s fused_silu=%s channels_last=%s",
        level,
        report.patched,
        report.fused_silu_dtypes or "off",
        report.channels_last,
    )
    return report


def uninstall_wan_vae_fastpath(vae: nn.Module) -> None:
    """Undo :func:`install_wan_vae_fastpath` on one VAE (intended for tests)."""
    undo = vae.__dict__.pop(_UNDO_ATTR, None)
    vae.__dict__.pop(REPORT_ATTR, None)
    for restore in reversed(undo or []):
        restore()


__all__ = [
    "REPORT_ATTR",
    "VAE_FAST_PATH_LEVELS",
    "WanVaeFastPathReport",
    "install_wan_vae_fastpath",
    "is_installed",
    "uninstall_wan_vae_fastpath",
]
