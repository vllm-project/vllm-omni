# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bit-exact decoder fast path for the diffusers HunyuanVideo-1.5 VAE.

Rebinds forwards on one VAE instance: causal convs use :func:`replicate_pad`,
RMS norms use the shared fused epilogue, with SiLU folded in only for dtypes
where the epilogue is verified exact.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MethodType

import torch
import torch.nn as nn
from diffusers import AutoencoderKLHunyuanVideo15
from diffusers.models.autoencoders.autoencoder_kl_hunyuanvideo15 import (
    HunyuanVideo15CausalConv3d,
    HunyuanVideo15ResnetBlock,
    HunyuanVideo15RMS_norm,
)
from vllm.logger import init_logger

from vllm_omni.diffusion.data import VAE_FAST_PATH_LEVELS
from vllm_omni.diffusion.layers import vae_norm_epilogue as rn
from vllm_omni.diffusion.layers.replicate_pad import replicate_pad
from vllm_omni.diffusion.layers.vae_rms_norm import rms_norm_fastpath

logger = init_logger(__name__)

REPORT_ATTR = "_vllm_omni_hv15_fastpath_report"
_FUSED_SILU_ATTR = "_vllm_omni_hv15_fused_silu_dtypes"


@dataclass(frozen=True)
class HunyuanVideo15VaeFastPathReport:
    """What :func:`install_hunyuan_video_15_vae_fastpath` did to one VAE instance."""

    level: str
    installed: bool
    reason: str | None = None
    patched: Mapping[str, int] = field(default_factory=dict)
    fused_silu_dtypes: tuple[str, ...] = ()


def _has_forward_hooks(*modules: nn.Module) -> bool:
    if nn.modules.module._global_forward_pre_hooks or nn.modules.module._global_forward_hooks:
        return True
    return any(m._forward_pre_hooks or m._forward_hooks for m in modules)


def causal_conv_forward(self: HunyuanVideo15CausalConv3d, hidden_states: torch.Tensor) -> torch.Tensor:
    if self.pad_mode != "replicate":
        return type(self).forward(self, hidden_states)
    return self.conv(replicate_pad(hidden_states, self.time_causal_padding))


def _is_dense(x: torch.Tensor) -> bool:
    expected = 1
    for stride, size in sorted((st, sz) for sz, st in zip(x.shape, x.stride()) if sz != 1):
        if stride != expected:
            return False
        expected *= size
    return True


def _fast_norm(norm: nn.Module, x: torch.Tensor, silu: bool = False) -> torch.Tensor | None:
    if torch.is_grad_enabled():
        return None
    if x.dtype is not torch.float32:
        # The reference reduces x.float(). ATen splits that reduction past 2**31 bytes,
        # changing its order, and it re-lays out a non-dense x; match both.
        if x.numel() * 4 > 2**31:
            return None
        if not _is_dense(x):
            x = torch.empty_like(x).copy_(x)
    return rms_norm_fastpath(norm, x, silu=silu)


def rms_norm_forward(self: HunyuanVideo15RMS_norm, x: torch.Tensor) -> torch.Tensor:
    out = _fast_norm(self, x)
    return out if out is not None else type(self).forward(self, x)


def _norm_act(block: HunyuanVideo15ResnetBlock, norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    act = block.nonlinearity
    if (
        type(norm) is HunyuanVideo15RMS_norm
        and type(act) is nn.SiLU
        and not act.inplace
        and not _has_forward_hooks(norm, act)
    ):
        fuse = x.dtype in getattr(block, _FUSED_SILU_ATTR)
        out = _fast_norm(norm, x, silu=fuse)
        if out is not None:
            return out if fuse else act(out)
    return act(norm(x))


def resnet_block_forward(self: HunyuanVideo15ResnetBlock, hidden_states: torch.Tensor) -> torch.Tensor:
    residual = hidden_states
    hidden_states = self.conv1(_norm_act(self, self.norm1, hidden_states))
    hidden_states = self.conv2(_norm_act(self, self.norm2, hidden_states))
    if self.conv_shortcut is not None:
        residual = self.conv_shortcut(residual)
    return hidden_states + residual


_REPLACEMENT_FORWARDS = {
    HunyuanVideo15CausalConv3d: causal_conv_forward,
    HunyuanVideo15ResnetBlock: resnet_block_forward,
    HunyuanVideo15RMS_norm: rms_norm_forward,
}


def _skip(level: str, reason: str) -> HunyuanVideo15VaeFastPathReport:
    logger.info("HunyuanVideo-1.5 VAE fast path (%s) not installed: %s", level, reason)
    return HunyuanVideo15VaeFastPathReport(level=level, installed=False, reason=reason)


def install_hunyuan_video_15_vae_fastpath(
    vae: nn.Module, *, level: str = "lossless"
) -> HunyuanVideo15VaeFastPathReport:
    """Install the decoder fast path on one loaded ``AutoencoderKLHunyuanVideo15``. Idempotent.

    ``channels_last`` is not implemented for this VAE and installs ``lossless``.
    """
    if level not in VAE_FAST_PATH_LEVELS:
        raise ValueError(f"vae_fast_path must be one of {list(VAE_FAST_PATH_LEVELS)}, got {level!r}")
    existing = getattr(vae, REPORT_ATTR, None)
    if existing is not None:
        return existing
    if level == "off":
        return HunyuanVideo15VaeFastPathReport(level=level, installed=False, reason="disabled")
    if level == "channels_last":
        logger.info("HunyuanVideo-1.5 VAE has no channels_last fast path; installing lossless")
        level = "lossless"
    if not isinstance(vae, AutoencoderKLHunyuanVideo15):
        return _skip(level, f"{type(vae).__name__} is not a diffusers AutoencoderKLHunyuanVideo15")
    decoder = getattr(vae, "decoder", None)
    if not isinstance(decoder, nn.Module):
        return _skip(level, "VAE has no decoder")

    bindings = []
    for name, module in decoder.named_modules():
        replacement = _REPLACEMENT_FORWARDS.get(type(module))
        if replacement is None:
            continue
        # The SiLU of a resnet block is bypassed when fused into the norm.
        checked = [module, module.nonlinearity] if replacement is resnet_block_forward else [module]
        for m in checked:
            forward = m.forward
            if not (isinstance(forward, MethodType) and forward.__self__ is m and forward.__func__ is type(m).forward):
                return _skip(level, f"decoder.{name} has a custom forward that the fast path would replace or bypass")
        bindings.append((module, replacement))

    device = next((p.device for p in decoder.parameters()), torch.device("cpu"))
    fused: frozenset[torch.dtype] = frozenset()
    if device.type in rn.KERNEL_DEVICE_TYPES:
        fused = frozenset(d for d in (torch.bfloat16, torch.float16) if rn.silu_epilogue_is_exact(device, d))
    patched: dict[str, int] = {}
    for module, replacement in bindings:
        module.forward = MethodType(replacement, module)
        if replacement is resnet_block_forward:
            setattr(module, _FUSED_SILU_ATTR, fused)
        patched[type(module).__name__] = patched.get(type(module).__name__, 0) + 1

    report = HunyuanVideo15VaeFastPathReport(
        level=level,
        installed=True,
        patched=patched,
        fused_silu_dtypes=tuple(str(d).removeprefix("torch.") for d in sorted(fused, key=str)),
    )
    setattr(vae, REPORT_ATTR, report)
    logger.info(
        "HunyuanVideo-1.5 VAE fast path (%s) installed: patched=%s fused_silu=%s",
        level,
        report.patched,
        report.fused_silu_dtypes or "off",
    )
    return report


__all__ = ["REPORT_ATTR", "HunyuanVideo15VaeFastPathReport", "install_hunyuan_video_15_vae_fastpath"]
