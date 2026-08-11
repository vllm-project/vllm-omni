# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quality-gated CUDA fast path for Diffusers Wan VAE decoders."""

from __future__ import annotations

from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.vae_optimizations.gate import (
    VaeFastPathGate,
    register_vae_fast_path_gate,
)

logger = init_logger(__name__)

try:
    from vllm_omni.diffusion.vae_optimizations.triton_wan_rmsnorm_silu import (
        wan_rmsnorm_silu,
    )

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised on builds without Triton
    _HAS_TRITON = False


class FusedWanRMSNormSiLU(nn.Module):
    """Quality-gated Wan RMSNorm + SiLU with an exact eager fallback."""

    def __init__(self, norm: nn.Module, gate: VaeFastPathGate) -> None:
        super().__init__()
        self.gamma = norm.gamma
        self.bias = norm.bias
        self.scale = float(norm.scale)
        self._vllm_gate = gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._vllm_gate.enabled and not torch.compiler.is_compiling():
            bias = self.bias if isinstance(self.bias, torch.Tensor) else None
            output = wan_rmsnorm_silu(x, self.gamma, bias, rms_scale=self.scale)
            if output is not None:
                return output

        needs_fp32_normalize = x.dtype in (torch.float16, torch.bfloat16) or any(
            token in str(x.dtype) for token in ("float4_", "float8_")
        )
        normalized = F.normalize(x.float() if needs_fp32_normalize else x, dim=1).to(x.dtype)
        return F.silu(normalized * self.scale * self.gamma + self.bias)


def _plain_silu(activation: object) -> bool:
    return isinstance(activation, nn.SiLU) and not activation.inplace


def _fusable_norm(norm: object, norm_cls: type) -> bool:
    return (
        type(norm) is norm_cls
        and getattr(norm, "channel_first", False)
        and isinstance(getattr(norm, "gamma", None), torch.Tensor)
    )


def _install_norm_silu(decoder: nn.Module, gate: VaeFastPathGate) -> int | None:
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanResidualBlock, WanRMS_norm

    residual_blocks = [module for module in decoder.modules() if isinstance(module, WanResidualBlock)]
    eligible = [
        module
        for module in residual_blocks
        if type(module) is WanResidualBlock
        and _plain_silu(module.nonlinearity)
        and _fusable_norm(module.norm1, WanRMS_norm)
        and _fusable_norm(module.norm2, WanRMS_norm)
    ]
    if len(eligible) != len(residual_blocks):
        logger.info(
            "Wan VAE has %d/%d non-standard residual blocks; skipping fast path",
            len(residual_blocks) - len(eligible),
            len(residual_blocks),
        )
        return None
    if not (
        _fusable_norm(getattr(decoder, "norm_out", None), WanRMS_norm)
        and _plain_silu(getattr(decoder, "nonlinearity", None))
    ):
        logger.info("Wan VAE has a non-standard output head; skipping fast path")
        return None

    count = 0
    for module in eligible:
        module.norm1 = FusedWanRMSNormSiLU(module.norm1, gate)
        module.norm2 = FusedWanRMSNormSiLU(module.norm2, gate)
        module.nonlinearity = nn.Identity()
        count += 2
    decoder.norm_out = FusedWanRMSNormSiLU(decoder.norm_out, gate)
    decoder.nonlinearity = nn.Identity()
    return count + 1


def _set_wan_decoder_memory_format(decoder: nn.Module, *, channels_last: bool) -> None:
    for module in decoder.modules():
        if isinstance(module, nn.Conv3d):
            module.to(memory_format=torch.channels_last_3d if channels_last else torch.contiguous_format)
        elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            module.to(memory_format=torch.channels_last if channels_last else torch.contiguous_format)


def _decoder_layout_forward(self: nn.Module, *args: Any, **kwargs: Any) -> torch.Tensor:
    use_channels_last = self._vllm_gate.enabled
    if use_channels_last != self._vllm_channels_last:
        _set_wan_decoder_memory_format(self, channels_last=use_channels_last)
        self._vllm_channels_last = use_channels_last
        logger.debug(
            "Wan VAE decoder switched to %s layout",
            "channels_last_3d" if use_channels_last else "contiguous",
        )
    return type(self).forward(self, *args, **kwargs)


def maybe_optimize_wan_vae(vae: nn.Module) -> nn.Module:
    """Install the Wan decoder fast path when its full structure is supported."""

    if not _HAS_TRITON or getattr(vae, "_vllm_vae_fast_path_installed", False):
        return vae

    from diffusers.models.autoencoders import AutoencoderKLWan
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanDecoder3d

    if not isinstance(vae, AutoencoderKLWan):
        return vae
    decoder = getattr(vae, "decoder", None)
    if type(decoder) is not WanDecoder3d:
        return vae

    distributed_executor = getattr(vae, "distributed_executor", None)
    parallel_mode = getattr(distributed_executor, "parallel_mode", "tile")
    if parallel_mode in ("auto", "spatial_shard_height", "spatial_shard_width"):
        logger.info("Wan spatial-shard decode is configured; skipping quality-gated VAE fast path")
        return vae

    gate = VaeFastPathGate()
    norm_count = _install_norm_silu(decoder, gate)
    if norm_count is None:
        return vae
    decoder._vllm_gate = gate
    decoder._vllm_channels_last = False
    decoder.forward = MethodType(_decoder_layout_forward, decoder)
    register_vae_fast_path_gate(vae, gate)
    vae._vllm_vae_fast_path_installed = True
    logger.info(
        "Wan VAE installed quality-gated decoder fast path (%d RMSNorm+SiLU fusions)",
        norm_count,
    )
    return vae


__all__ = ["FusedWanRMSNormSiLU", "maybe_optimize_wan_vae"]
