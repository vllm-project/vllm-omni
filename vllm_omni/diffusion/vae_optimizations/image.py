# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quality-gated CUDA fast paths for Diffusers-style 2D KL VAE decoders."""

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
    from vllm_omni.diffusion.vae_optimizations.triton_group_norm_silu import (
        group_norm_silu_4d,
        group_norm_silu_rows,
    )

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised on builds without Triton
    _HAS_TRITON = False


class FusedGroupNormSiLU(nn.Module):
    """Quality-gated GroupNorm + SiLU with an exact eager fallback."""

    def __init__(self, norm: nn.GroupNorm, gate: VaeFastPathGate) -> None:
        super().__init__()
        self.num_groups = norm.num_groups
        self.num_channels = norm.num_channels
        self.eps = norm.eps
        self.affine = norm.affine
        # Register parameters directly so state-dict FQNs do not change.
        self.weight = norm.weight
        self.bias = norm.bias
        self._vllm_gate = gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._vllm_gate.enabled and x.dim() == 4 and not torch.compiler.is_compiling():
            output = group_norm_silu_4d(
                x,
                self.weight,
                self.bias,
                self.num_groups,
                self.eps,
                apply_silu=True,
            )
            if output is not None:
                return output
        return F.silu(F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps))


def _install_norm_silu(decoder: nn.Module, resnet_cls: type[nn.Module], gate: VaeFastPathGate) -> int:
    def is_fusable(norm: object) -> bool:
        return type(norm) is nn.GroupNorm and norm.affine and norm.weight is not None and norm.bias is not None

    count = 0
    for module in decoder.modules():
        if (
            type(module) is resnet_cls
            and module.time_emb_proj is None
            and module.time_embedding_norm in ("default", "group")
            and module.upsample is None
            and module.downsample is None
            and isinstance(module.nonlinearity, nn.SiLU)
            and is_fusable(module.norm1)
            and is_fusable(module.norm2)
        ):
            module.norm1 = FusedGroupNormSiLU(module.norm1, gate)
            module.norm2 = FusedGroupNormSiLU(module.norm2, gate)
            module.nonlinearity = nn.Identity()
            count += 2

    if (
        type(getattr(decoder, "conv_norm_out", None)) is nn.GroupNorm
        and isinstance(getattr(decoder, "conv_act", None), nn.SiLU)
        and is_fusable(decoder.conv_norm_out)
    ):
        decoder.conv_norm_out = FusedGroupNormSiLU(decoder.conv_norm_out, gate)
        decoder.conv_act = nn.Identity()
        count += 1
    return count


_UPSAMPLE_TAP_MAP = {0: (2,), 1: (1, 2), 2: (0, 1), 3: (0,)}


def _fold_upsample2x_conv2d_weight(conv: nn.Conv2d) -> torch.Tensor:
    weight = conv.weight.detach().float()
    output_channels, input_channels = weight.shape[:2]
    folded = weight.new_zeros(input_channels, output_channels, 4, 4)
    for row in range(4):
        for column in range(4):
            value = weight.new_zeros(output_channels, input_channels)
            for source_row in _UPSAMPLE_TAP_MAP[row]:
                for source_column in _UPSAMPLE_TAP_MAP[column]:
                    value += weight[:, :, source_row, source_column]
            folded[:, :, row, column] = value.t()
    folded = folded.to(conv.weight.dtype)
    if conv.weight.is_contiguous(memory_format=torch.channels_last):
        folded = folded.contiguous(memory_format=torch.channels_last)
    return folded.to(conv.weight.device)


class FusedUpsample2xConv2d(nn.Module):
    """Quality-gated nearest-2x + Conv2d replacement using ConvTranspose2d."""

    def __init__(self, upsample: nn.Module, gate: VaeFastPathGate) -> None:
        super().__init__()
        # Keep only ``conv`` registered so checkpoint parameter names stay
        # ``...upsamplers.N.conv.*``. The original module is an eager fallback.
        object.__setattr__(self, "_original", upsample)
        self.conv = upsample.conv
        self.channels = upsample.channels
        self._vllm_gate = gate
        self.register_buffer("_fused_weight", None, persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_size: int | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not self._vllm_gate.enabled or output_size is not None or hidden_states.shape[1] != self.channels:
            return self._original(hidden_states, output_size=output_size, *args, **kwargs)

        weight = self._fused_weight
        if weight is None or weight.device != self.conv.weight.device or weight.dtype != self.conv.weight.dtype:
            weight = _fold_upsample2x_conv2d_weight(self.conv)
            self._fused_weight = weight
        return F.conv_transpose2d(hidden_states, weight, self.conv.bias, stride=2, padding=1)


def _install_fused_upsample(decoder: nn.Module, upsample_cls: type[nn.Module], gate: VaeFastPathGate) -> int:
    count = 0
    for block in decoder.up_blocks:
        upsamplers = getattr(block, "upsamplers", None)
        if not upsamplers:
            continue
        for index, upsample in enumerate(upsamplers):
            if type(upsample) is not upsample_cls:
                continue
            conv = getattr(upsample, "conv", None)
            if (
                upsample.use_conv
                and not upsample.use_conv_transpose
                and upsample.interpolate
                and upsample.norm is None
                and upsample.name == "conv"
                and type(conv) is nn.Conv2d
                and conv.kernel_size == (3, 3)
                and conv.stride == (1, 1)
                and conv.padding == (1, 1)
                and conv.dilation == (1, 1)
                and conv.groups == 1
                and conv.padding_mode == "zeros"
            ):
                upsamplers[index] = FusedUpsample2xConv2d(upsample, gate)
                count += 1
    return count


def _fold_attention_value_projection(module: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    value_weight = module.to_v.weight.detach().float()
    value_bias = module.to_v.bias.detach().float()
    output_weight = module.to_out[0].weight.detach().float()
    output_bias = module.to_out[0].bias.detach().float()
    dtype = module.to_v.weight.dtype
    return (
        (output_weight @ value_weight).to(dtype).contiguous(),
        (output_weight @ value_bias + output_bias).to(dtype),
    )


def _attention_fast_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    temb: torch.Tensor | None = None,
    **cross_attention_kwargs: Any,
) -> torch.Tensor:
    if (
        not self._vllm_gate.enabled
        or encoder_hidden_states is not None
        or attention_mask is not None
        or temb is not None
        or hidden_states.ndim != 4
    ):
        return type(self).forward(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
            **cross_attention_kwargs,
        )

    residual = hidden_states
    batch_size, channels, height, width = hidden_states.shape
    hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)

    if self.group_norm is not None:
        norm = self.group_norm
        normalized = None
        if not torch.compiler.is_compiling():
            normalized = group_norm_silu_rows(
                hidden_states,
                norm.weight,
                norm.bias,
                norm.num_groups,
                norm.eps,
                apply_silu=False,
            )
        hidden_states = normalized if normalized is not None else norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = self.to_q(hidden_states)
    key = self.to_k(hidden_states)
    folded_weight = self._vllm_folded_value_weight
    folded_bias = self._vllm_folded_value_bias
    if (
        folded_weight is None
        or folded_bias is None
        or folded_weight.device != hidden_states.device
        or folded_weight.dtype != self.to_v.weight.dtype
    ):
        folded_weight, folded_bias = _fold_attention_value_projection(self)
        self._vllm_folded_value_weight = folded_weight
        self._vllm_folded_value_bias = folded_bias
    value = F.linear(hidden_states, folded_weight, folded_bias)

    output = F.scaled_dot_product_attention(query.unsqueeze(1), key.unsqueeze(1), value.unsqueeze(1))
    output = output.squeeze(1).to(query.dtype)
    output = self.to_out[1](output)
    output = output.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
    if self.residual_connection:
        output = output + residual
    return output / self.rescale_output_factor


def _attention_fast_compatible(module: nn.Module, attention_cls: type, processor_cls: type) -> bool:
    return (
        type(module) is attention_cls
        and isinstance(module.processor, processor_cls)
        and module.heads == 1
        and module.scale_qk
        and module.spatial_norm is None
        and not module.norm_cross
        and getattr(module, "norm_q", None) is None
        and getattr(module, "norm_k", None) is None
        and getattr(module, "add_k_proj", None) is None
        and type(module.to_q) is nn.Linear
        and type(module.to_k) is nn.Linear
        and type(module.to_v) is nn.Linear
        and type(module.to_out[0]) is nn.Linear
        and isinstance(module.to_out[1], nn.Dropout)
        and module.to_v.bias is not None
        and module.to_out[0].bias is not None
    )


def _decoder_layout_forward(self: nn.Module, *args: Any, **kwargs: Any) -> torch.Tensor:
    use_channels_last = self._vllm_gate.enabled
    if use_channels_last != self._vllm_channels_last:
        memory_format = torch.channels_last if use_channels_last else torch.contiguous_format
        self.to(memory_format=memory_format)
        self._vllm_channels_last = use_channels_last
        logger.debug(
            "%s decoder switched to %s layout",
            self._vllm_label,
            "channels_last" if use_channels_last else "contiguous",
        )
    return type(self).forward(self, *args, **kwargs)


def maybe_optimize_image_vae(vae: nn.Module) -> nn.Module:
    """Install image VAE decoder fast paths when the structure is fully supported."""

    if not _HAS_TRITON or getattr(vae, "_vllm_vae_fast_path_installed", False):
        return vae

    from diffusers.models.attention_processor import Attention, AttnProcessor2_0
    from diffusers.models.autoencoders import AutoencoderKL
    from diffusers.models.autoencoders.vae import Decoder
    from diffusers.models.resnet import ResnetBlock2D
    from diffusers.models.upsampling import Upsample2D

    supported_types: tuple[type[nn.Module], ...] = (AutoencoderKL,)
    try:
        from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2

        supported_types += (AutoencoderKLFlux2,)
    except ImportError:  # pragma: no cover - older supported Diffusers builds
        pass

    if not isinstance(vae, supported_types):
        return vae
    decoder = getattr(vae, "decoder", None)
    if type(decoder) is not Decoder:
        return vae

    attention_modules = [
        module for module in decoder.modules() if _attention_fast_compatible(module, Attention, AttnProcessor2_0)
    ]
    attention_count = sum(isinstance(module, Attention) for module in decoder.modules())
    if len(attention_modules) != attention_count:
        logger.info(
            "%s has %d/%d attention blocks without a layout-safe rewrite; skipping VAE fast paths",
            type(vae).__name__,
            attention_count - len(attention_modules),
            attention_count,
        )
        return vae

    gate = VaeFastPathGate()
    decoder._vllm_gate = gate
    decoder._vllm_label = type(vae).__name__
    decoder._vllm_channels_last = False
    decoder.forward = MethodType(_decoder_layout_forward, decoder)
    upsample_count = _install_fused_upsample(decoder, Upsample2D, gate)
    for module in attention_modules:
        module._vllm_gate = gate
        module.register_buffer("_vllm_folded_value_weight", None, persistent=False)
        module.register_buffer("_vllm_folded_value_bias", None, persistent=False)
        module.forward = MethodType(_attention_fast_forward, module)
    norm_count = _install_norm_silu(decoder, ResnetBlock2D, gate)
    register_vae_fast_path_gate(vae, gate)
    vae._vllm_vae_fast_path_installed = True
    logger.info(
        "%s installed quality-gated VAE decoder fast paths "
        "(%d upsamplers, %d attention blocks, %d GroupNorm+SiLU fusions)",
        type(vae).__name__,
        upsample_count,
        len(attention_modules),
        norm_count,
    )
    return vae


__all__ = [
    "FusedGroupNormSiLU",
    "FusedUpsample2xConv2d",
    "maybe_optimize_image_vae",
]
