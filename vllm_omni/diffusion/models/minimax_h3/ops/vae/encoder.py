# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Callable
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

from .dispatch import CausalReflectPadOp, resolve_h3_vae_operators

logger = init_logger(__name__)

_Plan = list[tuple[nn.Module, Callable[..., torch.Tensor] | None, dict[str, object]]]


def _flag_off(value: object) -> bool:
    return value is False or type(value) is int and value == 0


def _has_forward_hooks(*modules: nn.Module) -> bool:
    if nn.modules.module._global_forward_pre_hooks or nn.modules.module._global_forward_hooks:
        return True
    return any(module._forward_pre_hooks or module._forward_hooks for module in modules)


def _causal_padding(conv: nn.Conv3d, frames: int) -> tuple[int, int, int]:
    pad_t, pad_h, pad_w = conv.padding
    front = 2 * pad_t if frames > 1 else conv.kernel_size[0] - 1
    return front, pad_h, pad_w


def _conv_padded(conv: nn.Conv3d, padded: torch.Tensor) -> torch.Tensor:
    return F.conv3d(padded, conv.weight, conv.bias, stride=conv.stride, padding=0, dilation=conv.dilation)


def _padded_conv_forward(self: nn.Conv3d, x: torch.Tensor) -> torch.Tensor:
    if not self.spatial_parallel and x.ndim == 5:
        front, pad_h, pad_w = _causal_padding(self, x.shape[2])
        padded = self._omni_causal_reflect_pad(
            x, front=front, top=pad_h, bottom=pad_h, left=pad_w, right=pad_w, apply_silu=False
        )
        if padded is not None:
            return _conv_padded(self, padded)
    return type(self).forward(self, x)


def _unpadded_conv_forward(self: nn.Conv3d, x: torch.Tensor) -> torch.Tensor:
    if not torch.compiler.is_compiling() and x.is_contiguous() and x.data_ptr() % 512 == 0:
        return F.conv3d(x, self.weight, self.bias, self.stride, (0, 0, 0), self.dilation, self.groups)
    return type(self).forward(self, x)


def _downsample_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    conv = self.conv
    if self.space_stride == 2 and not conv.spatial_parallel and x.ndim == 5 and not _has_forward_hooks(conv):
        front = _causal_padding(conv, x.shape[2])[0]
        padded = conv._omni_causal_reflect_pad(x, front=front, top=0, bottom=1, left=0, right=1, apply_silu=False)
        if padded is not None:
            return _conv_padded(conv, padded)
    return type(self).forward(self, x)


def _norm_silu_conv(norm: nn.GroupNorm, conv: nn.Conv3d, x: torch.Tensor) -> torch.Tensor:
    if (
        norm.spatial_parallel
        or conv.spatial_parallel
        or x.ndim != 5
        or torch.is_grad_enabled()
        or _has_forward_hooks(norm, conv)
    ):
        return conv(F.silu(norm(x)))
    batch, channels, frames, height, width = x.shape
    if norm._omni_time_isolated:
        merged = x.permute(0, 2, 1, 3, 4).contiguous().view(batch * frames, channels, 1, height, width)
        source = nn.GroupNorm.forward(norm, merged)
        del merged
        source = source.view(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)
    else:
        source = nn.GroupNorm.forward(norm, x)
    front, pad_h, pad_w = _causal_padding(conv, frames)
    padded = conv._omni_causal_reflect_pad(
        source, front=front, top=pad_h, bottom=pad_h, left=pad_w, right=pad_w, apply_silu=True
    )
    if padded is None:
        return conv(F.silu(source.contiguous()))
    del source
    return _conv_padded(conv, padded)


def _resnet_block_forward(self: nn.Module, x: torch.Tensor, zq: torch.Tensor | None = None) -> torch.Tensor:
    if zq is not None:
        return type(self).forward(self, x, zq)
    h = _norm_silu_conv(self.norm1, self.conv1, x)
    h = _norm_silu_conv(self.norm2, self.conv2, h)
    if self.in_channels != self.out_channels:
        x = self.nin_shortcut(x)
    return x + h


def _encoder_forward(self: nn.Module, x: torch.Tensor, zq: torch.Tensor | None = None) -> torch.Tensor:
    if zq is not None or self.training:
        return type(self).forward(self, x, zq)
    h = self.conv_in(x)
    for i_level in range(self.num_levels):
        for i_block in range(self.num_res_blocks[i_level]):
            h = self.down[i_level].block[i_block](h, zq)
        if hasattr(self.down[i_level], "downsample"):
            h = self.down[i_level].downsample(h)
    return _norm_silu_conv(self.norm_out, self.conv_out, h)


def _probe_matches(module: nn.Module, matches: Callable[[], bool]) -> bool:
    name = type(module).__name__
    try:
        with torch.no_grad():
            if matches():
                return True
        logger.debug("MiniMax-H3 VAE encoder fusion declined: %s probe mismatch", name)
    except (torch.OutOfMemoryError, torch.AcceleratorError):
        raise
    except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
        logger.debug("MiniMax-H3 VAE encoder fusion declined: %s probe raised %r", name, exc)
    return False


def _conv_plan(plan: _Plan, conv: nn.Module, pad_op: CausalReflectPadOp) -> bool:
    if not (
        type(conv).__name__ == "SpatialParallelConv3d"
        and isinstance(conv, nn.Conv3d)
        and _flag_off(getattr(conv, "spatial_parallel", None))
        and getattr(conv, "pad_mode", None) == "reflect"
        and getattr(conv, "pad_mode_t", None) == "constant"
        and getattr(conv, "causal", None) is True
        and conv.groups == 1
        and len(conv.padding) == 3
        and all(type(pad) is int and pad >= 0 for pad in conv.padding)
        and sum(conv.padding) > 0
        and max(conv.padding[1:]) <= 1
        and callable(getattr(conv, "_apply_padding", None))
    ):
        return False

    def matches() -> bool:
        for frames in (1, 3):
            probe = torch.arange(24 * frames, dtype=torch.float32, device="cpu").view(1, 2, frames, 3, 4)
            front, pad_h, pad_w = _causal_padding(conv, frames)
            expected = F.pad(F.pad(probe, (pad_w, pad_w, pad_h, pad_h, 0, 0), mode="reflect"), (0, 0, 0, 0, front, 0))
            actual = conv._apply_padding(probe)
            if not (isinstance(actual, torch.Tensor) and torch.equal(actual, expected)):
                return False
        return True

    plan.append((conv, _padded_conv_forward, {"_omni_causal_reflect_pad": pad_op}))
    return _probe_matches(conv, matches)


def _norm_plan(plan: _Plan, norm: nn.Module) -> bool:
    name = type(norm).__name__
    if not (
        name in ("TemporalIsolatedSpatialParallelGroupNorm", "SpatialParallelGroupNorm")
        and isinstance(norm, nn.GroupNorm)
        and _flag_off(getattr(norm, "spatial_parallel", None))
        and norm.affine
    ):
        return False
    time_isolated = name == "TemporalIsolatedSpatialParallelGroupNorm"
    channels = norm.num_channels
    cpu_probe = torch.randn(1, channels, 2, 2, 3, generator=torch.Generator().manual_seed(0), device="cpu")

    def matches() -> bool:
        probe = cpu_probe.to(device=norm.weight.device, dtype=norm.weight.dtype)
        expected = norm(probe)
        if time_isolated:
            merged = probe.permute(0, 2, 1, 3, 4).contiguous().view(2, channels, 1, 2, 3)
            actual = nn.GroupNorm.forward(norm, merged).view(1, 2, channels, 2, 3).permute(0, 2, 1, 3, 4)
        else:
            actual = nn.GroupNorm.forward(norm, probe)
        return isinstance(expected, torch.Tensor) and torch.equal(expected, actual.contiguous())

    plan.append((norm, None, {"_omni_time_isolated": time_isolated}))
    return _probe_matches(norm, matches)


def _encoder_fusion_plan(encoder: nn.Module, pad_op: CausalReflectPadOp) -> _Plan | None:
    plan: _Plan = [(encoder, _encoder_forward, {})]
    if not (
        type(encoder).__name__ == "EncoderFCN3D"
        and getattr(encoder, "use_fused_norm", None) is False
        and isinstance(getattr(encoder, "down", None), nn.ModuleList)
        and len(encoder.down) == getattr(encoder, "num_levels", None)
        and len(getattr(encoder, "num_res_blocks", ())) == encoder.num_levels
        and _conv_plan(plan, getattr(encoder, "conv_in", None), pad_op)
        and _conv_plan(plan, getattr(encoder, "conv_out", None), pad_op)
        and _norm_plan(plan, getattr(encoder, "norm_out", None))
    ):
        return None
    for i_level, level in enumerate(encoder.down):
        blocks = getattr(level, "block", None)
        if not isinstance(blocks, nn.ModuleList) or len(blocks) != encoder.num_res_blocks[i_level]:
            return None
        for block in blocks:
            if not (
                type(block).__name__ == "ResnetBlock3D"
                and getattr(block, "use_fused_norm", None) is False
                and _conv_plan(plan, getattr(block, "conv1", None), pad_op)
                and _conv_plan(plan, getattr(block, "conv2", None), pad_op)
                and _norm_plan(plan, getattr(block, "norm1", None))
                and _norm_plan(plan, getattr(block, "norm2", None))
            ):
                return None
            plan.append((block, _resnet_block_forward, {}))
            if block.in_channels != block.out_channels:
                shortcut = getattr(block, "nin_shortcut", None)
                if not (
                    type(shortcut).__name__ == "SpatialParallelConv3d"
                    and isinstance(shortcut, nn.Conv3d)
                    and shortcut.padding == (0, 0, 0)
                    and shortcut.padding_mode != "zeros"
                ):
                    return None
                plan.append((shortcut, _unpadded_conv_forward, {}))
        downsample = getattr(level, "downsample", None)
        if downsample is not None:
            conv = getattr(downsample, "conv", None)
            if not (
                type(downsample).__name__ == "Downsample3D"
                and getattr(downsample, "pad_mode", None) == "reflect"
                and getattr(downsample, "space_stride", None) in (1, 2, 3)
                and _conv_plan(plan, conv, pad_op)
                and conv.padding[1:] == (0, 0)
            ):
                return None
            plan.append((downsample, _downsample_forward, {}))
    return None if any("forward" in vars(module) for module, _, _ in plan) else plan


def install_h3_vae_encoder_fusion(encoder: nn.Module, *, device: torch.device) -> bool:
    if getattr(encoder, "_omni_h3_vae_encoder_fusion_installed", False):
        return True
    pad_op = getattr(resolve_h3_vae_operators(device), "causal_reflect_pad", None)
    plan = None if pad_op is None else _encoder_fusion_plan(encoder, pad_op)
    if plan is None:
        return False
    for module, forward, attributes in plan:
        for name, value in attributes.items():
            setattr(module, name, value)
        if forward is not None:
            module.forward = MethodType(forward, module)
    encoder._omni_h3_vae_encoder_fusion_installed = True
    return True


__all__ = ["install_h3_vae_encoder_fusion"]
