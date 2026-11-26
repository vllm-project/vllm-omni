# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in causal temporal padding inside the H3 NPU encoder's Conv3D."""

from __future__ import annotations

import os
import sys
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F

_ENABLE_ENV = "VLLM_OMNI_H3_VAE_TEMPORAL_PAD_FUSION"


def _enabled() -> bool:
    value = os.environ.get(_ENABLE_ENV, "0").strip().lower()
    if value in ("0", "false", "off", ""):
        return False
    if value in ("1", "true", "on"):
        return True
    raise ValueError(f"{_ENABLE_ENV} must be 0/1, false/true, or off/on")


def _is_npu_device(device: torch.device) -> bool:
    return device.type == "npu"


def _supported_conv(module: nn.Conv3d) -> bool:
    return (
        module.kernel_size[0] == 3
        and isinstance(module.padding, tuple)
        and len(module.padding) == 3
        and module.padding[0] == 1
        and all(type(p) is int and 0 <= p <= 1 for p in module.padding)
        and module.groups == 1
        and module.dilation == (1, 1, 1)
        and module.stride[0] in (1, 2)
        and getattr(module, "causal", None) is True
        and getattr(module, "pad_mode_t", None) == "constant"
        and getattr(module, "pad_mode", None) in ("constant", "reflect", "replicate")
        and not getattr(module, "spatial_parallel", True)
    )


def _padding_matches(module: nn.Conv3d) -> bool:
    """Check the loaded checkpoint's padding ABI without copying its weights."""
    ph, pw = module.padding[1:]
    try:
        with torch.no_grad():
            # Only D=17 is eligible, so check the ABI at that depth. T=1 uses the original forward.
            frames = 17
            probe = torch.arange(1, 20 * frames + 1, device="cpu", dtype=torch.float32).view(1, 1, frames, 4, 5)
            spatial = F.pad(probe, (pw, pw, ph, ph, 0, 0), mode=module.pad_mode)
            expected = F.pad(spatial, (0, 0, 0, 0, 2, 0))
            actual = module._apply_padding(probe)
            if not (
                isinstance(actual, torch.Tensor)
                and actual.dtype == expected.dtype
                and actual.device == expected.device
                and actual.stride() == expected.stride()
                and torch.equal(actual, expected)
            ):
                return False
    except (torch.OutOfMemoryError, torch.AcceleratorError):
        raise
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False
    return True


def _eligible(module: nn.Conv3d, x: torch.Tensor) -> bool:
    if not (
        x.ndim == 5
        and x.shape[2] == 17
        and _is_npu_device(x.device)
        and x.dtype == torch.float32
        and not torch.compiler.is_compiling()
        and not torch.is_grad_enabled()
        and not module.training
        and _supported_conv(module)
        and module.weight.device == x.device
        and module.weight.dtype == x.dtype
        and (module.bias is None or (module.bias.device == x.device and module.bias.dtype == x.dtype))
    ):
        return False
    return not (
        torch.is_autocast_enabled(x.device.type)
        or module._forward_pre_hooks
        or module._forward_hooks
        or nn.modules.module._global_forward_pre_hooks
        or nn.modules.module._global_forward_hooks
    )


def _forward(self: nn.Conv3d, x: torch.Tensor) -> torch.Tensor:
    if not _eligible(self, x):
        return self._omni_temporal_pad_forward(x)
    ph, pw = self.padding[1:]
    # Keep the checkpoint's spatial pad, including its zero-width call. That
    # copy removal is a separate optimization; only temporal padding changes.
    x = F.pad(x, (pw, pw, ph, ph, 0, 0), mode=self.pad_mode)
    output = F.conv3d(x, self.weight, self.bias, self.stride, (2, 0, 0), self.dilation, self.groups)
    # Symmetric padding adds trailing outputs. Keep the prefix obtained with
    # the original left-only two-frame pad, including temporal stride 2.
    frames = (x.shape[2] + 2 - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
    return output[:, :, :frames]


def install_npu_temporal_pad_fusion(model: nn.Module, *, device: torch.device) -> int:
    """Install once on the checkpoint encoder; disabled by default.

    The public NPU accuracy and latency gates are still required before
    changing this default. Decoder and quant_conv modules are out of scope.
    """
    if not _is_npu_device(device) or not _enabled():
        return 0
    encoder = getattr(model, "encoder", None)
    namespace = vars(sys.modules[type(encoder).__module__]) if isinstance(encoder, nn.Module) else {}
    if namespace.get("EncoderFCN3D") is not type(encoder):
        return 0
    conv_type = namespace.get("SpatialParallelConv3d")
    if not isinstance(conv_type, type) or not issubclass(conv_type, nn.Conv3d):
        return 0
    installed = 0
    for module in encoder.modules():
        if type(module) is not conv_type or not _supported_conv(module):
            continue
        if hasattr(module, "_omni_temporal_pad_forward"):
            installed += 1
            continue
        if "forward" in vars(module) or "_apply_padding" in vars(module) or not _padding_matches(module):
            continue
        module._omni_temporal_pad_forward = module.forward
        module.forward = MethodType(_forward, module)
        installed += 1
    return installed


__all__ = ["install_npu_temporal_pad_fusion"]
