# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in removal of empty spatial padding in H3 NPU video downsampling."""

from __future__ import annotations

import os
import sys
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F

_ENABLE_ENV = "VLLM_OMNI_H3_VAE_SKIP_NOOP_PAD"


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
        module.padding == (1, 0, 0)
        and module.kernel_size[0] == 3
        and module.groups == 1
        and module.dilation == (1, 1, 1)
        and getattr(module, "causal", None) is True
        and getattr(module, "pad_mode_t", None) == "constant"
        and not getattr(module, "spatial_parallel", True)
    )


def _padding_matches(module: nn.Conv3d) -> bool:
    """Verify that spatial padding is only an empty copy on this checkpoint."""
    try:
        with torch.no_grad():
            for frames in (3, 17):
                probe = torch.arange(1, 20 * frames + 1, device="cpu", dtype=torch.float32).view(1, 1, frames, 4, 5)
                expected = F.pad(probe, (0, 0, 0, 0, 2 * module.padding[0], 0))
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
        and x.shape[2] > 1
        and _is_npu_device(x.device)
        and x.dtype == torch.float32
        and x.is_contiguous()
        and not torch.compiler.is_compiling()
        and not torch.is_grad_enabled()
        and not module.training
        and _supported_conv(module)
        and module.weight.device == x.device
        and module.weight.dtype == x.dtype
    ):
        return False
    return not (
        torch.is_autocast_enabled(x.device.type)
        or module._forward_pre_hooks
        or module._forward_hooks
        or nn.modules.module._global_forward_pre_hooks
        or nn.modules.module._global_forward_hooks
    )


def _apply_padding(self: nn.Conv3d, x: torch.Tensor) -> torch.Tensor:
    if not _eligible(self, x):
        return self._omni_noop_spatial_pad(x)
    # The all-zero spatial F.pad is gone. Keep the original left temporal
    # zero pad and let the checkpoint's own forward perform the convolution.
    return F.pad(x, (0, 0, 0, 0, 2 * self.padding[0], 0), mode=self.pad_mode_t)


def install_npu_skip_noop_pad(model: nn.Module, *, device: torch.device) -> int:
    """Install on encoder convolutions with temporal-only padding, once.

    The caller keeps downsampling's outer right/bottom pad and the original
    Conv3D forward. T=1 images, decoder, and quant_conv are out of scope.
    """
    if not _is_npu_device(device) or not _enabled():
        return 0
    encoder = getattr(model, "encoder", None)
    if not isinstance(encoder, nn.Module):
        return 0
    encoder_module = sys.modules.get(type(encoder).__module__)
    if getattr(encoder_module, "EncoderFCN3D", None) is not type(encoder):
        return 0
    conv_type = getattr(encoder_module, "SpatialParallelConv3d", None)
    if not isinstance(conv_type, type) or not issubclass(conv_type, nn.Conv3d):
        return 0
    installed = 0
    for module in encoder.modules():
        if type(module) is not conv_type or not _supported_conv(module):
            continue
        if hasattr(module, "_omni_noop_spatial_pad"):
            installed += 1
            continue
        if "_apply_padding" in vars(module) or not _padding_matches(module):
            continue
        module._omni_noop_spatial_pad = module._apply_padding
        module._apply_padding = MethodType(_apply_padding, module)
        installed += 1
    return installed


__all__ = ["install_npu_skip_noop_pad"]
