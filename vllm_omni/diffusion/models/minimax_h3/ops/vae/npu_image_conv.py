# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Causal T=1 Conv3D reduction for the H3 NPU image encoder."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F

_IMAGE_ENCODE: ContextVar[dict[int, tuple[int | None, torch.Tensor]] | None] = ContextVar(
    "h3_npu_image_conv", default=None
)


def _supported_layout(module: nn.Conv3d) -> bool:
    return (
        getattr(module, "causal", False)
        and getattr(module, "pad_mode_t", None) == "constant"
        and module.kernel_size[0] in (1, 3)
        and module.padding[0] == module.kernel_size[0] // 2
        and getattr(module, "pad_mode", None) in ("constant", "reflect", "replicate")
    )


@contextmanager
def image_encode_context() -> Iterator[None]:
    """Keep packed weights only for this image's checkpoint encode."""
    cache: dict[int, tuple[int | None, torch.Tensor]] = {}
    token = _IMAGE_ENCODE.set(cache)
    try:
        yield
    finally:
        cache.clear()
        _IMAGE_ENCODE.reset(token)


def _eligible(module: nn.Conv3d, x: torch.Tensor) -> bool:
    return (
        _IMAGE_ENCODE.get() is not None
        and not torch.compiler.is_compiling()
        and not module.training
        and not torch.is_grad_enabled()
        and x.device.type == "npu"
        and x.dtype == torch.float32
        and x.ndim == 5
        and x.shape[2] == 1
        and module.weight.device == x.device
        and module.weight.dtype == x.dtype
        and (module.bias is None or (module.bias.device == x.device and module.bias.dtype == x.dtype))
        and module.groups == 1
        and module.dilation == (1, 1, 1)
        and not getattr(module, "spatial_parallel", False)
        and _supported_layout(module)
    )


def _weight(module: nn.Conv3d) -> torch.Tensor:
    source = module.weight
    cache = _IMAGE_ENCODE.get()
    if cache is None or source.shape[2] == 1:
        return source[:, :, -1].contiguous().unsqueeze(2)
    stream = None
    if source.device.type != "cpu":
        backend = torch.get_device_module(source.device)
        stream = getattr(backend.current_stream(source.device), "npu_stream", None)
        if stream is None:
            return source[:, :, -1].contiguous().unsqueeze(2)
    module_id = id(module)
    cached = cache.get(module_id)
    if cached is not None and cached[0] == stream:
        return cached[1]
    # The checkpoint keeps its parameters fixed throughout encode_images.
    # Repack on stream changes so the copy and convolution share a stream.
    packed = source[:, :, -1].contiguous().unsqueeze(2)
    cache[module_id] = (stream, packed)
    return packed


def _forward(self: nn.Conv3d, x: torch.Tensor) -> torch.Tensor:
    if not _eligible(self, x):
        return self._omni_image_conv_forward(x)
    # Keep the Conv3D backend and its current HF32 policy. For causal T=1,
    # preceding temporal slices multiply padded zeros. The derived cache
    # reuses only this exact contiguous slice within one image encode.
    ph, pw = self.padding[1:]
    if ph or pw:
        x = F.pad(x.squeeze(2), (pw, pw, ph, ph), mode=self.pad_mode).unsqueeze(2)
    weight = _weight(self)
    return F.conv3d(x, weight, self.bias, self.stride, 0, self.dilation, self.groups)


def install_npu_image_conv(model: nn.Module, *, device: torch.device) -> int:
    """Install once on supported checkpoint encoder convolutions."""
    if device.type != "npu":
        return 0
    encoder = getattr(model, "encoder", None)
    if not isinstance(encoder, nn.Module):
        return 0
    namespace = vars(sys.modules[type(encoder).__module__])
    # Bind the loaded H3 CNN's concrete convolution class, not an arbitrary
    # subclass with the same name. This adapter follows the checkpoint ABI:
    # causal constant temporal padding and ordinary non-parallel Conv3D.
    if namespace.get("EncoderFCN3D") is not type(encoder):
        return 0
    conv_type = namespace.get("SpatialParallelConv3d")
    if not isinstance(conv_type, type) or not issubclass(conv_type, nn.Conv3d):
        return 0
    installed = 0
    for module in encoder.modules():
        if type(module) is not conv_type or not _supported_layout(module):
            continue
        if not hasattr(module, "_omni_image_conv_forward"):
            if "forward" in module.__dict__:
                continue
            module._omni_image_conv_forward = module.forward
            module.forward = MethodType(_forward, module)
        installed += 1
    return installed
