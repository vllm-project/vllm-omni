# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in causal T=1 Conv3D reduction for the H3 NPU image encoder."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F

_IMAGE_ENCODE = ContextVar("h3_npu_image_conv", default=False)
_ENV = "VLLM_OMNI_H3_NPU_IMAGE_T1_CONV"
_CACHE_ENV = "VLLM_OMNI_H3_NPU_IMAGE_T1_WEIGHT_CACHE"


def _supported_module(module: nn.Module, conv_type: type[nn.Conv3d] | None = None) -> bool:
    if type(module) is nn.Conv3d:
        return module.kernel_size[0] == 1 and module.padding == (0, 0, 0) and module.padding_mode == "zeros"
    return type(module) is conv_type


@contextmanager
def image_encode_context() -> Iterator[None]:
    """The video and decoder paths never enter this scope."""
    token = _IMAGE_ENCODE.set(True)
    try:
        yield
    finally:
        _IMAGE_ENCODE.reset(token)


def _eligible(module: nn.Conv3d, x: torch.Tensor) -> bool:
    if not (
        _IMAGE_ENCODE.get()
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
    ):
        return False
    if type(module) is nn.Conv3d:
        return module.kernel_size[0] == 1 and module.padding == (0, 0, 0) and module.padding_mode == "zeros"
    return (
        module.causal
        and module.pad_mode_t == "constant"
        and module.kernel_size[0] in (1, 3)
        and module.padding[0] == module.kernel_size[0] // 2
        and module.pad_mode in ("constant", "reflect", "replicate")
    )


def _clear_weight(module: nn.Module, *args) -> None:
    module._omni_image_conv_weight = None


def _encoder_modules(model: nn.Module) -> Iterator[nn.Module]:
    for part in ("encoder", "quant_conv"):
        root = getattr(model, part, None)
        if isinstance(root, nn.Module):
            yield from root.modules()


def clear_npu_image_conv_cache(model: nn.Module) -> None:
    """Release derived encoder weights before H3 residency storage is rebound.

    Also call before edits through .data or external aliases, which can bypass
    PyTorch's version counter. No parameter or state_dict entry is added.
    """
    for module in _encoder_modules(model):
        if getattr(module, "_omni_image_conv_cache_enabled", False):
            _clear_weight(module)


def _apply(self: nn.Module, fn: Callable, recurse: bool = True):
    _clear_weight(self)
    return self._omni_image_conv_apply(fn, recurse=recurse)


def _weight_key(weight: torch.Tensor) -> tuple | None:
    try:
        version = weight._version
    except RuntimeError:
        # Inference tensors cannot report mutations: use the live slice.
        return None
    if weight.device.type == "cpu":
        stream = None
    else:
        backend = torch.get_device_module(weight.device)
        current = backend.current_stream(weight.device)
        stream = getattr(current, "npu_stream", None)
        if stream is None:
            return None
    return (
        version,
        weight.data_ptr(),
        weight.untyped_storage()._cdata,
        weight.device,
        weight.dtype,
        tuple(weight.shape),
        tuple(weight.stride()),
        stream,
    )


def _weight(module: nn.Conv3d) -> torch.Tensor:
    source = module.weight
    use_cache = getattr(module, "_omni_image_conv_cache_enabled", False) and source.shape[2] != 1
    key = _weight_key(source) if use_cache else None
    if use_cache:
        cached = module._omni_image_conv_weight
        if key is not None and cached is not None and cached[0] is source and cached[1] == key:
            return cached[2]
        _clear_weight(module)
    packed = source[:, :, -1, :, :].contiguous().unsqueeze(2)
    if key is not None:
        module._omni_image_conv_weight = (source, key, packed)
    return packed


def _forward(self: nn.Conv3d, x: torch.Tensor) -> torch.Tensor:
    if not _eligible(self, x):
        if getattr(self, "_omni_image_conv_cache_enabled", False):
            _clear_weight(self)
        return self._omni_image_conv_forward(x)
    # Keep the Conv3D backend and its current HF32 policy. For causal T=1,
    # preceding temporal slices multiply padded zeros. The optional cache
    # reuses only this exact contiguous slice within one residency interval.
    ph, pw = self.padding[1:]
    if ph or pw:
        x = F.pad(x.squeeze(2), (pw, pw, ph, ph), mode=self.pad_mode).unsqueeze(2)
    weight = _weight(self)
    return F.conv3d(x, weight, self.bias, self.stride, 0, self.dilation, self.groups)


def install_npu_image_conv(model: nn.Module, *, device: torch.device) -> int:
    """Install once on encoder/quant_conv only; default OFF on every device."""
    if device.type != "npu" or os.environ.get(_ENV, "0") != "1":
        return 0
    encoder = getattr(model, "encoder", None)
    namespace = vars(sys.modules[type(encoder).__module__]) if encoder is not None else {}
    # Bind the loaded H3 CNN's concrete convolution class, not an arbitrary
    # subclass with the same name. This adapter follows the checkpoint ABI:
    # causal constant temporal padding and ordinary non-parallel Conv3D.
    conv_type = namespace.get("SpatialParallelConv3d") if namespace.get("EncoderFCN3D") is type(encoder) else None
    if not isinstance(conv_type, type) or not issubclass(conv_type, nn.Conv3d):
        conv_type = None
    installed = 0
    cache_enabled = os.environ.get(_CACHE_ENV, "0") == "1"
    for module in _encoder_modules(model):
        if not isinstance(module, nn.Conv3d) or not _supported_module(module, conv_type):
            continue
        if not hasattr(module, "_omni_image_conv_forward"):
            if "forward" in module.__dict__:
                continue
            module._omni_image_conv_forward = module.forward
            module.forward = MethodType(_forward, module)
        if cache_enabled and not getattr(module, "_omni_image_conv_cache_enabled", False):
            module._omni_image_conv_cache_enabled = True
            module._omni_image_conv_weight = None
            module._omni_image_conv_apply = module._apply
            module._apply = MethodType(_apply, module)
            module.register_load_state_dict_pre_hook(_clear_weight)
        installed += 1
    return installed
