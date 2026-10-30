# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in GroupNorm/SiLU fusion for the H3 NPU image encoder."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from types import FunctionType, MethodType

import torch
import torch.nn as nn

_IMAGE_ENCODE = ContextVar("h3_npu_image_norm", default=False)
_ENV = "VLLM_OMNI_H3_NPU_IMAGE_GN_SILU"


@contextmanager
def image_encode_context() -> Iterator[None]:
    token = _IMAGE_ENCODE.set(True)
    try:
        yield
    finally:
        _IMAGE_ENCODE.reset(token)


def _eligible(x: torch.Tensor, norm: nn.Module, cond: torch.Tensor | None) -> bool:
    return (
        _IMAGE_ENCODE.get()
        and not torch.compiler.is_compiling()
        and cond is None
        and not norm.training
        and not torch.is_grad_enabled()
        and x.device.type == "npu"
        and x.dtype == torch.float32
        and x.ndim == 5
        and x.shape[2] == 1
        and x.is_contiguous()
        and getattr(norm, "_omni_image_norm_supported", False)
        and not norm._forward_hooks
        and not norm._forward_pre_hooks
        and "forward" not in norm.__dict__
        and not nn.modules.module._global_forward_hooks
        and not nn.modules.module._global_forward_pre_hooks
        and not getattr(norm, "spatial_parallel", False)
        and norm.weight is not None
        and norm.bias is not None
        and norm.weight.device == x.device
        and norm.bias.device == x.device
        and norm.weight.dtype == x.dtype
        and norm.bias.dtype == x.dtype
        and norm.eps > 0
        and x.shape[1] == norm.num_channels
        and x.numel() > 0
    )


def install_npu_image_norm(model: nn.Module, *, device: torch.device) -> int:
    """Bind private forward namespaces; never mutate the remote-code globals."""
    encoder = getattr(model, "encoder", None)
    if device.type != "npu" or os.environ.get(_ENV, "0") != "1" or not isinstance(encoder, nn.Module):
        return 0
    namespace = vars(sys.modules[type(encoder).__module__])
    if namespace.get("EncoderFCN3D") is not type(encoder):
        return 0
    original = namespace.get("norm_silu")
    factory = namespace.get("get_group_norm_3d")
    if not isinstance(original, FunctionType) or not isinstance(factory, FunctionType):
        return 0
    # These concrete checkpoint types use ordinary GroupNorm with spatial
    # parallelism disabled; merging T into N is an identity when T=1.
    norm_types = (nn.GroupNorm,) + tuple(
        cls
        for name in ("SpatialParallelGroupNorm", "TemporalIsolatedSpatialParallelGroupNorm")
        if isinstance(cls := factory.__globals__.get(name), type) and issubclass(cls, nn.GroupNorm)
    )
    import torch_npu

    fused = getattr(torch_npu, "npu_group_norm_silu", None)
    if not callable(fused):
        return 0
    for norm in encoder.modules():
        if isinstance(norm, nn.GroupNorm):
            norm._omni_image_norm_supported = type(norm) in norm_types

    def norm_silu(x: torch.Tensor, norm: nn.Module, cond: torch.Tensor | None = None) -> torch.Tensor:
        if _eligible(x, norm, cond):
            return fused(x, norm.weight, norm.bias, norm.num_groups, norm.eps)[0]
        return original(x, norm, cond)

    installed = 0
    block_types = (type(encoder), namespace.get("ResnetBlock3D"))
    for module in encoder.modules():
        if getattr(module, "_omni_image_norm_installed", False):
            installed += 1
            continue
        # These are the two official CNN call sites. Preserve their complete
        # forward body, including checkpointing, residual and downsample order.
        if type(module) not in block_types:
            continue
        forward = type(module).forward
        if not isinstance(forward, FunctionType) or forward.__closure__ or "forward" in module.__dict__:
            continue
        if forward.__globals__.get("norm_silu") is not original:
            continue

        namespace = dict(forward.__globals__, norm_silu=norm_silu)
        private_forward = FunctionType(forward.__code__, namespace, forward.__name__, forward.__defaults__)
        private_forward.__kwdefaults__ = forward.__kwdefaults__
        module.forward = MethodType(private_forward, module)
        module._omni_image_norm_installed = True
        installed += 1
    return installed
