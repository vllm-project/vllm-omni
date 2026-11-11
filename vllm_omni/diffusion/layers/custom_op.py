# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, TypeVar

import torch.nn as nn

from vllm_omni.platforms import current_omni_platform

_Op = TypeVar("_Op", bound="CustomOp")


class CustomOp(nn.Module):
    """
    Base class for custom ops.
    Dispatches the forward method to the appropriate backend.
    """

    _oot_registry: ClassVar[dict[type[CustomOp], type[CustomOp]]] = {}

    def __new__(cls, *args: Any, **kwargs: Any) -> CustomOp:
        implementation = cls
        if CustomOp._oot_registry and current_omni_platform.is_out_of_tree():
            implementation = CustomOp._oot_registry.get(cls, cls)
        return super().__new__(implementation)

    @classmethod
    def register_oot(cls, implementation: type[_Op]) -> type[_Op]:
        """Register an external subclass before constructing operators.

        Built-in platforms retain their dispatch. Registrations are keyed by
        class identity; duplicate registrations and unrelated types are errors.
        """
        if not issubclass(implementation, cls):
            raise TypeError(f"{implementation.__name__} must inherit {cls.__name__}")
        if cls in CustomOp._oot_registry:
            raise ValueError(f"An external implementation is already registered for {cls.__name__}")
        CustomOp._oot_registry[cls] = implementation
        return implementation

    def __init__(self) -> None:
        super().__init__()
        self._forward_method = self.dispatch_forward()

    def dispatch_forward(self) -> Callable:
        if current_omni_platform.is_rocm():
            return self.forward_hip
        elif current_omni_platform.is_cuda():
            return self.forward_cuda
        elif current_omni_platform.is_npu():
            return self.forward_npu
        elif current_omni_platform.is_xpu():
            return self.forward_xpu
        elif current_omni_platform.is_musa():
            return self.forward_musa
        elif current_omni_platform.is_out_of_tree():
            return self.forward_oot
        else:
            return self.forward_native

    def forward(self, *args, **kwargs) -> Any:
        return self._forward_method(*args, **kwargs)

    def forward_native(self, *args, **kwargs):
        """PyTorch-native implementation of the forward method.
        This method is optional. If implemented, it can be used with compilers
        such as torch.compile or PyTorch XLA. Also, it can be used for testing
        purposes.
        """
        raise NotImplementedError

    def forward_cuda(self, *args, **kwargs):
        raise NotImplementedError

    def forward_oot(self, *args, **kwargs):
        """External backend implementation; default to the native reference."""
        return self.forward_native(*args, **kwargs)

    def forward_npu(self, *args, **kwargs):
        raise NotImplementedError

    def forward_xpu(self, *args, **kwargs):
        # By default, we assume that XPU ops are compatible with the
        # PyTorch-native implementation.
        return self.forward_native(*args, **kwargs)

    def forward_hip(self, *args, **kwargs):
        # By default, we assume that HIP ops are compatible with CUDA ops.
        return self.forward_cuda(*args, **kwargs)

    def forward_musa(self, *args, **kwargs):
        # By default, we assume that MUSA ops are compatible with CUDA ops.
        return self.forward_cuda(*args, **kwargs)
