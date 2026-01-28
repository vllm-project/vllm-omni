from collections.abc import Callable
from typing import Any

import torch.nn as nn

from vllm_omni.platforms import current_omni_platform


class CustomOp(nn.Module):
    """
    Base class for custom ops.
    Dispatches the forward method to the appropriate backend.
    """

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

    def forward_npu(self, *args, **kwargs):
        raise NotImplementedError

    def forward_xpu(self, *args, **kwargs):
        raise NotImplementedError

    def forward_hip(self, *args, **kwargs):
        # By default, we assume that HIP ops are compatible with CUDA ops.
        return self.forward_cuda(*args, **kwargs)
