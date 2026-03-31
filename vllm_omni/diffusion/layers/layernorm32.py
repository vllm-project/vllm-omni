from importlib.util import find_spec

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

_HAS_MINDIESD = find_spec("mindiesd") is not None


class LayerNorm32(nn.LayerNorm):
    """FP32-safe layernorm implementation.

    Matches the numerical behavior of diffusers' ``FP32LayerNorm``:
    inputs and affine parameters are promoted to float32 for normalization,
    then cast back to the original dtype.
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        origin_dtype = x.dtype
        return F.layer_norm(
            x.float(),
            normalized_shape=[self.dim],
            weight=self.weight.float() if self.weight is not None else None,
            bias=self.bias.float() if self.bias is not None else None,
            eps=self.eps,
        ).to(origin_dtype)


class FastLayerNorm32(LayerNorm32):
    """Platform-dispatched FP32-safe fast layernorm.

    NPU:
        Uses ``mindiesd.fast_layernorm(self, x)`` when MindIE-SD is installed.
    CUDA / HIP / XPU / native:
        Falls back to the FP32-safe reference implementation.
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__(dim=dim, eps=eps, elementwise_affine=elementwise_affine)
        self._forward_method = self.dispatch_forward()

    def dispatch_forward(self):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_method(x)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)

    def forward_hip(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)

    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)

    def forward_npu(self, x: torch.Tensor) -> torch.Tensor:
        if _HAS_MINDIESD:
            try:
                from mindiesd import fast_layernorm

                return fast_layernorm(self, x)
            except ImportError as e:
                logger.warning_once(
                    "mindiesd.fast_layernorm import failed, falling back to FP32 layer_norm: %s",
                    e,
                )

        return self.forward_native(x)

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)


__all__ = [
    "LayerNorm32",
    "FastLayerNorm32",
]
